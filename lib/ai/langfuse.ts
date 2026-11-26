// Langfuse (Module 5): every chat becomes a trace in Langfuse, with an agent
// observation for each agent, a tool observation for each tool call, and a
// generation for each model call (model, tokens, cost and time). The evals
// add an AI judge's scores to those traces and run the golden set as Langfuse
// experiments, one per version of Ask Copper.
// Observations are OpenTelemetry spans sent to Langfuse's OTLP endpoint
// (/api/public/otel/v1/traces) with Langfuse's attribute names
// (https://langfuse.com/integrations/native/opentelemetry), one batch per chat
// turn. Scores go to the ingestion API, which still takes scores. With no
// keys set (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)
// nothing is sent and the app works as before.
import { AsyncLocalStorage } from "node:async_hooks";
import { randomBytes, randomUUID } from "node:crypto";

/**
 * Where new observations go: a trace, the observation they belong to, and
 * the trace-wide attributes every observation in it carries.
 */
export type TraceRef = {
  traceId: string;
  parentId?: string;
  attributes?: Record<string, unknown>;
};

/** One golden-set item in a Langfuse experiment (Module 5 evals). */
export type ExperimentItem = {
  /** Unique per experiment run. */
  id: string;
  name: string;
  description?: string;
  datasetId?: string;
  /** The Langfuse dataset item this trace answers. */
  itemId: string;
  expectedOutput?: unknown;
  metadata?: Record<string, unknown>;
};

type Kind = "span" | "agent" | "tool" | "generation" | "event";

type SpanRecord = {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  start: string;
  end: string;
  attributes: Record<string, unknown>;
  error?: string;
};

const DEFAULT_BASE_URL = "https://cloud.langfuse.com";

export function isLangfuseConfigured(): boolean {
  return Boolean(
    process.env.LANGFUSE_PUBLIC_KEY?.trim() &&
    process.env.LANGFUSE_SECRET_KEY?.trim(),
  );
}

export function langfuseBaseUrl(): string {
  return (process.env.LANGFUSE_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );
}

function authHeader(): string {
  const pair = `${process.env.LANGFUSE_PUBLIC_KEY?.trim()}:${process.env.LANGFUSE_SECRET_KEY?.trim()}`;
  return `Basic ${Buffer.from(pair).toString("base64")}`;
}

/** Calls a Langfuse public API endpoint, e.g. langfuseApi("/api/public/projects"). */
export async function langfuseApi(
  path: string,
  init: {
    method?: string;
    body?: unknown;
    headers?: Record<string, string>;
  } = {},
): Promise<unknown> {
  const response = await fetch(`${langfuseBaseUrl()}${path}`, {
    method: init.method ?? (init.body ? "POST" : "GET"),
    headers: {
      Authorization: authHeader(),
      "Content-Type": "application/json",
      ...init.headers,
    },
    body: init.body ? JSON.stringify(init.body) : undefined,
    signal: AbortSignal.timeout(15_000),
  });
  const text = await response.text();
  if (!response.ok)
    throw new Error(
      `Langfuse ${path}: ${response.status} ${text.slice(0, 300)}`,
    );
  return text ? JSON.parse(text) : null;
}

let projectId: Promise<string | null> | undefined;

/** A link to a trace in Langfuse (needs sign-in unless the trace is public). */
export async function traceUrl(traceId: string): Promise<string | null> {
  projectId ??= langfuseApi("/api/public/projects")
    .then((r) => (r as { data: { id: string }[] }).data[0]?.id ?? null)
    .catch(() => null);
  const project = await projectId;
  return project
    ? `${langfuseBaseUrl()}/project/${project}/traces/${traceId}`
    : null;
}

// OpenTelemetry ids: 32 hex characters for a trace, 16 for a span.
const newTraceId = () => randomUUID().replace(/-/g, "");
const newSpanId = () => randomBytes(8).toString("hex");
/** Now, in nanoseconds since 1970, as OTLP wants it. */
const nowNanos = () =>
  (
    BigInt(Math.round((performance.timeOrigin + performance.now()) * 1000)) *
    BigInt(1000)
  ).toString();
const isoToNanos = (iso: string) =>
  (BigInt(Date.parse(iso)) * BigInt(1_000_000)).toString();

const finished: SpanRecord[] = [];
const scores: Record<string, unknown>[] = [];

const json = (value: unknown) =>
  typeof value === "string" ? value : JSON.stringify(value);

/** The observation's own attributes: type, input, output, metadata. */
function observation(
  kind: Kind,
  fields: {
    input?: unknown;
    output?: unknown;
    metadata?: Record<string, unknown>;
  },
) {
  const attributes: Record<string, unknown> = {
    "langfuse.observation.type": kind,
  };
  if (fields.input !== undefined)
    attributes["langfuse.observation.input"] = json(fields.input);
  if (fields.output !== undefined)
    attributes["langfuse.observation.output"] = json(fields.output);
  for (const [key, value] of Object.entries(fields.metadata ?? {}))
    if (value !== undefined)
      attributes[`langfuse.observation.metadata.${key}`] = json(value);
  return attributes;
}

function record(
  parent: TraceRef,
  name: string,
  start: string,
  attributes: Record<string, unknown>,
  opts: { spanId?: string; error?: string } = {},
) {
  if (!isLangfuseConfigured()) return;
  finished.push({
    traceId: parent.traceId,
    spanId: opts.spanId ?? newSpanId(),
    parentSpanId: parent.parentId,
    name,
    start,
    end: nowNanos(),
    attributes: { ...parent.attributes, ...attributes },
    error: opts.error,
  });
}

// The observation that model calls in this async flow belong to, so
// lib/ai/call.ts can attach each generation to the right agent without
// passing it around.
const current = new AsyncLocalStorage<TraceRef>();

/** Runs `fn` with model calls inside it traced under `ref`. */
export function withTrace<T>(ref: TraceRef | undefined, fn: () => T): T {
  return ref ? current.run(ref, fn) : fn();
}

/** The trace (and observation) the running code belongs to, if any. */
export function currentTrace(): TraceRef | undefined {
  return current.getStore();
}

export type TraceBody = {
  name: string;
  input?: unknown;
  sessionId?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  /** Makes it a golden-set item in a Langfuse experiment. */
  experiment?: ExperimentItem;
};

export type Trace = TraceRef & {
  parentId: string;
  /** Ends the trace's root observation with its output and final metadata. */
  end(output?: unknown, metadata?: Record<string, unknown>): void;
};

/** Starts a trace (one chat turn): a root observation everything else goes under. */
export function trace(body: TraceBody): Trace {
  const traceId = newTraceId();
  const rootId = newSpanId();
  const start = nowNanos();
  const exp = body.experiment;
  // On every observation, as an OpenTelemetry baggage processor would.
  const attributes: Record<string, unknown> = {
    "langfuse.trace.name": body.name,
    ...(body.sessionId && { "langfuse.session.id": body.sessionId }),
    ...(body.tags?.length && { "langfuse.trace.tags": body.tags }),
    ...(exp && {
      "langfuse.environment": "experiment",
      "langfuse.experiment.id": exp.id,
      "langfuse.experiment.name": exp.name,
      ...(exp.datasetId && { "langfuse.experiment.dataset.id": exp.datasetId }),
      "langfuse.experiment.item.id": exp.itemId,
      "langfuse.experiment.item.root_observation_id": rootId,
      ...Object.fromEntries(
        Object.entries(exp.metadata ?? {}).map(([k, v]) => [
          `langfuse.experiment.metadata.${k}`,
          json(v),
        ]),
      ),
    }),
  };
  const ref = { traceId, attributes };
  return {
    traceId,
    parentId: rootId,
    attributes,
    end(output, metadata) {
      const all = { ...body.metadata, ...metadata };
      record(
        ref,
        body.name,
        start,
        {
          ...observation("span", { input: body.input, output }),
          ...Object.fromEntries(
            Object.entries(all)
              .filter(([, v]) => v !== undefined)
              .map(([k, v]) => [`langfuse.trace.metadata.${k}`, json(v)]),
          ),
          ...(exp?.description && {
            "langfuse.experiment.description": exp.description,
          }),
          ...(exp?.expectedOutput !== undefined && {
            "langfuse.experiment.item.expected_output": json(
              exp.expectedOutput,
            ),
          }),
        },
        { spanId: rootId },
      );
    },
  };
}

/** An observation with a start and an end: an agent's turn, or a tool call. */
export function startSpan(
  parent: TraceRef,
  name: string,
  input?: unknown,
  metadata?: Record<string, unknown>,
  kind: "span" | "agent" | "tool" = "span",
) {
  const spanId = newSpanId();
  const start = nowNanos();
  return {
    /** Where the observation's children go. */
    ref: {
      traceId: parent.traceId,
      parentId: spanId,
      attributes: parent.attributes,
    } satisfies TraceRef,
    end(output?: unknown, opts: { error?: string } = {}) {
      record(
        parent,
        name,
        start,
        observation(kind, { input, output, metadata }),
        { spanId, error: opts.error },
      );
    },
  };
}

/** A point in time inside a trace, like a handoff between agents. */
export function event(parent: TraceRef, name: string, input?: unknown) {
  record(parent, name, nowNanos(), observation("event", { input }));
}

export type GenerationBody = {
  name: string;
  model: string;
  startTime: string;
  input?: unknown;
  output?: unknown;
  inputTokens?: number | null;
  outputTokens?: number | null;
  costUsd?: number | null;
  error?: string;
  metadata?: Record<string, unknown>;
};

// Running totals per trace, so the evals can report cost and tokens.
const totals = new Map<
  string,
  { inputTokens: number; outputTokens: number; costUsd: number; calls: number }
>();

/** Tokens and cost of every model call recorded under a trace so far. */
export function traceTotals(traceId: string) {
  return (
    totals.get(traceId) ?? {
      inputTokens: 0,
      outputTokens: 0,
      costUsd: 0,
      calls: 0,
    }
  );
}

/** One model call: which model, its tokens, its cost and how long it took. */
export function generation(parent: TraceRef, body: GenerationBody) {
  const input = body.inputTokens ?? 0;
  const output = body.outputTokens ?? 0;
  const sum = traceTotals(parent.traceId);
  totals.set(parent.traceId, {
    inputTokens: sum.inputTokens + input,
    outputTokens: sum.outputTokens + output,
    costUsd: sum.costUsd + (body.costUsd ?? 0),
    calls: sum.calls + 1,
  });
  record(
    parent,
    body.name,
    isoToNanos(body.startTime),
    {
      ...observation("generation", body),
      "langfuse.observation.model.name": body.model,
      "langfuse.observation.usage_details": JSON.stringify({
        input,
        output,
        total: input + output,
      }),
      ...(body.costUsd != null && {
        "langfuse.observation.cost_details": JSON.stringify({
          total: body.costUsd,
        }),
      }),
    },
    { error: body.error },
  );
}

/** A score on a trace (or one observation in it), like the AI judge's grade. */
export function score(
  traceId: string,
  name: string,
  value: number,
  opts: {
    comment?: string;
    dataType?: "NUMERIC" | "BOOLEAN";
    observationId?: string;
  } = {},
) {
  if (!isLangfuseConfigured()) return;
  scores.push({
    id: randomUUID(),
    traceId,
    observationId: opts.observationId,
    name,
    value,
    dataType: opts.dataType ?? "NUMERIC",
    comment: opts.comment,
  });
}

/** Drops a trace that hasn't been sent yet (the evals drop a failed attempt before retrying). */
export function discardTrace(traceId: string) {
  for (let i = finished.length - 1; i >= 0; i--)
    if (finished[i].traceId === traceId) finished.splice(i, 1);
  for (let i = scores.length - 1; i >= 0; i--)
    if (scores[i].traceId === traceId) scores.splice(i, 1);
  totals.delete(traceId);
}

/** An OTLP attribute value. */
function anyValue(value: unknown): Record<string, unknown> {
  if (Array.isArray(value))
    return { arrayValue: { values: value.map(anyValue) } };
  if (typeof value === "boolean") return { boolValue: value };
  if (typeof value === "number")
    return Number.isInteger(value)
      ? { intValue: String(value) }
      : { doubleValue: value };
  return { stringValue: json(value) };
}

/** The finished observations as an OTLP/JSON export request. */
export function otlpPayload(spans: SpanRecord[]) {
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [
            { key: "service.name", value: { stringValue: "ask-copper" } },
          ],
        },
        scopeSpans: [
          {
            scope: { name: "copper-langfuse", version: "1.0.0" },
            spans: spans.map((s) => ({
              traceId: s.traceId,
              spanId: s.spanId,
              ...(s.parentSpanId && { parentSpanId: s.parentSpanId }),
              name: s.name,
              kind: 1,
              startTimeUnixNano: s.start,
              endTimeUnixNano: s.end,
              attributes: Object.entries({
                ...s.attributes,
                ...(s.error && {
                  "langfuse.observation.level": "ERROR",
                  "langfuse.observation.status_message": s.error,
                }),
              }).map(([key, value]) => ({ key, value: anyValue(value) })),
              status: s.error ? { code: 2, message: s.error } : { code: 1 },
            })),
          },
        ],
      },
    ],
  };
}

/** Sends everything finished so far. Never throws: tracing must not break a chat. */
export async function flushLangfuse(): Promise<void> {
  if (!isLangfuseConfigured()) return;
  const spans = finished.splice(0);
  const batch = scores.splice(0);
  const sends: Promise<unknown>[] = [];
  if (spans.length)
    sends.push(
      langfuseApi("/api/public/otel/v1/traces", {
        body: otlpPayload(spans),
        headers: { "x-langfuse-ingestion-version": "4" },
      }),
    );
  if (batch.length)
    sends.push(
      langfuseApi("/api/public/ingestion", {
        body: {
          batch: batch.map((body) => ({
            id: randomUUID(),
            timestamp: new Date().toISOString(),
            type: "score-create",
            body,
          })),
        },
      }).then((result) => {
        const errors = (result as { errors?: unknown[] } | null)?.errors;
        if (errors?.length)
          console.warn(
            "[langfuse] rejected scores:",
            JSON.stringify(errors[0]).slice(0, 300),
          );
      }),
    );
  for (const result of await Promise.allSettled(sends))
    if (result.status === "rejected")
      console.warn(
        "[langfuse] could not send:",
        result.reason instanceof Error ? result.reason.message : result.reason,
      );
}
