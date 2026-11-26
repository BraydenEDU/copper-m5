// Langfuse (Module 5): every chat becomes a trace in Langfuse, with a span
// for each agent, a span for each tool call, and a generation for each model
// call (model, tokens, cost and time). The evals add an AI judge's scores to
// those traces and group them into dataset runs.
// Events are queued here and sent in one batch by flushLangfuse(), through
// Langfuse's public API (https://api.reference.langfuse.com). With no keys set
// (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL) nothing is
// sent and the app works as before.
import { AsyncLocalStorage } from "node:async_hooks";

/** Where new observations go: a trace, and the span they belong to. */
export type TraceRef = { traceId: string; parentId?: string };

type IngestionEvent = {
  id: string;
  timestamp: string;
  type:
    | "trace-create"
    | "span-create"
    | "generation-create"
    | "event-create"
    | "score-create";
  body: Record<string, unknown>;
};

const DEFAULT_BASE_URL = "https://cloud.langfuse.com";

export function isLangfuseConfigured(): boolean {
  return Boolean(
    process.env.LANGFUSE_PUBLIC_KEY?.trim() &&
    process.env.LANGFUSE_SECRET_KEY?.trim(),
  );
}

function baseUrl(): string {
  return (process.env.LANGFUSE_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );
}

function authHeader(): string {
  const pair = `${process.env.LANGFUSE_PUBLIC_KEY?.trim()}:${process.env.LANGFUSE_SECRET_KEY?.trim()}`;
  return `Basic ${Buffer.from(pair).toString("base64")}`;
}

/** A link to a trace in the Langfuse app (works once you're signed in, or for anyone once the trace is public). */
export function traceUrl(traceId: string): string {
  return `${baseUrl()}/trace/${traceId}`;
}

export const newId = () => crypto.randomUUID();
const now = () => new Date().toISOString();

const queue: IngestionEvent[] = [];

function enqueue(type: IngestionEvent["type"], body: Record<string, unknown>) {
  if (!isLangfuseConfigured()) return;
  queue.push({ id: newId(), timestamp: now(), type, body });
}

/** Calls a Langfuse public API endpoint, e.g. langfuseApi("/api/public/scores", {...}). */
export async function langfuseApi(
  path: string,
  init: { method?: string; body?: unknown } = {},
): Promise<unknown> {
  const response = await fetch(`${baseUrl()}${path}`, {
    method: init.method ?? (init.body ? "POST" : "GET"),
    headers: {
      Authorization: authHeader(),
      "Content-Type": "application/json",
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

/** Sends everything queued so far. Never throws: tracing must not break a chat. */
export async function flushLangfuse(): Promise<void> {
  if (!queue.length || !isLangfuseConfigured()) return;
  const batch = queue.splice(0);
  try {
    const result = (await langfuseApi("/api/public/ingestion", {
      body: { batch },
    })) as { errors?: { message?: string; error?: unknown }[] } | null;
    if (result?.errors?.length)
      console.warn(
        "[langfuse]",
        `${result.errors.length} of ${batch.length} events were rejected:`,
        JSON.stringify(result.errors[0]).slice(0, 300),
      );
  } catch (error) {
    console.warn(
      "[langfuse] could not send traces:",
      error instanceof Error ? error.message : error,
    );
  }
}

// The span that model calls in this async flow belong to, so lib/ai/call.ts
// can attach each generation to the right agent without passing it around.
const current = new AsyncLocalStorage<TraceRef>();

/** Runs `fn` with model calls inside it traced under `ref`. */
export function withTrace<T>(ref: TraceRef | undefined, fn: () => T): T {
  return ref ? current.run(ref, fn) : fn();
}

/** The trace (and span) the running code belongs to, if any. */
export function currentTrace(): TraceRef | undefined {
  return current.getStore();
}

export type TraceBody = {
  name?: string;
  input?: unknown;
  output?: unknown;
  sessionId?: string;
  userId?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  public?: boolean;
};

/** Starts a trace (one chat turn). Call again with the same id to add its output. */
export function trace(body: TraceBody, id = newId()): TraceRef {
  enqueue("trace-create", { id, timestamp: now(), ...body });
  return { traceId: id };
}

/** Updates a trace that already exists (Langfuse merges by id). */
export function updateTrace(ref: TraceRef, body: TraceBody) {
  enqueue("trace-create", { id: ref.traceId, ...body });
}

/** A span: a named piece of work (an agent's turn, a tool call) with a start and an end. */
export function startSpan(
  parent: TraceRef,
  name: string,
  input?: unknown,
  metadata?: Record<string, unknown>,
) {
  const id = newId();
  const startTime = now();
  return {
    /** Where the span's children go. */
    ref: { traceId: parent.traceId, parentId: id } satisfies TraceRef,
    end(output?: unknown, opts: { error?: string } = {}) {
      enqueue("span-create", {
        id,
        traceId: parent.traceId,
        parentObservationId: parent.parentId,
        name,
        startTime,
        endTime: now(),
        input,
        output,
        metadata,
        ...(opts.error && { level: "ERROR", statusMessage: opts.error }),
      });
    },
  };
}

/** A point in time inside a trace, like a handoff between agents. */
export function event(
  parent: TraceRef,
  name: string,
  input?: unknown,
  metadata?: Record<string, unknown>,
) {
  enqueue("event-create", {
    id: newId(),
    traceId: parent.traceId,
    parentObservationId: parent.parentId,
    name,
    startTime: now(),
    input,
    metadata,
  });
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
  const cost = body.costUsd ?? 0;
  const sum = traceTotals(parent.traceId);
  totals.set(parent.traceId, {
    inputTokens: sum.inputTokens + input,
    outputTokens: sum.outputTokens + output,
    costUsd: sum.costUsd + cost,
    calls: sum.calls + 1,
  });
  enqueue("generation-create", {
    id: newId(),
    traceId: parent.traceId,
    parentObservationId: parent.parentId,
    name: body.name,
    model: body.model,
    startTime: body.startTime,
    endTime: now(),
    input: body.input,
    output: body.output,
    metadata: body.metadata,
    usageDetails: { input, output, total: input + output },
    ...(body.costUsd != null && {
      costDetails: { total: body.costUsd },
    }),
    ...(body.error && { level: "ERROR", statusMessage: body.error }),
  });
}

/** A score on a trace, like the AI judge's grade. */
export function score(
  traceId: string,
  name: string,
  value: number,
  opts: { comment?: string; dataType?: "NUMERIC" | "BOOLEAN" } = {},
) {
  enqueue("score-create", {
    id: newId(),
    traceId,
    name,
    value,
    dataType: opts.dataType ?? "NUMERIC",
    comment: opts.comment,
  });
}
