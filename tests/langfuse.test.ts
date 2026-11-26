// @vitest-environment node
// Module 5: every chat is traced in Langfuse (which agent answered, its tool
// calls, tokens, cost and time), the single support agent does everything
// with no handoff, and the eval's tool checks grade tool use. Mock model and
// a fake Langfuse, so no keys are needed.
import type { LanguageModelV4StreamPart as ModelStreamPart } from "@ai-sdk/provider";
import { convertArrayToReadableStream, MockLanguageModelV4 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { checkTools } from "@/evals/judge";
import { GOLDEN_SET } from "@/evals/golden-set";
import { finalValue } from "@/lib/ai/live";
import { toolCallsOf } from "@/lib/copper/assistant/progress";

let model: MockLanguageModelV4;
vi.mock("@ai-sdk/google", () => ({
  createGoogleGenerativeAI: () => () => model,
}));

type StreamPart = Parameters<
  typeof convertArrayToReadableStream<ModelStreamPart>
>[0][number];

const usage = {
  inputTokens: {
    total: 1000,
    noCache: 1000,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 100, text: 100, reasoning: undefined },
};

const finish = (reason: "stop" | "tool-calls"): StreamPart => ({
  type: "finish",
  finishReason: { unified: reason, raw: undefined },
  usage,
});

const says = (text: string): StreamPart[] => [
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: text },
  { type: "text-end", id: "t1" },
  finish("stop"),
];

const calls = (toolName: string, input: unknown): StreamPart[] => [
  { type: "tool-input-start", id: "call-1", toolName },
  { type: "tool-input-end", id: "call-1" },
  {
    type: "tool-call",
    toolCallId: "call-1",
    toolName,
    input: JSON.stringify(input),
  },
  finish("tool-calls"),
];

function modelStreaming(...steps: StreamPart[][]) {
  let step = 0;
  return new MockLanguageModelV4({
    doStream: async () => ({
      stream: convertArrayToReadableStream(steps[step++] ?? says("")),
    }),
  });
}

type Span = {
  name: string;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  attrs: Record<string, unknown>;
};
let spans: Span[];
let scores: Record<string, unknown>[];

/** An OTLP attribute value back to plain JSON. */
function plain(value: Record<string, unknown>): unknown {
  if ("arrayValue" in value)
    return (
      value.arrayValue as { values: Record<string, unknown>[] }
    ).values.map(plain);
  return Object.values(value)[0];
}

/** Fake Langfuse: records the OpenTelemetry spans and scores the app sends. */
function fakeLangfuse() {
  spans = [];
  scores = [];
  vi.stubEnv("LANGFUSE_PUBLIC_KEY", "pk-lf-test");
  vi.stubEnv("LANGFUSE_SECRET_KEY", "sk-lf-test");
  vi.stubEnv("LANGFUSE_BASE_URL", "https://langfuse.test");
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (url, init) => {
    const body = JSON.parse(String(init?.body));
    if (String(url).endsWith("/otel/v1/traces"))
      for (const s of body.resourceSpans[0].scopeSpans[0].spans)
        spans.push({
          ...s,
          attrs: Object.fromEntries(
            s.attributes.map(
              (a: { key: string; value: Record<string, unknown> }) => [
                a.key,
                plain(a.value),
              ],
            ),
          ),
        });
    else scores.push(...body.batch.map((e: { body: unknown }) => e.body));
    return new Response(JSON.stringify({ successes: [], errors: [] }), {
      status: 207,
    });
  });
}

const desk = { limitUsd: 50, refunds: [] };

async function ask(
  request: Parameters<typeof import("@/lib/copper/assistant/ask").askCopper>[0],
) {
  vi.resetModules();
  vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
  const { askCopper } = await import("@/lib/copper/assistant/ask");
  const { progress, traceId, observationId } = await askCopper(request);
  return { final: await finalValue(progress), traceId, observationId };
}

const named = (name: string) => spans.filter((s) => s.name === name);
const typed = (type: string) =>
  spans.filter((s) => s.attrs["langfuse.observation.type"] === type);
const read = (s: Span, key: string) =>
  JSON.parse(String(s.attrs[`langfuse.observation.${key}`]));

beforeEach(() => {
  vi.spyOn(console, "info").mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("Langfuse tracing", () => {
  it("traces a refund conversation: each agent, its tools, tokens, cost and time", async () => {
    const fetch = fakeLangfuse();
    model = modelStreaming(
      calls("transfer_to_refund_agent", { reason: "refund order 1001" }),
      calls("request_refund", { orderId: "1001" }),
      says("Order #1001 is refunded."),
    );
    const { final, traceId, observationId } = await ask({
      messages: [{ role: "user", content: "Refund order 1001" }],
      useTools: true,
      desk,
      mode: "multi",
      sessionId: "s-1",
    });

    expect(final.reply).toBe("Order #1001 is refunded.");
    expect(fetch).toHaveBeenCalledWith(
      "https://langfuse.test/api/public/otel/v1/traces",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "x-langfuse-ingestion-version": "4",
        }),
      }),
    );
    // Every observation is in this turn's trace, and carries its name,
    // session and tags.
    expect(spans.every((s) => s.traceId === traceId)).toBe(true);
    expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    for (const s of spans)
      expect(s.attrs).toMatchObject({
        "langfuse.trace.name": "ask-copper",
        "langfuse.session.id": "s-1",
        "langfuse.trace.tags": ["multi-agent"],
      });
    // The root observation is the trace: the question, the answer, and who
    // answered.
    const [root] = named("ask-copper").filter((s) => !s.parentSpanId);
    expect(root.spanId).toBe(observationId);
    expect(root.attrs["langfuse.observation.input"]).toBe("Refund order 1001");
    expect(root.attrs["langfuse.observation.output"]).toBe(
      "Order #1001 is refunded.",
    );
    expect(
      JSON.parse(String(root.attrs["langfuse.trace.metadata.agents"])),
    ).toEqual(["Copper", "Refund agent"]);
    expect(root.attrs["langfuse.trace.metadata.answeredBy"]).toBe(
      "Refund agent",
    );
    expect(
      JSON.parse(String(root.attrs["langfuse.trace.metadata.tools"])),
    ).toEqual(["transfer_to_refund_agent", "request_refund"]);
    // An agent observation per agent, the tools inside the agent that called them.
    const [copper] = named("Copper");
    const [refund] = named("Refund agent");
    expect(typed("agent")).toEqual([copper, refund]);
    expect(copper.parentSpanId).toBe(root.spanId);
    const [handoff] = named("transfer_to_refund_agent");
    expect(handoff.parentSpanId).toBe(copper.spanId);
    const [requested] = named("request_refund");
    expect(requested).toMatchObject({ parentSpanId: refund.spanId });
    expect(requested.attrs["langfuse.observation.type"]).toBe("tool");
    expect(read(requested, "input")).toEqual({ orderId: "1001" });
    expect(read(requested, "output")).toMatchObject({ status: "refunded" });
    expect(read(named("handoff")[0], "input")).toEqual({
      from: "Copper",
      to: "Refund agent",
    });
    // Each model call: model, tokens, cost and time, under its agent.
    const generations = typed("generation");
    expect(generations.map((g) => g.name)).toEqual([
      "ask-copper",
      "refund-agent",
    ]);
    const [, last] = generations;
    expect(last.parentSpanId).toBe(refund.spanId);
    expect(last.attrs["langfuse.observation.model.name"]).toBe(
      "google/gemini-3.5-flash-lite",
    );
    expect(read(last, "usage_details")).toEqual({
      input: 2000,
      output: 200,
      total: 2200,
    });
    expect(read(last, "cost_details").total).toBeGreaterThan(0);
    for (const s of spans)
      expect(
        BigInt(
          String((s as unknown as { endTimeUnixNano: string }).endTimeUnixNano),
        ),
      ).toBeGreaterThanOrEqual(
        BigInt(
          String(
            (s as unknown as { startTimeUnixNano: string }).startTimeUnixNano,
          ),
        ),
      );
  });

  it("sends nothing without Langfuse keys, and the chat still works", async () => {
    const fetch = vi.spyOn(globalThis, "fetch");
    model = modelStreaming(says("Hello!"));
    const { final } = await ask({
      messages: [{ role: "user", content: "Hi" }],
      useTools: true,
    });
    expect(final.reply).toBe("Hello!");
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("eval traces in a Langfuse experiment", () => {
  it("marks the turn as a golden-set item and scores its root observation", async () => {
    fakeLangfuse();
    model = modelStreaming(says("Polished Copper, Patina Teal and Night Sky."));
    const { traceId, observationId } = await ask({
      messages: [{ role: "user", content: "What colors?" }],
      useTools: true,
      desk,
      experiment: {
        id: "exp-1",
        name: "multi-agent abc1234",
        description: "Two agents",
        datasetId: "ds-1",
        itemId: "golden-colors",
        expectedOutput: { good: "Lists the three colors." },
      },
    });
    const root = spans.find((s) => s.spanId === observationId)!;
    expect(root.attrs).toMatchObject({
      "langfuse.environment": "experiment",
      "langfuse.experiment.id": "exp-1",
      "langfuse.experiment.name": "multi-agent abc1234",
      "langfuse.experiment.dataset.id": "ds-1",
      "langfuse.experiment.item.id": "golden-colors",
      "langfuse.experiment.item.root_observation_id": observationId,
      "langfuse.experiment.description": "Two agents",
    });
    expect(
      JSON.parse(
        String(root.attrs["langfuse.experiment.item.expected_output"]),
      ),
    ).toEqual({ good: "Lists the three colors." });
    // Child observations carry the experiment too, like OTel baggage.
    for (const s of spans)
      expect(s.attrs["langfuse.experiment.item.id"]).toBe("golden-colors");

    const { score, flushLangfuse } = await import("@/lib/ai/langfuse");
    score(traceId, "passed", 1, { dataType: "BOOLEAN", observationId });
    await flushLangfuse();
    expect(scores).toEqual([
      expect.objectContaining({
        traceId,
        observationId,
        name: "passed",
        value: 1,
        dataType: "BOOLEAN",
      }),
    ]);
  });
});

describe("one support agent (AGENT_MODE=single)", () => {
  it("refunds with its own tools and no handoff", async () => {
    fakeLangfuse();
    model = modelStreaming(
      calls("request_refund", { orderId: "1004" }),
      says("Order #1004 needs a supervisor's review."),
    );
    const { final } = await ask({
      messages: [{ role: "user", content: "Refund order 1004" }],
      useTools: true,
      desk,
      mode: "single",
    });
    expect(final.agent).toBe("support");
    expect(final.steps.some((s) => s.type === "handoff")).toBe(false);
    expect(toolCallsOf(final)[0]).toMatchObject({
      name: "request_refund",
      output: { status: "needs_approval", amountUsd: 149 },
    });
    // It has every tool, and no handoff tool.
    const tools = (
      model.doStreamCalls[0].tools as { name: string }[] | undefined
    )?.map((t) => t.name);
    expect(tools).toEqual(
      expect.arrayContaining([
        "get_products",
        "search_docs",
        "get_orders",
        "request_refund",
      ]),
    );
    expect(tools).not.toContain("transfer_to_refund_agent");
    expect(spans[0].attrs["langfuse.trace.tags"]).toEqual(["single-agent"]);
    expect(named("Support agent")).toHaveLength(1);
  });

  it("AGENT_MODE=single picks it in the app", async () => {
    vi.stubEnv("AGENT_MODE", "single");
    model = modelStreaming(says("Hi there."));
    const { final } = await ask({
      messages: [{ role: "user", content: "Hi" }],
      useTools: true,
      desk,
    });
    expect(final.agent).toBe("support");
  });
});

describe("the golden set's tool checks", () => {
  const refund = GOLDEN_SET.find((c) => c.id === "refund-under-limit")!;
  const which = GOLDEN_SET.find((c) => c.id === "refund-which-order")!;

  it("needs the right tool with the right order", () => {
    expect(
      checkTools(refund, [
        { name: "request_refund", input: { orderId: "#1001" } },
      ]),
    ).toEqual([]);
    expect(
      checkTools(refund, [
        { name: "request_refund", input: { orderId: "1002" } },
      ]),
    ).toEqual(['didn\'t call request_refund with {"orderId":"1001"}']);
  });

  it("flags a tool a good answer must not call", () => {
    expect(checkTools(which, [{ name: "get_orders" }])).toEqual([]);
    expect(
      checkTools(which, [
        { name: "request_refund", input: { orderId: "1001" } },
      ]),
    ).toEqual(["called request_refund, which it shouldn't"]);
  });

  it("has a unique id and a description of a good answer for every case", () => {
    const ids = GOLDEN_SET.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(GOLDEN_SET.length).toBeGreaterThanOrEqual(12);
    for (const c of GOLDEN_SET) expect(c.good.length).toBeGreaterThan(20);
  });
});
