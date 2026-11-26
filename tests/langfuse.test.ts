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

type Event = { type: string; body: Record<string, unknown> };
let sent: Event[];

/** Fake Langfuse: records every batch the app sends. */
function fakeLangfuse() {
  sent = [];
  vi.stubEnv("LANGFUSE_PUBLIC_KEY", "pk-lf-test");
  vi.stubEnv("LANGFUSE_SECRET_KEY", "sk-lf-test");
  vi.stubEnv("LANGFUSE_BASE_URL", "https://langfuse.test");
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (_url, init) => {
      sent.push(...JSON.parse(String(init?.body)).batch);
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
  const { progress, traceId } = await askCopper(request);
  return { final: await finalValue(progress), traceId };
}

const of = (type: string) => sent.filter((e) => e.type === type);
const named = (type: string, name: string) =>
  of(type).filter((e) => e.body.name === name);

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
    const { final, traceId } = await ask({
      messages: [{ role: "user", content: "Refund order 1001" }],
      useTools: true,
      desk,
      mode: "multi",
      sessionId: "s-1",
    });

    expect(final.reply).toBe("Order #1001 is refunded.");
    expect(fetch).toHaveBeenCalledWith(
      "https://langfuse.test/api/public/ingestion",
      expect.objectContaining({ method: "POST" }),
    );
    // One trace for the turn, with its question, answer and who answered.
    const traces = of("trace-create").filter((e) => e.body.id === traceId);
    expect(traces[0].body).toMatchObject({
      name: "ask-copper",
      input: "Refund order 1001",
      sessionId: "s-1",
      tags: ["multi-agent"],
    });
    expect(traces.at(-1)!.body).toMatchObject({
      output: "Order #1001 is refunded.",
      metadata: expect.objectContaining({
        agents: ["Copper", "Refund agent"],
        answeredBy: "Refund agent",
        tools: ["transfer_to_refund_agent", "request_refund"],
        handoffs: 1,
      }),
    });
    // A span per agent, the tools inside the agent that called them.
    const copper = named("span-create", "Copper")[0].body;
    const refund = named("span-create", "Refund agent")[0].body;
    expect(
      named("span-create", "transfer_to_refund_agent")[0].body,
    ).toMatchObject({ parentObservationId: copper.id });
    expect(named("span-create", "request_refund")[0].body).toMatchObject({
      parentObservationId: refund.id,
      input: { orderId: "1001" },
      output: expect.objectContaining({ status: "refunded" }),
    });
    expect(named("event-create", "handoff")[0].body.input).toEqual({
      from: "Copper",
      to: "Refund agent",
    });
    // Each model call: model, tokens, cost and time, under its agent.
    const generations = of("generation-create").map((e) => e.body);
    expect(generations.map((g) => g.name)).toEqual([
      "ask-copper",
      "refund-agent",
    ]);
    expect(generations[1]).toMatchObject({
      parentObservationId: refund.id,
      model: "google/gemini-3.5-flash-lite",
      usageDetails: { input: 2000, output: 200, total: 2200 },
      costDetails: { total: expect.any(Number) },
      startTime: expect.any(String),
      endTime: expect.any(String),
    });
    expect(
      sent.every((e) => !("traceId" in e.body) || e.body.traceId === traceId),
    ).toBe(true);
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
    expect(of("trace-create")[0].body.tags).toEqual(["single-agent"]);
    expect(named("span-create", "Support agent")).toHaveLength(1);
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
