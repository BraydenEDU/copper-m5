"use server";

// The server action the Ask Copper chat calls. It streams the conversation to
// the model through callStream (lib/ai/call.ts), with or without the tools,
// and sends back each step as it happens, so the panel can show Copper
// thinking and using its tools.
// Module 4: on the Support page the chat also sends the Refund Console's
// "desk" (the auto-refund limit and the refunds so far) and which agent is
// talking. Copper can hand the conversation to the Refund agent, a second
// agent with its own instructions and tools, and the Refund agent can hand it
// back. The agent that ends the turn answers the next message too.
// Module 5: every turn is a trace in Langfuse (lib/ai/langfuse.ts): a span for
// each agent, a span for each tool call, and each model call with its tokens,
// cost and time. AGENT_MODE=single (or `mode: "single"`) swaps the two agents
// for one support agent with all the tools and no handoff.
import { hasToolCall, stepCountIs } from "ai";
import { callStream, THOUGHT_SUMMARIES } from "@/lib/ai/call";
import {
  event,
  flushLangfuse,
  startSpan,
  trace,
  updateTrace,
  withTrace,
} from "@/lib/ai/langfuse";
import { createLive, type Live } from "@/lib/ai/live";
import {
  COPPER_HANDOFF,
  COPPER_INSTRUCTIONS,
  REFUND_AGENT_INSTRUCTIONS,
  SUPPORT_AGENT_INSTRUCTIONS,
} from "@/lib/copper/assistant/instructions";
import {
  type Agent,
  AGENT_NAMES,
  type AgentMode,
  type Progress,
  type ToolStep,
} from "@/lib/copper/assistant/progress";
import {
  copperTools,
  refundAgentTools,
  supportAgentTools,
  transferToRefundAgent,
} from "@/lib/copper/assistant/tools";
import type { Desk } from "@/lib/copper/orders";

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type AskCopperRequest = {
  messages: ChatMessage[];
  useTools: boolean;
  /** The Refund Console's limit and refunds (Support page only). */
  desk?: Desk;
  /** Who answers first: whoever ended the last turn. Copper by default. */
  agent?: Agent;
  /** Two agents (default) or one support agent. Defaults to AGENT_MODE. */
  mode?: AgentMode;
  /** Groups a conversation's turns in Langfuse. */
  sessionId?: string;
  /** Extra Langfuse tags, like "eval". */
  tags?: string[];
};

export type AskCopperResult = {
  progress: Live<Progress>;
  /** This turn's trace in Langfuse. */
  traceId: string;
};

/** AGENT_MODE=single runs one support agent; anything else, the two agents. */
function defaultMode(): AgentMode {
  return process.env.AGENT_MODE?.trim() === "single" ? "single" : "multi";
}

/** Waits at most this long to send the trace, so a slow Langfuse can't hold up a chat. */
const FLUSH_TIMEOUT_MS = 3000;

/** At most this many model steps (tool calls plus the reply) per agent. */
const MAX_STEPS = 5;

/** Copper → Refund agent → Copper at most, so a turn can't bounce forever. */
const MAX_HANDOFFS = 2;

const HANDOFF_TOOLS: Record<string, Agent> = {
  transfer_to_refund_agent: "refund",
  transfer_to_copper: "copper",
};

// A handoff ends the agent's turn: the other agent answers.
const STOP = [
  stepCountIs(MAX_STEPS),
  hasToolCall("transfer_to_refund_agent", "transfer_to_copper"),
];

const SORRY = "Sorry, something went wrong on my end. Please try again.";

/** One agent's call: its instructions and only its own tools. */
function agentCall(agent: Agent, messages: ChatMessage[], desk?: Desk) {
  if (agent === "support" && desk)
    return callStream(
      { module: 5, feature: "support-agent" },
      {
        messages,
        providerOptions: THOUGHT_SUMMARIES,
        instructions: SUPPORT_AGENT_INSTRUCTIONS,
        tools: supportAgentTools(desk),
        stopWhen: STOP,
      },
    );
  if (agent === "refund" && desk)
    return callStream(
      { module: 4, feature: "refund-agent" },
      {
        messages,
        providerOptions: THOUGHT_SUMMARIES,
        instructions: REFUND_AGENT_INSTRUCTIONS,
        tools: refundAgentTools(desk),
        stopWhen: STOP,
      },
    );
  return callStream(
    { module: 2, feature: "ask-copper" },
    {
      messages,
      providerOptions: THOUGHT_SUMMARIES,
      instructions: desk
        ? `${COPPER_INSTRUCTIONS}\n\n${COPPER_HANDOFF}`
        : COPPER_INSTRUCTIONS,
      tools: desk
        ? { ...copperTools, transfer_to_refund_agent: transferToRefundAgent }
        : copperTools,
      stopWhen: STOP,
    },
  );
}

export async function askCopper({
  messages,
  useTools,
  desk,
  agent = "copper",
  mode = defaultMode(),
  sessionId,
  tags = [],
}: AskCopperRequest): Promise<AskCopperResult> {
  const { live, update, done } = createLive<Progress>();
  const first: Agent =
    desk && useTools
      ? mode === "single"
        ? "support"
        : agent === "support"
          ? "copper"
          : agent
      : "copper";
  const turn = trace({
    name: "ask-copper",
    input: messages.at(-1)?.content,
    sessionId,
    tags: [mode === "single" ? "single-agent" : "multi-agent", ...tags],
    metadata: {
      mode,
      useTools,
      page: desk ? "support" : "home",
      limitUsd: desk?.limitUsd,
      history: messages.slice(0, -1),
    },
  });
  const progress: Progress = {
    steps: [],
    reply: "",
    status: "working",
    agent: first,
  };
  // A fresh copy each time, so earlier updates never change underneath the
  // panel.
  const snapshot = (): Progress => ({
    ...progress,
    steps: progress.steps.map((step) => ({ ...step })),
  });
  // Tool call ids are per agent call, so each call's are kept apart.
  let call = 0;
  const toolStep = (id: string) =>
    progress.steps.find(
      (step): step is ToolStep =>
        step.type === "tool" && step.id === `${call}:${id}`,
    );

  // Each tool call is a span in Langfuse, from its call to its result.
  const toolSpans = new Map<string, ReturnType<typeof startSpan>>();
  const agentsUsed: Agent[] = [];

  /** Reads one agent's stream into progress. Returns who to hand off to. */
  async function follow(
    current: Agent,
    result: ReturnType<typeof callStream>,
    span: ReturnType<typeof startSpan>,
  ): Promise<Agent | undefined> {
    let handoff: Agent | undefined;
    for await (const part of result.fullStream) {
      switch (part.type) {
        case "reasoning-delta": {
          const last = progress.steps.at(-1);
          if (last?.type === "thinking" && last.agent === current)
            last.text += part.text;
          else
            progress.steps.push({
              type: "thinking",
              text: part.text,
              agent: current,
            });
          break;
        }
        case "tool-input-start":
          progress.steps.push({
            type: "tool",
            id: `${call}:${part.id}`,
            agent: current,
            name: part.toolName,
            state: "calling",
          });
          break;
        case "tool-call": {
          toolSpans.set(
            `${call}:${part.toolCallId}`,
            startSpan(span.ref, part.toolName, part.input, { agent: current }),
          );
          const step = toolStep(part.toolCallId);
          if (step) step.input = part.input;
          else
            progress.steps.push({
              type: "tool",
              id: `${call}:${part.toolCallId}`,
              agent: current,
              name: part.toolName,
              state: "calling",
              input: part.input,
            });
          break;
        }
        case "tool-result": {
          const step = toolStep(part.toolCallId);
          if (step) Object.assign(step, { output: part.output, state: "done" });
          toolSpans.get(`${call}:${part.toolCallId}`)?.end(part.output);
          if (part.toolName in HANDOFF_TOOLS)
            handoff = HANDOFF_TOOLS[part.toolName];
          break;
        }
        case "tool-error": {
          const step = toolStep(part.toolCallId);
          const message =
            part.error instanceof Error
              ? part.error.message
              : String(part.error);
          if (step) Object.assign(step, { output: message, state: "error" });
          toolSpans
            .get(`${call}:${part.toolCallId}`)
            ?.end(undefined, { error: message });
          break;
        }
        case "text-delta":
          progress.reply += part.text;
          break;
        case "error":
          throw part.error;
        default:
          continue;
      }
      update(snapshot());
    }
    return handoff;
  }

  /** One agent's turn, traced as a span with its model call and tools inside. */
  async function runAgent(
    current: Agent,
    run: () => ReturnType<typeof callStream>,
  ) {
    agentsUsed.push(current);
    const span = startSpan(turn, AGENT_NAMES[current], messages.at(-1), {
      agent: current,
      mode,
    });
    try {
      const next = await withTrace(span.ref, () =>
        follow(current, run(), span),
      );
      span.end(progress.reply || undefined);
      return next;
    } catch (error) {
      span.end(undefined, {
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /** Adds the outcome to the trace and sends it to Langfuse. */
  async function finishTrace() {
    const tools = progress.steps.flatMap((step) =>
      step.type === "tool" ? [step.name] : [],
    );
    updateTrace(turn, {
      output: progress.reply,
      metadata: {
        mode,
        useTools,
        page: desk ? "support" : "home",
        limitUsd: desk?.limitUsd,
        history: messages.slice(0, -1),
        agents: agentsUsed.map((a) => AGENT_NAMES[a]),
        answeredBy: AGENT_NAMES[progress.agent ?? "copper"],
        tools,
        handoffs: progress.steps.filter((s) => s.type === "handoff").length,
        status: progress.status,
      },
    });
    await Promise.race([
      flushLangfuse(),
      new Promise((resolve) => setTimeout(resolve, FLUSH_TIMEOUT_MS)),
    ]);
  }

  // Read the stream in the background, so the panel gets the first update now.
  void (async () => {
    try {
      if (!useTools) {
        // Without tools, send no instructions either: "always use your
        // tools" with no tools makes the model answer with nothing.
        await runAgent("copper", () =>
          callStream(
            { module: 2, feature: "ask-copper" },
            { messages, providerOptions: THOUGHT_SUMMARIES },
          ),
        );
      } else {
        let current = first;
        for (let handoffs = 0; ; handoffs++) {
          const agentNow = current;
          const next = await runAgent(agentNow, () =>
            agentCall(agentNow, messages, desk),
          );
          if (!next || next === current || handoffs >= MAX_HANDOFFS) break;
          event(turn, "handoff", {
            from: AGENT_NAMES[current],
            to: AGENT_NAMES[next],
          });
          progress.steps.push({ type: "handoff", from: current, to: next });
          progress.agent = current = next;
          progress.reply = "";
          call++;
          update(snapshot());
        }
      }
      progress.status = "done";
      if (!progress.reply.trim()) progress.reply = SORRY;
    } catch {
      progress.status = "error";
      if (!progress.reply.trim()) progress.reply = SORRY;
    }
    await finishTrace();
    done(snapshot());
  })();

  return { progress: live, traceId: turn.traceId };
}
