// Runs the golden set (Module 5) on Ask Copper, has the AI judge grade every
// answer, and keeps it all in Langfuse:
// - each answer is a normal Ask Copper trace, tagged "eval";
// - the judge's grade goes on that trace as scores (judge_score, passed,
//   tool_use);
// - the golden set is a Langfuse dataset, and each version's run (two agents
//   or one support agent) is a Langfuse experiment on it, so Langfuse compares
//   their scores, cost and time side by side.
import { execSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  discardTrace,
  type ExperimentItem,
  flushLangfuse,
  isLangfuseConfigured,
  langfuseApi,
  langfuseBaseUrl,
  score,
  trace,
  traceTotals,
  withTrace,
} from "@/lib/ai/langfuse";
import { finalValue } from "@/lib/ai/live";
import { askCopper } from "@/lib/copper/assistant/ask";
import {
  type AgentMode,
  type Progress,
  toolCallsOf,
} from "@/lib/copper/assistant/progress";
import { DEFAULT_LIMIT_USD } from "@/lib/copper/orders";
import { GOLDEN_SET, type GoldenCase } from "@/evals/golden-set";
import { checkTools, judge, type Verdict } from "@/evals/judge";

export const DATASET = "ask-copper-golden-set";

export const MODE_LABELS: Record<AgentMode, string> = {
  multi: "Two agents (Copper + Refund agent)",
  single: "One support agent",
};

export type CaseResult = {
  id: string;
  category: GoldenCase["category"];
  question: string;
  reply: string;
  tools: string[];
  toolProblems: string[];
  verdict: Verdict | null;
  passed: boolean;
  latencyMs: number;
  costUsd: number;
  tokens: number;
  traceId: string;
  error?: string;
};

export type ModeResult = {
  mode: AgentMode;
  runName: string;
  cases: CaseResult[];
  passRate: number;
  avgJudgeScore: number;
  avgCostUsd: number;
  avgLatencyMs: number;
  totalCostUsd: number;
  totalTokens: number;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function gitLabel(): string {
  if (process.env.EVAL_RUN_LABEL) return process.env.EVAL_RUN_LABEL;
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "local";
  }
}

/** The golden set as a Langfuse dataset (created once, items kept in sync). Returns its id. */
async function syncDataset(): Promise<string | undefined> {
  if (!isLangfuseConfigured()) return;
  try {
    await langfuseApi("/api/public/datasets", {
      body: {
        name: DATASET,
        description:
          "Ask Copper golden set (Module 5): customer questions and what a good answer does.",
      },
    });
  } catch {
    // Already there.
  }
  for (const c of GOLDEN_SET) {
    try {
      await langfuseApi("/api/public/dataset-items", {
        body: {
          datasetName: DATASET,
          id: `golden-${c.id}`,
          input: { messages: c.messages, desk: c.desk, agent: c.agent },
          expectedOutput: {
            good: c.good,
            mustCall: c.mustCall,
            mustNotCall: c.mustNotCall,
          },
          metadata: { category: c.category },
        },
      });
    } catch (error) {
      console.warn("[evals] dataset item", c.id, (error as Error).message);
    }
  }
  try {
    const dataset = (await langfuseApi(
      `/api/public/v2/datasets/${encodeURIComponent(DATASET)}`,
    )) as { id: string };
    return dataset.id;
  } catch {
    return undefined;
  }
}

/** One golden-set case as an item of this run's Langfuse experiment. */
function experimentItem(
  experiment: Omit<ExperimentItem, "itemId">,
  c: GoldenCase,
): ExperimentItem {
  return {
    ...experiment,
    itemId: `golden-${c.id}`,
    expectedOutput: {
      good: c.good,
      mustCall: c.mustCall,
      mustNotCall: c.mustNotCall,
    },
  };
}

/** Asks Ask Copper one golden-set question, the way the Support page does. */
async function answer(
  mode: AgentMode,
  c: GoldenCase,
  runName: string,
  experiment: ExperimentItem,
) {
  const started = performance.now();
  const { progress, traceId, observationId } = await askCopper({
    messages: c.messages,
    useTools: true,
    desk: c.desk ?? { limitUsd: DEFAULT_LIMIT_USD, refunds: [] },
    agent: c.agent,
    mode,
    sessionId: runName,
    tags: ["eval", `case:${c.id}`],
    experiment,
    holdTrace: true,
  });
  const final: Progress = await finalValue(progress);
  return {
    final,
    traceId,
    observationId,
    latencyMs: Math.round(performance.now() - started),
  };
}

async function runCase(
  mode: AgentMode,
  c: GoldenCase,
  runName: string,
  experiment: Omit<ExperimentItem, "itemId">,
): Promise<CaseResult> {
  const item = experimentItem(experiment, c);
  let attempt = 0;
  let result = await answer(mode, c, runName, item);
  // A busy model (rate limit) can fail a turn; try it again before grading.
  // Only the final attempt is graded and kept: a Langfuse experiment has one
  // trace per item.
  while (result.final.status === "error" && attempt++ < 2) {
    discardTrace(result.traceId);
    await sleep(10_000 * attempt);
    result = await answer(mode, c, runName, item);
  }
  const { final, traceId, observationId, latencyMs } = result;
  const calls = toolCallsOf(final);
  const toolProblems = checkTools(c, calls);

  let verdict: Verdict | null = null;
  let error: string | undefined;
  if (final.status === "error") error = "Ask Copper failed to answer";
  else {
    // The judge's own call goes in its own trace, so it doesn't count toward
    // the answer's cost or time.
    const judgeTrace = trace({
      name: "eval-judge",
      input: { case: c.id, reply: final.reply },
      tags: ["eval-judge", mode === "single" ? "single-agent" : "multi-agent"],
      metadata: { answerTraceId: traceId, mode },
    });
    for (let tries = 0; !verdict && tries < 3; tries++) {
      try {
        verdict = await withTrace(judgeTrace, () =>
          judge(c, calls, final.reply),
        );
      } catch (e) {
        error = `Judge failed: ${(e as Error).message}`;
        await sleep(5_000 * (tries + 1));
      }
    }
    if (verdict) error = undefined;
    judgeTrace.end(verdict ?? error);
  }

  const passed = Boolean(verdict?.pass) && toolProblems.length === 0;
  const comment = [verdict?.reasoning, ...toolProblems]
    .filter(Boolean)
    .join(" ");
  // On the trace's root observation, where Langfuse experiments read them.
  if (verdict)
    score(traceId, "judge_score", (verdict.score - 1) / 4, {
      comment: `${verdict.score}/5. ${verdict.reasoning}`,
      observationId,
    });
  score(traceId, "tool_use", toolProblems.length ? 0 : 1, {
    dataType: "BOOLEAN",
    comment: toolProblems.join("; ") || "Used the right tools.",
    observationId,
  });
  score(traceId, "passed", passed ? 1 : 0, {
    dataType: "BOOLEAN",
    comment: error ?? comment,
    observationId,
  });
  await flushLangfuse();

  const totals = traceTotals(traceId);
  return {
    id: c.id,
    category: c.category,
    question: c.messages.at(-1)!.content,
    reply: final.reply,
    tools: calls.map((t) => t.name),
    toolProblems,
    verdict,
    passed,
    latencyMs,
    costUsd: totals.costUsd,
    tokens: totals.inputTokens + totals.outputTokens,
    traceId,
    error,
  };
}

const average = (values: number[]) =>
  values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;

/** Runs the whole golden set on one version of Ask Copper. */
export async function runMode(
  mode: AgentMode,
  cases: GoldenCase[] = GOLDEN_SET,
  datasetId?: string,
): Promise<ModeResult> {
  const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
  const runName = `${mode}-agent ${gitLabel()} ${stamp}`;
  // This run is one Langfuse experiment on the golden-set dataset.
  const experiment = {
    id: randomUUID(),
    name: runName,
    description: MODE_LABELS[mode],
    datasetId,
    metadata: { mode, commit: gitLabel() },
  };
  const results: CaseResult[] = [];
  for (const c of cases) {
    const r = await runCase(mode, c, runName, experiment);
    console.info(
      `[evals] ${mode} ${r.passed ? "PASS" : "FAIL"} ${c.id} (${r.verdict?.score ?? "-"}/5, ${(r.latencyMs / 1000).toFixed(1)}s, $${r.costUsd.toFixed(5)})${r.passed ? "" : ` ${r.error ?? r.toolProblems.join("; ") ?? ""} ${r.verdict?.reasoning ?? ""}`}`,
    );
    results.push(r);
  }
  return {
    mode,
    runName,
    cases: results,
    passRate: results.filter((r) => r.passed).length / results.length,
    avgJudgeScore: average(
      results.filter((r) => r.verdict).map((r) => r.verdict!.score),
    ),
    avgCostUsd: average(results.map((r) => r.costUsd)),
    avgLatencyMs: average(results.map((r) => r.latencyMs)),
    totalCostUsd: results.reduce((a, r) => a + r.costUsd, 0),
    totalTokens: results.reduce((a, r) => a + r.tokens, 0),
  };
}

export async function runEvals(modes: AgentMode[]) {
  const datasetId = await syncDataset();
  const results: ModeResult[] = [];
  // EVAL_CASES=colors,stock runs just those cases (for a quick check).
  const only = process.env.EVAL_CASES?.split(",").map((id) => id.trim());
  const cases = only
    ? GOLDEN_SET.filter((c) => only.includes(c.id))
    : GOLDEN_SET;
  for (const mode of modes) results.push(await runMode(mode, cases, datasetId));
  return results;
}

/** A link to the golden-set dataset in Langfuse, where the runs are compared. */
export async function datasetUrl(): Promise<string | null> {
  if (!isLangfuseConfigured()) return null;
  try {
    const projects = (await langfuseApi("/api/public/projects")) as {
      data: { id: string }[];
    };
    const dataset = (await langfuseApi(
      `/api/public/v2/datasets/${encodeURIComponent(DATASET)}`,
    )) as { id: string };
    return `${langfuseBaseUrl()}/project/${projects.data[0].id}/datasets/${dataset.id}`;
  } catch {
    return null;
  }
}

const pct = (n: number) => `${Math.round(n * 100)}%`;
const usd = (n: number) => `$${n.toFixed(5)}`;
const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`;

/** The results as Markdown, for the pull request's checks and the terminal. */
export function summaryMarkdown(
  results: ModeResult[],
  opts: { minPassRate: number; gateMode: AgentMode; link: string | null },
): string {
  const lines = [
    "## Ask Copper evals (golden set, AI judge)",
    "",
    "| Version | Pass rate | Judge score (avg, 1-5) | Cost per chat (avg) | Time per chat (avg) | Tokens (total) |",
    "|---|---|---|---|---|---|",
    ...results.map(
      (r) =>
        `| ${MODE_LABELS[r.mode]}${r.mode === opts.gateMode ? " (ships)" : ""} | **${pct(r.passRate)}** (${r.cases.filter((c) => c.passed).length}/${r.cases.length}) | ${r.avgJudgeScore.toFixed(2)} | ${usd(r.avgCostUsd)} | ${secs(r.avgLatencyMs)} | ${r.totalTokens.toLocaleString("en-US")} |`,
    ),
    "",
    `Merging needs a pass rate of at least ${pct(opts.minPassRate)} for the version that ships (${MODE_LABELS[opts.gateMode]}).`,
    opts.link
      ? `Scores, traces and the side-by-side comparison of runs are in Langfuse: [${DATASET}](${opts.link}).`
      : "",
    "",
    "<details><summary>Every case</summary>",
    "",
    `| Case | ${results.map((r) => MODE_LABELS[r.mode]).join(" | ")} |`,
    `|---|${results.map(() => "---").join("|")}|`,
    ...GOLDEN_SET.map((c) => {
      const cells = results.map((r) => {
        const x = r.cases.find((y) => y.id === c.id);
        if (!x) return "-";
        const why = x.passed
          ? ""
          : ` ${(x.error ?? [x.verdict?.reasoning, ...x.toolProblems].filter(Boolean).join(" ")).replace(/\|/g, "/").replace(/\n/g, " ")}`;
        return `${x.passed ? "✅" : "❌"} ${x.verdict?.score ?? "-"}/5${why}`;
      });
      return `| \`${c.id}\` | ${cells.join(" | ")} |`;
    }),
    "",
    "</details>",
  ];
  return lines.join("\n");
}
