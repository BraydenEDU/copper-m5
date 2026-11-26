// @vitest-environment node
// npm run eval: runs the golden set on Ask Copper and has the AI judge grade
// every answer (Module 5). Fails when the version that ships passes fewer
// than EVAL_MIN_PASS_RATE of the cases, so CI blocks the merge.
// EVAL_MODES picks the versions: "multi" (two agents, the default that
// ships), "single" (one support agent), or "multi,single" to compare them.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { AgentMode } from "@/lib/copper/assistant/progress";

if (existsSync(".env.local")) process.loadEnvFile(".env.local");

const modes = (process.env.EVAL_MODES ?? "multi,single")
  .split(",")
  .map((m) => m.trim())
  .filter((m): m is AgentMode => m === "multi" || m === "single");
const minPassRate = Number(process.env.EVAL_MIN_PASS_RATE ?? "0.8");
const gateMode: AgentMode = "multi";

describe("Ask Copper golden set", () => {
  it(
    `passes at least ${Math.round(minPassRate * 100)}% of the cases`,
    async () => {
      expect(
        process.env.GOOGLE_GENERATIVE_AI_API_KEY,
        "GOOGLE_GENERATIVE_AI_API_KEY is needed to run the evals",
      ).toBeTruthy();
      const { datasetUrl, runEvals, summaryMarkdown } =
        await import("@/evals/run");
      const results = await runEvals(modes);
      const summary = summaryMarkdown(results, {
        minPassRate,
        gateMode,
        link: await datasetUrl(),
      });
      mkdirSync("eval-results", { recursive: true });
      writeFileSync("eval-results/summary.md", summary);
      writeFileSync(
        "eval-results/results.json",
        JSON.stringify({ minPassRate, gateMode, results }, null, 2),
      );
      if (process.env.GITHUB_STEP_SUMMARY)
        writeFileSync(process.env.GITHUB_STEP_SUMMARY, `${summary}\n`, {
          flag: "a",
        });
      console.info(`\n${summary}\n`);

      const gate = results.find((r) => r.mode === gateMode) ?? results[0];
      expect(
        gate.passRate,
        `${gate.mode} pass rate is below ${minPassRate}`,
      ).toBeGreaterThanOrEqual(minPassRate);
    },
    30 * 60_000,
  );
});
