// npm run eval: the golden-set evals (Module 5), kept apart from npm test
// because they call the real model and take a few minutes.
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { tsconfigPaths: true },
  test: {
    environment: "node",
    include: ["evals/**/*.eval.ts"],
    testTimeout: 30 * 60_000,
  },
});
