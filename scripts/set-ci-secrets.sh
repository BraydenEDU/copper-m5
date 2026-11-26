#!/bin/sh
# Copies the Gemini and Langfuse keys from .env.local into this repo's GitHub
# Actions secrets, so the evals can run on pull requests (Module 5).
# Nothing is printed. Run it once: sh scripts/set-ci-secrets.sh
cd "$(dirname "$0")/.." || exit 1
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner) || exit 1
grep -E '^(GOOGLE_GENERATIVE_AI_API_KEY|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_BASE_URL)=' .env.local |
  gh secret set -R "$repo" -f /dev/stdin >/dev/null || exit 1
echo "Saved these secrets to $repo:"
gh secret list -R "$repo"
