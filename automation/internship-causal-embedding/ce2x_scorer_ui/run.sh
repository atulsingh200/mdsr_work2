#!/usr/bin/env bash
# Run the CrossEncoder2x Scorer UI
# Usage:  bash run.sh [port]
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8000}"

echo "Starting CrossEncoder2x Scorer UI on http://0.0.0.0:${PORT}"
echo "Open in browser: http://localhost:${PORT}"

cd "$REPO_ROOT"
uv run uvicorn ce2x_scorer_ui.app:app --host 0.0.0.0 --port "$PORT" --reload
