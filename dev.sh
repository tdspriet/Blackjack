#!/usr/bin/env bash
# Start the ONE Blackjack Lab backend (FastAPI, auto-reload) and the Vite dev server together.
# Ctrl-C stops both.   Ports: BACKEND_PORT (default 8000), FRONTEND_PORT (default 5173).
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm (Node.js 20+) is required" >&2; exit 1; }
[ -d .venv ] || uv sync
[ -d frontend/node_modules ] || (cd frontend && npm install)

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping…"
  kill 0 2>/dev/null || true  # the whole process group: uvicorn (+ reloader), npm, vite
}
trap cleanup INT TERM EXIT

uv run uvicorn onebj.server.app:app --reload --reload-dir backend --port "$BACKEND_PORT" &
BACKEND=$!
(cd frontend && ONEBJ_BACKEND="http://localhost:$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT" --strictPort) &
FRONTEND=$!

echo "Backend  → http://localhost:$BACKEND_PORT  (API docs at /docs)"
echo "Frontend → http://localhost:$FRONTEND_PORT"

# macOS ships bash 3.2 (no `wait -n`): exit as soon as either server stops
while kill -0 "$BACKEND" 2>/dev/null && kill -0 "$FRONTEND" 2>/dev/null; do
  sleep 1
done
