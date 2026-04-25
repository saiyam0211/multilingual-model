#!/usr/bin/env bash
# End-to-end smoke test: install, run tests, hit live endpoints.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "→ pytest"
pytest -x -q

echo "→ starting uvicorn in background"
uvicorn polyglot_redteam.server:app --host 127.0.0.1 --port 8000 &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT
sleep 3

echo "→ /health"
curl -fsS http://127.0.0.1:8000/health | python -m json.tool

echo "→ /reset"
EID=$(curl -fsS -X POST http://127.0.0.1:8000/reset -H 'content-type: application/json' \
  -d '{"seed":42}' | python -c "import sys,json; print(json.load(sys.stdin)['episode_id'])")
echo "episode_id=$EID"

echo "→ /step"
curl -fsS -X POST http://127.0.0.1:8000/step -H 'content-type: application/json' \
  -d "{\"episode_id\":\"$EID\",\"action\":\"नमस्ते, यह एक परीक्षण प्रॉम्प्ट है।\"}" \
  | python -m json.tool

echo "✓ smoke ok"
