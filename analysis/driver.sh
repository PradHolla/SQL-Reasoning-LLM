#!/usr/bin/env bash
# Everything left in one unattended pass.
#
# Written as one chained job rather than five launched by hand, because the
# gaps between hand-launched jobs are what let the idle-shutdown reclaim the
# box mid-session: it fires after 30 minutes at <5% GPU with nobody logged in,
# and a finished job with the next one not yet started looks exactly like an
# abandoned machine. Chaining removes the gaps entirely.
#
# The manual hold is released by a trap, not by a line at the bottom, so a
# crash halfway through cannot leave the guardrail disabled on a running box.

set -uo pipefail
trap 'rm -f /opt/sql-llm/.no-autoshutdown; echo "[driver] idle-shutdown hold released"' EXIT

cd /opt/sql-llm/repo
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # stdout is a file here, so print() would block-buffer

step() { echo; echo "=== $* ==="; date -u +'    %Y-%m-%dT%H:%M:%SZ'; }

step "1/5  vote curve + calibration (k=1..16)"
uv run python -m analysis.vote_curve \
    --votes results/test/grpo-coder15-vote16.json --split test \
    --out results/analysis/vote_curve.json

step "2/5  demo capture"
uv run uvicorn sqlrl.serving.api:app --host 127.0.0.1 --port 8000 --log-level warning &
SERVER=$!
for _ in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 5
done
uv run python -m analysis.demo --db-id bakery_1 --samples 8 \
    --transcript results/analysis/demo.json \
    --questions \
      "What is the most expensive cake and its flavor?" \
      "Give me a list of all the distinct items bought by the customer number 15." \
      "Give me the first name and last name of customers who have bought apple flavor Tart."
kill "$SERVER" 2>/dev/null
wait "$SERVER" 2>/dev/null

step "3/5  demo gif"
uv run python -m analysis.demo_gif \
    --transcript results/analysis/demo.json \
    --out results/analysis/charts/demo.gif

step "4/5  full-scale latency (n=2147, the long one)"
uv run python -m sqlrl.serving.bench --model grpo-coder15 --n 2147 \
    --modes greedy,vote8,retry3 --json results/analysis/latency.json

step "5/5  render every chart"
uv run python -m analysis.charts

echo
echo "[driver] ALL DONE"
