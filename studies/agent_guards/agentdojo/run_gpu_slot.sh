#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# AgentDojo 0.1.35 x local Qwen3.5-9B Q8_0 (llama.cpp b11742 llama-server) -- FOR THE GPU SLOT ONLY.
# NOT RUN YET (written 2026-10-21). Start it only in a GPU slot of the queue:
#     bash run_gpu_slot.sh --go [extra args for run_agentdojo_local.py]
# e.g. bash run_gpu_slot.sh --go --subset small --defences none tool_guard
# Steps (each logged with a UTC stamp in runs_gpu/<stamp>/slot.log):
#   1. checks: GGUF files match FILES.sha256; the port is free; harness files present;
#   2. llama-server on 127.0.0.1:$PORT only (--no-webui, -np 1, -c $CTX, -ngl 99, -fa off, --jinja,
#      thinking off with --reasoning-budget 0, --temp 0 as server default, 2 CPU threads), waits for /health;
#   3. /v1/models and a request with a wrong model name (llama-server in single-model mode should ignore the name;
#      upstream's tool_filter sends the pipeline name as model);
#   4. run_agentdojo_local.py (CPU side: no GPU, 2 threads, nice 19) on the chosen subset;
#   5. stops the server it started (only that PID), logs GPU memory before / after.
set -u
H=${REPO_ROOT}/studies/agent_guards/agentdojo
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
G=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742
M=$G/Qwen3.5-9B-Q8_0.gguf
PORT=${PORT:-18081}
CTX=${CTX:-65536}
ALIAS=qwen3.5-9b-q8_0
PY=$H/.venv/bin/python

[ "${1:-}" = "--go" ] || { echo "usage: bash $0 --go [run_agentdojo_local.py args]   (GPU slot only; nothing done)"; exit 2; }
shift
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$H/runs_gpu/$STAMP
mkdir -p "$RUN" || exit 1
log=$RUN/slot.log
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$log"; }
spid=""
cleanup() { if [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null; then kill "$spid"; wait "$spid" 2>/dev/null; say "server $spid stopped"; fi; }
trap cleanup EXIT INT TERM

say "start; args: $*; PORT=$PORT CTX=$CTX"
(cd "$G" && sha256sum -c --quiet FILES.sha256) >> "$log" 2>&1 || { say "STOP: GGUF files do not match FILES.sha256"; exit 1; }
[ -x "$B/llama-server" ] || { say "STOP: llama-server missing"; exit 1; }
[ -x "$PY" ] || { say "STOP: venv python missing"; exit 1; }
if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then say "STOP: something already answers on port $PORT"; exit 1; fi
say "GPU before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1)"

CUDA_VISIBLE_DEVICES=0 "$B/llama-server" -m "$M" --alias "$ALIAS" --host 127.0.0.1 --port "$PORT" --no-webui \
  -np 1 -c "$CTX" -ngl 99 -fa off --jinja --reasoning-budget 0 --temp 0 -t 2 -tb 2 > "$RUN/server.log" 2>&1 &
spid=$!
say "server pid $spid"
ok=0
for i in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$spid" 2>/dev/null || break
  sleep 5
done
[ "$ok" = 1 ] || { say "STOP: server not healthy; tail: $(tail -n 5 "$RUN/server.log" | tr '\n' ' ')"; exit 1; }
say "health: $(curl -s -m 5 "http://127.0.0.1:$PORT/health")"
say "models: $(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | head -c 600)"
say "wrong-model-name check: $(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"local-qwen3.5-9b-q8_0-tool_filter","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' | head -c 400)"

cd "$RUN" || exit 1
env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 19 "$PY" -I -B "$H/run_agentdojo_local.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$ALIAS" \
  --out "$RUN/agentdojo" "$@" >> "$log" 2>&1
e=$?
say "driver exit $e"
cleanup
spid=""
say "GPU after: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1)"
say "CUDA errors in server log: $(grep -c 'CUDA error' "$RUN/server.log")"
say "SLOT DONE"
