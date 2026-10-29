#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# End-to-end smoke of serve_qwen_agent.sh with the REAL binary and the Q8_0 model on the CPU only
# (SERVE_CPU_TEST=1: CUDA_VISIBLE_DEVICES= and -ngl 0, 2 threads, nice 19; the GPU is never touched; CPU gate 4.0 cores):
#   1. start with the defaults (E2/E4, reasoning off since 04:33Z) on port 18098 (all of start's checks:
#      the pinned hashes, the gate, /health, /v1/models, /props with the pinned build_info);
#   2. smoke_serve_checks.py: props, both renderings, a short reply without the thinking switch, one tool call;
#   3. status, stop (no llama-server left, the port free);
#   4. start again with --reasoning auto; the rendering checks only; stop.
# State in tests/runs/smoke_cpu_<stamp>/. Run: bash tests/smoke_serve_cpu.sh
set -u
L=${REPO_ROOT}/studies/agent_guards/llm
S=$L/serve_qwen_agent.sh
T=$L/tests
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
SERVE_DIR=$T/runs/smoke_cpu_$(date -u +%Y%m%dT%H%M%SZ)
export SERVE_DIR SERVE_CPU_TEST=1
P=18098
fail=0
step() { echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
trap 'bash "$S" stop > /dev/null 2>&1' EXIT
[ -z "$(pgrep -x llama-server)" ] || { echo "a llama-server is running; the smoke needs none"; exit 1; }

step "start (defaults: reasoning off)"
bash "$S" start --port $P --note "CPU smoke" || { echo "FAIL: start"; exit 1; }
step "checks (reasoning off)"
CUDA_VISIBLE_DEVICES='' "$PY" -I "$T/smoke_serve_checks.py" --port $P --alias qwen3.5-9b-q8_0 --np 1 --ctx 65536 \
  --reasoning off --out "$SERVE_DIR/checks_off.json" || fail=1
step "status"
bash "$S" status
step "stop"
bash "$S" stop || fail=1
if [ -z "$(pgrep -x llama-server)" ]; then echo "PASS no llama-server left"; else echo "FAIL a llama-server is left"; fail=1; fi
if (exec 3<>"/dev/tcp/127.0.0.1/$P") 2>/dev/null; then echo "FAIL port $P still open"; fail=1; else echo "PASS port $P free"; fi

step "start (reasoning auto)"
bash "$S" start --port $P --reasoning auto --note "CPU smoke, reasoning auto" || { echo "FAIL: start (reasoning auto)"; exit 1; }
step "rendering checks (reasoning auto)"
CUDA_VISIBLE_DEVICES='' "$PY" -I "$T/smoke_serve_checks.py" --port $P --alias qwen3.5-9b-q8_0 --np 1 --ctx 65536 \
  --reasoning auto --render-only --out "$SERVE_DIR/checks_auto.json" || fail=1
step "stop"
bash "$S" stop || fail=1
if [ -z "$(pgrep -x llama-server)" ]; then echo "PASS no llama-server left"; else echo "FAIL a llama-server is left"; fail=1; fi

step "smoke result: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
