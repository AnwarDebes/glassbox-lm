#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# READY TO RUN, NOT RUN BY THE PREPARER.
# tau2-bench v1.0.1 (commit fc0055dc4e0a316c3f83133267fbd6faaa770992) on a small task subset, with the agent, the
# user simulator and the NL-assertion judge all on ONE local OpenAI-compatible llama-server (Qwen3.5-9B Q8_0).
#
# This script never starts a server. Start llama-server first (the GPU queue owns the GPU), for example:
#   CUDA_VISIBLE_DEVICES=0 ../tools/llama.cpp/b11742/build-cuda129/bin/llama-server \
#     -m ../tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf \
#     --host 127.0.0.1 --port 18080 --no-webui --jinja --reasoning-budget 0 -fa off -ngl 99 \
#     --alias qwen3.5-9b -np 2 -c 65536
#   (--jinja is required for tool calls; -np 2 keeps one KV slot for the agent prompt and one for the user prompt;
#    -c is split across slots, so 65536 gives 32768 tokens per slot. These are suggestions, not tested here.)
#
# Usage (all variables optional):
#   PORT=18080 DOMAIN=retail SPLIT=test NUM_TASKS=5 TRIALS=1 \
#     bash studies/agent_guards/taubench/scripts/run_local_subset.sh
#   TASK_IDS="0 1 2"  -> run these task ids instead of the first NUM_TASKS of SPLIT
set -euo pipefail
export CUDA_VISIBLE_DEVICES=            # tau2 itself needs no GPU; only llama-server uses it

F=${REPO_ROOT}/studies/agent_guards/taubench
PY="$F/.venv/bin/python"
PORT="${PORT:-18080}"
BASE="http://127.0.0.1:${PORT}/v1"
DOMAIN="${DOMAIN:-retail}"              # airline | retail | telecom
SPLIT="${SPLIT:-test}"                  # train | test | base (telecom also: small, full)
NUM_TASKS="${NUM_TASKS:-5}"
TRIALS="${TRIALS:-1}"
CONC="${CONC:-1}"                       # simulations in parallel; keep <= server slots / 2
TIMEOUT_S="${TIMEOUT_S:-1800}"          # wall-clock cap per simulation (premature end scores 0)
TEMP="${TEMP:-0.0}"                     # tau2 default for agent and user
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="${RUN:-local_${DOMAIN}_${SPLIT}_n${NUM_TASKS}_t${TRIALS}_${STAMP}}"
OUT="$F/runs/$RUN"
mkdir -p "$F/runs" "$F/.tmp"

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$F/runs/run_local_subset.log"; }

# 1. A local server must already answer; never start one here.
MODELS_JSON="$(curl -sf -m 5 --noproxy "*" "$BASE/models" || true)"
if [ -z "$MODELS_JSON" ]; then
  say "STOP: no OpenAI-compatible server at $BASE (start llama-server first; this script never starts it)"
  exit 2
fi
MODEL="${MODEL:-$(printf '%s' "$MODELS_JSON" | "$PY" -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')}"
say "server $BASE serves model id: $MODEL"

# 2. CPU check (third field = cores used by this cgroup; wait if above 3.5).
LOAD_PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
LOAD_SCRIPT=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
for i in 1 2 3 4 5 6; do
  L="$($LOAD_PY -I "$LOAD_SCRIPT" 10)"; CORES="$(echo "$L" | awk '{print $3}')"
  say "cpu: $L"
  awk -v c="$CORES" 'BEGIN{exit !(c <= 3.5)}' && break
  [ "$i" = 6 ] && { say "STOP: cgroup still busy (> 3.5 cores)"; exit 4; }
  sleep 120
done

# 3. Environment: every LLM call goes to the local server; anything else hits a dead proxy and fails fast.
export OPENAI_API_KEY=sk-local-no-key
export OPENAI_BASE_URL="$BASE" OPENAI_API_BASE="$BASE"      # fallback for any openai/... call without api_base
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
for v in $(compgen -e | grep '_API_KEY$' | grep -v '^OPENAI_API_KEY$'); do unset "$v"; done  # every other provider key
export LITELLM_LOCAL_MODEL_COST_MAP=True                   # no cost-map download at import
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR="$F/.tmp"

LLM_ARGS="{\"api_base\": \"$BASE\", \"api_key\": \"sk-local-no-key\", \"temperature\": $TEMP, \"max_tokens\": 2048, \"timeout\": 600}"
SEL=(--task-split-name "$SPLIT" --num-tasks "$NUM_TASKS")
if [ -n "${TASK_IDS:-}" ]; then
  read -r -a IDS <<< "$TASK_IDS"        # no glob expansion (telecom ids contain [ ] and |)
  SEL=(--task-split-name "$SPLIT" --task-ids "${IDS[@]}")
fi

say "start: domain=$DOMAIN split=$SPLIT ${TASK_IDS:+task_ids=$TASK_IDS }num_tasks=$NUM_TASKS trials=$TRIALS conc=$CONC out=$OUT"
set +e
nice -n 19 "$PY" -I -B "$F/scripts/tau2_local.py" --base "$BASE" --model "$MODEL" -- \
  run --domain "$DOMAIN" "${SEL[@]}" --num-trials "$TRIALS" \
  --agent llm_agent --agent-llm "openai/$MODEL" --agent-llm-args "$LLM_ARGS" \
  --user user_simulator --user-llm "openai/$MODEL" --user-llm-args "$LLM_ARGS" \
  --max-concurrency "$CONC" --timeout "$TIMEOUT_S" --seed 300 \
  --save-to "$OUT" --auto-resume --log-level WARNING \
  > "$F/runs/$RUN.log" 2>&1
RC=$?
set -e
say "tau2 run exit $RC (log: $F/runs/$RUN.log, results: $OUT/results.json)"

# 4. Judge-free score of the same trajectories (DB / ENV_ASSERTION / ACTION / COMMUNICATE; never calls an LLM).
if [ -f "$OUT/results.json" ]; then
  nice -n 19 "$PY" -I -B "$F/scripts/rescore_nonllm.py" --results "$OUT/results.json" --domain "$DOMAIN" \
    --out "$OUT/rescore_nonllm.json" > "$OUT/rescore_nonllm.txt" 2>&1 || true
  say "judge-free rescore: $(grep -E '"mean_(recorded|nonllm)_reward"' "$OUT/rescore_nonllm.txt" | tr -d ' \n')"
fi
exit $RC
