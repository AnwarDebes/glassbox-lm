#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Offline reward-function check on the leaderboard trajectories shipped in the repo (no LLM, network blocked).
set -u
export CUDA_VISIBLE_DEVICES=
F=${REPO_ROOT}/studies/agent_guards/taubench
R=${REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/results/final
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 NO_PROXY=
for v in $(compgen -e | grep '_API_KEY$'); do unset "$v"; done  # no provider key at all
export LITELLM_LOCAL_MODEL_COST_MAP=True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=$F/.tmp
run() { # file domain limit tag extra
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $4"
  nice -n 19 $F/.venv/bin/python -I -B $F/scripts/rescore_nonllm.py --results "$R/$1" --domain "$2" ${3:+--limit $3} $5 \
     --out "$F/results/rescore_$4.json" > "$F/results/rescore_$4.txt" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $4 exit $?"
}
run gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json airline "" gpt41_airline_embedded ""
run gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json airline "" gpt41_airline_fresh "--fresh-tasks"
run gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json retail 200 gpt41_retail_embedded_200 ""
run gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json telecom 200 gpt41_telecom_embedded_200 ""
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) all done"
