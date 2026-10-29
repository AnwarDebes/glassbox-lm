#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# PLAN_llm_dev.md Q1: start the local Qwen server, answer 500 dev rows of ABCD and of STAR, stop the server.
# Run inside one exclusive GPU slot (run_queue.sh cmd: line). Usage: llm_job.sh OUTDIR
set -uo pipefail
out="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
srv=${REPO_ROOT}/studies/agent_guards/llm/serve_qwen_agent.sh
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "${out}"
echo "$(stamp) server start" >> "${out}/job.log"
bash "${srv}" start --note "headline PLAN_llm_dev Q1 (dev only), alone on the GPU" >> "${out}/job.log" 2>&1 || {
    echo "$(stamp) server start failed" >> "${out}/job.log"; exit 3; }
rc=0
for ds in abcd star; do
    echo "$(stamp) ${ds} start" >> "${out}/job.log"
    OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/llm_next_action.py" --dataset "${ds}" --n 500 --k 10 --seed 1 \
        --out "${out}/${ds}" >> "${out}/${ds}.log" 2>&1
    r=$?
    echo "$(stamp) ${ds} exit ${r}" >> "${out}/job.log"
    [ "${r}" -ne 0 ] && rc="${r}"
done
bash "${srv}" stop >> "${out}/job.log" 2>&1
echo "$(stamp) server stopped; job exit ${rc}" >> "${out}/job.log"
exit "${rc}"
