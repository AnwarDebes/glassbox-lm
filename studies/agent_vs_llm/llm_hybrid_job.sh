#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# PLAN_hybrid_dev.md: on 500 fresh dev rows per dataset (seed 2, Q1's rows excluded): Hybrid-k3, Hybrid-k5, LLM alone.
# Runs inside one exclusive GPU slot (run_queue.sh cmd: line). Usage: llm_hybrid_job.sh OUTDIR
set -uo pipefail
out="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
srv=${REPO_ROOT}/studies/agent_guards/llm/serve_qwen_agent.sh
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "${out}"
echo "$(stamp) server start" >> "${out}/job.log"
bash "${srv}" start --note "headline PLAN_hybrid_dev (dev only), alone on the GPU" >> "${out}/job.log" 2>&1 || {
    echo "$(stamp) server start failed" >> "${out}/job.log"; exit 3; }
rc=0
for ds in abcd star; do
    common=(--dataset "${ds}" --n 500 --k 10 --seed 2 --exclude "${here}/runs/llm_q1_dev/${ds}/preds.jsonl")
    for arm in k3 k5 alone; do
        extra=()
        [ "${arm}" = k3 ] && extra=(--shortlist "${here}/runs/agent_${ds}_c4_tm_s1/tm.npz" --k-short 3)
        [ "${arm}" = k5 ] && extra=(--shortlist "${here}/runs/agent_${ds}_c4_tm_s1/tm.npz" --k-short 5)
        echo "$(stamp) ${ds} ${arm} start" >> "${out}/job.log"
        OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/llm_next_action.py" "${common[@]}" "${extra[@]}" \
            --out "${out}/${ds}_${arm}" >> "${out}/${ds}_${arm}.log" 2>&1
        r=$?
        echo "$(stamp) ${ds} ${arm} exit ${r}" >> "${out}/job.log"
        [ "${r}" -ne 0 ] && rc="${r}"
    done
done
bash "${srv}" stop >> "${out}/job.log" 2>&1
echo "$(stamp) server stopped; job exit ${rc}" >> "${out}/job.log"
exit "${rc}"
