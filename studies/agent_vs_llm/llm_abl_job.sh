#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# PLAN_ablation_dev.md: L9 and L35 (thinking off) WITHOUT the written policy, on the fresh rows (seed 2, Q1 excluded).
# Usage: llm_abl_job.sh OUTDIR
set -uo pipefail
out="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
M9=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf
M35=${WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742/Qwen3.6-35B-A3B-Q4_K_M.gguf
S9="$(grep Q8_0 "$(dirname "${M9}")/FILES.sha256" | cut -d' ' -f1)"
S35="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "${out}"
rc=0
for arm in L9 L35; do
    if [ "${arm}" = L9 ]; then model="${M9}"; sha="${S9}"; else model="${M35}"; sha="${S35}"; fi
    echo "$(stamp) ${arm}-noG server start" >> "${out}/job.log"
    bash "${here}/serve_headline.sh" start "${model}" "${sha}" 65536 1 0 >> "${out}/job.log" 2>&1 || {
        echo "$(stamp) ${arm} server start failed" >> "${out}/job.log"; rc=3; continue; }
    for ds in abcd star; do
        echo "$(stamp) ${arm}-noG ${ds} start" >> "${out}/job.log"
        OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/llm_next_action.py" --dataset "${ds}" --n 500 --k 10 --seed 2 \
            --exclude "${here}/runs/llm_q1_dev/${ds}/preds.jsonl" --no-guidelines --url http://127.0.0.1:18083 \
            --out "${out}/${arm}/${ds}" >> "${out}/${arm}_${ds}.log" 2>&1
        r=$?
        echo "$(stamp) ${arm}-noG ${ds} exit ${r}" >> "${out}/job.log"
        [ "${r}" -ne 0 ] && rc="${r}"
    done
    bash "${here}/serve_headline.sh" stop >> "${out}/job.log" 2>&1
done
echo "$(stamp) job exit ${rc}" >> "${out}/job.log"
exit "${rc}"
