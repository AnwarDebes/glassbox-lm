#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# PLAN_llm_q3_dev.md: one LLM arm on the 500 fresh dev rows of each dataset (seed 2, Q1's rows excluded).
#   llm_q3_job.sh ARM OUTDIR [HEAD]     ARM: L9t | L35 | L35t
set -uo pipefail
arm="$1"; out="$2"; head="${3:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
M9=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf
S9="$(grep Q8_0 "$(dirname "${M9}")/FILES.sha256" | cut -d' ' -f1)"   # read from the model folder's own list
M35=${WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742/Qwen3.6-35B-A3B-Q4_K_M.gguf
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
case "${arm}" in
    L9t) model="${M9}"; sha="${S9}"; budget=-1; think=(--think) ;;
    L35) model="${M35}"; sha="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"; budget=0; think=() ;;
    L35t) model="${M35}"; sha="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"; budget=-1; think=(--think) ;;
    L35k50) model="${M35}"; sha="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"; budget=0; think=(); kk=50 ;;
    L35k100) model="${M35}"; sha="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"; budget=0; think=(); kk=100 ;;
    *) echo "unknown arm ${arm}"; exit 2 ;;
esac
mkdir -p "${out}"
echo "$(stamp) ${arm} server start ${model}" >> "${out}/job.log"
bash "${here}/serve_headline.sh" start "${model}" "${sha}" 65536 1 "${budget}" >> "${out}/job.log" 2>&1 || {
    echo "$(stamp) server start failed" >> "${out}/job.log"; exit 3; }
rc=0
for ds in ${Q3_DATASETS:-abcd star}; do   # Q3_DATASETS: optional subset and order (note NOTE_q3_order.md)
    extra=()
    excl=()
    [ -f "${here}/runs/llm_q1_dev/${ds}/preds.jsonl" ] && excl=(--exclude "${here}/runs/llm_q1_dev/${ds}/preds.jsonl")
    [ -n "${head}" ] && extra=(--head "${head}")
    echo "$(stamp) ${arm} ${ds} start" >> "${out}/job.log"
    OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/llm_next_action.py" --dataset "${ds}" --n 500 --k ${kk:-10} --seed 2 \
        "${excl[@]}" --url http://127.0.0.1:18083 "${think[@]}" "${extra[@]}" \
        --out "${out}/${ds}" >> "${out}/${ds}.log" 2>&1
    r=$?
    echo "$(stamp) ${arm} ${ds} exit ${r}" >> "${out}/job.log"
    [ "${r}" -ne 0 ] && rc="${r}"
done
bash "${here}/serve_headline.sh" stop >> "${out}/job.log" 2>&1
echo "$(stamp) ${arm} server stopped; job exit ${rc}" >> "${out}/job.log"
exit "${rc}"
