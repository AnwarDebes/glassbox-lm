#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# PLAN_logsize_dev.md: LLM*(N) (L35k50's arguments, retrieval pool of N training dialogues) on the 500 dev rows.
#   logsize_llm_job.sh SIZE [DATASET ...]      (default: star abcd multiwoz)
# One server session per dataset (serve_headline.sh: sha256 of the model checked, 65,536 context, one slot, thinking
# off); a second attempt when the client stops on request errors (exit 4). Restart-safe: the client keeps finished rows.
set -uo pipefail
size="$1"; shift
dss=("$@"); [ "${#dss[@]}" -eq 0 ] && dss=(star abcd multiwoz)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
M35=${WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742/Qwen3.6-35B-A3B-Q4_K_M.gguf
sha="$(grep Q4_K_M "$(dirname "${M35}")/FILES.sha256" | cut -d' ' -f1)"
out="${here}/runs/logsize/llm_N${size}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "${out}"
rc=0
for ds in "${dss[@]}"; do
    excl=()
    [ -f "${here}/runs/llm_q1_dev/${ds}/preds.jsonl" ] && excl=(--exclude "${here}/runs/llm_q1_dev/${ds}/preds.jsonl")
    r=3
    for attempt in 1 2; do
        echo "$(stamp) ${ds} server start (attempt ${attempt})" >> "${out}/job.log"
        if ! bash "${here}/serve_headline.sh" start "${M35}" "${sha}" 65536 1 0 >> "${out}/job.log" 2>&1; then
            echo "$(stamp) ${ds} server start failed" >> "${out}/job.log"; r=3; sleep 60; continue
        fi
        echo "$(stamp) ${ds} start" >> "${out}/job.log"
        OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/logsize_llm.py" --size "${size}" -- --dataset "${ds}" \
            --n 500 --k 50 --seed 2 "${excl[@]}" --url http://127.0.0.1:18083 --out "${out}/${ds}" \
            >> "${out}/${ds}.log" 2>&1
        r=$?
        echo "$(stamp) ${ds} exit ${r}" >> "${out}/job.log"
        bash "${here}/serve_headline.sh" stop >> "${out}/job.log" 2>&1
        [ "${r}" -eq 4 ] || break
        sleep 60
    done
    [ "${r}" -ne 0 ] && rc="${r}"
done
exit "${rc}"
