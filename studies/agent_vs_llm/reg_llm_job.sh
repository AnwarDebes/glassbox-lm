#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# REG_agent_vs_llm: one LLM* test run on the registered rows; a second attempt if request errors remain (exit 4).
#   reg_llm_job.sh DATASET MODEL_GGUF QUANT BUDGET ROWS OUTDIR REG [flags for llm_next_action.py]
# The server: serve_headline.sh (pinned build, sha256 of the model checked at start, one slot, 65,536 context).
set -uo pipefail
ds="$1"; model="$2"; q="$3"; budget="$4"; rows="$5"; out="$6"; reg="$7"; shift 7
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
sha="$(grep "${q}" "$(dirname "${model}")/FILES.sha256" | cut -d' ' -f1)"
rc=1
for attempt in 1 2; do
    if ! bash "${here}/serve_headline.sh" start "${model}" "${sha}" 65536 1 "${budget}"; then rc=3; sleep 60; continue; fi
    OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${here}/llm_next_action.py" --dataset "${ds}" --n "${rows}" \
        --seed 20261027 "$@" --url http://127.0.0.1:18083 --test-registration "${reg}" --out "${out}"
    rc=$?
    bash "${here}/serve_headline.sh" stop
    [ "${rc}" -eq 4 ] || break
    sleep 60
done
exit "${rc}"
