#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# PLAN_zipf_synth.md grid, one CPU job at a time (2 threads, nice 19), each after the CPU gate. Restart-safe: a run whose
# output exists is skipped. Overrides for tests: ZIPF_PY, ZIPF_SCRIPT, ZIPF_OUT, ZIPF_SEEDS, ZIPF_SVALS, ZIPF_BUDGETS,
# ZIPF_GATE_CMD.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ZIPF_PY:-${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python}"
SCRIPT="${ZIPF_SCRIPT:-${here}/zipf_synth.py}"
OUT="${ZIPF_OUT:-${here}/runs/zipf}"
GATE=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
GATE_CMD="${ZIPF_GATE_CMD:-OMP_NUM_THREADS=1 nice -n 19 ${PY} -I ${GATE} 10}"
log="${OUT}/run.log"
mkdir -p "${OUT}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gate() {
    local g cores
    while :; do
        g="$(bash -c "${GATE_CMD}")"
        cores="$(sed -n 's/.* cores \([0-9.]*\) .*/\1/p' <<< "${g}")"
        if [ -n "${cores}" ] && awk -v x="${cores}" 'BEGIN { exit !(x <= 4.5) }'; then return; fi
        echo "$(stamp) gate closed: ${g}" >> "${log}"
        sleep 60
    done
}
run() {   # run NAME ARGS...: skip if the output exists; log the true exit code
    local name="$1" rc o
    shift
    o="${OUT}/${name}.json"
    [ -f "${o}" ] && return
    gate
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 "${PY}" -I "${SCRIPT}" "$@" --out "${o}" >> "${log}" 2>&1
    rc=$?
    echo "$(stamp) ${name} exit ${rc}" >> "${log}"
}
for seed in ${ZIPF_SEEDS:-1 2 3}; do
    run "lr_seed${seed}" --lr --seed "${seed}"
    for budget in ${ZIPF_BUDGETS:-320 80 20}; do
        for sv in ${ZIPF_SVALS:-2 4 8 16 32 64}; do
            run "s${sv}_c${budget}_seed${seed}" --s "${sv}" --clauses "${budget}" --seed "${seed}"
        done
    done
done
echo "$(stamp) grid done" >> "${log}"
