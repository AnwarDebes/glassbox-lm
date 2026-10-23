#!/usr/bin/env bash
# T0 final runs (registrations/T0.md and its tuning amendment).
#   scripts/t0_final.sh S_COAL E_COAL S_FLAT E_FLAT
# Seeds 1 to 10 of both machines, each in one CPU slot at nice 10; after each coalesced seed its
# logistic-regression arm runs on the GPU under the exclusive lock. Logs in logs/t0_final_*.out.
# The verdict (src/tally/t0/report.py) is run by hand once every run has finished.
set -euo pipefail
[ "$#" -eq 4 ] || { echo "usage: t0_final.sh S_COAL E_COAL S_FLAT E_FLAT" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scripts="${here}/../tm_language_model/scripts"
py="${here}/../tm_language_model/.venv-tm/bin/python"
torch_py="${here}/../clause_lm/.venv-cdllm-gpu/bin/python"
reg="${here}/registrations/T0.md"
mkdir -p "${here}/logs"
cd "${here}/src"
for seed in 1 2 3 4 5 6 7 8 9 10; do
    (
        out="${here}/logs/t0_final_coal_seed${seed}.out"
        nice -n 10 "${scripts}/cpu_slot.sh" 1 -- "${py}" -m tally.t0.run --mode final --machine coalesced \
            --clauses 2000 --seed "${seed}" --s "$1" --epochs "$2" --registration "${reg}" >"${out}" 2>&1
        run_id="$(sed -n 's/^run //p' "${out}" | head -n 1)"
        "${scripts}/gpu_run.sh" exclusive -- env PYTHONPATH=. "${torch_py}" -m tally.t0.logreg \
            "${here}/results/${run_id}" >"${here}/logs/t0_final_logreg_seed${seed}.out" 2>&1
    ) &
    nice -n 10 "${scripts}/cpu_slot.sh" 1 -- "${py}" -m tally.t0.run --mode final --machine flat \
        --clauses 100 --seed "${seed}" --s "$3" --epochs "$4" --registration "${reg}" \
        >"${here}/logs/t0_final_flat_seed${seed}.out" 2>&1 &
done
wait
