#!/usr/bin/env bash
# B1 full grid: 10 seeds x 7 team sizes x all arms, one process per run, each in one CPU slot.
#
#   scripts/b1_launch.sh registrations/B1_<date>.md [JOBS]
#
# Needs the frozen registration (registrations/<name>.frozen.json next to it). JOBS parallel
# runs (default 5), each through cpu_slot.sh 1 under nice 10 with NUMBA_NUM_THREADS=1.
# Phase 1: self-checks, flat TM tuning (one per size), oracle, shared, selfish, marginal, dual,
#          greedy, random. Phase 2 (needs phase 1): flat (reads its tuning run) and ab (reads the
#          dual run of the same size and seed). Largest sizes first. Job logs: results/b1_launch_<UTC>/.
set -euo pipefail

reg="${1:?usage: b1_launch.sh registrations/<file>.md [JOBS]}"
jobs="${2:-5}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study="$(cd "${here}/.." && pwd)"
slot="${study}/../tm_language_model/scripts/cpu_slot.sh"
py="${study}/../tm_language_model/.venv-tm/bin/python"
reg="$(cd "$(dirname "${reg}")" && pwd)/$(basename "${reg}")"
frozen="${reg%.md}.frozen.json"
[ -f "${frozen}" ] || { echo "not frozen: ${frozen} missing" >&2; exit 2; }

export PYTHONPATH="${study}/src"
export NUMBA_NUM_THREADS=1
logs="${study}/results/b1_launch_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${logs}"

"${py}" -m tally.b1.selfcheck > "${logs}/selfcheck.txt" || { echo "self-checks failed" >&2; exit 1; }

sizes="512 256 128 64 32 16 8"
seeds="0 1 2 3 4 5 6 7 8 9"
phase1="${logs}/phase1.txt"
phase2="${logs}/phase2.txt"
: > "${phase1}"
: > "${phase2}"
for M in ${sizes}; do
    echo "flat_tune ${M} 10000" >> "${phase1}"
done
for M in ${sizes}; do
    for arm in dual marginal shared selfish greedy random oracle; do
        for s in ${seeds}; do
            echo "${arm} ${M} ${s}" >> "${phase1}"
        done
    done
    for arm in flat ab; do
        for s in ${seeds}; do
            echo "${arm} ${M} ${s}" >> "${phase2}"
        done
    done
done

run_one() {
    arm="$1"; M="$2"; s="$3"
    nice -n 10 "${slot}" 1 -- "${py}" -m tally.b1.run --arm "${arm}" --M "${M}" --seed "${s}" \
        --label reg --registration "${reg}" > "${logs}/${arm}_M${M}_s${s}.txt" 2>&1 \
        || echo "FAILED ${arm} ${M} ${s}" >> "${logs}/failed.txt"
}
export -f run_one
export slot py reg logs

echo "phase 1: $(wc -l < "${phase1}") runs, ${jobs} at a time; logs in ${logs}"
xargs -P "${jobs}" -L 1 bash -c 'run_one "$@"' _ < "${phase1}"
echo "phase 2: $(wc -l < "${phase2}") runs"
xargs -P "${jobs}" -L 1 bash -c 'run_one "$@"' _ < "${phase2}"
"${py}" -m tally.b1.report --label reg --out "${logs}/report.json" > "${logs}/report.txt" 2>&1 || true
echo "done; failed runs: $(cat "${logs}/failed.txt" 2>/dev/null | wc -l)"
