#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# C-core job runner (PLAN_part3.md section 6, "Files and run rules"; Part 3 CPU rule).
#
#   scripts/run_stage.sh <stage> [args...]
#
# Stages (in order): step0 | prep | twin | learners | gates   (real runs need the frozen R1 record;
#                    the same stages with --dry use pseudo splits from train blocks only)
#                    selftest_tm | selftest_dl                  (self-tests, under 2 minutes, no CPU gate)
# Every job runs as `python -I -B` at nice 19 with 2 threads (OMP / MKL / OPENBLAS / NUMBA / RAYON = 2),
# CUDA_VISIBLE_DEVICES empty, and NUMBA_CACHE_DIR, TMPDIR, JOBLIB_TEMP_FOLDER, MPLCONFIGDIR and
# XDG_CACHE_HOME inside C/tmp/. Before and after each job the imported folders are listed with
# `ls -la --time-style=full-iso` (no recursion) and compared; any change makes the runner exit 4.
# Self-tests and --dry runs (tiny data) skip the CPU gate and are killed after 120 s. Every real stage waits for the CPU gate: cpu_load.py (10 s) at most 4.0 cores in use
# and 0 throttled periods, checked every 2 minutes; every check goes to runs/gate_log.tsv and every job
# start and end to runs/jobs_log.tsv (the gate-open hours of the K4 time box). One C job at a time
# (flock on C/tmp/c_job.lock); one Part 3 job at a time overall is the scheduler's rule.
set -euo pipefail

C=${REPO_ROOT}/studies/language_part3/C
STUDIES=${REPO_ROOT}/studies
PY_TM=$STUDIES/tm_language_model/.venv-tm/bin/python
PY_DL=$STUDIES/decision_layer/.venv-dl/bin/python
CPU_LOAD=$STUDIES/decision_layer/scripts/cpu_load.py
MAX_CORES=4.0
CHECK_EVERY=120

stage="${1:?usage: run_stage.sh <stage> [args...]}"
shift
case "$stage" in
    step0|prep|twin|gates) py=$PY_TM; script=$C/scripts/$stage.py; gate=1 ;;
    learners)              py=$PY_DL; script=$C/scripts/$stage.py; gate=1 ;;
    selftest_tm)           py=$PY_TM; script=$C/tests/selftest_tm.py; gate=0 ;;
    selftest_dl)           py=$PY_DL; script=$C/tests/selftest_dl.py; gate=0 ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

mkdir -p "$C/tmp/numba" "$C/tmp/tmp" "$C/tmp/joblib" "$C/tmp/mpl" "$C/tmp/xdg" "$C/runs/listings"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2
export NUMBA_CACHE_DIR=$C/tmp/numba TMPDIR=$C/tmp/tmp JOBLIB_TEMP_FOLDER=$C/tmp/joblib
export MPLCONFIGDIR=$C/tmp/mpl XDG_CACHE_HOME=$C/tmp/xdg
export CUDA_VISIBLE_DEVICES= CUDA_CACHE_DISABLE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false

LISTED=(
    "$STUDIES/tally/src/tally"
    "$STUDIES/tally/src/tally/__pycache__"
    "$STUDIES/tally/src/tally/floor"
    "$STUDIES/tally/src/tally/floor/__pycache__"
    "$STUDIES/decision_layer/src/dl"
    "$STUDIES/decision_layer/src/dl/__pycache__"
    "$STUDIES/tm_language_model/src/tmlm/eval/baselines"
    "$STUDIES/decision_layer/data/tables/c4words_S1000_v1"
    "$STUDIES/decision_layer/data/tables/c4words_S1000_v1/__pycache__"
    "${WS_ROOT}/tools/kenlm/bin"
)
listing() {
    local d
    for d in "${LISTED[@]}"; do
        echo "== $d"
        ls -la --time-style=full-iso "$d" 2>&1
    done
}

gate_open() {
    local out cores thr
    out=$(nice -n 19 "$PY_DL" -I -B "$CPU_LOAD" 10)
    cores=$(echo "$out" | awk '{print $3}')
    thr=$(echo "$out" | sed -n 's/.*throttled \([0-9][0-9]*\) of.*/\1/p')
    if awk -v c="$cores" -v m="$MAX_CORES" 'BEGIN { exit !(c <= m) }' && [ "$thr" = "0" ]; then
        printf '%s\t%s\t%s\t%s\topen\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$cores" "$thr" >> "$C/runs/gate_log.tsv"
        return 0
    fi
    printf '%s\t%s\t%s\t%s\tclosed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$cores" "$thr" >> "$C/runs/gate_log.tsv"
    return 1
}

exec 9>"$C/tmp/c_job.lock"
flock -n 9 || { echo "another C job holds $C/tmp/c_job.lock" >&2; exit 3; }

for arg in "$@"; do
    [ "$arg" = "--dry" ] && gate=0
done
limit=()
[ "$gate" = "0" ] && limit=(timeout 120)
if [ "$gate" = "1" ]; then
    until gate_open; do
        sleep $((CHECK_EVERY - 10))
    done
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
before=$C/runs/listings/${stamp}_${stage}_before.txt
after=$C/runs/listings/${stamp}_${stage}_after.txt
listing > "$before"
t_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
s_start=$(date +%s)
set +e
"${limit[@]}" nice -n 19 "$py" -I -B "$script" "$@"
rc=$?
set -e
listing > "$after"
t_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if diff "$before" "$after" > "$C/runs/listings/${stamp}_${stage}_diff.txt"; then
    listed="no change"
else
    listed="CHANGED"
    [ "$rc" = "0" ] && rc=4
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t_start" "$t_end" "$stage" "$*" "$rc" "$(( $(date +%s) - s_start ))" "$listed" >> "$C/runs/jobs_log.tsv"
echo "run_stage: $stage $* exit $rc; listing before/after: $listed ($before)"
exit "$rc"
