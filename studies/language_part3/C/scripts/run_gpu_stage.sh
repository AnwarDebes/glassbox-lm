#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# C-core GPU stage runner (step 9 of R1 [C-core], PLAN_part3.md section 6; written under the step-9 amendment). The
# entry point of every GPU stage; the CPU stages keep scripts/run_stage.sh.
#
#   scripts/run_gpu_stage.sh <stage> [--dry]       stages, in order: tm_smoke, tm_grid, tm_seeds, tm_c0
#
# Before the job, in this order:
#   - refuses (exit 2) unless the go file C/.gpu_go exists (no code here creates it);
#   - one C job at a time: flock -n on C/tmp/c_job.lock, the lock of scripts/run_stage.sh (busy: exit 75);
#   - refuses (exit 75, try later) when `nvidia-smi --query-compute-apps=pid,process_name,used_memory
#     --format=csv,noheader` lists any GPU compute process, and (exit 2) when that list cannot be read.
# The job: python -I -B of .venv-dl (torch 2.6.0+cu124) at nice 19 with OMP / MKL / OPENBLAS / NUMBA / RAYON threads 2,
# CUDA_VISIBLE_DEVICES=0, CUDA_CACHE_DISABLE=1, NUMBA_CACHE_DIR, TMPDIR, JOBLIB_TEMP_FOLDER, MPLCONFIGDIR and
# XDG_CACHE_HOME inside C/tmp/, and C_GPU_RUNNER=1 (each stage refuses to start without it). The imported folders are
# listed before and after (ls -la --time-style=full-iso, no recursion) and compared: any change gives exit 4, except a
# change confined to ".." (parent folder) lines, which is a note (code review issue 6). Start and end go to
# runs/jobs_log.tsv. No CPU gate here: the GPU queue reads cpu_load.py before it starts a job.
# Exit codes: 0 done, decision PASS; 3 a registered STOP or a C-K3 HOLD (data/tm/<stage>_decision.json);
# 4 the imported folders changed; 2 usage error or refused start; 75 busy (a GPU compute process or C's lock);
# anything else: the stage failed or refused (its run folder's log.txt says why).
# Self-tests only: C_STEP9_TEST_ROOT=<a folder under C/tmp/ that holds .step9_test_root> moves the go file, the GPU
# process list (<root>/fake_nvidia_smi.sh), the lock, the listings and the job log into that folder and hides the GPU
# (CUDA_VISIBLE_DEVICES empty); the stages then run on the CPU with the test settings and cannot read or write C/data,
# C/data_dry or C/runs.
set -euo pipefail

C=${REPO_ROOT}/studies/language_part3/C
STUDIES=${REPO_ROOT}/studies
PY_DL=$STUDIES/decision_layer/.venv-dl/bin/python

usage() { echo "usage: run_gpu_stage.sh tm_smoke|tm_grid|tm_seeds|tm_c0 [--dry]" >&2; exit 2; }
[ "$#" -ge 1 ] || usage
stage=$1
shift
case "$stage" in
    tm_smoke|tm_grid|tm_seeds|tm_c0) script=$C/scripts/$stage.py ;;
    *) echo "unknown GPU stage: $stage" >&2; usage ;;
esac
for arg in "$@"; do
    [ "$arg" = "--dry" ] || { echo "unknown argument: $arg" >&2; usage; }
done

root=${C_STEP9_TEST_ROOT:-}
if [ -n "$root" ]; then
    ctmp=$(realpath -e "$C/tmp")
    root=$(realpath -e "$root") || { echo "no test root ${C_STEP9_TEST_ROOT}" >&2; exit 2; }
    case "$root/" in "$ctmp/"?*) ;; *) echo "test root $root is not a folder under $ctmp" >&2; exit 2 ;; esac
    [ -e "$root/.step9_test_root" ] || { echo "test root $root has no .step9_test_root marker" >&2; exit 2; }
    export C_STEP9_TEST_ROOT=$root
    GO=$root/.gpu_go
    NVSMI=(bash "$root/fake_nvidia_smi.sh")
    LOCK=$root/tmp/c_job.lock
    RUNS=$root/runs
    CVD=
else
    unset C_STEP9_TEST_ROOT
    GO=$C/.gpu_go
    NVSMI=(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
    LOCK=$C/tmp/c_job.lock
    RUNS=$C/runs
    CVD=0
fi

if [ ! -e "$GO" ]; then
    echo "no go file $GO: the GPU queue creates it when it gives the slot; nothing was run" >&2
    exit 2
fi
mkdir -p "$C/tmp/numba" "$C/tmp/tmp" "$C/tmp/joblib" "$C/tmp/mpl" "$C/tmp/xdg" "$RUNS/listings" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "another C job holds $LOCK" >&2; exit 75; }

set +e
procs=$("${NVSMI[@]}" 2>&1)
nrc=$?
set -e
if [ "$nrc" != "0" ]; then
    echo "cannot read the GPU compute-process list (exit $nrc): $procs" >&2
    exit 2
fi
if [ -n "$(printf '%s' "$procs" | tr -d '[:space:]')" ]; then
    echo "GPU compute processes present; nothing was run:" >&2
    printf '%s\n' "$procs" >&2
    exit 75
fi

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2
export NUMBA_CACHE_DIR=$C/tmp/numba TMPDIR=$C/tmp/tmp JOBLIB_TEMP_FOLDER=$C/tmp/joblib
export MPLCONFIGDIR=$C/tmp/mpl XDG_CACHE_HOME=$C/tmp/xdg
export CUDA_VISIBLE_DEVICES=$CVD CUDA_CACHE_DISABLE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false C_GPU_RUNNER=1

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
not_parent() { awk '$NF != ".."' "$1"; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
before=$RUNS/listings/${stamp}_${stage}_before.txt
after=$RUNS/listings/${stamp}_${stage}_after.txt
listing > "$before"
t_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
s_start=$(date +%s)
set +e
nice -n 19 "$PY_DL" -I -B "$script" "$@"
rc=$?
set -e
listing > "$after"
t_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
stage_rc=$rc
if diff "$before" "$after" > "$RUNS/listings/${stamp}_${stage}_diff.txt"; then
    listed="no change"
elif diff <(not_parent "$before") <(not_parent "$after") > /dev/null; then
    listed="note: only '..' (parent folder) lines changed"
else
    listed="CHANGED"
    rc=4
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t_start" "$t_end" "$stage" "$*" "$rc" "$(( $(date +%s) - s_start ))" \
    "$listed" >> "$RUNS/jobs_log.tsv"
echo "run_gpu_stage: $stage $* exit $rc (stage exit $stage_rc); listing before/after: $listed ($before)"
exit "$rc"
