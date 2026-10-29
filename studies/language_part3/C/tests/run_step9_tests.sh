#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Self-tests of C-core's step-9 code (written under the step-9 amendment). Synthetic data only; no number of C.
#
#   bash studies/language_part3/C/tests/run_step9_tests.sh
#
# Waits for nothing: it refuses (exit 2) unless cpu_load.py (10 s) shows at most 4.0 cores and 0 throttled periods.
# Then runs, each as python -I -B at nice 19 with the environment of scripts/run_stage.sh (OMP / MKL / OPENBLAS /
# NUMBA / RAYON threads 2, caches and temp files in C/tmp/, CUDA_VISIBLE_DEVICES empty, CUDA_CACHE_DISABLE=1):
#   1. tests/selftest_step9.py      (.venv-dl)  unit tests
#   2. tests/selftest_step9_e2e.py  (.venv-dl)  end-to-end smoke through scripts/run_gpu_stage.sh in self-test mode
#   3. tests/selftest_tm.py         (.venv-tm)  the older self-tests, on the changed code
#   4. tests/selftest_dl.py         (.venv-dl)
# Not through scripts/run_stage.sh, so nothing is written under C/runs/ (no listing file, no jobs_log.tsv line):
# each output goes to tests/out/step9_<name>_<UTC>.txt, with its sha256 printed. The imported folders are listed
# before and after (the folders of run_stage.sh) and compared: exit 4 on a change. Exit 1 if any file failed.
set -uo pipefail

C=${REPO_ROOT}/studies/language_part3/C
STUDIES=${REPO_ROOT}/studies
PY_TM=$STUDIES/tm_language_model/.venv-tm/bin/python
PY_DL=$STUDIES/decision_layer/.venv-dl/bin/python
OUT=$C/tests/out

load=$(nice -n 19 "$PY_DL" -I -B "$STUDIES/decision_layer/scripts/cpu_load.py" 10)
cores=$(echo "$load" | awk '{print $3}')
thr=$(echo "$load" | sed -n 's/.*throttled \([0-9][0-9]*\) of.*/\1/p')
echo "CPU gate: $load"
if ! awk -v c="$cores" 'BEGIN { exit !(c <= 4.0) }' || [ "$thr" != "0" ]; then
    echo "CPU gate closed (needs at most 4.0 cores and 0 throttled periods): not run"
    exit 2
fi

mkdir -p "$C/tmp/numba" "$C/tmp/tmp" "$C/tmp/joblib" "$C/tmp/mpl" "$C/tmp/xdg" "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2
export NUMBA_CACHE_DIR=$C/tmp/numba TMPDIR=$C/tmp/tmp JOBLIB_TEMP_FOLDER=$C/tmp/joblib
export MPLCONFIGDIR=$C/tmp/mpl XDG_CACHE_HOME=$C/tmp/xdg
export CUDA_VISIBLE_DEVICES= CUDA_CACHE_DISABLE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
unset C_STEP9_TEST_ROOT C_GPU_RUNNER

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
watched() {   # C's real records: the tests must not change them
    ls -la --time-style=full-iso "$C/data" "$C/data_dry" "$C/runs" "$C/R1_C-core.frozen.json" "$C/R1_C-core.md" 2>&1
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
before=$(listing)
wbefore=$(watched)
fails=0
run_one() {
    local name=$1 py=$2 file=$3 out rc
    out=$OUT/step9_${name}_${stamp}.txt
    (cd "$C" && nice -n 19 "$py" -I -B "$file") > "$out" 2>&1
    rc=$?
    tail -n 1 "$out"
    echo "  exit $rc; $(sha256sum "$out")"
    [ "$rc" = "0" ] || fails=$((fails + 1))
}
run_one unit "$PY_DL" tests/selftest_step9.py
run_one e2e "$PY_DL" tests/selftest_step9_e2e.py
run_one selftest_tm "$PY_TM" tests/selftest_tm.py
run_one selftest_dl "$PY_DL" tests/selftest_dl.py
after=$(listing)
wafter=$(watched)
rc=0
if [ "$before" = "$after" ]; then
    echo "listing of the imported folders: no change"
else
    echo "listing of the imported folders: CHANGED"
    diff <(echo "$before") <(echo "$after")
    rc=4
fi
if [ "$wbefore" = "$wafter" ]; then
    echo "C/data, C/data_dry, C/runs and the frozen record: no change"
else
    echo "C/data, C/data_dry, C/runs or the frozen record CHANGED"
    diff <(echo "$wbefore") <(echo "$wafter")
    rc=4
fi
[ "$fails" = "0" ] || rc=1
echo "run_step9_tests: $((4 - fails)) of 4 files passed; exit $rc (outputs tests/out/step9_*_${stamp}.txt)"
exit "$rc"
