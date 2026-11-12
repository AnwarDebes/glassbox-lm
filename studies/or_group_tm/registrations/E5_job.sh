#!/usr/bin/env bash
# The whole job of the registration E5 (registrations/E5_wikipedia_test.md), from the tests to the report, in one go:
#
#   registrations/E5_job.sh dry          dry run of the pipeline: DEV pairs, throwaway plan, runs under runs_dry/
#   registrations/E5_job.sh registered   the registered runs on the test pairs
#
# Order: unit tests -> length and sha256 of the run log, then the freeze of the plan with every pinned file -> check
# of the record -> the runs, in three parallel streams, one per data version (sa, np, ns_all; one thread each; the
# seeds of a stream in order) -> checks and report -> sha256 of the outputs.
# The job takes no setting from the environment: the interpreter is .venv/bin/python of this study, and the CPU rule
# of the machine (at most 4.5 cores in use over 10 s, measured with ../decision_layer/scripts/cpu_load.py, a pinned
# file) is applied before each run and before the report; it changes only when they start.
# The job stops at the first failure. All seeds are run whatever the earlier ones show; none is added or replaced.
# Restart after a stop: a run whose folder holds the five files that the report requires is skipped; a folder without
# them is renamed <folder>_CRASHED (and logged here) and the run is repeated; a run that ended before writing a folder
# is repeated. Every output of every attempt is kept under its own UTC stamp.
set -euo pipefail
MODE="${1:?usage: E5_job.sh dry|registered}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
CPU_LOAD=../decision_layer/scripts/cpu_load.py
case "$MODE" in
  dry)
    PLAN=registrations/dry/E5_dry.md; DRY=(--dry); RUNS=runs_dry; PREFIX=e5dry_; SPLIT=dev; SEEDS="1"; TAG=e5_dry ;;
  registered)
    PLAN=registrations/E5_wikipedia_test.md; DRY=(); RUNS=runs; PREFIX=e5_; SPLIT=test; SEEDS="11 12 13 14 15"; TAG=e5 ;;
  *) echo "usage: E5_job.sh dry|registered" >&2; exit 2 ;;
esac
RECORD="${PLAN%.md}.frozen.json"
OUT="logs/$TAG"
mkdir -p "$OUT"
stamp() { date -u +%Y%m%dT%H%M%SZ; }

# one thread per process, lowest priority, no GPU; compiled code is built from the files of this job, not taken
# from an older cache
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
unset TMLM_RUN_DIR
NUMBA_CACHE_DIR="$(mktemp -d)"; export NUMBA_CACHE_DIR
run() { nice -n 19 "$PY" -I "$@"; }
wait_cpu() {  # the CPU rule of the machine
  local load
  while true; do
    load="$(run "$CPU_LOAD" 10 | sed -E 's/.*cores ([0-9.]+) over.*/\1/')"
    if [[ "$load" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v l="$load" 'BEGIN { exit !(l <= 4.5) }'; then return 0; fi
    echo "== $(date -u +%FT%TZ) CPU in use: ${load:-not measured} cores; waiting"
    sleep 20
  done
}
complete() {  # the five files that src/e5_report.py requires of a run folder
  local f
  for f in config.json metrics.json state.npz groups.txt "scores_$SPLIT.npz"; do [ -f "$1/$f" ] || return 1; done
}

echo "== $(date -u +%FT%TZ) unit tests"
wait_cpu
T="$(stamp)"
run tests/test_orgroup_tm.py > "$OUT/test_orgroup_tm_$T.out" 2>&1
run tests/test_e5_units.py > "$OUT/test_e5_units_$T.out" 2>&1

if [ ! -e "$RECORD" ]; then
  echo "== $(date -u +%FT%TZ) freeze"
  run src/e5_common.py --runlog-prefix > "$OUT/runs_wiki_prefix.json"
  mapfile -t EXTRAS < <(run src/e5_common.py --extras "${DRY[@]}")
  [ "${#EXTRAS[@]}" -gt 0 ]
  run src/freeze.py "$PLAN" "${EXTRAS[@]}" | tee "$OUT/freeze.out"
fi
echo "== $(date -u +%FT%TZ) check of the record"
run src/e5_common.py --check "$RECORD" "${DRY[@]}" > "$OUT/check_before_runs_$(stamp).json"

stream() {  # the seeds of one data version, in order
  local data=$1 s d have T
  for s in $SEEDS; do
    have=()
    for d in "$RUNS"/*_"$PREFIX${data}_s$s"; do [ -d "$d" ] && have+=("$d"); done
    if [ "${#have[@]}" -gt 1 ]; then
      echo "== $(date -u +%FT%TZ) $data seed $s: more than one run folder (${have[*]}); stop" >&2
      return 1
    fi
    if [ "${#have[@]}" -eq 1 ]; then
      if complete "${have[0]}"; then
        echo "== $data seed $s: complete run folder exists, not repeated (${have[0]})"
        continue
      fi
      mv "${have[0]}" "${have[0]}_CRASHED"
      echo "== $(date -u +%FT%TZ) $data seed $s: incomplete run folder renamed ${have[0]}_CRASHED; the run is repeated"
    fi
    wait_cpu
    T="$(stamp)"
    echo "== $(date -u +%FT%TZ) $data seed $s (output $OUT/run_${data}_s${s}_$T.out)"
    run src/run_wiki.py --data "$data" --seed "$s" --F 1024 --clauses 2000 --K 1 --T 40 --mode side --init-state -3 \
        --seed-state 0 --r 2 --p-rec 1 --p-rep 1 --q-seed 1 --max-seed 1 --reject weakest --reinforce share \
        --p-decay 0 --tied --wmax 64 --neg-frac 0 --smin -32 --smax 31 --passes 10 --eval-every 10 --n-train 0 \
        --n-dev 0 --top 40 --split "$SPLIT" --registered "$RECORD" --name "$PREFIX${data}_s$s" --out-root "$RUNS" \
        > "$OUT/run_${data}_s${s}_$T.out" 2>&1
  done
}
PIDS=()
I=0
for D in sa np ns_all; do
  ( sleep $((30 * I)); stream "$D" ) &   # staggered, so that each start sees the load of the streams before it
  PIDS+=("$!")
  I=$((I + 1))
done
FAIL=0
for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "== $(date -u +%FT%TZ) a run stream failed; see $OUT/run_*.out" >&2
  exit 1
fi

echo "== $(date -u +%FT%TZ) checks and report"
wait_cpu
RC=0
T="$(stamp)"
run src/e5_report.py --frozen "$RECORD" "${DRY[@]}" --out "logs/${TAG}_report.json" > "$OUT/report_$T.out" 2>&1 || RC=$?
date -u +%FT%TZ >> "$OUT/finished_utc.txt"
{ ls logs/"${TAG}"_report.json* 2>/dev/null; find "$OUT" -maxdepth 1 -type f ! -name 'SHA256SUMS*' | sort; echo logs/RUNS_wiki.md; } \
  | xargs sha256sum > "$OUT/SHA256SUMS_$T"
echo "== finished with exit code $RC; report: logs/${TAG}_report.json; outputs and hashes: $OUT/ (SHA256SUMS_$T)"
exit "$RC"
