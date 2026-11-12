#!/usr/bin/env bash
# The whole job of the registration E3 (registrations/E3_slide10_text.md), from the tests to the report, in one go:
#
#   registrations/E3_job.sh dry          dry run of the pipeline: DEV sentences, throwaway plan, runs under runs_dry/
#   registrations/E3_job.sh registered   the registered runs on the EVAL sentences
#
# Order: unit tests -> freeze of the plan with every pinned file -> check of the record -> the registered seeds,
# each followed by its probe -> the further seeds -> checks and report -> hashes of the outputs.
# The job stops at the first failure. All seeds are run whatever the earlier ones show; none is added or replaced.
# A seed whose complete run folder exists is skipped, so that the job can be continued after a crash: the crash is
# logged first and the folder of the crashed run is renamed <folder>_CRASHED.
set -euo pipefail
MODE="${1:?usage: E3_job.sh dry|registered}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${E3_PYTHON:-.venv/bin/python}"
H=../hierarchical_tm
EXTRAS=("$H/data/slide10/pairs_slide10.npz" "$H/data/slide10/slide10_meta.json" "$H/data/slide10/vocab_slide10.json"
        "$H/data/slide10/text/train.npz" "$H/data/slide10/text/dev.npz" "$H/src/slide10.py" "$H/src/htm_driver.py"
        "$H/src/recovered/e3fb_deletion_probe.py" registrations/E3_job.sh env/requirements.txt)
case "$MODE" in
  dry)
    PLAN=registrations/dry/E3_dry.md; DRY=(--dry); RUNS=runs_dry; PREFIX=e3dry_text_s; SPLIT=dev
    SEEDS="1 2 3 4 5"; FURTHER="6 7"; TAG=e3_dry ;;
  registered)
    PLAN=registrations/E3_slide10_text.md; DRY=(); RUNS=runs; PREFIX=e3_text_s; SPLIT=eval
    SEEDS="11 12 13 14 15"; FURTHER="$(seq -s ' ' 16 35)"; TAG=e3
    EXTRAS+=(logs/e3_dry_report.json) ;;   # the report of the dry run is pinned with the registration
  *) echo "usage: E3_job.sh dry|registered" >&2; exit 2 ;;
esac
RECORD="${PLAN%.md}.frozen.json"
OUT="logs/$TAG"
mkdir -p "$OUT"

# one thread, lowest priority, no GPU; compiled code is built from the files of this job, not taken from an older cache
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
unset TMLM_RUN_DIR
NUMBA_CACHE_DIR="$(mktemp -d)"; export NUMBA_CACHE_DIR
run() { nice -n 19 "$PY" -I "$@"; }

echo "== $(date -u +%FT%TZ) unit tests"
run tests/test_orgroup_tm.py > "$OUT/test_orgroup_tm.out" 2>&1
run tests/test_e3_units.py > "$OUT/test_e3_units.out" 2>&1

if [ ! -e "$RECORD" ]; then
  echo "== $(date -u +%FT%TZ) freeze"
  run src/freeze.py "$PLAN" "${EXTRAS[@]}" | tee "$OUT/freeze.out"
fi
echo "== $(date -u +%FT%TZ) check of the record"
run src/e3_common.py --check "$RECORD" "${DRY[@]}" > "$OUT/check_before_runs.json"

for S in $SEEDS $FURTHER; do
  DONE="$(ls -d "$RUNS"/*_"$PREFIX$S" 2>/dev/null | head -n 1 || true)"
  if [ -n "$DONE" ] && [ -f "$DONE/metrics.json" ]; then
    echo "== seed $S: run folder exists, not repeated ($DONE)"
  else
    echo "== $(date -u +%FT%TZ) seed $S"
    run src/run_slide10.py --version text --seed "$S" --clauses 20 --K 12 --T 8 --mode near --init-state -1 \
        --seed-state 0 --r 2 --p-rec 1 --p-rep 1 --q-seed 1 --max-seed 1 --reject weakest --reinforce critical \
        --neg-frac 0 --p-decay 0 --seed-all --passes 20 --n-train 0 --split "$SPLIT" --registered "$RECORD" \
        --name "$PREFIX$S" --out-root "$RUNS" > "$OUT/run_s$S.out" 2>&1
    DONE="$(ls -d "$RUNS"/*_"$PREFIX$S")"
  fi
  if [ ! -f "$DONE/probe_$SPLIT.json" ]; then
    run src/probe_slide10.py "$DONE" --split "$SPLIT" > "$OUT/probe_s$S.out" 2>&1
  fi
done

echo "== $(date -u +%FT%TZ) checks and report"
RC=0
run src/e3_report.py --frozen "$RECORD" "${DRY[@]}" --out "logs/${TAG}_report.json" > "$OUT/report.out" 2>&1 || RC=$?
date -u +%FT%TZ > "$OUT/finished_utc.txt"
sha256sum logs/"${TAG}"_report.json* "$OUT"/*.out "$OUT"/*.json > "$OUT/SHA256SUMS"
echo "== finished with exit code $RC; report: logs/${TAG}_report.json; outputs and hashes: $OUT/"
exit "$RC"
