#!/usr/bin/env bash
# GPU-sharing test + evaluation-schedule check (see the gates.md addenda).
# Purpose: decide "1 job or 2-3 jobs at once" for the queue of 2 Oct 2026, decide "--eval-every 5" for E4/E4s,
# and whether saved models can be re-scored exactly (so seeds 1-3 need no retraining for the final test planned for 3 Oct 2026).
# Order: (1) per setting: seed 1 alone, seed 1 alone again, seeds 1-2 shared, seeds 1-3 shared;
#        (2) eval-schedule check on the 2,000-clause setting (10 passes, eval every pass vs every 5th);
#        (3) LAST, the MPS arm (2 jobs per setting), so an MPS crash cannot spoil (1) and (2). The daemon is stopped after.
# Under MPS, jobs stop only through --max-minutes; the outer timeout is a far-away last resort (a killed MPS client
# can take down the others on Volta).
set -u
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=$STUDY; D=$H/src/htm_driver.py; OUT=$H/queue/share_test; mkdir -p $OUT
MPS=$H/queue/mps; mkdir -p $MPS/pipe $MPS/log
pgrep -f '^bash ./run_queue.sh' >/dev/null && { echo "queue still running; not starting"; exit 1; }
run() {  # gpu-mode run-name seed outer-timeout-min driver-args...
  local mode=$1 name=$2 seed=$3 cap=$4; shift 4
  local cfg; cfg=$(mktemp "$OUT/cfg_XXXX.json")
  echo "{\"name\": \"$name\", \"purpose\": \"GPU-sharing / eval-schedule test (exploratory)\", \"gpu_mode\": \"$mode\"}" > "$cfg"
  local lock=$mode; [ "$mode" = mps ] && lock=shared
  ( cd "$STUDY/../tm_language_model" && source env/env.sh >/dev/null 2>&1
    if [ "$mode" = mps ]; then export CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe CUDA_MPS_LOG_DIRECTORY=$MPS/log; fi
    timeout --kill-after=60 "${cap}m" scripts/mem_guard.py --scope group --limit-gib 12 -- \
    scripts/run_logged.py --name "$name" --config "$cfg" --results-dir $H/results -- \
    scripts/gpu_run.sh "$lock" -- .venv-htm/bin/python -u $D "$@" --seed "$seed" ) > "$OUT/$name.log" 2>&1
}
args() {
  if [ "$1" = granmo ]; then echo "--data sa --clauses 50 --A 64 --K 2 --T 3200 --s 100 --passes $2"
  else echo "--data sa --clauses 2000 --A 8 --K 2 --T 4000 --s 40 --passes $2"; fi
}
# (1) plain time-slicing
for setting in best granmo; do
  A=$(args $setting 4)
  t0=$(date +%s); run exclusive share_${setting}_n1_s1_exploratory 1 30 $A
  echo "$setting n=1 wall_s $(( $(date +%s) - t0 ))" >> $OUT/summary.txt
  run exclusive share_${setting}_n1b_s1_exploratory 1 30 $A
  for n in 2 3; do
    t0=$(date +%s)
    for s in $(seq 1 $n); do run shared share_${setting}_n${n}_s${s}_exploratory $s 30 $A & done
    wait
    echo "$setting n=$n wall_s $(( $(date +%s) - t0 ))" >> $OUT/summary.txt
  done
done
# (2) evaluation schedule: identical results with --eval-every 5?
A=$(args best 10)
run exclusive evalsched_best_every1_s1_exploratory 1 40 $A --eval-every 1 --save-model
run exclusive evalsched_best_every5_s1_exploratory 1 40 $A --eval-every 5
# (2b) save/load check: reload the saved model and re-score dev (must equal the in-run final dev numbers)
RD=$(ls -td $H/results/*_evalsched_best_every1_s1_exploratory | head -1)
( cd "$STUDY/../tm_language_model" && source env/env.sh >/dev/null 2>&1 && cfg=$(mktemp "$OUT/cfg_XXXX.json") && \
  echo '{"name": "saveload_check_exploratory", "purpose": "reload a saved model and re-score dev"}' > "$cfg" && \
  timeout --kill-after=60 20m scripts/mem_guard.py --scope group --limit-gib 12 -- \
  scripts/run_logged.py --name saveload_check_exploratory --config "$cfg" --results-dir $H/results -- \
  scripts/gpu_run.sh exclusive -- .venv-htm/bin/python -u $H/src/score_saved.py "$RD" model_final.pkl dev ) > $OUT/saveload_check.log 2>&1
# (3) MPS arm, last
if CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe CUDA_MPS_LOG_DIRECTORY=$MPS/log nvidia-cuda-mps-control -d; then
  echo "MPS daemon started" >> $OUT/summary.txt
  for setting in best granmo; do
    A=$(args $setting 4)
    t0=$(date +%s)
    for s in 1 2; do run mps share_${setting}_mps2_s${s}_exploratory $s 120 $A --max-minutes 25 & done
    sleep 60
    { echo "--- MPS check during $setting:"; echo get_server_list | CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe nvidia-cuda-mps-control;
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> $OUT/summary.txt 2>&1
    wait
    echo "$setting mps n=2 wall_s $(( $(date +%s) - t0 ))" >> $OUT/summary.txt
  done
  echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe nvidia-cuda-mps-control && echo "MPS daemon stopped" >> $OUT/summary.txt
else
  echo "MPS daemon could not start; MPS arm skipped" >> $OUT/summary.txt
fi
"$STUDY/../tm_language_model/.venv-tm/bin/python" $H/queue/analyse_share_test.py | tee $OUT/RESULT.md
