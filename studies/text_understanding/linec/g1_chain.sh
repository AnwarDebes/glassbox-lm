#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Line C G1, the TM arbiter (EXPLORATORY, validation only; G0 PASSED 16:12:25Z; rules in LOG.md before any run).
# Waits for linec/.g1_go (made on the queue's go for G1's GPU slot), then, one job at a time on the GPU:
# the grid (g1_select.py grid: 6 jobs), the 10,000-clause run of the grid's best (wide), seeds 2-3 at the pick
# (seeds), each job list written by g1_select.py from the rows at that point; then g1_select.py words (.venv-tm,
# for tmlm's bootstrap) -> linec/g1_words.txt. Hashes are checked before every job; a failed job is retried once
# after 10 minutes, a second failure stops the chain. Before every job it waits while linec/.g1_pause exists
# (made on the queue's go, e.g. when the decision_layer R22 is frozen: the current job finishes, then G1 waits).
# Log linec/runs_g1.log; rows linec/results_g1.jsonl.
lc=${REPO_ROOT}/studies/text_understanding/linec
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
tpy=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
dl=${REPO_ROOT}/studies/decision_layer/src/dl
log=$lc/runs_g1.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $lc/g1_tm.py 25205f5887d226de; check $lc/g1_select.py d696bbe50a310d38
  check $dl/tm.py f649043f5801605b; check $dl/tm_init.py 557ff69706529738
  check $lc/g0/rows_arbiter.npz 055779011ef4f585; check $lc/g0/rows_calibration.npz f4e159099bcc019c
  check $lc/g0/rows_validation.npz de5b40da42a2c315; check $lc/g0/val_bits_per_position.npz d8c77d7a94623fa3
  check $lc/g0/results.json c61e3faec6787708; }
runjob() { # $1 = stage|T|s|clauses|seed
  local stage T s clauses seed try e
  IFS='|' read -r stage T s clauses seed <<< "$1"
  for try in 1 2; do
    while [ -e $lc/.g1_pause ]; do sleep 60; done
    checks
    echo "$(ts) start (try $try) $1" >> "$log"
    "$gpu" shared -- "$py" -I -B $lc/g1_tm.py --T "$T" --s "$s" --clauses "$clauses" --seed "$seed" --stage "$stage" < /dev/null >> "$log" 2>&1
    e=$?
    echo "$(ts) end exit $e (try $try)" >> "$log"
    [ $e -eq 0 ] && return 0
    [ $try = 1 ] && sleep 600
  done
  stop "job failed twice: $1"
}
stage() {
  "$py" -I -B $lc/g1_select.py "$1" < /dev/null > $lc/.g1_jobs_$1.txt 2>> "$log" || stop "g1_select.py $1 failed"
  echo "$(ts) wrote $(wc -l < $lc/.g1_jobs_$1.txt) jobs for $1" >> "$log"
  local job
  while IFS= read -r job; do runjob "$job"; done < $lc/.g1_jobs_$1.txt
}
if [ "$1" = --dry ]; then [ -e $lc/.g1_go ] && echo "dry: go" || echo "dry: would wait for .g1_go"
  [ -e $lc/.g1_pause ] && echo "dry: paused" || echo "dry: not paused"; exit 0; fi
until [ -e $lc/.g1_go ]; do sleep 60; done
echo "$(ts) go: $(head -c 300 $lc/.g1_go)" >> "$log"
for st in grid wide seeds; do stage $st; done
checks
(cd /tmp && "$tpy" -I -B $lc/g1_select.py words < /dev/null >> "$log" 2>&1) || stop "g1_select.py words failed"
echo "$(ts) G1 done" >> "$log"
