#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# t4b_queue.sh: test 4 exploration, second round (val only), two lines so that at most two of my jobs run
set -uo pipefail
here=${REPO_ROOT}/studies/text_understanding
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
log=$here/runs_T4.log
tm() {  # parts tag
  echo "$(date -u +%H:%M:%SZ) start tm $1 batch 128" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I -B "$here/slot_tm.py" --parts "$1" --seed 1 --batch 128 \
    --epochs 30 --patience 6 --emb-dir "$emb" --tag "$2" 2>&1 | grep -v -i warning | tail -1 >> "$log"
}
( tm W+S+P+N+G E4b_WSPNG_b128_s1 ) &
( echo "$(date -u +%H:%M:%SZ) start lr W+S+P+N+G" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 "$py" -I -B "$here/slot_lr.py" --parts W+S+P+N+G --emb-dir "$emb" 2>&1 | grep -v -i warning | tail -1 >> "$log"
  tm W+S+P+N E4b_WSPN_b128_s1 ) &
wait
echo "$(date -u +%H:%M:%SZ) round 2 done" >> "$log"
