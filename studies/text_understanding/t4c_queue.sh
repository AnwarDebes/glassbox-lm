#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# t4c_queue.sh: test 4 exploration, round 3 (val only), two jobs side by side
set -uo pipefail
here=${REPO_ROOT}/studies/text_understanding
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
log=$here/runs_T4.log
run() {  # tag, extra args...
  tag=$1; shift
  echo "$(date -u +%H:%M:%SZ) start tm $tag $*" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I -B "$here/slot_tm.py" --parts W+S+P+N+G --seed 1 --batch 128 \
    --epochs 30 --patience 6 --emb-dir "$emb" --tag "$tag" "$@" 2>&1 | grep -v -i warning | tail -1 >> "$log"
}
run E4c_WSPNG_c8000_s1 --clauses 8000 --T 8000 --mem 0.2 &
run E4c_WSPNG_other33_s1 --other-keep 0.33 &
wait
echo "$(date -u +%H:%M:%SZ) round 3 done" >> "$log"
