#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# t4_queue.sh: test 4 exploration (val only), started after the test 3 chains so that at most two of my jobs run:
# the CPU line (LR twins) waits for the bash of chain C1, the GPU line (MiniLM word vectors, then TM runs) for C2.
set -uo pipefail
here=${REPO_ROOT}/studies/text_understanding
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
log=$here/runs_T4.log
cpu_line() {
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
  for p in W S S+P+N W+S+P+N W+S+P+N+PW+NW W+S+P+N+PW+NW+G minilm minilm3; do
    if [[ $p == minilm* ]]; then until [ -f "$emb/massive_en__slots_minilm.npz" ]; do sleep 20; done; fi
    echo "$(date -u +%H:%M:%SZ) start lr $p" >> "$log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 "$py" -I -B "$here/slot_lr.py" --parts "$p" --emb-dir "$emb" 2>&1 | grep -v -i warning | tail -1 >> "$log"
  done
  echo "$(date -u +%H:%M:%SZ) cpu line done" >> "$log"
}
gpu_line() {
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
  echo "$(date -u +%H:%M:%SZ) start minilm_slots" >> "$log"
  OMP_NUM_THREADS=2 "$gpu" shared -- "$py" -I -B "$here/minilm_slots.py" --emb-dir "$emb" 2>&1 | grep -v -i warning | tail -1 >> "$log"
  for p in W+S+P+N W; do
    t=$(echo "$p" | tr -d '+')
    echo "$(date -u +%H:%M:%SZ) start tm $p" >> "$log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I -B "$here/slot_tm.py" --parts "$p" --seed 1 --emb-dir "$emb" --tag "E4_${t}_s1" 2>&1 | grep -v -i warning | tail -1 >> "$log"
  done
  echo "$(date -u +%H:%M:%SZ) gpu line done" >> "$log"
}
cpu_line "$1" &
gpu_line "$2" &
wait
echo "$(date -u +%H:%M:%SZ) batch done" >> "$log"
