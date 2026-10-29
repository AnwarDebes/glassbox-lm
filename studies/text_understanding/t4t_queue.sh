#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# t4t_queue.sh LANE: registered run T4 (REGISTRATION_T4.md). Lane A: the LR test rows, then TM seeds 1, 2, 3.
# Lane B (slot 2, after the test 5 pilot): TM seeds 3, 2. Each seed runs once (mkdir claims).
set -uo pipefail
lane=$1
here=${REPO_ROOT}/studies/text_understanding
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
log=$here/runs_T4t.log
cfg="--clauses 8000 --T 8000 --mem 0.2 --epochs 60 --patience 8 --other-keep 0.33"
check() { [ "$(sha256sum "$here/$1" | cut -c1-16)" = "$2" ] || { echo "$(date -u +%H:%M:%SZ) STOP lane $lane: $1 changed" >> "$log"; exit 1; }; }
tm() {
  s=$1; mkdir "$here/.t4claim_s$s" 2>/dev/null || return 0
  check slot_tm.py 18bce7a2bc1619b2; check slot_feats.py 70045d1f5db00335; check slot_eval.py f468f9c3b1e3be42
  echo "$(date -u +%H:%M:%SZ) lane $lane start tm seed $s" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I -B "$here/slot_tm.py" --parts W+S+P+N+G --seed "$s" --batch 128 \
    $cfg --emb-dir "$emb" --tag "T4_s$s" --test --rules 5 > "$here/runs_T4t_s$s.log" 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%SZ) lane $lane end tm seed $s (exit $rc): $(tail -1 "$here/runs_T4t_s$s.log" | cut -c1-400)" >> "$log"
}
lr() {
  check slot_lr.py 38b89d92a1d5cb63; check slot_feats.py 70045d1f5db00335; check slot_eval.py f468f9c3b1e3be42
  echo "$(date -u +%H:%M:%SZ) lane $lane start lr $1" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$py" -I -B "$here/slot_lr.py" --parts "$1" --emb-dir "$emb" --test > "$here/runs_T4t_lr_$1.log" 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%SZ) lane $lane end lr $1 (exit $rc): $(tail -1 "$here/runs_T4t_lr_$1.log" | cut -c1-400)" >> "$log"
}
if [ "$lane" = A ]; then
  lr W+S+P+N+G; lr minilm3; lr W
  tm 1; tm 2; tm 3
else
  tm 3; tm 2
fi
echo "$(date -u +%H:%M:%SZ) lane $lane done" >> "$log"
