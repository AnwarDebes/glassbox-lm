#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# step0b_chain.sh: logic-reader step 0b, EXPLORATORY, val only: stronger teachers (bge-base, bert-base) for the
# headroom check of step 0 (one job slot; the embedding runs under the shared GPU lock)
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_headroom.log
for t in bge-base bert-base; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start teacher_slots.py --teacher $t (teacher_slots.py $(sha256sum $here/teacher_slots.py | cut -c1-16))" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/teacher_slots.py" --emb-dir "$here/emb" --teacher "$t" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end teacher_slots.py --teacher $t exit $?" >> "$log"
done
for t in bge-base bert-base; do
  for b in W+S+P+N+G none; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start headroom_lr.py --teacher $t --extra minilm3 --bits $b (headroom_lr.py $(sha256sum $here/headroom_lr.py | cut -c1-16))" >> "$log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$py" -I -B "$here/headroom_lr.py" --emb-dir "$here/emb" --teacher "$t" --extra minilm3 --bits "$b" >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end headroom_lr.py --teacher $t --bits $b exit $?" >> "$log"
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) step 0b chain done" >> "$log"
