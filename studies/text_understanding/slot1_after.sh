#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# slot1_after.sh: after the process WAIT_PID ends (default 1903021, t5/slot1_next.sh; WAIT_PID=0 runs at once): logic-reader step 0,
# EXPLORATORY, val only (headroom_lr.py: LR on the test 4 bits with and without frozen MiniLM word vectors)
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$here/runs_headroom.log
while [ -e /proc/${WAIT_PID:-1903021} ]; do sleep 60; done
for extra in minilm3 none; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start headroom_lr.py --extra $extra (headroom_lr.py $(sha256sum $here/headroom_lr.py | cut -c1-16))" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$py" -I -B "$here/headroom_lr.py" --emb-dir "$here/emb" --extra "$extra" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end headroom_lr.py --extra $extra exit $?" >> "$log"
done
