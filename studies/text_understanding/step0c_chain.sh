#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# step0c_chain.sh: logic-reader step 0c, EXPLORATORY, val only: the static part of the bert-base headroom
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$here/runs_headroom.log
for b in W+S+P+N+G none; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start headroom_lr.py --teacher bert-base --static --bits $b (headroom_lr.py $(sha256sum $here/headroom_lr.py | cut -c1-16))" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$py" -I -B "$here/headroom_lr.py" --emb-dir "$here/emb" --teacher bert-base --static --extra minilm3 --bits "$b" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end headroom_lr.py --teacher bert-base --static --bits $b exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) step 0c chain done" >> "$log"
