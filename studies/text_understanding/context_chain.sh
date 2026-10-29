#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# context_chain.sh: slot context checks (EXPLORATORY, val only) after the process WAIT_PID ends (slot 2)
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$here/runs_headroom.log
while [ -e /proc/${WAIT_PID:-0} ]; do sleep 20; done
for spec in "W+S+P+N+P2+N2+G -" "W+S+P+N+G --prev"; do
  set -- $spec
  flag=$2; [ "$flag" = - ] && flag=""
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start prevlabel_lr.py --bits $1 $flag (prevlabel_lr.py $(sha256sum $here/prevlabel_lr.py | cut -c1-16))" >> "$log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$py" -I -B "$here/prevlabel_lr.py" --emb-dir "$here/emb" --bits "$1" $flag >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end prevlabel_lr.py --bits $1 $flag exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) context chain done" >> "$log"
