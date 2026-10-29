#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Run a command only when this cgroup uses <= 3.5 cores (cpu_load.py over 10 s), checking every ~60 s, at most
# MAX_WAIT_MIN minutes (default 90). Every check is logged to e4/.tmp/cpu_wait.log. Usage: when_cpu_free.sh <cmd...>
LOAD_PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
LOAD=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
LOG=${REPO_ROOT}/studies/agent_guards/e4/.tmp/cpu_wait.log
MAX=${MAX_WAIT_MIN:-90}
start=$(date +%s)
while true; do
  L="$(CUDA_VISIBLE_DEVICES= $LOAD_PY -I -B $LOAD 10)"; C="$(echo "$L" | awk '{print $3}')"
  echo "$L | waiting for: $*" >> "$LOG"
  if awk -v c="$C" 'BEGIN{exit !(c <= 3.5)}'; then
    echo "$(date -u +%FT%TZ) START $*" >> "$LOG"
    "$@"; rc=$?
    echo "$(date -u +%FT%TZ) END rc=$rc $*" >> "$LOG"
    exit $rc
  fi
  if [ $(( $(date +%s) - start )) -gt $(( MAX * 60 )) ]; then
    echo "$(date -u +%FT%TZ) GAVE UP after $MAX min: $*" >> "$LOG"; exit 75
  fi
  CUDA_VISIBLE_DEVICES= $LOAD_PY -I -B -c "import time; time.sleep(50)"
done
