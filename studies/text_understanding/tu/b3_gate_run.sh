#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Run one CPU job behind the CPU gate (an ops helper for step 3b; tu/PLAN_3b.md, the CPU conditions of
# 07:15Z): wait until scripts/cpu_load.py over 10 s shows at most LIMIT cores (one check every 60 s, at most TRIES
# checks), then run the command at nice 19 with 2 threads and no GPU, its output and exit code in LOG, and print the
# log's sha256. It signals nothing and never overwrites LOG.
# Usage: b3_gate_run.sh LIMIT TRIES LOG -- command [args...]
set -u
LIMIT=${1:?limit in cores}; TRIES=${2:?tries}; LOG=${3:?log file}; shift 3
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command" >&2; exit 2; }
[ -e "$LOG" ] && { echo "refusing to overwrite $LOG" >&2; exit 2; }
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${B3_GATE_LOAD:-$py -I ${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py 10}
sleep_s=${B3_GATE_SLEEP:-60}
for i in $(seq 1 "$TRIES"); do
  line=$($load 2>&1); cores=$(echo "$line" | awk '{print $3}')
  if [ -n "$cores" ] && awk -v c="$cores" -v l="$LIMIT" 'BEGIN { exit !(c <= l) }'; then
    { echo "gate: $line (limit $LIMIT, check $i)"; echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ): $*"; } > "$LOG"
    nice -n 19 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$@" >> "$LOG" 2>&1
    e=$?
    echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $e" >> "$LOG"
    sha256sum "$LOG"
    exit $e
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gate held: $line (limit $LIMIT, check $i of $TRIES)"
  [ "$i" -lt "$TRIES" ] && sleep "$sleep_s"
done
echo "gate never opened in $TRIES checks: not run"
exit 4
