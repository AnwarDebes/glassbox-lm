#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# wait_cpu.sh [MAXCORES] [MAXWAIT_S]: poll the decision_layer cpu_load.py (10 s window, read-only) until the cgroup uses at most
# MAXCORES cores (default 3.5); print each reading; exit 0 when below, 1 after MAXWAIT_S seconds (default 1800).
maxc="${1:-3.5}"; maxw="${2:-1800}"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
cl=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
t0=$(date +%s)
while true; do
  m=$(nice -n 19 "$py" -I "$cl" 10)
  echo "$m"
  c=$(awk '{print $3}' <<< "$m")
  awk -v c="$c" -v m="$maxc" 'BEGIN { exit !(c <= m) }' && exit 0
  [ $(( $(date +%s) - t0 )) -ge "$maxw" ] && exit 1
  sleep 20
done
