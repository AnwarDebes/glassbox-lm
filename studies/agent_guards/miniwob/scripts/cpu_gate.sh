#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Wait until the cgroup CPU load (cpu_load.py 10, third field) is <= 3.5 cores; give up after N tries (default 60 = ~10 min).
# Exit 0 = go, 3 = still busy. Each try measures for 10 s (no sleep needed).
N=${1:-60}
for i in $(seq 1 "$N"); do
  L=$(${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python -I ${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py 10)
  C=$(echo "$L" | awk '{print $3}')
  if awk "BEGIN{exit !($C <= 3.5)}"; then echo "cpu gate OK: $L"; exit 0; fi
  echo "cpu gate wait ($i/$N): $L"
done
echo "cpu gate: still above 3.5 cores after $N tries"; exit 3
