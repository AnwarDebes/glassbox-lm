#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Wait until this cgroup uses less than LIMIT cores (default 3.5) over a 10 s window; give up after N tries (default 60).
# Prints the last measurement; exit 0 when quiet, 1 when it gave up. Read-only (cpu_load.py reads /sys/fs/cgroup).
LIMIT=${1:-3.5}; N=${2:-60}
P=${REPO_ROOT}/studies/decision_layer
for _ in $(seq 1 "$N"); do
  r=$(CUDA_VISIBLE_DEVICES='' "$P/.venv-dl/bin/python" -I "$P/scripts/cpu_load.py" 10)
  c=$(echo "$r" | awk '{print $3}')
  if awk -v c="$c" -v l="$LIMIT" 'BEGIN{exit !(c<l)}'; then echo "quiet: $r"; exit 0; fi
done
echo "gave up: $r"; exit 1
