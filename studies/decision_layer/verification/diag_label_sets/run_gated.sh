#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Runs label_sets.py once the cgroup's CPU load (scripts/cpu_load.py, 10 s) is at most 4.5 cores; 1 thread, nice 19, no GPU.
set -uo pipefail
S=${REPO_ROOT}/studies/decision_layer
D=${REPO_ROOT}/studies/decision_layer/verification/diag_label_sets
cd "$S" || exit 1
while true; do
  m=$(.venv-dl/bin/python scripts/cpu_load.py 10)
  echo "$m" >> "$D/gate.log"
  c=$(awk '{print $3}' <<< "$m")
  awk -v c="$c" 'BEGIN { exit !(c <= 4.5) }' && break
  sleep 50
done
echo "$(date -u +%FT%TZ) start label_sets.py" >> "$D/gate.log"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
  nice -n 19 .venv-dl/bin/python -I "$D/label_sets.py" "$D/label_sets_mwoz.json" > "$D/label_sets.out" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" >> "$D/gate.log"
exit $rc
