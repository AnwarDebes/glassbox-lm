#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# run_batch.sh JOBFILE LOG: runs each line of JOBFILE (arguments for static_tm.py) one after the other
# under the shared GPU lock of the project; appends the last output line of each run to LOG.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" =~ ^# ]] && continue
  echo "$(date -u +%H:%M:%SZ) start $line" >> "$2"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I "$here/static_tm.py" --emb-dir "$emb" $line 2>&1 \
    | grep -v -i -E "warning|loading weights" | tail -1 >> "$2"
done < "$1"
echo "$(date -u +%H:%M:%SZ) batch done" >> "$2"
