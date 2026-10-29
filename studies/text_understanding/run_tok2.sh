#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# run_tok2.sh JOBFILE LOG [WAIT_PID]: as run_tok.sh (each line of JOBFILE = arguments for tok_tm.py, one after the
# other under the shared GPU lock, last output line appended to LOG), plus: waits first until WAIT_PID has exited
# (so that at most two of my jobs run at once), and before every job checks the sha256 of the code the
# registrations froze; on a mismatch it writes "Error: hash mismatch" to LOG and stops.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
emb=${REPO_ROOT}/studies/text_understanding/emb_potion_test2
dl=${REPO_ROOT}/studies/decision_layer/src/dl
declare -A want=(
  ["$here/tok_tm.py"]=f17ef85c8f2c2c6d ["$here/graph_feats.py"]=cd0725094c7e014d
  ["$dl/tm.py"]=f649043f5801605b ["$dl/tm_init.py"]=557ff69706529738 ["$dl/boolean.py"]=b5f5917f390da8a6 ["$dl/data.py"]=0c4f8d22576e817c
)
if [ "$#" -ge 3 ]; then
  while kill -0 "$3" 2>/dev/null; do sleep 20; done
fi
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" =~ ^# ]] && continue
  for f in "${!want[@]}"; do
    got=$(sha256sum "$f" | cut -c1-16)
    if [ "$got" != "${want[$f]}" ]; then
      echo "$(date -u +%H:%M:%SZ) Error: hash mismatch $f $got (want ${want[$f]}); chain stopped before: $line" >> "$2"
      exit 1
    fi
  done
  echo "$(date -u +%H:%M:%SZ) start $line" >> "$2"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$gpu" shared -- "$py" -I "$here/tok_tm.py" --emb-dir "$emb" $line 2>&1 \
    | grep -v -i -E "warning|loading weights" | tail -1 >> "$2"
done < "$1"
echo "$(date -u +%H:%M:%SZ) batch done" >> "$2"
