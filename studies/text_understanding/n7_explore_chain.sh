#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_explore_chain.sh: N7 smoke before the registration freeze (train and our carved val split only; the test files
# are not opened): build the explore caches, then one 10-epoch TM run per dataset for timing. One job at a time.
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_n7_explore.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for ds in trec mr r52 r8 sst2; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start build $ds explore (n7_tm.py $(sha256sum $here/n7_tm.py | cut -c1-16))" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --build --dataset "$ds" --mode explore >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end build $ds exit $?" >> "$log"
done
for ds in trec mr r52 r8 sst2; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start smoke $ds bow+existsT_potion seed 1, 10 epochs" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --dataset "$ds" --mode explore --view bow+existsT_potion --seed 1 --epochs 10 >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end smoke $ds exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) n7 explore chain done" >> "$log"
