#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_explore4_chain.sh: N7 smoke with the FROZEN code (train and carved val only; the test files are not opened):
# rebuild the explore caches (GloVe 42B, Yadav's vocabulary sizes) for the record, timing and the per-set GloVe OOV
# share, and one 10-epoch run of bow+existsT_glove on TREC for timing. Nothing changes whatever this shows.
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_n7_explore.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for ds in trec mr sst2 r52 r8; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start frozen-code build $ds explore (n7_tm.py $(sha256sum $here/n7_tm.py | cut -c1-16))" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --build --dataset "$ds" --mode explore >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end frozen-code build $ds exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start frozen-code smoke trec bow+existsT_glove seed 1, 10 epochs" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --dataset trec --mode explore --view bow+existsT_glove --seed 1 --epochs 10 >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end frozen-code smoke exit $?" >> "$log"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) n7 explore4 chain done" >> "$log"
