#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_explore3_chain.sh: N7 smoke, part 3 (train and carved val only): does the published TM setting of Yadav et al.
# (T 80, s 9) fix the bag-of-words anchor that test 2's setting (T 4000, s 8) breaks on short texts? Seed 1, 100 epochs.
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_n7_explore.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for ds in trec mr r52; do
  for v in bow bow+existsT_potion; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start T80s9 $ds $v seed 1 (n7_tm.py $(sha256sum $here/n7_tm.py | cut -c1-16))" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --dataset "$ds" --mode explore --view "$v" --seed 1 --T 80 --s 9 >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end T80s9 $ds $v exit $?" >> "$log"
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) n7 explore3 chain done" >> "$log"
