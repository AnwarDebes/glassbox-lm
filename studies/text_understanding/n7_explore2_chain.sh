#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_explore2_chain.sh: N7 smoke, part 2 (train and our carved val only; test files not opened): saturation of the
# max over tokens (n7_saturation.py), then 100-epoch val runs of the anchor (bow) and of bow+existsT_potion, seed 1.
# Every dataset is registered and run whatever these show; they fix nothing. One job at a time.
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_n7_explore.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
while pgrep -f "^bash ${REPO_ROOT}/studies/text_understanding/n7_explore_chain.sh" > /dev/null; do sleep 10; done
for ds in clinc150 trec mr sst2 r52 r8; do
  for t in potion glove; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start saturation $ds $t (n7_saturation.py $(sha256sum $here/n7_saturation.py | cut -c1-16))" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/n7_saturation.py" --dataset "$ds" --table "$t" >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end saturation $ds $t exit $?" >> "$log"
  done
done
for ds in trec mr sst2 r52 r8; do
  for v in bow bow+existsT_potion; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start val100 $ds $v seed 1" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --dataset "$ds" --mode explore --view "$v" --seed 1 >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end val100 $ds $v exit $?" >> "$log"
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) n7 explore2 chain done" >> "$log"
