#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_test_chain.sh LANE: the registered N7 runs (REGISTRATION_N7.md). Lane A first checks the data files once and
# builds the five test-mode caches; then lanes A and B take runs from the same list (5 sets x 5 views x 5 seeds =
# 125), each run once (mkdir claims). Before every build and run the frozen code hashes are checked; on a mismatch
# the lane stops. Shared GPU lock. Log runs_n7_test.log; rows results_n7_test.jsonl (by n7_tm.py).
lane=$1
here=${REPO_ROOT}/studies/text_understanding
dl=${REPO_ROOT}/studies/decision_layer/src/dl
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_n7_test.log
claims=$here/.n7claims
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
stop() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STOP lane $lane: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() {
  check "$here/n7_tm.py" 3d715305b119b57b; check "$here/glove_feats.py" 7b5e8760e1f5e107
  check "$here/exists_tm.py" 7da0aee9f4eef796; check "$dl/tm.py" f649043f5801605b
  check "$dl/tm_init.py" 557ff69706529738; check "$dl/boolean.py" b5f5917f390da8a6
}
mkdir -p "$claims"
if [ "$lane" = A ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane A: data checks" >> "$log"
  (cd "$here/data_untrusted/glove42B" && sha256sum -c --quiet PARSED.sha256) || stop "GloVe 42B parsed files changed"
  for d in trec_setfit mr_dxgp sst2_setfit r52_dxgp r8_dxgp; do
    (cd "$here/data_untrusted/$d" && tail -n +2 SOURCE.txt | sha256sum -c --quiet) || stop "data $d changed"
  done
  for ds in trec mr sst2 r52 r8; do
    checks
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane A start build $ds test" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --build --dataset "$ds" --mode test >> "$log" 2>&1 || stop "build $ds failed"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane A end build $ds" >> "$log"
  done
  touch "$claims/caches_ready"
else
  while [ ! -e "$claims/caches_ready" ]; do sleep 20; done
fi
for ds in trec mr sst2 r52 r8; do
  for seed in 1 2 3 4 5; do
    for v in bow existsT_potion existsT_glove bow+existsT_potion bow+existsT_glove; do
      mkdir "$claims/${ds}_${v}_s$seed" 2>/dev/null || continue
      checks
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane $lane start $ds $v seed $seed" >> "$log"
      "$gpu" shared -- "$py" -I -B "$here/n7_tm.py" --dataset "$ds" --mode test --view "$v" --seed "$seed" >> "$log" 2>&1
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane $lane end $ds $v seed $seed exit $?" >> "$log"
    done
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane $lane done" >> "$log"
