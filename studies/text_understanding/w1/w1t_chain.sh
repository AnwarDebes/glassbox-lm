#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1-T (REGISTRATION_W1T.md): the C4-word arm on N7's test splits (second use, declared). Lane A builds the five
# test-mode caches (each checks that its train and val bits equal the W1-S val-run cache), touches .w1t/caches_ready;
# both lanes take the 25 runs (5 sets x 5 seeds) by mkdir claims. Code and table hashes are checked before every build
# and run; on a mismatch the lane stops. Shared GPU lock. Log runs_w1t.log; rows results_w1t_test.jsonl.
lane=$1
here=${REPO_ROOT}/studies/text_understanding/w1
emb=${REPO_ROOT}/studies/text_understanding
dl=${REPO_ROOT}/studies/decision_layer/src/dl
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1t.log
cl=$here/.w1t
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP lane $lane: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $here/w1_n7_test.py 91906d0248803783; check $here/w1_n7w.py b54688135df9f106; check $here/w1_n7.py c8bcdd17ab999579
  check $emb/n7_tm.py 3d715305b119b57b; check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/glove_feats.py 7b5e8760e1f5e107
  check $dl/tm.py f649043f5801605b; check $dl/tm_init.py 557ff69706529738; check $dl/boolean.py b5f5917f390da8a6
  check $here/tables/ppw_S1000_d500_e0.5.npz e1771d9eaf4a07f7; }
mkdir -p "$cl"
if [ "$lane" = A ]; then
  for ds in trec mr sst2 r52 r8; do
    checks
    echo "$(ts) lane A start build $ds test" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/w1_n7_test.py" --build --dataset "$ds" >> "$log" 2>&1 || stop "build $ds failed (or its train / val bits differ from the val-run cache)"
    echo "$(ts) lane A end build $ds" >> "$log"
  done
  touch "$cl/caches_ready"
else
  while [ ! -e "$cl/caches_ready" ]; do grep -q 'STOP lane A' "$log" 2>/dev/null && exit 1; sleep 20; done
fi
for ds in trec mr sst2 r52 r8; do
  for seed in 1 2 3 4 5; do
    mkdir "$cl/${ds}_s$seed" 2>/dev/null || continue
    checks
    echo "$(ts) lane $lane start $ds seed $seed" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/w1_n7_test.py" --dataset "$ds" --seed "$seed" >> "$log" 2>&1
    echo "$(ts) lane $lane end $ds seed $seed exit $?" >> "$log"
  done
done
echo "$(ts) lane $lane done" >> "$log"
