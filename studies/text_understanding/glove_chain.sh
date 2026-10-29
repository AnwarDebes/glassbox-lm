#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# glove_chain.sh: N7 GloVe arm (val only; rule in LOG.md 18:50:25Z): GloVe features for the three intent datasets,
# then existsT x seeds 1-3 with the test 2 harness (glove_exists_tm.py), ONE job at a time, shared GPU lock
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
out=$here/emb_glove6B300
log=$here/runs_glove.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
check() { [ "$(sha256sum "$here/$1" | cut -c1-16)" = "$2" ] || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Error: $1 changed; chain stopped" >> "$log"; exit 1; }; }
for ds in clinc150 banking77 hwu64; do
  check glove_feats.py 7b5e8760e1f5e107
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start glove_feats.py $ds" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/glove_feats.py" --dataset "$ds" --table glove6B300 --out-dir "$out" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end glove_feats.py $ds exit $?" >> "$log"
done
for seed in 1 2 3; do
  for ds in clinc150 banking77 hwu64; do
    check glove_exists_tm.py 8c77c3b265421de3; check exists_tm.py 7da0aee9f4eef796
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start glove_exists_tm.py $ds existsT seed $seed" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/glove_exists_tm.py" --dataset "$ds" --view existsT --seed "$seed" --emb-dir "$out" --tag "N7g_${ds}_s$seed" >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end glove_exists_tm.py $ds seed $seed exit $?" >> "$log"
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) glove chain done" >> "$log"
