#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# n7_followups_chain.sh: N7 follow-ups (EXPLORATORY; rules in LOG.md before any run). F2: GloVe 42B features for
# the three intent sets, then existsT x seeds 1-3 with the test 2 harness (glove_exists_tm.py), as glove_chain.sh did
# for GloVe 6B. F1: tmu's standard TM on TREC and MR bag of words (carved val). ONE job at a time; GPU jobs through
# the shared lock; the W1 step 0 chain holds my other slot. Log runs_n7_followups.log.
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
pytm=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
out=$here/emb_glove42B300
log=$here/runs_n7_followups.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
check() { [ "$(sha256sum "$here/$1" | cut -c1-16)" = "$2" ] || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Error: $1 changed; chain stopped" >> "$log"; exit 1; }; }
for ds in clinc150 banking77 hwu64; do
  check glove42_feats.py 915a21525c488ba3
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start glove42_feats.py $ds" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/glove42_feats.py" --dataset "$ds" --table glove42B300 --out-dir "$out" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end glove42_feats.py $ds exit $?" >> "$log"
done
for seed in 1 2 3; do
  for ds in clinc150 banking77 hwu64; do
    check glove_exists_tm.py 8c77c3b265421de3; check exists_tm.py 7da0aee9f4eef796
    grep -q '"table": "glove.42B.300d"' "$out/${ds}__p32tok_K512.meta.json" 2>/dev/null || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Error: no GloVe 42B cache for $ds; chain stopped" >> "$log"; exit 1; }
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start glove_exists_tm.py $ds existsT seed $seed (GloVe 42B)" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/glove_exists_tm.py" --dataset "$ds" --view existsT --seed "$seed" --emb-dir "$out" --tag "N7F2_${ds}_s$seed" >> "$log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end glove_exists_tm.py $ds seed $seed exit $?" >> "$log"
  done
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) F2 done" >> "$log"
for ds in trec mr; do
  check n7_tmu_calib.py e082d197a09869c1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start n7_tmu_calib.py $ds seed 1" >> "$log"
  "$pytm" -I -B "$here/n7_tmu_calib.py" --dataset "$ds" --seed 1 >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end n7_tmu_calib.py $ds exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) followups done" >> "$log"
