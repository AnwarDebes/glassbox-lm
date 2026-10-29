#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 step 0 (PLAN_tm_word_table.md): C4 download, corpus build with dedupe and slice hashes, the harness-equivalence
# check on train/val features (no TM trained on val, no accuracy), the speed pilot on the S100 train slice (timing
# and memory only, no table), and one short learning check on S1 with the held-out C4 check only. One job at a time,
# GPU through gpu_run.sh shared. Log w1/runs_w1_step0.log.
set -u
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_step0.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP step0: $1" >> "$log"; exit 1; }
echo "$(ts) step0 start: download" >> "$log"
bash "$here/get_c4.sh" >> "$log" 2>&1 || stop "download failed"
echo "$(ts) step0: corpus smoke (3,000 documents; nothing kept)" >> "$log"
smoke=$("$py" -I -B "$here/build_corpus_c4.py" --shards 00000,00001 --smoke 3000 2>> "$log" | tail -n 1)
echo "$smoke" >> "$log"
# gate fixed before the smoke: stop for a review if the filter drops more than 2% of the sentences
echo "$smoke" | "$py" -I -c "import json,sys; d=json.loads(sys.stdin.read()); f=d['filter']; r=(f['key']+f['ngram'])/max(f['sentences'],1); print('drop share', round(r,5)); sys.exit(0 if r <= 0.02 else 3)" >> "$log" 2>&1 \
  || stop "the filter drops more than 2% of the smoke sentences (or the smoke failed): review the rule"
echo "$(ts) step0: corpus build" >> "$log"
"$py" -I -B "$here/build_corpus_c4.py" --shards 00000,00001 >> "$log" 2>&1 || stop "corpus build failed"
CUDA_VISIBLE_DEVICES= "$py" -I -B "$here/make_potion_table.py" >> "$log" 2>&1 || stop "make_potion_table failed"
for ds in clinc150 banking77 hwu64; do
  for kind in potion tables/ref_potion_full.npz; do  # potion's own path, then the table path with potion's rows
    out="$here/feats/check_$(basename "$kind" .npz)"
    echo "$(ts) step0: equivalence $ds $kind" >> "$log"
    "$gpu" shared -- "$py" -I -B "$here/w1_feats.py" --dataset "$ds" --table "$kind" --out-dir "$out" >> "$log" 2>&1 || stop "w1_feats $ds $kind failed"
    "$gpu" shared -- "$py" -I -B "$here/check_equiv.py" --dataset "$ds" --new "$out" --ref ${REPO_ROOT}/studies/text_understanding/emb >> "$log" 2>&1 || stop "check_equiv $ds $kind failed"
  done
done
echo "$(ts) step0: speed pilot (S100 train slice)" >> "$log"
for cfg in "4000 1024 20000" "2000 1024 20000" "8000 1024 20000" "4000 256 20000" "4000 1024 10000"; do
  set -- $cfg
  "$gpu" shared -- "$py" -I -B "$here/mo_tm.py" --slice S100 --pilot 200 --tag "pilot_N$1_b$2_F$3" --clauses "$1" --T "$1" \
    --batch "$2" --features "$3" >> "$log" 2>&1 || echo "$(ts) pilot N$1 b$2 F$3 failed" >> "$log"
done
echo "$(ts) step0: learning check (S1, held-out C4 check only)" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/mo_tm.py" --slice S1 --tag dev_S1_learncheck --epochs 3 --eval-every 150 --patience 0 \
  >> "$log" 2>&1 || echo "$(ts) learning check failed" >> "$log"
echo "$(ts) step0 done" >> "$log"
