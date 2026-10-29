#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W3 step 1, the GPU part again (EXPLORATORY, intent VAL only; rules in LOG.md before any run). w3_chain.sh
# (18b88f9546d8fa6f) had a bug: its checks() looped over the tables with the global variable t, so every call
# reset the main loop's t to CFrand, and the GPU part would have run CFrand three times and never CF or RF. It was
# stopped at 16:29:53Z after one valid CFrand job (CLINC150 features, TM seed 1; correctly labelled). This copy is
# w3_chain.sh's GPU part with two changes and nothing else: checks() uses a local loop variable, and a job whose
# output already exists is skipped (a feature cache whose meta names the right table hash; a TM row with the same
# tag; an LR row with the same name and set), so no row is made twice. The CPU part is not repeated: the pairs file
# and the tables are checked against the hashes logged at 14:39:41Z and 15:31:57Z. Log w3/runs_w3.log.
emb=${REPO_ROOT}/studies/text_understanding
w3=$emb/w3
w1=$emb/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
oewn=$emb/data_untrusted/oewn2025/wndb/oewn2025
log=$w3/runs_w3.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PAIRS_H=681c7605c6512f5b
TAB_CF=3ffe09679133cb62; TAB_RF=4d84890ce8d864df; TAB_CFrand=e2a17a98d67194c5
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { local tt h
  check $w3/lexicon_pairs.py 95d14dbff2f9a746; check $w3/specialise.py 1f1872981e4f412a
  check $w3/w3_exists_tm.py 25b2a0c385e8b455; check $w3/w3_lr.py 1e11c93439fa69e4; check $w3/verdict_w3.py 88c6271a87780bc7
  check $w1/wordtab_feats.py 001a93b259a1b34f; check $w1/ppmi_svd2.py fe37efe876686391; check $w1/mo_tm.py 4d85663b1556ccf2
  check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/glove_feats.py 7b5e8760e1f5e107
  check $w1/tables/ppw_S1000_d500_e0.5.npz e1771d9eaf4a07f7
  check $oewn/data.noun 97c1c4e4f323e2b2; check $oewn/data.verb 2ef76318838583e5
  check $oewn/data.adj 98651304f9060235; check $oewn/data.adv c40c0bf2eb0c0ba6
  check $w3/pairs/oewn_pairs.npz "$PAIRS_H"
  for tt in CF RF CFrand; do eval "h=\$TAB_$tt"; check $w3/tables/${tt}_d500_e0.5.npz "$h"; done; return 0; }
run() { echo "$(ts) start $*" >> "$log"; "$gpu" shared -- "$py" -I -B "$@" < /dev/null >> "$log" 2>&1; local e=$?; echo "$(ts) end $(basename "$1") exit $e" >> "$log"; return $e; }
n9_running() { pgrep -f "glassbox-lm/studies/text_understanding/n9_sweep.py" > /dev/null; }
has_feats() { local f=$w3/feats/$1_d500_e0.5/$2__p32tok_K512; eval "local h=\$TAB_$1"
  [ -e $f.npz ] && grep -q "\"table_sha256\": \"$h\"" $f.meta.json; }
has_tm() { grep -q "\"tag\": \"W3_$1_$2_s$3\"" $w3/results_w3_val.jsonl 2>/dev/null; }
has_lr() { grep -q "\"name\": \"$1\", \"dataset\": \"$2\"" $w3/results_w3_lr.jsonl 2>/dev/null; }
if [ "$1" = --dry ]; then
  for t in CF RF CFrand; do for ds in clinc150 banking77 hwu64; do
    echo "dry: $t $ds feats $(has_feats $t $ds && echo have || echo run) TM $(for s in 1 2 3; do has_tm $t $ds $s && echo -n 'have ' || echo -n 'run '; done)LR $(has_lr $t $ds && echo have || echo run)"
  done; done; exit 0
fi
echo "$(ts) w3_chain2.sh (the GPU part again, with the checks() fix and skips of existing outputs) starts" >> "$log"
[ -e $emb/.n9/e74_over ] || stop "the e74 marker is missing"
touch $emb/.n9/pause
trap 'rm -f $emb/.n9/pause; echo "$(ts) pause removed" >> "$log"' EXIT
echo "$(ts) pause set for N9" >> "$log"
until ! n9_running; do sleep 30; done
echo "$(ts) no N9 job runs; GPU part starts" >> "$log"
for t in CF RF CFrand; do
  for ds in clinc150 banking77 hwu64; do
    checks
    if has_feats $t $ds; then echo "$(ts) skip: features $t $ds exist" >> "$log"
    else run $w1/wordtab_feats.py --dataset $ds --table wordtab:$w3/tables/${t}_d500_e0.5.npz --out-dir $w3/feats/${t}_d500_e0.5 || stop "wordtab_feats $t $ds"; fi
    for seed in 1 2 3; do
      checks
      if has_tm $t $ds $seed; then echo "$(ts) skip: TM row W3_${t}_${ds}_s$seed exists" >> "$log"; continue; fi
      run $w3/w3_exists_tm.py --dataset $ds --view existsT --seed $seed --emb-dir $w3/feats/${t}_d500_e0.5 --tag "W3_${t}_${ds}_s$seed" || stop "TM $t $ds $seed"
    done
    checks
    if has_lr $t $ds; then echo "$(ts) skip: LR row $t $ds exists" >> "$log"
    else run $w3/w3_lr.py --name $t --dataset $ds --emb-dir $w3/feats/${t}_d500_e0.5 || stop "LR $t $ds"; fi
  done
done
checks
echo "$(ts) start verdict_w3.py" >> "$log"
"$py" -I -B $w3/verdict_w3.py < /dev/null >> "$log" 2>&1 || stop "verdict_w3.py failed"
echo "$(ts) W3 step 1 done" >> "$log"
