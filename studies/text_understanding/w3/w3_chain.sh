#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W3 step 1 (EXPLORATORY, intent VAL only; rules in LOG.md before any run). CPU part: when this cgroup's CPU load
# without it is at most 4.5 cores (cpu_load.py over 10 s, re-checked every 5 min): lexicon_pairs.py, then
# specialise.py (2 threads). GPU part, after the queue's go that e74 is over (.n9/e74_over): touch .n9/pause
# (N9 waits between its jobs while it exists; it is removed when this script exits, whatever the exit), wait until
# no n9_sweep.py runs, then for CF, RF and CFrand on CLINC150, Banking77 and HWU64: w1/wordtab_feats.py (unchanged),
# the TM seeds 1-3 (w3_exists_tm.py) and the LR probe (w3_lr.py); then verdict_w3.py. Every hash is checked before
# every step; the pairs file and the tables are hashed when made and checked after. Log w3/runs_w3.log.
emb=${REPO_ROOT}/studies/text_understanding
w3=$emb/w3
w1=$emb/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
oewn=$emb/data_untrusted/oewn2025/wndb/oewn2025
log=$w3/runs_w3.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $w3/lexicon_pairs.py 95d14dbff2f9a746; check $w3/specialise.py 1f1872981e4f412a
  check $w3/w3_exists_tm.py 25b2a0c385e8b455; check $w3/w3_lr.py 1e11c93439fa69e4; check $w3/verdict_w3.py 88c6271a87780bc7
  check $w1/wordtab_feats.py 001a93b259a1b34f; check $w1/ppmi_svd2.py fe37efe876686391; check $w1/mo_tm.py 4d85663b1556ccf2
  check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/glove_feats.py 7b5e8760e1f5e107
  check $w1/tables/ppw_S1000_d500_e0.5.npz e1771d9eaf4a07f7
  check $oewn/data.noun 97c1c4e4f323e2b2; check $oewn/data.verb 2ef76318838583e5
  check $oewn/data.adj 98651304f9060235; check $oewn/data.adv c40c0bf2eb0c0ba6
  [ -n "$PAIRS_H" ] && check $w3/pairs/oewn_pairs.npz "$PAIRS_H"
  for t in CF RF CFrand; do eval "h=\$TAB_$t"; [ -n "$h" ] && check $w3/tables/${t}_d500_e0.5.npz "$h"; done; return 0; }
run() { echo "$(ts) start $*" >> "$log"; "$gpu" shared -- "$py" -I -B "$@" < /dev/null >> "$log" 2>&1; local e=$?; echo "$(ts) end $(basename "$1") exit $e" >> "$log"; return $e; }
n9_running() { pgrep -f "glassbox-lm/studies/text_understanding/n9_sweep.py" > /dev/null; }
if [ "$1" = --dry ]; then
  [ -e $emb/.n9/e74_over ] && echo "dry: e74 over" || echo "dry: would wait for e74"
  n9_running && echo "dry: an N9 job runs" || echo "dry: no N9 job"; exit 0
fi
# CPU part
while true; do
  line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}' && break
  sleep 300
done
checks
echo "$(ts) start lexicon_pairs.py" >> "$log"
(cd /tmp && "$py" -I -B $w3/lexicon_pairs.py < /dev/null >> "$log" 2>&1) || stop "lexicon_pairs.py failed"
PAIRS_H=$(sha256sum $w3/pairs/oewn_pairs.npz | cut -c1-16); echo "$(ts) pairs file $PAIRS_H" >> "$log"
checks
echo "$(ts) start specialise.py" >> "$log"
(cd /tmp && CUDA_VISIBLE_DEVICES= "$py" -I -B $w3/specialise.py --device cpu < /dev/null >> "$log" 2>&1) || stop "specialise.py failed"
for t in CF RF CFrand; do eval "TAB_$t=$(sha256sum $w3/tables/${t}_d500_e0.5.npz | cut -c1-16)"; done
echo "$(ts) tables CF $TAB_CF RF $TAB_RF CFrand $TAB_CFrand" >> "$log"
# GPU part
until [ -e $emb/.n9/e74_over ]; do sleep 60; done
echo "$(ts) e74 is over: $(head -c 200 $emb/.n9/e74_over)" >> "$log"
touch $emb/.n9/pause
trap 'rm -f $emb/.n9/pause; echo "$(ts) pause removed" >> "$log"' EXIT
echo "$(ts) pause set for N9" >> "$log"
until ! n9_running; do sleep 30; done
echo "$(ts) no N9 job runs; GPU part starts" >> "$log"
for t in CF RF CFrand; do
  for ds in clinc150 banking77 hwu64; do
    checks
    run $w1/wordtab_feats.py --dataset $ds --table wordtab:$w3/tables/${t}_d500_e0.5.npz --out-dir $w3/feats/${t}_d500_e0.5 || stop "wordtab_feats $t $ds"
    for seed in 1 2 3; do
      checks
      run $w3/w3_exists_tm.py --dataset $ds --view existsT --seed $seed --emb-dir $w3/feats/${t}_d500_e0.5 --tag "W3_${t}_${ds}_s$seed" || stop "TM $t $ds $seed"
    done
    checks
    run $w3/w3_lr.py --name $t --dataset $ds --emb-dir $w3/feats/${t}_d500_e0.5 || stop "LR $t $ds"
  done
done
checks
echo "$(ts) start verdict_w3.py" >> "$log"
"$py" -I -B $w3/verdict_w3.py < /dev/null >> "$log" 2>&1 || stop "verdict_w3.py failed"
echo "$(ts) W3 step 1 done" >> "$log"
