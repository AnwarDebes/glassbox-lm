#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W2 H-B step 0 part 2 (EXPLORATORY, VAL only; rules in LOG.md before any run). GPU order (decided 07:36Z): after N8.
# Waits until runs_n8c.log holds 'lane A done' or a STOP line and no n8_sweep.py runs, then until this cgroup's CPU
# load without it is at most 4.5 cores (cpu_load.py over 10 s, re-checked every 5 min; the gate was 3.0 in
# w2_part2_chain.sh 690a6d3ece54f2d4, raised at 13:03Z; this copy differs only in the gate). Every hash is checked before
# every step. Steps: (0) LR harness check: w2_lr.py on test 2's potion caches must give posthoc_lr.jsonl's
# exists_thermo rows again (within 0.002), else STOP; (1) pair_table.py (exit 5 = too little free GPU memory: wait
# 10 min, at most 12 tries); (2) for J and Jshift, per intent set: w1/wordtab_feats.py (unchanged), the TM seeds 1-3
# (w2_exists_tm.py) and the LR probe (w2_lr.py); (3) the LR probe on ppw_S1000's W1-S caches; (4) the same for the
# diagnostic table P (a failure there is logged, not fatal); (5) verdict_p2.py. Log w2/runs_w2p2.log.
w2=${REPO_ROOT}/studies/text_understanding/w2
emb=${REPO_ROOT}/studies/text_understanding
w1=$emb/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
potion=$emb/emb_potion_test2
log=$w2/runs_w2p2.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $w2/pair_table.py c18950834c5bb103; check $w2/w2_exists_tm.py e326b2a2ed8252b7; check $w2/w2_lr.py f424ba4c054f4a7c
  check $w2/verdict_p2.py fcf97e392da189e2; check $w1/wordtab_feats.py 001a93b259a1b34f; check $w1/ppmi_svd2.py fe37efe876686391
  check $w1/mo_tm.py 4d85663b1556ccf2; check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/glove_feats.py 7b5e8760e1f5e107
  check $w2/corpus/pairs_S1000.jsonl.gz 11f98abae10e5a3a; check $w1/tables/ppw_S1000_d500_e0.5.npz e1771d9eaf4a07f7; }
run() { echo "$(ts) start $*" >> "$log"; "$gpu" shared -- "$py" -I -B "$@" < /dev/null >> "$log" 2>&1; local e=$?; echo "$(ts) end $(basename "$1") exit $e" >> "$log"; return $e; }
n8_over() { grep -q 'lane A done\|STOP lane' $emb/runs_n8c.log 2>/dev/null; }
n8_running() { pgrep -f "glassbox-lm/studies/text_understanding/n8_sweep.py" > /dev/null; }
if [ "$1" = --dry ]; then
  n8_over && echo "dry: N8 over" || echo "dry: would wait for N8"; n8_running && echo "dry: n8_sweep.py running" || echo "dry: no n8_sweep.py"; exit 0
fi
until n8_over; do sleep 60; done
until ! n8_running; do sleep 30; done
echo "$(ts) N8 is over: $(grep -m1 'lane A done\|STOP lane' $emb/runs_n8c.log)" >> "$log"
while true; do
  line=$("$py" -I "$load" 10)
  c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}' && break
  sleep 300
done
checks
# (0) LR harness check
for ds in clinc150 banking77 hwu64; do run $w2/w2_lr.py --name potion --dataset $ds --emb-dir $potion || stop "LR potion $ds"; done
"$py" -I - >> "$log" 2>&1 <<'EOF' || stop "LR harness check failed (w2_lr.py does not give posthoc_lr.jsonl's exists_thermo rows)"
import json
emb = "${REPO_ROOT}/studies/text_understanding"
old = {r["dataset"]: r["lr_val"] for r in map(json.loads, open(emb + "/posthoc_lr.jsonl")) if r["bits"] == "exists_thermo"}
new = {}
for r in map(json.loads, open(emb + "/w2/results_w2_lr.jsonl")):
    if r["name"] == "potion":
        new[r["dataset"]] = r["lr_val"]
d = {k: round(new[k] - old[k], 4) for k in old}
print("LR harness check (w2_lr.py potion - posthoc_lr.jsonl exists_thermo):", d)
raise SystemExit(0 if all(abs(x) <= 0.002 for x in d.values()) else 1)
EOF
echo "$(ts) LR harness check passed" >> "$log"
# (1) tables
e=1
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  checks
  run $w2/pair_table.py; e=$?
  [ $e -eq 0 ] && break
  [ $e -eq 5 ] || stop "pair_table.py exit $e"
  sleep 600
done
[ $e -eq 0 ] || stop "too little free GPU memory after 12 tries"
# (2) J and Jshift; (3) the reference; (4) P
probe() { # $1 = table name, $2 = fatal (1) or not (0)
  local t=$1 ds seed
  for ds in clinc150 banking77 hwu64; do
    checks
    if ! run $w1/wordtab_feats.py --dataset $ds --table wordtab:$w2/tables/${t}_d500_e0.5.npz --out-dir $w2/feats/${t}_d500_e0.5; then
      [ "$2" = 1 ] && stop "wordtab_feats $t $ds"; echo "$(ts) $t $ds: features failed (diagnostic table, not fatal)" >> "$log"; continue
    fi
    for seed in 1 2 3; do
      checks
      run $w2/w2_exists_tm.py --dataset $ds --view existsT --seed $seed --emb-dir $w2/feats/${t}_d500_e0.5 --tag "W2P2_${t}_${ds}_s$seed" \
        || { [ "$2" = 1 ] && stop "TM $t $ds $seed"; echo "$(ts) $t $ds $seed: TM failed (not fatal)" >> "$log"; }
    done
    checks
    run $w2/w2_lr.py --name $t --dataset $ds --emb-dir $w2/feats/${t}_d500_e0.5 \
      || { [ "$2" = 1 ] && stop "LR $t $ds"; echo "$(ts) $t $ds: LR failed (not fatal)" >> "$log"; }
  done
}
probe J 1
probe Jshift 1
for ds in clinc150 banking77 hwu64; do checks; run $w2/w2_lr.py --name ppw_S1000 --dataset $ds --emb-dir $w1/feats/ppw_S1000_d500_e0.5 || stop "LR ppw_S1000 $ds"; done
probe P 0
# (5) verdict
checks
echo "$(ts) start verdict_p2.py" >> "$log"
"$py" -I -B $w2/verdict_p2.py < /dev/null >> "$log" 2>&1; e=$?
echo "$(ts) end verdict_p2.py exit $e" >> "$log"
[ $e -eq 0 ] || stop "verdict_p2.py exit $e"
echo "$(ts) part 2 done" >> "$log"
