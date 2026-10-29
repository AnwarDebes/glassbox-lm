#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Step 0b chain (PLAN_text_understanding.md section 1b, rules fixed there; VAL only; the test split is never read).
# Waits for tu/.0b/gpu_go (made on the queue's go that the GPU is free for 0b; one big GPU job at a time), then
# the screen (seed 1; s 2/4/8/12/16/24; C4 table ppw_S1000 view, then potion; CLINC150, Banking77, HWU64), the pick
# (b0_select.py), the seeds (s 8 and the pick on seeds 4-6) and the verdict (b0_verdict.py). Before every run: the
# hashes, and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the last check is over 5 min old).
# Restart-safe: a run whose row already exists is skipped. Log tu/runs_0b.log.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$tu/runs_0b.log
go=$tu/.0b/gpu_go
declare -A DIR=([ppw_S1000_d500_e0.5]=$emb/w1/feats/ppw_S1000_d500_e0.5 [potion-base-32M]=$emb/emb_potion_test2)
TABLES="ppw_S1000_d500_e0.5 potion-base-32M"
SETS="clinc150 banking77 hwu64"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $tu/b0_exists_tm.py 9fbc334af9aea8ff; check $tu/b0_select.py 87a5c2d5e3d36fbf; check $tu/b0_verdict.py db15e85dca29323e
  check $emb/exists_tm.py 7da0aee9f4eef796; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120
  done; }
have() { local t=$1 d=$2 s=$3 sd=$4 tag=$5
  [ -e $tu/results_0b_val.jsonl ] || return 1
  "$py" -I -c "
import json, sys
t, d, s, sd, tag, f = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
rows = [json.loads(l) for l in open(f)]
sys.exit(0 if any(r.get('tag') == tag and r['table'] == t and r['dataset'] == d and float(r['s']) == s and r['seed'] == sd for r in rows) else 1)
" "$t" "$d" "$s" "$sd" "$tag" $tu/results_0b_val.jsonl; }
run() { local t=$1 d=$2 s=$3 sd=$4 tag=$5 e
  if have "$t" "$d" "$s" "$sd" "$tag"; then echo "$(ts) skip (row exists): $tag $t $d s $s seed $sd" >> "$log"; return 0; fi
  [ "$DRY" = 1 ] && { echo "dry: would run $tag $t $d s $s seed $sd"; return 0; }
  gate; checks
  echo "$(ts) start $tag $t $d s $s seed $sd" >> "$log"
  (cd /tmp && "$py" -I -B $tu/b0_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir ${DIR[$t]} --tag $tag < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end exit $e" >> "$log"
  [ $e -eq 0 ] || stop "run exit $e ($tag $t $d s $s seed $sd)"; }
DRY=0; [ "$1" = --dry ] && DRY=1
if [ $DRY = 1 ]; then checks; echo "dry: hashes ok"; [ -e $go ] && echo "dry: GPU go exists" || echo "dry: would wait for $go"; else
  until [ -e $go ]; do sleep 60; done
  echo "$(ts) GPU go: $(head -c 300 $go)" >> "$log"
  echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
for t in $TABLES; do for d in $SETS; do for s in 2 4 8 12 16 24; do run $t $d $s 1 0b-screen; done; done; done
[ $DRY = 1 ] && { echo "dry: then b0_select.py, the seed runs (s 8 and the pick on seeds 4-6) and b0_verdict.py"; exit 0; }
if [ ! -e $tu/b0_picks.json ]; then checks; (cd /tmp && "$py" -I -B $tu/b0_select.py >> "$log" 2>&1) || stop "b0_select.py failed"; fi
for t in $TABLES; do for d in $SETS; do
  p=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['pick'])" $tu/b0_picks.json "$t|$d") || stop "no pick for $t $d"
  for sd in 4 5 6; do run $t $d 8 $sd 0b-seeds; [ "$p" = 8.0 ] || run $t $d $p $sd 0b-seeds; done
done; done
if [ ! -e $tu/b0_verdict.json ]; then checks; (cd /tmp && "$py" -I -B $tu/b0_verdict.py >> "$log" 2>&1) || stop "b0_verdict.py failed"; fi
echo "$(ts) 0b done" >> "$log"
