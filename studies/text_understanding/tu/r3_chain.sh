#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# R3 after 3a (the binding rule R3 of PLAN_text_understanding.md: re-tune the reader after each accepted change; VAL
# only). Runs ONLY if 3a's verdict is PASS. Waits for tu/.r3/gpu_go (the queue's go; one big GPU job at
# a time) and for tu/a3/verdict_a3.json. Then e74's rule on the a3B table at lambda* (tu/a3/feats/a3B_d768): the screen
# at seed 1 over s 2 / 4 / 8 / 12 / 16 / 24 on CLINC150, Banking77 and HWU64 (r3_exists_tm.py, tag r3-screen), the
# pick (r3_select.py -> tu/r3/r3_picks.json), and seeds 4-6 at the pick (tag r3-seeds; when the pick is 3a's s, these
# repeat 3a's a3B rows and must equal them). Before every run: the hashes and the CPU gate (cpu_load.py over 10 s at
# most 4.5 cores, re-checked when the last check is over 5 min old). Restart-safe. Log tu/r3/runs_r3.log.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
r3=$tu/r3
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$r3/runs_r3.log
go=$tu/.r3/gpu_go
feats=$tu/a3/feats/a3B_d768
SETS="clinc150 banking77 hwu64"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false  # env-only CPU change, outcome-neutral, before the run (decided 00:42Z; NOTE_2026-10-21_env.md)
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $tu/r3_exists_tm.py 1a73897b52873790; check $tu/r3_select.py e2dff9593ff72773; check $emb/exists_tm.py 7da0aee9f4eef796; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120
  done; }
have() { local d=$1 s=$2 sd=$3 tag=$4
  [ -e $r3/results_r3_val.jsonl ] || return 1
  "$py" -I -c "
import json, sys
d, s, sd, tag, f = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(f)]
sys.exit(0 if any(r.get('tag') == tag and r['dataset'] == d and float(r['s']) == s and r['seed'] == sd for r in rows) else 1)
" "$d" "$s" "$sd" "$tag" $r3/results_r3_val.jsonl; }
run() { local d=$1 s=$2 sd=$3 tag=$4 e
  if have "$d" "$s" "$sd" "$tag"; then echo "$(ts) skip (row exists): $tag $d s $s seed $sd" >> "$log"; return 0; fi
  [ "$DRY" = 1 ] && { echo "dry: would run $tag $d s $s seed $sd"; return 0; }
  gate; checks
  echo "$(ts) start $tag a3B_d768 $d s $s seed $sd" >> "$log"
  (cd /tmp && "$py" -I -B $tu/r3_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir $feats --tag $tag < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end exit $e" >> "$log"
  [ $e -eq 0 ] || stop "run exit $e ($tag $d s $s seed $sd)"; }
DRY=0; [ "$1" = --dry ] && DRY=1
mkdir -p $r3
if [ $DRY = 1 ]; then checks; echo "dry: hashes ok"; else
  until [ -e $go ]; do sleep 60; done
  echo "$(ts) GPU go: $(head -c 300 $go)" >> "$log"
  until [ -e $tu/a3/verdict_a3.json ]; do sleep 60; done
  v=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))['verdict'])" $tu/a3/verdict_a3.json)
  echo "$(ts) 3a verdict: $v" >> "$log"
  [ "$v" = PASS ] || { echo "$(ts) R3 not needed (3a did not pass); nothing run" >> "$log"; exit 0; }
  echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
for d in $SETS; do for s in 2 4 8 12 16 24; do run $d $s 1 r3-screen; done; done
[ $DRY = 1 ] && { echo "dry: then r3_select.py and seeds 4-6 at the pick"; exit 0; }
if [ ! -e $r3/r3_picks.json ]; then checks; (cd /tmp && "$py" -I -B $tu/r3_select.py >> "$log" 2>&1) || stop "r3_select.py failed"; fi
for d in $SETS; do
  p=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['pick'])" $r3/r3_picks.json "a3B_d768|$d") || stop "no pick for $d"
  for sd in 4 5 6; do run $d $p $sd r3-seeds; done
done
"$py" -I -c "
import json, sys
p = json.load(open(sys.argv[1])); r3 = [json.loads(l) for l in open(sys.argv[2])]; a3 = [json.loads(l) for l in open(sys.argv[3])]
for d in ('clinc150', 'banking77', 'hwu64'):
    s = float(p['a3B_d768|' + d]['pick'])
    for sd in (4, 5, 6):
        m = [r['val_acc'] for r in r3 if r.get('tag') == 'r3-seeds' and r['dataset'] == d and r['seed'] == sd and float(r['s']) == s]
        o = [r['val_acc'] for r in a3 if r.get('tag') == '3a-tm' and r['table'] == 'a3B_d768' and r['dataset'] == d and r['seed'] == sd and float(r['s']) == s]
        print(d, 's', s, 'seed', sd, 'r3', m, '3a', o, 'equal' if (m and o and m == o) else ('3a has no row at this s' if not o else 'DIFFERENT'))
" $r3/r3_picks.json $r3/results_r3_val.jsonl $tu/a3/results_a3_val.jsonl >> "$log" 2>&1
echo "$(ts) R3 done" >> "$log"
