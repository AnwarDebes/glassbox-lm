#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Step 3a chain (tu/PLAN_3a.md; VAL only; the test split is never read). Waits for tu/.a3/gpu_go (made on the
# word: the plan approved and 3a's turn on the GPU, one big job at a time) and for 0b's verdict (its seed
# rows are 3a's TM reference; its picks set the TM's s). Then, each step skipped when its output exists (restart-safe):
# the teacher (a3_teacher.py), the fit (a3_fit.py), 6 feature builds (w1/wordtab_feats.py, unchanged), 18 TM runs
# (a3_exists_tm.py at the C4 view's 0b pick, seeds 4-6; a3B first), 9 LR fits (a3_lr.py: ppw_S1000's own features first,
# the fresh LR reference of the fix 3, then the two tables) and the verdict
# (a3_verdict.py). Before every step: the hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked
# when the last check is over 5 min old). Log tu/a3/runs_a3.log.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
a3=$tu/a3
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$a3/runs_a3.log
go=$tu/.a3/gpu_go
SETS="clinc150 banking77 hwu64"
ARMS="a3B_d768 a3A_d768"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $tu/a3_teacher.py 7539e26231b77a17; check $tu/a3_fit.py e199064180b37637; check $tu/a3_exists_tm.py 24a2eb9c83cfc3e0
  check $tu/a3_lr.py 4fd99da0cd209e3c; check $tu/a3_verdict.py 806a15d693631791; check $a3/teacher_cpu64.npy f8c506b7c0c76686
  check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/w1/wordtab_feats.py 001a93b259a1b34f
  check $emb/glove_feats.py 7b5e8760e1f5e107; check $emb/w1/tables/ppw_S1000_d500_e0.5.npz e1771d9eaf4a07f7
  check $a3/c4_sent_train_f.txt 00481f8a27d5a08f; check $a3/c4_sent_held_f.txt 3207323aca2a59e9; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120
  done; }
step() { local name=$1 e; shift
  [ "$DRY" = 1 ] && { echo "dry: would run $name: $*"; return 0; }
  gate; checks
  echo "$(ts) start $name: $*" >> "$log"
  (cd /tmp && "$py" -I -B "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e" >> "$log"
  [ $e -eq 0 ] || stop "$name exit $e"; }
have() { local f=$1 t=$2 d=$3 s=$4 sd=$5
  [ -e "$f" ] || return 1
  "$py" -I -c "
import json, sys
f, t, d, s, sd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(f)]
ok = any(r.get('table') == t and r['dataset'] == d and (s == '-' or float(r.get('s', -1)) == float(s)) and (sd == '-' or r.get('seed') == int(sd)) for r in rows)
sys.exit(0 if ok else 1)
" "$f" "$t" "$d" "$s" "$sd"; }
DRY=0; [ "$1" = --dry ] && DRY=1
if [ $DRY = 1 ]; then checks; echo "dry: hashes ok"; else
  until [ -e $go ]; do sleep 60; done
  echo "$(ts) GPU go: $(head -c 300 $go)" >> "$log"
  until [ -e $tu/b0_verdict.json ]; do sleep 60; done
  echo "$(ts) 0b verdict found; GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
[ -e $a3/teacher_meta.json ] || step teacher $tu/a3_teacher.py
[ -e $a3/fit_meta.json ] || step fit $tu/a3_fit.py
for t in $ARMS; do for d in $SETS; do
  [ -e $a3/feats/$t/${d}__p32tok_K512.npz ] || step feats $emb/w1/wordtab_feats.py --dataset $d --table wordtab:$a3/tables/$t.npz --out-dir $a3/feats/$t
done; done
for t in $ARMS; do for d in $SETS; do
  if [ $DRY = 1 ]; then s=PICK; else
    s=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['pick'])" $tu/b0_picks.json "ppw_S1000_d500_e0.5|$d") || stop "no pick for $d"; fi
  for sd in 4 5 6; do
    have $a3/results_a3_val.jsonl $t $d $s $sd || step tm $tu/a3_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir $a3/feats/$t --tag 3a-tm
  done
done; done
for d in $SETS; do
  have $a3/results_a3_lr.jsonl ppw_S1000_d500_e0.5 $d - - || step lr $tu/a3_lr.py --name ppw_S1000 --dataset $d --emb-dir $emb/w1/feats/ppw_S1000_d500_e0.5
done
for t in $ARMS; do for d in $SETS; do
  have $a3/results_a3_lr.jsonl $t $d - - || step lr $tu/a3_lr.py --name $t --dataset $d --emb-dir $a3/feats/$t
done; done
[ -e $a3/verdict_a3.json ] || step verdict $tu/a3_verdict.py
[ $DRY = 1 ] || echo "$(ts) 3a done" >> "$log"
