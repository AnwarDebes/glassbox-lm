#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# R3 on the 3c stack (tu/r3c/NOTE_2026-10-22_R3c.md; PLAN_3c "On PASS": the reader is re-tuned on the augmented stack;
# the binding rule R3 of PLAN_text_understanding.md; VAL only). tu/r3_chain.sh with these changes and nothing else:
# - the features are 3c's registered kept_F cache (tu/c3/feats/F; the thermometer is fit on the originals, as in 3c);
# - the runner is r3c_exists_tm.py and the pick is r3c_select.py (the same rule: e74's, seed 1 over s 2 / 4 / 8 / 12 /
#   16 / 24, the best unless it is less than 0.3 above s 8);
# - it waits for tu/.r3c/gpu_go and needs 3c's verdict PASS;
# - the seed rows (4-6 at the pick) are compared with 3c's own 3c-tm rows at the same s, which they must equal where
#   that s is 3c's;
# - every way out appends "r3c chain end: <UTC> exit <code>" to the log.
# Before every run: the hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the last
# check is over 5 min old). Restart-safe. Log tu/r3c/runs_r3c.log.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
r3=$tu/r3c
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$r3/runs_r3c.log
go=$tu/.r3c/gpu_go
feats=$tu/c3/feats/F
SETS="clinc150 banking77 hwu64"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
last_gate=-1000
DRY=0; [ "${1:-}" = --dry ] && DRY=1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p $r3
end_line() { local e=$?; [ "$DRY" = 1 ] || echo "r3c chain end: $(ts) exit $e" >> "$log"; exit "$e"; }
trap end_line EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $tu/r3c_exists_tm.py 607c3aab513fe3ad; check $tu/r3c_select.py 97cff2213ab7a84a; check $emb/exists_tm.py 7da0aee9f4eef796
  check $feats/clinc150__p32tok_K512.npz 4d5cbe9a477e6cae; check $feats/banking77__p32tok_K512.npz ecba00634a411878
  check $feats/hwu64__p32tok_K512.npz c3efdae37318adaa; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120 & wait $!
  done; }
have() { local d=$1 s=$2 sd=$3 tag=$4
  [ -e $r3/results_r3c_val.jsonl ] || return 1
  "$py" -I -c "
import json, sys
d, s, sd, tag, f = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(f)]
sys.exit(0 if any(r.get('tag') == tag and r['dataset'] == d and float(r['s']) == s and r['seed'] == sd for r in rows) else 1)
" "$d" "$s" "$sd" "$tag" $r3/results_r3c_val.jsonl; }
run() { local d=$1 s=$2 sd=$3 tag=$4 e
  if have "$d" "$s" "$sd" "$tag"; then echo "$(ts) skip (row exists): $tag $d s $s seed $sd" >> "$log"; return 0; fi
  [ "$DRY" = 1 ] && { echo "dry: would run $tag $d s $s seed $sd"; return 0; }
  gate; checks
  echo "$(ts) start $tag 3c-F $d s $s seed $sd" >> "$log"
  (cd /tmp && "$py" -I -B $tu/r3c_exists_tm.py --dataset $d --view existsT --seed $sd --s $s --emb-dir $feats --tag $tag < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end exit $e" >> "$log"
  [ $e -eq 0 ] || stop "run exit $e ($tag $d s $s seed $sd)"; }
if [ $DRY = 1 ]; then checks; echo "dry: hashes ok"; else
  checks
  until [ -e $go ]; do sleep 60 & wait $!; done
  echo "$(ts) GPU go: $(head -c 300 $go)" >> "$log"
  echo "$(ts) chain $(sha256sum "$0" | cut -c1-16)" >> "$log"
  v=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))['verdict'])" $tu/c3/verdict_c3.json)
  echo "$(ts) 3c verdict: $v" >> "$log"
  [ "$v" = PASS ] || { echo "$(ts) R3c not needed (3c did not pass); nothing run" >> "$log"; exit 0; }
  echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
for d in $SETS; do for s in 2 4 8 12 16 24; do run $d $s 1 r3c-screen; done; done
[ $DRY = 1 ] && { echo "dry: then r3c_select.py and seeds 4-6 at the pick"; exit 0; }
if [ ! -e $r3/r3c_picks.json ]; then checks; (cd /tmp && "$py" -I -B $tu/r3c_select.py >> "$log" 2>&1) || stop "r3c_select.py failed"; fi
for d in $SETS; do
  p=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['pick'])" $r3/r3c_picks.json "a3B_d768|$d") || stop "no pick for $d"
  for sd in 4 5 6; do run $d $p $sd r3c-seeds; done
done
"$py" -I -c "
import json, sys
p = json.load(open(sys.argv[1])); r3 = [json.loads(l) for l in open(sys.argv[2])]; c3 = [json.loads(l) for l in open(sys.argv[3])]
for d in ('clinc150', 'banking77', 'hwu64'):
    s = float(p['a3B_d768|' + d]['pick'])
    for sd in (4, 5, 6):
        m = [r['val_acc'] for r in r3 if r.get('tag') == 'r3c-seeds' and r['dataset'] == d and r['seed'] == sd and float(r['s']) == s]
        o = [r['val_acc'] for r in c3 if r.get('tag') == '3c-tm' and r['dataset'] == d and r['seed'] == sd and float(r['s']) == s]
        print(d, 's', s, 'seed', sd, 'r3c', m, '3c', o, 'equal' if (m and o and m == o) else ('3c has no row at this s' if not o else 'DIFFERENT'))
" $r3/r3c_picks.json $r3/results_r3c_val.jsonl $tu/c3/results_c3_val.jsonl >> "$log" 2>&1
echo "$(ts) R3c done" >> "$log"
exit 0
