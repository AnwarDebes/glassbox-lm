#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M1, the registered test of the text-understanding stack (tu/PLAN_M1.md): one GPU job, alone, after the speed block's
# first part, on the queue's go (the go file below). Step 0 (m1_overlap.py) ran on the CPU before the freeze; its
# output is pinned.
# Steps (each skipped when its output exists, so the job can be started again; nothing is overwritten, and a run whose
# test predictions exist cannot run again: one test read per run):
#   1. m1_feats.py per set and table, ONE k-means each: a3B (arms a3B, a3B_3c, a3B_twin), C4 (C4, C4_3c), potion;
#   2. code check: 3c's CLINC150 rebuilt-original cache with "test" := val (m1_check_cache.py, the stand-in dataset
#      m1_standin.py:load; no test text or label is read): m1_tm.py (seed 4, s 8) and m1_lr.py must give 3c's 3c-ref and
#      ref_rebuilt rows exactly, and the same number again on that "test"; else stop;
#   3. TM (m1_tm.py, 4,000 clauses, T 4,000, 40 epochs, s 8, seeds 7-9): C4, a3B, potion, a3B_3c, C4_3c, a3B_twin and
#      a3B repeat 5, on every set;
#   4. LR (m1_lr.py): C4, a3B, potion, a3B_3c, C4_3c, a3B_twin, on every set;
#   5. m1_verdict.py.
# Before every step: the pinned hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the
# last check is over 5 min old). Every way out appends "m1 chain end: <branch> <UTC> exit <code>" to the log (the
# handoff line for the queue). Log tu/m1/runs_m1.log. Test hooks (tests/test_m1_chain.sh only): M1_DIR, M1_GO,
# M1_PINS, M1_STEP_CMD, M1_LOAD, M1_GATE_SLEEP, M1_GO_SLEEP, M1_SKIP_CODECHECK, M1_NO_NVSMI. Sleeps run as
# 'sleep & wait', so TERM ends a waiting chain at once; to end a running step too, signal the chain's process group.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
c3=$tu/c3
cx=$tu/c3x
m1=${M1_DIR:-$tu/m1}
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$m1/runs_m1.log
go=${M1_GO:-$tu/.m1/gpu_go}
pins=${M1_PINS:-$m1/pins.sh}  # written at the freeze: check lines for every code and input file
read -r -a STEP_CMD <<< "${M1_STEP_CMD:-$py -I -B}"
read -r -a LOADCMD <<< "${M1_LOAD:-$py -I $load 10}"
gate_sleep=${M1_GATE_SLEEP:-120}
SETS="clinc150 banking77 hwu64"
A3B=$tu/a3/tables/a3B_d768.npz
C4=$emb/w1/tables/ppw_S1000_d500_e0.5.npz
STANDIN=$tu/m1_standin.py:load
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DRY=0; [ "${1:-}" = --dry ] && DRY=1
BRANCH=start
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$m1"
end_line() { local e=$? line
  line="m1 chain end: $BRANCH $(ts) exit $e"
  if [ "$DRY" = 1 ]; then echo "$line"; else echo "$line" >> "$log"; fi
  exit "$e"; }
trap end_line EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stop() { if [ "$DRY" = 1 ]; then echo "STOP: $1"; else echo "$(ts) STOP: $1" >> "$log"; fi; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
# shellcheck source=/dev/null
checks() { [ -e "$pins" ] || stop "no pins file"; . "$pins"; }
gate() { local line c
  (( SECONDS - last_gate < 300 )) && return 0
  while true; do
    line=$("${LOADCMD[@]}"); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}'; then echo "$(ts) load check: $line" >> "$log"; last_gate=$SECONDS; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep "$gate_sleep" & wait $!
  done; }
step() { local name=$1 e; shift
  [ "$DRY" = 1 ] && { echo "dry: would run $name: $*"; return 0; }
  gate; checks
  echo "$(ts) start $name: $*" >> "$log"
  (cd /tmp && "${STEP_CMD[@]}" "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e" >> "$log"
  [ $e -eq 0 ] || stop "$name exit $e"; }
have_tm() { [ -e "$1/results_m1_tm.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, tag, arm, d, sd, rep = sys.argv[1:7]
ok = any(r.get('tag') == tag and r.get('arm') == arm and r['dataset'] == d and r['seed'] == int(sd) and r.get('repeat', 1) == int(rep)
         for r in map(json.loads, open(f)))
sys.exit(0 if ok else 1)
" "$1/results_m1_tm.jsonl" "${@:2}"; }
have_lr() { [ -e "$1/results_m1_lr.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, name, d = sys.argv[1:4]
sys.exit(0 if any(r.get('name') == name and r['dataset'] == d for r in map(json.loads, open(f))) else 1)
" "$1/results_m1_lr.jsonl" "${@:2}"; }
BRANCH=pins
if [ $DRY = 1 ]; then checks; echo "dry: pins ok"; else
  checks
  [ -e "$m1/overlap_m1.json" ] || stop "no step-0 overlap file"
  BRANCH=wait-go
  until [ -e "$go" ]; do sleep "${M1_GO_SLEEP:-60}" & wait $!; done
  echo "$(ts) GPU go: $(head -c 300 "$go")" >> "$log"
  echo "$(ts) chain $(sha256sum "$0" | cut -c1-16), pins $(sha256sum "$pins" | cut -c1-16)" >> "$log"
  [ -n "${M1_NO_NVSMI:-}" ] || echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
# 1. features, one k-means per set and table
BRANCH=feats
for d in $SETS; do
  [ -e "$m1/feats/a3B/${d}__p32tok_K512.npz" ] || step feats $tu/m1_feats.py --dataset "$d" --table wordtab:$A3B \
    --arms "a3B:none,a3B_3c:$c3/kept_F.jsonl,a3B_twin:$cx/twin_kept.jsonl" --out-root "$m1/feats"
  [ -e "$m1/feats/C4/${d}__p32tok_K512.npz" ] || step feats $tu/m1_feats.py --dataset "$d" --table wordtab:$C4 \
    --arms "C4:none,C4_3c:$c3/kept_F.jsonl" --out-root "$m1/feats"
  [ -e "$m1/feats/potion/${d}__p32tok_K512.npz" ] || step feats $tu/m1_feats.py --dataset "$d" --table potion \
    --arms "potion:none" --out-root "$m1/feats"
done
# 2. code check on 3c's own rebuilt-original cache, "test" := val (no test text or label is read)
BRANCH=code-check
ck=$m1/check
[ -e "$ck/clinc150__p32tok_K512.npz" ] || step check-cache $tu/m1_check_cache.py --src $c3/feats/orig --out "$ck"
have_tm "$ck" m1-check orig clinc150 4 1 || step tm $tu/m1_tm.py --dataset clinc150 --view existsT --seed 4 --s 8 \
  --emb-dir "$ck" --tag m1-check --out-dir "$ck" --load "$STANDIN"
have_lr "$ck" check clinc150 || step lr $tu/m1_lr.py --name check --dataset clinc150 --emb-dir "$ck" --out-dir "$ck" --load "$STANDIN"
if [ $DRY = 0 ] && [ -z "${M1_SKIP_CODECHECK:-}" ]; then
  "$py" -I -c "
import json, sys
c3, ck = sys.argv[1:3]
a = [r for r in map(json.loads, open(ck + '/results_m1_tm.jsonl')) if r.get('tag') == 'm1-check']
b = [r for r in map(json.loads, open(c3 + '/results_c3_val.jsonl')) if r.get('tag') == '3c-ref' and r['dataset'] == 'clinc150' and r['seed'] == 4]
la = [r for r in map(json.loads, open(ck + '/results_m1_lr.jsonl')) if r.get('name') == 'check']
lb = [r for r in map(json.loads, open(c3 + '/results_c3_lr.jsonl')) if r.get('name') == 'ref_rebuilt' and r['dataset'] == 'clinc150']
assert len(a) == len(b) == len(la) == len(lb) == 1, (len(a), len(b), len(la), len(lb))
print('code check', a[0]['val_acc'], b[0]['val_acc'], a[0]['test_acc'], la[0]['lr_val'], lb[0]['lr_val'], la[0]['lr_test'])
ok = a[0]['val_acc'] == b[0]['val_acc'] == a[0]['test_acc'] and la[0]['lr_val'] == lb[0]['lr_val'] == la[0]['lr_test']
sys.exit(0 if ok else 1)
" "$c3" "$ck" >> "$log" 2>&1 || stop "code check: m1_tm.py or m1_lr.py does not give 3c's rows"
  echo "$(ts) code check passed" >> "$log"
fi
# 3. TM, seeds 7-9, s 8
BRANCH=tm
for d in $SETS; do
  for ar in C4:1 a3B:1 potion:1 a3B_3c:1 C4_3c:1 a3B_twin:1 a3B:5; do
    arm=${ar%%:*}; rep=${ar##*:}
    for sd in 7 8 9; do
      have_tm "$m1" m1 "$arm" "$d" "$sd" "$rep" || step tm $tu/m1_tm.py --dataset "$d" --view existsT --seed "$sd" --s 8 \
        --emb-dir "$m1/feats/$arm" --tag m1 --repeat "$rep" --out-dir "$m1"
    done
  done
done
# 4. LR
BRANCH=lr
for d in $SETS; do
  for arm in C4 a3B potion a3B_3c C4_3c a3B_twin; do
    have_lr "$m1" "$arm" "$d" || step lr $tu/m1_lr.py --name "$arm" --dataset "$d" --emb-dir "$m1/feats/$arm" --out-dir "$m1"
  done
done
# 5. verdict
BRANCH=verdict
[ -e "$m1/verdict_m1.json" ] || step verdict $tu/m1_verdict.py --m1 "$m1" --overlap "$m1/overlap_m1.json"
BRANCH="done"
[ $DRY = 1 ] || echo "$(ts) m1 done" >> "$log"
exit 0
