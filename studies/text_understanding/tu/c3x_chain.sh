#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Step 3c follow-ups (a), (b), (b') and the one-k-means replication F2: one GPU job, after the decision_layer e81 and e81x, on the
# word (its 11:01Z queue; its 11:39Z message lets this job create the go file itself once the decision_layer
# e81_chain.log has an "e81 chain end:" line and nvidia-smi shows no compute app). Descriptive: no bar. (a), (b') and
# F2 are post hoc; (b) is the plan's follow-up. Notes, frozen before the job: tu/c3x/NOTE_2026-10-21_followups.md
# and its v2 addendum (the pins file below is v2).
# Steps (each skipped when its output exists, so the job can be started again; nothing is overwritten):
#   1. c3x_feats.py per set: arms orig2 (rebuilt originals), F2 (3c's kept_F), twin (EDA copies), and on CLINC150 F_rk
#      (kept_F minus the reader-key and order-free copies of evaluation texts), all from ONE k-means per set;
#   2. code checks: c3x_exists_tm.py and c3x_lr.py on 3c's own rebuilt-original cache (tu/c3/feats/orig, CLINC150, seed 4,
#      s 8) must give 3c's 3c-ref and ref_rebuilt rows exactly;
#   3. TM (c3x_exists_tm.py, 4,000 clauses, T 4,000, 40 epochs, at the base's s, seeds 4-6): orig2, F2, twin and
#      rep5 (orig2 with --repeat 5) on every set, F_rk on CLINC150;
#   4. LR (c3x_lr.py, with each C's val accuracy kept): orig2, F2, twin on every set, F_rk on CLINC150;
#   5. c3x_report.py.
# Before every step: the pinned hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the
# last check is over 5 min old). Every way out (done, a failed step, a refused pin check, TERM, INT) appends one line
# "c3x chain end: <branch> <UTC> exit <code>" to the log (the handoff to its speed block).
# Log tu/c3x/runs_c3x.log. Test hooks (tests/test_c3x_chain.sh only): C3X_DIR, C3X_GO, C3X_PINS, C3X_BASE, C3X_STEP_CMD,
# C3X_LOAD, C3X_GATE_SLEEP, C3X_GO_SLEEP, C3X_SKIP_CODECHECK, C3X_NO_NVSMI. Sleeps run as 'sleep & wait', so TERM
# ends a waiting chain at once; to end a running step too, signal the chain's process group.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
c3=$tu/c3
cx=${C3X_DIR:-$tu/c3x}
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$cx/runs_c3x.log
go=${C3X_GO:-$tu/.c3x/gpu_go}
pins=${C3X_PINS:-$cx/pins_v2.sh}  # written with the v2 note, at the freeze: check lines for every code and input file
base_json=${C3X_BASE:-$c3/base.json}
read -r -a STEP_CMD <<< "${C3X_STEP_CMD:-$py -I -B}"
read -r -a LOADCMD <<< "${C3X_LOAD:-$py -I $load 10}"
gate_sleep=${C3X_GATE_SLEEP:-120}
SETS="clinc150 banking77 hwu64"
TABLE=$tu/a3/tables/a3B_d768.npz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
DRY=0; [ "${1:-}" = --dry ] && DRY=1
BRANCH=start
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$cx"
end_line() { local e=$? line
  line="c3x chain end: $BRANCH $(ts) exit $e"
  if [ "$DRY" = 1 ]; then echo "$line"; else echo "$line" >> "$log"; fi
  exit $e; }
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
bj() { "$py" -I -c "import json, sys; v = json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)" "$base_json" "$@"; }
have_tm() { [ -e "$cx/results_c3x_val.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, tag, arm, d, sd, rep = sys.argv[1:7]
ok = any(r.get('tag') == tag and r.get('arm') == arm and r['dataset'] == d and r['seed'] == int(sd) and r.get('repeat', 1) == int(rep)
         for r in map(json.loads, open(f)))
sys.exit(0 if ok else 1)
" "$cx/results_c3x_val.jsonl" "$@"; }
have_lr() { [ -e "$cx/results_c3x_lr.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, name, d = sys.argv[1:4]
sys.exit(0 if any(r.get('name') == name and r['dataset'] == d for r in map(json.loads, open(f))) else 1)
" "$cx/results_c3x_lr.jsonl" "$@"; }
BRANCH=pins
if [ $DRY = 1 ]; then checks; echo "dry: pins ok"; else
  checks
  BRANCH=wait-go
  until [ -e "$go" ]; do sleep "${C3X_GO_SLEEP:-60}" & wait $!; done
  echo "$(ts) GPU go: $(head -c 300 "$go")" >> "$log"
  echo "$(ts) chain $(sha256sum "$0" | cut -c1-16), pins $(sha256sum "$pins" | cut -c1-16)" >> "$log"
  [ -n "${C3X_NO_NVSMI:-}" ] || echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
# 1. features, one k-means per set
BRANCH=feats
for d in $SETS; do
  arms="orig2:none,F2:$c3/kept_F.jsonl,twin:$cx/twin_kept.jsonl"
  [ "$d" = clinc150 ] && arms="$arms,F_rk:$cx/kept_F_rk_clinc150.jsonl"
  [ -e "$cx/feats/orig2/${d}__p32tok_K512.npz" ] || step feats $tu/c3x_feats.py --dataset $d --table wordtab:$TABLE --arms "$arms" --out-root "$cx/feats"
done
# 2. code checks on 3c's own rebuilt-original cache
BRANCH=code-check
s1=$(bj s clinc150)
have_tm 3cx-check orig clinc150 4 1 || step tm $tu/c3x_exists_tm.py --dataset clinc150 --view existsT --seed 4 --s "$s1" --emb-dir $c3/feats/orig --tag 3cx-check
have_lr check_orig clinc150 || step lr $tu/c3x_lr.py --name check_orig --dataset clinc150 --emb-dir $c3/feats/orig
if [ $DRY = 0 ] && [ -z "${C3X_SKIP_CODECHECK:-}" ]; then
  "$py" -I -c "
import json, sys
c3, cx = sys.argv[1:3]
a = [r for r in map(json.loads, open(cx + '/results_c3x_val.jsonl')) if r.get('tag') == '3cx-check']
b = [r for r in map(json.loads, open(c3 + '/results_c3_val.jsonl')) if r.get('tag') == '3c-ref' and r['dataset'] == 'clinc150' and r['seed'] == 4]
la = [r for r in map(json.loads, open(cx + '/results_c3x_lr.jsonl')) if r.get('name') == 'check_orig']
lb = [r for r in map(json.loads, open(c3 + '/results_c3_lr.jsonl')) if r.get('name') == 'ref_rebuilt' and r['dataset'] == 'clinc150']
assert len(a) == len(b) == len(la) == len(lb) == 1, (len(a), len(b), len(la), len(lb))
print('code check', a[0]['val_acc'], b[0]['val_acc'], la[0]['lr_val'], lb[0]['lr_val'])
sys.exit(0 if a[0]['val_acc'] == b[0]['val_acc'] and la[0]['lr_val'] == lb[0]['lr_val'] else 1)
" "$c3" "$cx" >> "$log" 2>&1 || stop "code check: c3x_exists_tm.py or c3x_lr.py does not give 3c's rows"
  echo "$(ts) code check passed" >> "$log"
fi
# 3. TM
BRANCH=tm
for d in $SETS; do
  s=$(bj s $d)
  arms="orig2 F2 twin"; [ "$d" = clinc150 ] && arms="$arms F_rk"
  for arm in $arms; do for sd in 4 5 6; do
    have_tm 3cx $arm $d $sd 1 || step tm $tu/c3x_exists_tm.py --dataset $d --view existsT --seed $sd --s "$s" --emb-dir "$cx/feats/$arm" --tag 3cx
  done; done
  for sd in 4 5 6; do
    have_tm 3cx orig2 $d $sd 5 || step tm $tu/c3x_exists_tm.py --dataset $d --view existsT --seed $sd --s "$s" --emb-dir "$cx/feats/orig2" --tag 3cx --repeat 5
  done
done
# 4. LR
BRANCH=lr
for d in $SETS; do
  arms="orig2 F2 twin"; [ "$d" = clinc150 ] && arms="$arms F_rk"
  for arm in $arms; do have_lr $arm $d || step lr $tu/c3x_lr.py --name $arm --dataset $d --emb-dir "$cx/feats/$arm"; done
done
# 5. report
BRANCH=report
[ -e "$cx/report_c3x.json" ] || step report $tu/c3x_report.py --c3x "$cx"
BRANCH="done"
[ $DRY = 1 ] || echo "$(ts) c3x done" >> "$log"
exit 0
