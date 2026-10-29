#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Step 3d, teacher-fitted pair rows (tu/PLAN_3d.md; VAL only, no test split is read): one GPU job, alone, at its turn in
# the GPU queue, on the go file below (written by tu/d3_wait_go.sh, then renamed when read, so a
# restart waits for a new go). Launch: setsid nohup nice -n 19 bash tu/d3_chain.sh > /dev/null 2>&1 < /dev/null &
# Steps (each skipped when its output exists, judged by the file written last, so the job can be started again;
# nothing is overwritten, and a half-written step stops the chain loudly):
#   1. code check: d3_exists_tm.py (seed 4, s 8) and d3_lr.py on 3c's CLINC150 rebuilt-original cache must give 3c's
#      3c-ref and ref_rebuilt val rows exactly; else stop;
#   2. fit check: d3_fit.py --check-a3B (no pairs) must reproduce 3a (check_P0.json "pass"); else stop;
#   3. tables: d3_fit.py (d3J; exit 4, a stop, when lambda*'s CG did not converge), d3_mw.py (d3MW), d3_twins.py (d3MID,
#      d3BJ);
#   4. features (d3_feats.py) per set: a3B, J, Jwc, JnoP, MW, MID, BJ, and the k-means spread arms a3B_k1, a3B_k2,
#      J_k1, J_k2; then two checks, else stop: the feature check (the a3B arm's train and val arrays equal M1's a3B
#      caches within 1e-4) and the pair-arm check (every pair arm has its pair tokens and d3J's pair count; JnoP and
#      a3B have none; Jwc's centroids equal JnoP's within 1e-4 and its exists values are never below JnoP's);
#   5. TM (d3_exists_tm.py, 4,000 clauses, T 4,000, 40 epochs, s 8): seeds 7-9 for the seven arms, seed 7 for the
#      k-means spread arms; every set;
#   6. LR (d3_lr.py): every arm on every set;
#   7. d3_verdict.py.
# Before every step: the pinned hashes (pins.sh; pins_big.sh, the large inputs, at the start, before the fit steps and
# before the feature check) and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the last check
# is over 5 min old). Every way out appends "d3 chain end: <branch> <UTC> exit <code>" to the log (the handoff line for
# the queue). Log tu/d3/runs_d3.log. Test hooks (tests/test_d3_chain.sh only, refused in the real run): D3_DIR, D3_GO,
# D3_PINS, D3_PINS_BIG, D3_STEP_CMD, D3_LOAD, D3_GATE_SLEEP, D3_GO_SLEEP, D3_SKIP_CHECKS, D3_NO_NVSMI, D3_M1, D3_C3.
# Sleeps run as 'sleep & wait', so TERM ends a waiting chain at once; to end a running step too, signal the chain's
# process group.
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
real=$tu/d3
c3=${D3_C3:-$tu/c3}
m1=${D3_M1:-$tu/m1}
d3=${D3_DIR:-$real}
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$d3/runs_d3.log
go=${D3_GO:-$tu/.d3/gpu_go}
pins=${D3_PINS:-$d3/pins.sh}  # written at the freeze: check lines for every code and small input file
pins_big=${D3_PINS_BIG:-$d3/pins_big.sh}  # the large inputs
read -r -a STEP_CMD <<< "${D3_STEP_CMD:-$py -I -B}"
read -r -a LOADCMD <<< "${D3_LOAD:-$py -I $load 10}"
gate_sleep=${D3_GATE_SLEEP:-120}
SETS="clinc150 banking77 hwu64"
A3B=$tu/a3/tables/a3B_d768.npz
TJ=$d3/tables/d3J_d768.npz  # d3_fit.py, d3_mw.py and d3_twins.py write to tu/d3 (= $d3 outside the tests)
TMW=$d3/tables/d3MW_d768.npz
TMID=$d3/tables/d3MID_d768.npz
TBJ=$d3/tables/d3BJ_d768.npz
# arm:table:pair-tokens:kmeans:kmeans-seed
ARMS="a3B:$A3B:off:all:0 J:$TJ:on:all:0 Jwc:$TJ:on:words:0 JnoP:$TJ:off:all:0 MW:$TMW:on:all:0 MID:$TMID:on:all:0 BJ:$TBJ:on:all:0"
KARMS="a3B_k1:$A3B:off:all:1 a3B_k2:$A3B:off:all:2 J_k1:$TJ:on:all:1 J_k2:$TJ:on:all:2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DRY=0; [ "${1:-}" = --dry ] && DRY=1
BRANCH=start
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$d3"
end_line() { local e=$? line
  line="d3 chain end: $BRANCH $(ts) exit $e"
  if [ "$DRY" = 1 ]; then echo "$line"; else echo "$line" >> "$log"; fi
  exit "$e"; }
trap end_line EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stop() { if [ "$DRY" = 1 ]; then echo "STOP: $1"; else echo "$(ts) STOP: $1" >> "$log"; fi; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
# shellcheck source=/dev/null
checks() { [ -e "$pins" ] || stop "no pins file"; . "$pins"; }
# shellcheck source=/dev/null
checks_big() { [ "$DRY" = 1 ] && return 0; [ -e "$pins_big" ] || stop "no big pins file"; . "$pins_big"; }
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
have_tm() { [ -e "$1/results_d3_val.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, tag, arm, d, sd = sys.argv[1:6]
ok = any(r.get('tag') == tag and r.get('arm') == arm and r['dataset'] == d and r['seed'] == int(sd) for r in map(json.loads, open(f)))
sys.exit(0 if ok else 1)
" "$1/results_d3_val.jsonl" "${@:2}"; }
have_lr() { [ -e "$1/results_d3_lr.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, name, d = sys.argv[1:4]
sys.exit(0 if any(r.get('name') == name and r['dataset'] == d for r in map(json.loads, open(f))) else 1)
" "$1/results_d3_lr.jsonl" "${@:2}"; }
BRANCH=hooks
hooks=$(env | grep -o '^D3_[A-Z0-9_]*' | sort | tr '\n' ' ')
if [ "$d3" = "$real" ] && [ -n "$hooks" ]; then log=/dev/stderr; stop "test hooks set in the real run: $hooks"; fi  # the real log stays untouched
BRANCH=pins
if [ $DRY = 1 ]; then checks; [ -e "$pins_big" ] || stop "no big pins file"; echo "dry: pins ok"; else
  checks; checks_big
  [ -n "$hooks" ] && echo "$(ts) test hooks: $hooks" >> "$log"
  renice -n 19 -p $$ > /dev/null 2>&1 || true
  BRANCH=wait-go
  until [ -e "$go" ]; do sleep "${D3_GO_SLEEP:-60}" & wait $!; done
  echo "$(ts) GPU go: $(head -c 300 "$go")" >> "$log"
  mv "$go" "$go.used_$(date -u +%Y%m%dT%H%M%SZ)"  # used up: a restart waits for a new go
  echo "$(ts) chain $(sha256sum "$0" | cut -c1-16), pins $(sha256sum "$pins" | cut -c1-16), big pins $(sha256sum "$pins_big" | cut -c1-16)" >> "$log"
  echo "$(ts) torch/cuda: $("$py" -I -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)" >> "$log"
  [ -n "${D3_NO_NVSMI:-}" ] || echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
# 1. code check on 3c's own rebuilt-original cache (val only)
BRANCH=code-check
ck=$d3/check
mkdir -p "$ck"
have_tm "$ck" d3-check orig clinc150 4 || step tm $tu/d3_exists_tm.py --dataset clinc150 --view existsT --seed 4 --s 8 \
  --emb-dir "$c3/feats/orig" --tag d3-check --out-dir "$ck"
have_lr "$ck" check clinc150 || step lr $tu/d3_lr.py --name check --dataset clinc150 --emb-dir "$c3/feats/orig" --out-dir "$ck"
if [ $DRY = 0 ] && [ -z "${D3_SKIP_CHECKS:-}" ]; then
  "$py" -I -c "
import json, sys
c3, ck = sys.argv[1:3]
a = [r for r in map(json.loads, open(ck + '/results_d3_val.jsonl')) if r.get('tag') == 'd3-check']
b = [r for r in map(json.loads, open(c3 + '/results_c3_val.jsonl')) if r.get('tag') == '3c-ref' and r['dataset'] == 'clinc150' and r['seed'] == 4]
la = [r for r in map(json.loads, open(ck + '/results_d3_lr.jsonl')) if r.get('name') == 'check']
lb = [r for r in map(json.loads, open(c3 + '/results_c3_lr.jsonl')) if r.get('name') == 'ref_rebuilt' and r['dataset'] == 'clinc150']
assert len(a) == len(b) == len(la) == len(lb) == 1, (len(a), len(b), len(la), len(lb))
print('code check', a[0]['val_acc'], b[0]['val_acc'], la[0]['lr_val'], lb[0]['lr_val'])
sys.exit(0 if a[0]['val_acc'] == b[0]['val_acc'] and la[0]['lr_val'] == lb[0]['lr_val'] else 1)
" "$c3" "$ck" >> "$log" 2>&1 || stop "code check: d3_exists_tm.py or d3_lr.py does not give 3c's rows"
  echo "$(ts) code check passed" >> "$log"
fi
# 2. fit check: no pairs must reproduce 3a
BRANCH=fit-check
[ -e "$d3/check_P0.json" ] || { checks_big; step fit-check $tu/d3_fit.py --check-a3B; }
if [ $DRY = 0 ] && [ -z "${D3_SKIP_CHECKS:-}" ]; then
  "$py" -I -c "
import json, sys
c = json.load(open(sys.argv[1]))['check']
print('fit check', json.dumps(c))
sys.exit(0 if c['pass'] else 1)
" "$d3/check_P0.json" >> "$log" 2>&1 || stop "fit check: the no-pair fit does not reproduce 3a"
  echo "$(ts) fit check passed" >> "$log"
fi
# 3. the tables (each skipped when its meta, written last, exists)
BRANCH=fit
[ -e "$d3/fit_meta_d3J.json" ] || { checks_big; step fit $tu/d3_fit.py; }
[ -e "$d3/mw_meta.json" ] || { checks_big; step mw $tu/d3_mw.py; }
[ -e "$d3/twins_meta.json" ] || step twins $tu/d3_twins.py
# 4. features, then the feature check against M1's a3B cache and the pair-arm check
BRANCH=feats
for d in $SETS; do
  for spec in $ARMS $KARMS; do
    IFS=: read -r arm tab pt km ks <<< "$spec"
    [ -e "$d3/feats/$arm/${d}__p32tok_K512.meta.json" ] || step feats $tu/d3_feats.py --dataset "$d" --arm "$arm" \
      --table "$tab" --pair-tokens "$pt" --kmeans "$km" --kmeans-seed "$ks" --out-root "$d3/feats"
  done
done
if [ $DRY = 0 ] && [ -z "${D3_SKIP_CHECKS:-}" ]; then
  BRANCH=feat-check
  checks_big
  "$py" -I -c "
import sys
import numpy as np
d3, m1 = sys.argv[1:3]
worst = 0.0
for d in ('clinc150', 'banking77', 'hwu64'):
    a = np.load(f'{d3}/feats/a3B/{d}__p32tok_K512.npz')
    b = np.load(f'{m1}/feats/a3B/{d}__p32tok_K512.npz')
    for k in ('train__exists', 'val__exists', 'y_train', 'y_val'):
        assert a[k].shape == b[k].shape, (d, k)
        worst = max(worst, float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()))
print('feature check: max abs diff', worst)
sys.exit(0 if worst <= 1e-4 else 1)
" "$d3" "$m1" >> "$log" 2>&1 || stop "feature check: d3_feats.py's a3B arm differs from M1's a3B cache"
  echo "$(ts) feature check passed" >> "$log"
  BRANCH=pair-check
  "$py" -I -c "
import json, sys
import numpy as np
d3 = sys.argv[1]
P = json.load(open(f'{d3}/fit_meta_d3J.json'))['pairs']
bad = []
for d in ('clinc150', 'banking77', 'hwu64'):
    m = {a: json.load(open(f'{d3}/feats/{a}/{d}__p32tok_K512.meta.json')) for a in ('a3B', 'J', 'Jwc', 'JnoP', 'MW', 'MID', 'BJ', 'J_k1', 'J_k2')}
    for a in ('J', 'MW', 'MID', 'BJ', 'J_k1', 'J_k2'):
        if not (m[a]['pair_tokens'] and m[a]['table_pairs'] == P and m[a]['train_vocab_pairs'] > 0):
            bad.append((d, a, 'pairs'))
    if not (m['Jwc']['pair_tokens'] and m['Jwc']['table_pairs'] == P and m['Jwc']['train_vocab_pairs'] == 0):
        bad.append((d, 'Jwc', 'pairs or k-means types'))
    for a in ('a3B', 'JnoP'):
        if m[a]['pair_tokens']:
            bad.append((d, a, 'has pair tokens'))
    cw = np.load(f'{d3}/feats/Jwc/{d}__p32tok_K512.centroids.npy')
    cn = np.load(f'{d3}/feats/JnoP/{d}__p32tok_K512.centroids.npy')
    fw = np.load(f'{d3}/feats/Jwc/{d}__p32tok_K512.npz')
    fn = np.load(f'{d3}/feats/JnoP/{d}__p32tok_K512.npz')
    if cw.shape != cn.shape or float(np.abs(cw - cn).max()) > 1e-4:
        bad.append((d, 'Jwc', 'centroids differ from JnoP'))
    for sp in ('train', 'val'):
        if float((fw[sp + '__exists'] - fn[sp + '__exists']).min()) < -1e-5:
            bad.append((d, 'Jwc', sp + ' exists below JnoP'))
print('pair-arm check: pairs', P, 'problems', bad)
sys.exit(1 if bad else 0)
" "$d3" >> "$log" 2>&1 || stop "pair-arm check failed"
  echo "$(ts) pair-arm check passed" >> "$log"
fi
# 5. TM, s 8: seeds 7-9 for the seven arms, seed 7 for the k-means spread arms
BRANCH=tm
for d in $SETS; do
  for spec in $ARMS $KARMS; do
    arm=${spec%%:*}
    seeds="7 8 9"; case "$arm" in *_k[12]) seeds=7 ;; esac
    for sd in $seeds; do
      have_tm "$d3" d3 "$arm" "$d" "$sd" || step tm $tu/d3_exists_tm.py --dataset "$d" --view existsT --seed "$sd" --s 8 \
        --emb-dir "$d3/feats/$arm" --tag d3 --out-dir "$d3"
    done
  done
done
# 6. LR
BRANCH=lr
for d in $SETS; do
  for spec in $ARMS $KARMS; do
    arm=${spec%%:*}
    have_lr "$d3" "$arm" "$d" || step lr $tu/d3_lr.py --name "$arm" --dataset "$d" --emb-dir "$d3/feats/$arm" --out-dir "$d3"
  done
done
# 7. verdict
BRANCH=verdict
[ -e "$d3/verdict_d3.json" ] || step verdict $tu/d3_verdict.py --d3 "$d3" --m1 "$m1"
BRANCH="done"
[ $DRY = 1 ] || echo "$(ts) d3 done" >> "$log"
exit 0
