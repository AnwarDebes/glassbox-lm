#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M2, the registered test on MTOP English (tu/PLAN_M2.md): one GPU job, alone, on the queue's go (the go file
# below), after the freeze. Launch: setsid nohup nice -n 19 bash tu/m2_chain.sh > /dev/null 2>&1 &
# tu/m1_chain.sh (1db3b6461cfe3bab) with these changes (one dataset, M2's arms) and nothing else:
#   - one set, mtop_en (the loader m2_data.load); arms C4, a3B and potion only; no paraphrase, twin or repeat arm;
#   - the code check runs FIRST, before any step that reads an MTOP file (in M1 it ran after the features), so a broken
#     copy stops the chain while MTOP's test split is untouched;
#   - before the go: the frozen matched list must exist (M1 checked its step-0 file);
#   - the pins are full sha256 values (M1 compared the first 16 hex characters);
#   - names: tu/m2, runs_m2.log, results_m2_*.jsonl, verdict_m2.*, tu/.m2/gpu_go, hooks M2_*.
#   After the review of draft 1 (2026-10-25):
#   - every step and the code-check comparison run under nice -n 19 (data MAJOR 2; M1 had no nice);
#   - the code-check comparison is tu/m2_codecheck.py (tested; it adds best_epoch, the full curve and C to M1's
#     inline comparison; design MINOR 10);
#   - tu/m2_freeze_check.py runs before the go (and in --dry): the pins file must equal the frozen plan's pins block
#     literally, and it must pin PLAN_M2.md itself (design MAJOR 2);
#   - the verdict also reads fresh/mtop_overlap.json, for the c4_fit line (no rule).
# Steps (each skipped when its output exists, so the job can be started again; nothing is overwritten, and a run whose
# test predictions exist cannot run again: one test read per run):
#   1. code check: 3c's CLINC150 cache with "test" := val (m2_check_cache.py, the stand-in dataset m2_standin.py:load,
#      which loads CLINC150's test split into memory and replaces it with val before any use; no MTOP file is read):
#      m2_tm.py (seed 4, s 8) and m2_lr.py must give 3c's 3c-ref and ref_rebuilt rows exactly (m2_codecheck.py);
#   2. m2_feats.py on mtop_en, ONE k-means per table: C4 (ppw_S1000), a3B (a3B_d768), potion;
#   3. TM (m2_tm.py, 4,000 clauses, T 4,000, 40 epochs, batch 128, s 8, seeds 7-9): C4, a3B, potion;
#   4. LR (m2_lr.py, C in 1/10/100 on en/eval): C4, a3B, potion;
#   5. m2_verdict.py (H1, H2, the lines with no rule).
# Before every step: the pinned hashes and the CPU gate (cpu_load.py over 10 s at most 4.5 cores, re-checked when the
# last check is over 5 min old). Every way out appends "m2 chain end: <branch> <UTC> exit <code>" to the log (the
# handoff line for the queue). Log tu/m2/runs_m2.log. Test hooks (tests/test_m2_chain.sh only): M2_DIR, M2_GO,
# M2_PINS, M2_PLAN, M2_MATCHED, M2_STEP_CMD, M2_LOAD, M2_GATE_SLEEP, M2_GO_SLEEP, M2_SKIP_CODECHECK, M2_NO_NVSMI. Sleeps
# run as 'sleep & wait', so TERM ends a waiting chain at once; to end a running step too, signal the process group.
# Dry listing (no step runs, nothing is logged; the pins and the freeze check are read): bash tu/m2_chain.sh --dry
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
c3=$tu/c3
m2=${M2_DIR:-$tu/m2}
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$m2/runs_m2.log
go=${M2_GO:-$tu/.m2/gpu_go}
pins=${M2_PINS:-$m2/pins.sh}  # written at the freeze: check lines for every code and input file
plan=${M2_PLAN:-$tu/PLAN_M2.md}
matched=${M2_MATCHED:-$emb/fresh/mtop_test_matched_v1.json}
overlap=$emb/fresh/mtop_overlap.json
read -r -a STEP_CMD <<< "${M2_STEP_CMD:-$py -I -B}"
read -r -a LOADCMD <<< "${M2_LOAD:-$py -I $load 10}"
gate_sleep=${M2_GATE_SLEEP:-120}
SET=mtop_en
A3B=$tu/a3/tables/a3B_d768.npz
C4=$emb/w1/tables/ppw_S1000_d500_e0.5.npz
STANDIN=$tu/m2_standin.py:load
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DRY=0; [ "${1:-}" = --dry ] && DRY=1
BRANCH=start
last_gate=-1000
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$m2"
end_line() { local e=$? line
  line="m2 chain end: $BRANCH $(ts) exit $e"
  if [ "$DRY" = 1 ]; then echo "$line"; else echo "$line" >> "$log"; fi
  exit "$e"; }
trap end_line EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stop() { if [ "$DRY" = 1 ]; then echo "STOP: $1"; else echo "$(ts) STOP: $1" >> "$log"; fi; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || stop "$1 changed"; }
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
  (cd /tmp && nice -n 19 "${STEP_CMD[@]}" "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e" >> "$log"
  [ $e -eq 0 ] || stop "$name exit $e"; }
have_tm() { [ -e "$1/results_m2_tm.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, tag, arm, d, sd, rep = sys.argv[1:7]
ok = any(r.get('tag') == tag and r.get('arm') == arm and r['dataset'] == d and r['seed'] == int(sd) and r.get('repeat', 1) == int(rep)
         for r in map(json.loads, open(f)))
sys.exit(0 if ok else 1)
" "$1/results_m2_tm.jsonl" "${@:2}"; }
have_lr() { [ -e "$1/results_m2_lr.jsonl" ] || return 1
  "$py" -I -c "
import json, sys
f, name, d = sys.argv[1:4]
sys.exit(0 if any(r.get('name') == name and r['dataset'] == d for r in map(json.loads, open(f))) else 1)
" "$1/results_m2_lr.jsonl" "${@:2}"; }
BRANCH=pins
if [ $DRY = 1 ]; then
  checks
  "$py" -I -B "$tu/m2_freeze_check.py" --plan "$plan" --pins "$pins" || stop "freeze check: the pins differ from the registration"
  echo "dry: pins ok"
else
  checks
  "$py" -I -B "$tu/m2_freeze_check.py" --plan "$plan" --pins "$pins" >> "$log" 2>&1 || stop "freeze check: the pins differ from the registration"
  [ -e "$matched" ] || stop "no frozen matched list"
  BRANCH=wait-go
  until [ -e "$go" ]; do sleep "${M2_GO_SLEEP:-60}" & wait $!; done
  echo "$(ts) GPU go: $(head -c 300 "$go")" >> "$log"
  echo "$(ts) chain $(sha256sum "$0" | cut -c1-64), pins $(sha256sum "$pins" | cut -c1-64)" >> "$log"
  [ -n "${M2_NO_NVSMI:-}" ] || echo "$(ts) GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$log"
fi
# 1. code check on 3c's own rebuilt-original cache, "test" := val (no MTOP file is read)
BRANCH=code-check
ck=$m2/check
[ -e "$ck/clinc150__p32tok_K512.npz" ] || step check-cache $tu/m2_check_cache.py --src $c3/feats/orig --out "$ck"
have_tm "$ck" m2-check orig clinc150 4 1 || step tm $tu/m2_tm.py --dataset clinc150 --view existsT --seed 4 --s 8 \
  --emb-dir "$ck" --tag m2-check --out-dir "$ck" --load "$STANDIN"
have_lr "$ck" check clinc150 || step lr $tu/m2_lr.py --name check --dataset clinc150 --emb-dir "$ck" --out-dir "$ck" --load "$STANDIN"
if [ $DRY = 0 ] && [ -z "${M2_SKIP_CODECHECK:-}" ]; then
  nice -n 19 "$py" -I -B "$tu/m2_codecheck.py" --c3 "$c3" --ck "$ck" >> "$log" 2>&1 \
    || stop "code check: m2_tm.py or m2_lr.py does not give 3c's rows"
  echo "$(ts) code check passed" >> "$log"
fi
# 2. features, one k-means per table, on MTOP English (the first step that reads an MTOP file)
BRANCH=feats
[ -e "$m2/feats/C4/${SET}__p32tok_K512.npz" ] || step feats $tu/m2_feats.py --dataset "$SET" --table wordtab:$C4 \
  --arms "C4:none" --out-root "$m2/feats"
[ -e "$m2/feats/a3B/${SET}__p32tok_K512.npz" ] || step feats $tu/m2_feats.py --dataset "$SET" --table wordtab:$A3B \
  --arms "a3B:none" --out-root "$m2/feats"
[ -e "$m2/feats/potion/${SET}__p32tok_K512.npz" ] || step feats $tu/m2_feats.py --dataset "$SET" --table potion \
  --arms "potion:none" --out-root "$m2/feats"
# 3. TM, seeds 7-9, s 8
BRANCH=tm
for arm in C4 a3B potion; do
  for sd in 7 8 9; do
    have_tm "$m2" m2 "$arm" "$SET" "$sd" 1 || step tm $tu/m2_tm.py --dataset "$SET" --view existsT --seed "$sd" --s 8 \
      --emb-dir "$m2/feats/$arm" --tag m2 --out-dir "$m2"
  done
done
# 4. LR
BRANCH=lr
for arm in C4 a3B potion; do
  have_lr "$m2" "$arm" "$SET" || step lr $tu/m2_lr.py --name "$arm" --dataset "$SET" --emb-dir "$m2/feats/$arm" --out-dir "$m2"
done
# 5. verdict
BRANCH=verdict
[ -e "$m2/verdict_m2.json" ] || step verdict $tu/m2_verdict.py --m2 "$m2" --matched "$matched" --overlap "$overlap"
BRANCH="done"
[ $DRY = 1 ] || echo "$(ts) m2 done" >> "$log"
exit 0
