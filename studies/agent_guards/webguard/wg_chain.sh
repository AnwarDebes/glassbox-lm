#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# WebGuard veto test chain (webguard/PLAN_wg_veto.md; CPU only, nice 19, 2 threads, no GPU: CUDA_VISIBLE_DEVICES is
# empty). Steps, each skipped when its output exists: the bits (wg_feats.py), the speed test (wg_run.py --speed; the
# chain stops when the projected total is over 3.5 hours (raised from 2 h on the queue's go, 23:29Z; see
# NOTE_2026-10-20_limit.md), and the TM then waits for a GPU slot after 3c), the fits
# (wg_run.py), the scoring (wg_eval.py, the test part once). Before every step: the hashes and the CPU gate
# (cpu_load.py over 10 s at most 4.0 cores without it: 2 threads). Log webguard/runs_wg.log.
W=${REPO_ROOT}/studies/agent_guards/webguard
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$W/runs_wg.log
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $W/wg_feats.py 6b3121e04fc46d28; check $W/wg_run.py 7aa06d45bd4f16f7; check $W/wg_eval.py ce0a49d34a8bbfd4
  check $W/wg_split.json a31de065415fb9d7; check ${REPO_ROOT}/studies/decision_layer/scripts/verify_policy.py c10c7722bb9f3def; check ${REPO_ROOT}/studies/decision_layer/src/dl/tm.py f649043f5801605b; }
gate() { local line c
  while true; do
    line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.0)}'; then echo "$(ts) load check: $line" >> "$log"; return 0; fi
    echo "$(ts) load check (waiting): $line" >> "$log"; sleep 120
  done; }
step() { local name=$1 e; shift
  gate; checks
  echo "$(ts) start $name: $*" >> "$log"
  (cd /tmp && nice -n 19 "$py" -I -B "$@" < /dev/null >> "$log" 2>&1)
  e=$?
  echo "$(ts) end $name exit $e" >> "$log"
  [ $e -eq 0 ] || stop "$name exit $e"; }
[ -e $W/feats/wg_bits.npz ] || step feats $W/wg_feats.py
[ -e $W/results/wg_speed.json ] || step speed $W/wg_run.py --speed
h=$("$py" -I -c "import json, sys; print(json.load(open(sys.argv[1]))['projected_hours'])" $W/results/wg_speed.json)
awk -v h="$h" 'BEGIN{exit !(h <= 3.5)}' || stop "projected TM time $h h > 3.5 h: the TM waits for a GPU slot after 3c"
[ -e $W/results/wg_preds.npz ] || step run $W/wg_run.py
[ -e $W/results/wg_eval.json ] || step eval $W/wg_eval.py
echo "$(ts) WG done" >> "$log"
