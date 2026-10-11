#!/usr/bin/env bash
# H-AO HIERARCHY DIAGNOSTICS (docs/13 A8 item 3, rows (i) to (vi); exploratory, world seeds 100 then 101).
# Config experiments/cvtm/configs/ssl_r2_hao_diagnostics.json (reading notes fixed before any output), script
# experiments/ssl_r2_hao_diagnostics.py. Package stages run twice: under .venv-tm (PyHierarchicalTsetlinMachineCUDA
# c85ca4e7) and under .venv-htm (HEAD 5b4a223); the script refuses a venv/commit mismatch. CPU stages (valiant, controls,
# report) use .venv-tm.
# Order: the HEAD selftest first (--stage selftest --expect-tag head under .venv-htm: the oracle write round trips,
# a tiny smoke of the package stages, tiny seeded demos and the demo source checks on HEAD); if it fails, every HEAD
# stage is skipped with a message and the c85 and CPU stages still run. Then Granmo's demos (A8 3 (vi): positive
# controls first), then per world seed Valiant and the controls (CPU, about 15 min), then per world seed the oracle
# stability, seen/unseen and positive-literal stages for both commits; the epochs-to-threshold instance (world seed
# 100, at most 2 GPU-hours, the best tuned hierarchical config = HEAD htm_hier_neg) after the seed-100 block; the
# report last. A failed step does not stop the chain (every stage is independent; failed checks are in each run's
# metrics.json and the report marks unverified oracle cells as invalid). The selftest run is not given to the report.
# Runs inside ONE CPU slot; each GPU step takes the shared GPU lock inside `step` (CPU slot first, docs/12 F25).
# mem_guard limits (process group): 3 GiB for GPU stages and the selftest, 2 GiB valiant, 4 GiB controls, 1 GiB
# report [derived: 380 MiB peak RSS measured for the seed-100 data build plus the oracle reference evaluation without
# CUDA, 200 MiB for the tiny GPU tests; the CUDA context and pycuda add a few hundred MiB; controls: dense float32
# copies of 8,000 x 2,000 bags for sklearn and tmu's clause banks, not measured, hence the wider limit]. Cost estimate
# (derived from logged epoch and fit times): about 8.5 h in total (demos 2.7 h, oracle 2 h, seen/unseen and pospair
# 0.8 h, threshold at most 2 h, controls about 0.5 h, selftest a few minutes).
#
# Launch (detached; usage: chain_hao_diag.sh [SEEDS],
# default "100 101"):
#   cd studies/tm_language_model && setsid nohup scripts/cpu_slot.sh 1 -- bash \
#     launch/chain_hao_diag.sh \
#     > launch/chain_hao_diag.out 2>&1 < /dev/null &
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
SEEDS=${1:-"100 101"}
CFG=experiments/cvtm/configs/ssl_r2_hao_diagnostics.json
TM=.venv-tm/bin/python
HTM=.venv-htm/bin/python
DG="-u experiments/ssl_r2_hao_diagnostics.py --config $CFG"
NAMES=()
run() {  # run NAME LIMIT GPU VENV_PY -- args...
  local name=$1 lim=$2 gpu=$3 py=$4; shift 5
  NAMES+=("$name")
  step "$name" $CFG "$lim" "$gpu" -- $py $DG "$@"
}
venv_of() { [ "$1" = head ] && echo $HTM || echo $TM; }
[ -x $HTM ] || { echo "CHAIN STOP: $HTM missing (env/htm_install.sh builds .venv-htm)"; exit 1; }
CHAIN_T0=$(date -u +%Y%m%dT%H%M%SZ)
echo "CHAIN hao_diag start $(date -u +%FT%TZ) seeds $SEEDS"
# the HEAD selftest (called with `step`, not `run`: its run is never handed to the report)
VERSIONS="c85 head"
if ! step ssl_r2_hao_diag_selftest_head $CFG 3 1 -- $HTM $DG --stage selftest --expect-tag head; then
  echo "HEAD SELFTEST FAILED $(date -u +%FT%TZ): every HEAD stage is skipped, see $(latest ssl_r2_hao_diag_selftest_head)/log.txt"
  VERSIONS="c85"
fi
for V in $VERSIONS; do
  run ssl_r2_hao_diag_demos_${V}_exploratory 3 1 "$(venv_of $V)" -- --stage demos
done
for S in $SEEDS; do
  run ssl_r2_hao_diag_valiant_seed${S}_exploratory 2 0 $TM -- --stage valiant --world-seed $S
  run ssl_r2_hao_diag_controls_seed${S}_exploratory 4 0 $TM -- --stage controls --world-seed $S
done
for S in $SEEDS; do
  for V in $VERSIONS; do
    run ssl_r2_hao_diag_oracle_${V}_seed${S}_exploratory 3 1 "$(venv_of $V)" -- --stage oracle --world-seed $S --replay-check
  done
  for V in $VERSIONS; do
    run ssl_r2_hao_diag_seen_unseen_${V}_seed${S}_exploratory 3 1 "$(venv_of $V)" -- --stage seen_unseen --world-seed $S --replay-check
    run ssl_r2_hao_diag_pospair_${V}_seed${S}_exploratory 3 1 "$(venv_of $V)" -- --stage pospair --world-seed $S
  done
  if [ "$S" = 100 ] && [[ " $VERSIONS " == *" head "* ]]; then
    run ssl_r2_hao_diag_threshold_head_seed100_exploratory 3 1 $HTM -- --stage threshold --world-seed 100
  fi
done
mine() {  # the latest run dir of NAME, only if this chain made it (its timestamp is not older than CHAIN_T0)
  local d b; d=$(latest "$1"); [ -n "$d" ] || return 1; b=$(basename "$d"); b=${b%%_*}
  [[ "$b" > "$CHAIN_T0" || "$b" == "$CHAIN_T0" ]] && echo "$d"
}
RUNS=$(for n in "${NAMES[@]}"; do d=$(mine "$n") && [ -f "$d/cells.jsonl" ] && echo "$d"; done | paste -sd, -)
for n in "${NAMES[@]}"; do d=$(mine "$n"); echo "RUN $n: ${d:-none} $(grep -o '"exit_status": [0-9]*' "$d/run.json" 2>/dev/null)"; done
if [ -n "$RUNS" ]; then
  step ssl_r2_hao_diag_report_exploratory $CFG 1 0 -- $TM $DG --stage report --runs "$RUNS"
else
  echo "no run with cells.jsonl: no report"
fi
echo "CHAIN hao_diag DONE $(date -u +%FT%TZ)"
