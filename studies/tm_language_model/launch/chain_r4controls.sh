#!/usr/bin/env bash
# R4 reported learning-credit controls (docs/12 F29 item 2, A8 item 6 diagnostics; experiments/ssl_r4_controls.py, config
# experiments/cvtm/configs/ssl_r4_controls.json). Same stages, limits and locks as the config's launch.lane_per_seed and
# launch.report; the registered inputs are the confirmatory runs ssl_r4_gpu_seedN_confirmatory and
# ssl_r4_probe_seedN_confirmatory. Lanes: seed1 | seed2 | seed3 (replay, twin, randbank, diag on the shared GPU lock, then
# the CPU probe) and report. A lane waits, outside any CPU slot, until the confirmatory GPU lane (R4 gpu seeds, R2 gpu,
# the exclusive R2 control) is done, so the registered runs go first. RESUME: a clean run of a name is not rerun.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?usage: chain_r4controls.sh seed1|seed2|seed3|report}
NIGHT=$LAUNCH_DIR
SELF=$NIGHT/chain_r4controls.sh
CFG=experiments/cvtm/configs/ssl_r4_controls.json
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
case "$LANE" in seed1|seed2|seed3|report) ;; *) echo "unknown lane $LANE"; exit 2 ;; esac
if [ -z "${R4C_IN_SLOT:-}" ]; then
  echo "CHAIN r4controls lane $LANE launched $(date -u +%FT%TZ)"
  rm -f "$NIGHT/r4c_${LANE}.done"
  until [ -f "$NIGHT/r2r4_gpu.done" ]; do sleep 300; done
  if [ "$LANE" = report ]; then
    for L in seed1 seed2 seed3; do until [ -f "$NIGHT/r4c_$L.done" ] && [ -f "$NIGHT/r2r4_r4$L.done" ]; do sleep 300; done; done
  fi
  export R4C_IN_SLOT=1
  exec scripts/cpu_slot.sh 1 -- bash "$SELF" "$LANE"
fi
echo "CHAIN r4controls lane $LANE start $(date -u +%FT%TZ) (inside a CPU slot)"
rstep() { local name=$1; local d; d=$(latest "$name"); ok "$d" && { echo "RESUME $name: $d"; return 0; }; step "$@"; }
if [ "$LANE" = report ]; then
  RUNS=$(for S in 1 2 3; do for p in ssl_r4_gpu_seed${S}_confirmatory ssl_r4_probe_seed${S}_confirmatory ssl_r4c_replay_seed$S \
           ssl_r4c_twin_seed$S ssl_r4c_randbank_seed$S ssl_r4c_diag_seed$S ssl_r4c_probe_seed$S; do d=$(latest $p); ok "$d" && echo "$d"; done; done | paste -sd, -)
  rstep ssl_r4c_report $CFG 4 0 -- $PY -u experiments/ssl_r4_controls.py --stage report --config $CFG --runs "$RUNS"
else
  S=${LANE#seed}; T=seed$S
  G=$(latest ssl_r4_gpu_seed${S}_confirmatory)
  ok "$G" || { echo "CHAIN STOP: no clean registered gpu run for seed $S"; echo STOP > "$NIGHT/r4c_${LANE}.done"; exit 1; }
  X="$PY -u experiments/ssl_r4_controls.py --config $CFG --gpu-run $G"
  rstep ssl_r4c_replay_$T $CFG 12 1 -- $X --stage replay || { echo "CHAIN STOP: replay failed (replay check or run error)"; echo STOP > "$NIGHT/r4c_${LANE}.done"; exit 1; }
  rstep ssl_r4c_twin_$T $CFG 12 1 -- $X --stage twin
  rstep ssl_r4c_randbank_$T $CFG 12 1 -- $X --stage randbank
  C="--replay-run $(latest ssl_r4c_replay_$T) --twin-run $(latest ssl_r4c_twin_$T) --randbank-run $(latest ssl_r4c_randbank_$T)"
  rstep ssl_r4c_diag_$T $CFG 12 1 -- $X --stage diag $C
  rstep ssl_r4c_probe_$T $CFG 16 0 -- $X --stage probe $C
fi
echo "DONE $(date -u +%FT%TZ)" > "$NIGHT/r4c_${LANE}.done"
echo "CHAIN r4controls lane $LANE DONE $(date -u +%FT%TZ)"
