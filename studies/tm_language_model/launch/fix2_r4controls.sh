#!/usr/bin/env bash
# 2026-09-30, second fix of the R4 controls: the controls probe loads the ladder, twin and randbank codes next to the
# registered ones and was stopped by mem_guard at 18.05 GiB (limit 16, copied from the registered probe). Waits for the
# first-fix lane of this seed to end (its randbank and diag steps are reused when clean), then runs the probe at 32 GiB;
# lane report runs the report after the three probes. A failed step stops its lane.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?}
NIGHT=$LAUNCH_DIR
CFG=experiments/cvtm/configs/ssl_r4_controls.json
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
fail() { echo "LANE STOP $LANE $(date -u +%FT%TZ): $*"; echo "STOP $*" > "$NIGHT/r4cfix2_${LANE}.done"; exit 1; }
rm -f "$NIGHT/r4cfix2_${LANE}.done"
if [ -z "${F2_IN_SLOT:-}" ]; then   # wait outside any CPU slot
  if [ "$LANE" = report ]; then
    for L in seed1 seed2 seed3; do until [ -f "$NIGHT/r4cfix2_$L.done" ]; do sleep 120; done; done
  else
    until [ -f "$NIGHT/r4cfix_${LANE}.done" ]; do sleep 60; done
  fi
  export F2_IN_SLOT=1; cd "$LAUNCH_DIR/.." && exec scripts/cpu_slot.sh 1 -- bash "$0" "$LANE"
fi
if [ "$LANE" = report ]; then
  grep -l STOP $NIGHT/r4cfix2_seed*.done && fail "a seed lane stopped"
  RUNS=$(for S in 1 2 3; do for p in ssl_r4_gpu_seed${S}_confirmatory ssl_r4_probe_seed${S}_confirmatory ssl_r4c_replay_seed$S \
           ssl_r4c_twin_seed$S ssl_r4c_randbank_seed$S ssl_r4c_diag_seed$S ssl_r4c_probe_seed$S; do d=$(latest $p); ok "$d" || { echo "NOTCLEAN $p" >&2; exit 1; }; echo "$d"; done; done | paste -sd, -) || fail "an input is not clean"
  step ssl_r4c_report $CFG 8 0 -- $PY -u experiments/ssl_r4_controls.py --stage report --config $CFG --runs "$RUNS" || fail "report"
else
  S=${LANE#seed}; T=seed$S
  G=$(latest ssl_r4_gpu_seed${S}_confirmatory); R=$(latest ssl_r4c_replay_$T); W=$(latest ssl_r4c_twin_$T)
  B=$(latest ssl_r4c_randbank_$T); D=$(latest ssl_r4c_diag_$T)
  for x in "$G" "$R" "$W" "$B" "$D"; do ok "$x" || fail "input not clean: $x"; done
  [[ "$(basename $D)" > "$(basename $B)" ]] || fail "the latest diag run is older than the latest randbank run"
  P=$(latest ssl_r4c_probe_$T)
  if ok "$P"; then echo "RESUME probe $P"; else
    step ssl_r4c_probe_$T $CFG 32 0 -- $PY -u experiments/ssl_r4_controls.py --config $CFG --gpu-run $G --stage probe \
      --replay-run $R --twin-run $W --randbank-run $B || fail "probe"
  fi
fi
echo "DONE $(date -u +%FT%TZ)" > "$NIGHT/r4cfix2_${LANE}.done"
echo "LANE $LANE DONE $(date -u +%FT%TZ)"
