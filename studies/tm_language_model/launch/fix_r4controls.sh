#!/usr/bin/env bash
# 2026-09-30 fix of the R4 controls chain: the randbank stage was stopped by mem_guard at 12 GiB on all three seeds
# (12.13 to 12.78 GiB; the smoke used 2,000 images), the probe then failed on the killed randbank run's missing metrics,
# and diag had read the killed run's random_bank.npz. Same commands; randbank at 40 GiB and one seed at a time (flock, so
# three copies cannot pass the 85 GiB project cap); then a NEW diag run on the clean randbank run, the probe, and the
# report. Replay and twin runs are clean and reused. A failed step stops its lane.
# Lanes: seed1 | seed2 | seed3 | report (report waits for the three seed lanes).
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?}
NIGHT=$LAUNCH_DIR
CFG=experiments/cvtm/configs/ssl_r4_controls.json
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
fail() { echo "LANE STOP $LANE $(date -u +%FT%TZ): $*"; echo "STOP $*" > "$NIGHT/r4cfix_${LANE}.done"; exit 1; }
rm -f "$NIGHT/r4cfix_${LANE}.done"
if [ "$LANE" = report ]; then
  for L in seed1 seed2 seed3; do until [ -f "$NIGHT/r4cfix_$L.done" ]; do sleep 120; done; done
  grep -l STOP $NIGHT/r4cfix_seed*.done && fail "a seed lane stopped"
  RUNS=$(for S in 1 2 3; do for p in ssl_r4_gpu_seed${S}_confirmatory ssl_r4_probe_seed${S}_confirmatory ssl_r4c_replay_seed$S \
           ssl_r4c_twin_seed$S ssl_r4c_randbank_seed$S ssl_r4c_diag_seed$S ssl_r4c_probe_seed$S; do d=$(latest $p); ok "$d" || fail "$p not clean"; echo "$d"; done; done | paste -sd, -)
  step ssl_r4c_report $CFG 8 0 -- $PY -u experiments/ssl_r4_controls.py --stage report --config $CFG --runs "$RUNS" || fail "report"
else
  S=${LANE#seed}; T=seed$S
  G=$(latest ssl_r4_gpu_seed${S}_confirmatory); R=$(latest ssl_r4c_replay_$T); W=$(latest ssl_r4c_twin_$T)
  ok "$G" && ok "$R" && ok "$W" || fail "gpu, replay or twin run not clean"
  X="$PY -u experiments/ssl_r4_controls.py --config $CFG --gpu-run $G"
  ( exec {l}>>"$NIGHT/r4c_randbank.lock"; flock -x "$l"
    step ssl_r4c_randbank_$T $CFG 40 1 -- $X --stage randbank ) || fail "randbank"
  B=$(latest ssl_r4c_randbank_$T); ok "$B" || fail "randbank not clean"
  C="--replay-run $R --twin-run $W --randbank-run $B"
  step ssl_r4c_diag_$T $CFG 12 1 -- $X --stage diag $C || fail "diag"
  step ssl_r4c_probe_$T $CFG 16 0 -- $X --stage probe $C || fail "probe"
fi
echo "DONE $(date -u +%FT%TZ)" > "$NIGHT/r4cfix_${LANE}.done"
echo "LANE $LANE DONE $(date -u +%FT%TZ)"
