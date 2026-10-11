#!/usr/bin/env bash
# 2026-10-01, third fix of the R4 controls probe. The probes of all three seeds fitted every ladder rung and the twin, then
# were stopped by mem_guard while loading the random bank (16.8 to 34.1 GiB): its code has 832 million nonzeros (about 54,600
# distinct codes per image against about 8,700 for the trained bank). Same probe code, split by representation:
#   lane mainN  (N = 1, 2, 3; parallel): --reps ladder.c0y,ladder.pct5,ladder.pct10,ladder.epoch1,twin, 32 GiB, run name
#               ssl_r4c_probe_seedN (the reps that fitted before)
#   lane rand   after every main lane: --reps randbank, seeds 1, 2, 3 one at a time and alone on the machine, 78 GiB (under
#               the 85 GiB project cap), run name ssl_r4c_probe_randbank_seedN
#   then the report over every run (the report pools the cells of all control runs; no cell appears twice).
# A failed step stops its lane; RESUME skips a clean run of the same name.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?}
NIGHT=$LAUNCH_DIR
CFG=experiments/cvtm/configs/ssl_r4_controls.json
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
fail() { echo "LANE STOP $LANE $(date -u +%FT%TZ): $*"; echo "STOP $*" > "$NIGHT/r4cfix3_${LANE}.done"; exit 1; }
rstep() { local d; d=$(latest "$1"); ok "$d" && { echo "RESUME $1: $d"; return 0; }; step "$@"; }
inputs() {  # S -> sets G R W B
  G=$(latest ssl_r4_gpu_seed${1}_confirmatory); R=$(latest ssl_r4c_replay_seed$1); W=$(latest ssl_r4c_twin_seed$1); B=$(latest ssl_r4c_randbank_seed$1)
  for x in "$G" "$R" "$W" "$B"; do ok "$x" || fail "input not clean: $x"; done
}
rm -f "$NIGHT/r4cfix3_${LANE}.done"
if [ -z "${F3_IN_SLOT:-}" ]; then
  [ "$LANE" = rand ] && for L in main1 main2 main3; do until [ -f "$NIGHT/r4cfix3_$L.done" ]; do sleep 120; done; done
  export F3_IN_SLOT=1; cd "$LAUNCH_DIR/.." && exec scripts/cpu_slot.sh 1 -- bash "$0" "$LANE"
fi
case "$LANE" in
  main1|main2|main3)
    S=${LANE#main}; inputs $S
    rstep ssl_r4c_probe_seed$S $CFG 32 0 -- $PY -u experiments/ssl_r4_controls.py --config $CFG --gpu-run $G --stage probe \
      --replay-run $R --twin-run $W --randbank-run $B --reps ladder.c0y,ladder.pct5,ladder.pct10,ladder.epoch1,twin || fail "probe seed $S" ;;
  rand)
    grep -l STOP $NIGHT/r4cfix3_main*.done && fail "a main lane stopped"
    for S in 1 2 3; do
      inputs $S
      rstep ssl_r4c_probe_randbank_seed$S $CFG 78 0 -- $PY -u experiments/ssl_r4_controls.py --config $CFG --gpu-run $G --stage probe \
        --replay-run $R --twin-run $W --randbank-run $B --reps randbank || fail "randbank probe seed $S"
    done
    RUNS=$(for S in 1 2 3; do for p in ssl_r4_gpu_seed${S}_confirmatory ssl_r4_probe_seed${S}_confirmatory ssl_r4c_replay_seed$S \
             ssl_r4c_twin_seed$S ssl_r4c_randbank_seed$S ssl_r4c_diag_seed$S ssl_r4c_probe_seed$S ssl_r4c_probe_randbank_seed$S; do
             d=$(latest $p); ok "$d" || { echo "NOTCLEAN $p" >&2; exit 1; }; echo "$d"; done; done | paste -sd, -) || fail "an input is not clean"
    rstep ssl_r4c_report $CFG 8 0 -- $PY -u experiments/ssl_r4_controls.py --stage report --config $CFG --runs "$RUNS" || fail "report" ;;
esac
echo "DONE $(date -u +%FT%TZ)" > "$NIGHT/r4cfix3_${LANE}.done"
echo "LANE $LANE DONE $(date -u +%FT%TZ)"
