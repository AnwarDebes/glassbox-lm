#!/usr/bin/env bash
# CONFIRMATORY R2 (H17.5, seeds 1 to 5, experiments/cvtm/configs/ssl_r2_hao.json, package c85ca4e7 as docs/13 A8 item 3
# registers it) and R4 (H17.10, seeds 1 to 3, experiments/cvtm/configs/ssl_r4_images.json), after gate G-R1 read "fail and
# below bag of words: weight moves to R2 and R4" (2026-09-30). Commands, configs and mem limits as in the drivers'
# docstrings; run names carry the suffix _confirmatory. Neither driver touches a test split (R2's test bags are synthetic
# held-out combinations; R4 never downloads the Fashion-MNIST test file).
# Lanes (one invocation each; every lane takes ONE CPU slot, CPU slot before GPU lock, docs/12 F25):
#   gpu      waits for the c85 demo rerun to finish; R4 gpu seeds 1, 2, 3; R2 gpu (5 seeds in one run); then the R2
#            control (Granmo's NoisyParity demo) under the EXCLUSIVE GPU lock: its pass rule is bounded by a 7,200 s wall
#            cap, and GPU sharing truncated such a cap before (c85 demos, 2026-09-25)
#   r4seedN  (N = 1, 2, 3) waits for its gpu run, then probe and probe_tm
#   r2cpu    the R2 CPU stage (tmu and sklearn models, 5 seeds)
#   report   waits for every lane's done marker, then the R4 and R2 reports
# RESUME: a step whose latest run of that name exited 0 is not rerun. Failures go to $NIGHT/r2r4_skipped.txt.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?usage: chain_r2r4_confirm.sh gpu|r4seed1|r4seed2|r4seed3|r2cpu|report}
NIGHT=$LAUNCH_DIR
SELF=$NIGHT/chain_r2r4_confirm.sh
C2=experiments/cvtm/configs/ssl_r2_hao.json
C4=experiments/cvtm/configs/ssl_r4_images.json
R2="$PY -u experiments/ssl_r2_hao.py --config $C2"
R4="$PY -u experiments/ssl_r4_images.py --config $C4"
SKIPPED=$NIGHT/r2r4_skipped.txt
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
note() { echo "SKIP/FAIL lane $LANE $(date -u +%FT%TZ): $*" | tee -a "$SKIPPED"; }
done_marker() { echo "$1 $(date -u +%FT%TZ)" > "$NIGHT/r2r4_${LANE}.done"; }
wait_clean() {  # wait_clean NAME MARKER_LANE: until run NAME is clean, or the lane that makes it is done without it
  local i=0
  while :; do
    ok "$(latest "$1")" && return 0
    [ -f "$NIGHT/r2r4_$2.done" ] && { note "$1 has no clean run and lane $2 is done"; return 1; }
    i=$((i + 1)); [ $((i % 30)) -eq 1 ] && echo "WAIT $1 $(date -u +%FT%TZ)"
    sleep 120
  done
}

case "$LANE" in gpu|r4seed1|r4seed2|r4seed3|r2cpu|report) ;; *) echo "unknown lane $LANE"; exit 2 ;; esac
if [ -z "${R2R4_IN_SLOT:-}" ]; then
  echo "CHAIN r2r4 lane $LANE launched $(date -u +%FT%TZ)"
  rm -f "$NIGHT/r2r4_${LANE}.done"
  if [ "$LANE" = gpu ]; then   # the c85 demo rerun uses the GPU and has a wall cap: do not share with it
    until grep -q 'C85 RERUN DONE\|c85 demos FAILED' $NIGHT/rerun_c85_demos.out 2>/dev/null || ! pgrep -f rerun_c85_demos.sh >/dev/null; do sleep 120; done
  fi
  case "$LANE" in r4seed*)   # wait for this seed's gpu run outside any CPU slot
    until ok "$(latest ssl_r4_gpu_seed${LANE#r4seed}_confirmatory)" || [ -f "$NIGHT/r2r4_gpu.done" ]; do sleep 120; done ;;
  esac
  if [ "$LANE" = report ]; then
    for L in gpu r4seed1 r4seed2 r4seed3 r2cpu; do until [ -f "$NIGHT/r2r4_$L.done" ]; do sleep 300; done; done
  fi
  export R2R4_IN_SLOT=1
  exec scripts/cpu_slot.sh 1 -- bash "$SELF" "$LANE"
fi
echo "CHAIN r2r4 lane $LANE start $(date -u +%FT%TZ) (inside a CPU slot)"

rstep() {  # rstep NAME CFG LIMIT GPU(0|1|x) -- command: lib.sh step with RESUME, x = exclusive GPU lock
  local name=$1 cfg=$2 lim=$3 gpu=$4; shift 5
  local d; d=$(latest "$name"); ok "$d" && { echo "RESUME $name: clean run $d exists, not rerun"; return 0; }
  if [ "$gpu" = x ]; then
    headroom "$lim" || { note "$name not started (memory headroom)"; return 1; }
    echo "STEP $name start $(date -u +%FT%TZ) (exclusive GPU lock)"
    scripts/gpu_run.sh exclusive -- nice -n 19 $PY scripts/mem_guard.py --scope group --limit-gib "$lim" -- scripts/run_logged.py --name "$name" --config "$cfg" -- "$@"
    local rc=$?; echo "STEP $name exit $rc $(date -u +%FT%TZ) dir $(latest "$name")"
  else
    step "$name" "$cfg" "$lim" "$gpu" -- "$@"; local rc=$?
  fi
  [ $rc -eq 0 ] || note "$name exit $rc, see $(latest "$name")/log.txt"
  return $rc
}

case "$LANE" in
  gpu)
    for S in 1 2 3; do rstep ssl_r4_gpu_seed${S}_confirmatory $C4 12 1 -- $R4 --stage gpu --seed $S; done
    rstep ssl_r2_hao_gpu_confirmatory $C2 8 1 -- $R2 --stage gpu --replay-check
    rstep ssl_r2_hao_control_confirmatory $C2 8 x -- $R2 --stage control
    done_marker DONE ;;
  r4seed1|r4seed2|r4seed3)
    S=${LANE#r4seed}
    if wait_clean ssl_r4_gpu_seed${S}_confirmatory gpu; then
      G=$(latest ssl_r4_gpu_seed${S}_confirmatory)
      rstep ssl_r4_probe_seed${S}_confirmatory $C4 16 0 -- $R4 --stage probe --gpu-run "$G"
      rstep ssl_r4_probe_tm_seed${S}_confirmatory $C4 16 0 -- $R4 --stage probe_tm --gpu-run "$G"
    fi
    done_marker DONE ;;
  r2cpu)
    rstep ssl_r2_hao_cpu_confirmatory $C2 8 0 -- $R2 --stage cpu --replay-check
    done_marker DONE ;;
  report)
    R4RUNS=$(for S in 1 2 3; do for n in probe probe_tm gpu; do d=$(latest ssl_r4_${n}_seed${S}_confirmatory); ok "$d" && echo "$d"; done; done | paste -sd, -)
    if [ -n "$R4RUNS" ]; then rstep ssl_r4_report_confirmatory $C4 8 0 -- $R4 --stage report --runs "$R4RUNS"; else note "no clean R4 run: no R4 report"; fi
    R2RUNS=$(for n in gpu cpu control; do d=$(latest ssl_r2_hao_${n}_confirmatory); ok "$d" && echo "$d"; done | paste -sd, -)
    if [ -n "$R2RUNS" ]; then rstep ssl_r2_hao_report_confirmatory $C2 4 0 -- $R2 --stage report --runs "$R2RUNS"; else note "no clean R2 run: no R2 report"; fi
    done_marker DONE ;;
esac
echo "CHAIN r2r4 lane $LANE DONE $(date -u +%FT%TZ)"
