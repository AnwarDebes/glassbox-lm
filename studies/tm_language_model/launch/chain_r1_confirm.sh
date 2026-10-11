#!/usr/bin/env bash
# CONFIRMATORY R1 (docs/13 A7 family H17 + A8 items 1, 6, 7 + the A9 readings, binding; config
# experiments/cvtm/configs/ssl_r1_confirm.json, script experiments/ssl_r1_confirm.py).
# Recovered 2026-09-28: rebuilt from the saved record (the text of 2026-09-25 12:52 UTC and its three later changes of 12:59 and
# 14:02 UTC); the /tmp original was lost in the host reboot of 2026-09-27 ~02:10 UTC. Two changes only: (1) the lane state
# lives in the folder of this script (persistent) instead of /tmp; (2) RESUME: a step whose latest run of the same name
# exited 0 is not rerun (its logged run is used, exactly as a completed step would have been). Scheduling only: every
# command, input and lock is unchanged, and a counted test access that already happened is never repeated.
# One LANE per invocation:
#   seed1 | seed2 | seed3   per seed, in this order (registered-threshold inputs first, Q_pred last, A8 item 7):
#                           A3, A3T, A2, A2T, A0, A1: train, census (A8 item 6), features (dev doc set), probe (dev,
#                           models frozen); for A3 and A2 also fullfeat and the full-pool probe (A9 item 8); then
#                           confirm (preflight, ONE counted access per task); then qcon and qint of A3 and A2 (qcon
#                           waits for the common lane's PPMI-SVD); then A6 and A3R (A9 item 9) the same way; then
#                           qpred of every run.
#   common                  common (G+, PPMI-SVD, doc set) -> probe; randbank seeds 1..3 -> probe -> confirm; then the
#                           common full-pool fullfeat and probe (A9 item 8) -> the common confirm.
#   report                  waits until every work lane has written its done marker, then the report (every verdict,
#                           and the list of skipped or failed steps).
# Every lane (A9 item 11):
#   1. WAITS, without a cap and before taking a CPU slot, until the exploratory control-study and configuration-arm
#      chains have finished (see exploratory_finished; the manual override is the file $NIGHT/r1_confirm_release);
#   2. takes its CPU slot (re-executes itself under scripts/cpu_slot.sh 1);
#   3. runs the testhash stage first unless a clean testhash run and both reference files exist (under an exclusive
#      flock, so the lanes run it once): the A9 item 10 references from the host's metadata, no counted access. Without
#      them no counted access runs at all (confirm and qcon are skipped and recorded); the other steps still run, and
#      a relaunch skips every step that already has a clean run (RESUME, 2026-09-28).
# Locks: every step that makes counted accesses (confirm, qcon) holds an EXCLUSIVE flock on
# $NIGHT/r1_confirm_testaccess.lock; census, qpred and qint hold it SHARED (their stages assert that the test-access
# log does not change, or that their own process made no access). The lock is taken before the GPU lock.
# No silent skips: GPU-memory and memory-headroom waits poll for up to 12 h, and every skip or failure is
# appended to $NIGHT/r1_confirm_skipped.txt, which the report lists. A failed A3 or A2 train does NOT stop the lane:
# the other arms continue (A0 and A1 feed H17.4), the failure is logged, the missing rows are INCOMPLETE.
# The test discipline is enforced by the script itself (confirm refuses before any access unless the preflight
# passes; a second access of a configuration and task is refused without a logged reason). The chain adds: confirm
# only after a clean probe step of the same configuration (and its full-pool probe for A3, A2 and common).
#
# Launch (detached; each lane waits outside any CPU slot, then takes one):
#   cd studies/tm_language_model && for L in common seed1 seed2 seed3 report; do setsid nohup bash \
#     launch/chain_r1_confirm.sh $L \
#     > launch/chain_r1_confirm_$L.out 2>&1 < /dev/null & done
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
LANE=${1:?usage: chain_r1_confirm.sh seed1|seed2|seed3|common|report}
NIGHT=$LAUNCH_DIR
SELF=$NIGHT/chain_r1_confirm.sh
CFG=experiments/cvtm/configs/ssl_r1_confirm.json
RC="$PY -u experiments/ssl_r1_confirm.py --config $CFG"
SKIPPED=$NIGHT/r1_confirm_skipped.txt
TA_LOCK=$NIGHT/r1_confirm_testaccess.lock
TH_LOCK=$NIGHT/r1_confirm_testhash.lock
WAIT_CAP_MIN=720          # 12 h for GPU-memory and memory-headroom waits
GPU_FREE_GIB=12
case "$LANE" in seed1|seed2|seed3|common|report) ;; *) echo "unknown lane $LANE"; exit 2 ;; esac
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
note() {  # every skip or failure: to the lane output and to $SKIPPED (the report lists it)
  echo "SKIP/FAIL lane $LANE $(date -u +%FT%TZ): $*" | tee -a "$SKIPPED"
}

# ------------------------------------------------------------------ 1. wait for the exploratory chains (A9 item 11)
exploratory_state() {  # prints the conditions not yet met (empty: finished)
  local s=""
  grep -q "CHAIN controls DONE" "$NIGHT/chain_controls_100.out" 2>/dev/null || s="$s controls_100"
  grep -q "CHAIN controls DONE" "$NIGHT/chain_controls_101.out" 2>/dev/null || s="$s controls_101"
  # the configuration-arm chain: DONE in chain_config_arms.out; since 12:47 UTC it is split (seed 101 in
  # chain_config_arms_101.out; stop_cfg100_after_last.sh stops the seed-100 shell at its last step, so that file may
  # never print DONE): then the stop script's record and DONE in chain_config_arms_101.out count instead
  if ! grep -Eq '^CHAIN .*DONE' "$NIGHT/chain_config_arms.out" 2>/dev/null; then
    grep -Eq 'stopped chain shell|chain shell already gone' "$NIGHT/stop_cfg100.out" 2>/dev/null || s="$s config_arms"
  fi
  if [ -e "$NIGHT/chain_config_arms_101.out" ]; then
    grep -Eq '^CHAIN .*DONE' "$NIGHT/chain_config_arms_101.out" 2>/dev/null || s="$s config_arms_101"
  fi
  # and no exploratory stage process still running (the seed-100 shell's last step outlives the stopped shell)
  pgrep -f 'experiments/ssl_r1_(controls|config_arms)\.py' >/dev/null 2>&1 && s="$s exploratory_process_running"
  echo "$s"
}
wait_exploratory() {  # no cap; polls every 5 min; a STOP line is reported once (the wait continues)
  local s i=0 warned=""
  while :; do
    s=$(exploratory_state)
    [ -z "$s" ] && { echo "WAIT exploratory chains finished $(date -u +%FT%TZ)"; return 0; }
    if [ -e "$NIGHT/r1_confirm_release" ]; then
      note "exploratory-chain wait released by $NIGHT/r1_confirm_release with pending:$s"
      return 0
    fi
    if [ -z "$warned" ] && grep -q "CHAIN STOP" "$NIGHT"/chain_controls_10[01].out "$NIGHT"/chain_config_arms*.out 2>/dev/null; then
      note "an exploratory chain printed CHAIN STOP: still waiting (rerun it to DONE, or create $NIGHT/r1_confirm_release)"
      warned=1
    fi
    [ $((i % 12)) -eq 0 ] && echo "WAIT $(date -u +%FT%TZ) pending:$s"
    i=$((i + 1))
    sleep 300
  done
}

# ------------------------------------------------------------------ 2. CPU slot: wait first, then re-execute inside one
if [ -z "${R1C_IN_SLOT:-}" ]; then
  echo "CHAIN r1_confirm lane $LANE launched $(date -u +%FT%TZ)"
  [ "$LANE" != report ] && rm -f "$NIGHT/r1_confirm_${LANE}.done"   # a work lane clears its own stale marker first
  wait_exploratory
  if [ "$LANE" = report ]; then
    sleep 60
    for L in common seed1 seed2 seed3; do      # waits outside any CPU slot; 72 h cap, recorded
      i=0; while [ ! -f "$NIGHT/r1_confirm_$L.done" ]; do
        i=$((i + 1)); [ $i -gt 864 ] && { note "lane $L not done after 72 h: the report runs without it (its rows INCOMPLETE)"; break; }
        sleep 300
      done
    done
  fi
  export R1C_IN_SLOT=1
  exec scripts/cpu_slot.sh 1 -- bash "$SELF" "$LANE"
fi
echo "CHAIN r1_confirm lane $LANE start $(date -u +%FT%TZ) (inside a CPU slot)"
echo "=== lane $LANE start $(date -u +%FT%TZ)" >> "$SKIPPED"

# ------------------------------------------------------------------ waits and steps (overrides of lib.sh: 12 h caps, recorded)
headroom() {  # wait until non-reclaimable memory + need < 0.80 * memory.max (lib.sh rule), at most 12 h
  local need=$1 i=0 cur max
  while :; do
    cur=$(awk '$1=="anon"||$1=="shmem"||$1=="kernel"||$1=="file_mapped"{s+=$2} END{printf "%d", s}' /sys/fs/cgroup/memory.stat); max=$(cat /sys/fs/cgroup/memory.max)
    if [ $(( cur + need * 1073741824 )) -lt $(( max * 8 / 10 )) ]; then return 0; fi
    i=$((i + 1)); [ $i -gt $(( WAIT_CAP_MIN * 2 )) ] && { note "memory headroom of $need GiB not reached in $WAIT_CAP_MIN min"; return 1; }
    sleep 30
  done
}
gpu_wait() {  # poll every 60 s, at most 12 h, until nvidia-smi reports >= GPU_FREE_GIB GiB free
  local name=$1 i=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    if [ -n "$free" ] && [ "$free" -ge $(( GPU_FREE_GIB * 1024 )) ]; then return 0; fi
    i=$((i + 1)); [ $i -gt $WAIT_CAP_MIN ] && { note "$name not started: GPU free ${free:-unknown} MiB < $GPU_FREE_GIB GiB after $WAIT_CAP_MIN min"; return 1; }
    sleep 60
  done
}
cstep() {  # cstep NAME LIMIT_GIB GPU(0|1) -- command: lib.sh `step` with the headroom wait for the FULL limit (lib.sh caps
           # it at 30 GiB) and every failure recorded
  local name=$1 lim=$2 gpu=$3; shift 4
  already_clean "$name" && return 0
  headroom "$lim" || { note "$name not started (memory headroom)"; return 1; }
  echo "STEP $name start $(date -u +%FT%TZ)"
  if [ "$gpu" = 1 ]; then
    scripts/gpu_run.sh shared -- nice -n 19 $PY scripts/mem_guard.py --scope group --limit-gib "$lim" -- scripts/run_logged.py --name "$name" --config "$CFG" -- "$@"
  else
    nice -n 19 $PY scripts/mem_guard.py --scope group --limit-gib "$lim" -- scripts/run_logged.py --no-gpu-sampler --name "$name" --config "$CFG" -- "$@"
  fi
  local rc=$?; echo "STEP $name exit $rc $(date -u +%FT%TZ) dir $(latest "$name")"
  [ $rc -eq 0 ] || note "$name exit $rc, see $(latest "$name")/log.txt"
  return $rc
}
already_clean() {  # RESUME (2026-09-28): the latest run of this name exited 0, so it is not rerun
  local d; d=$(latest "$1")
  ok "$d" || return 1
  echo "RESUME $1: clean run $d exists, not rerun $(date -u +%FT%TZ)"
}
gpu_step() { local name=$1 lim=$2; shift 3; already_clean "$name" && return 0; gpu_wait "$name" || return 1; cstep "$name" "$lim" 1 -- "$@"; }
cpu_step() { local name=$1 lim=$2; shift 3; cstep "$name" "$lim" 0 -- "$@"; }
with_lock() {  # with_lock -x|-s LOCKFILE CMD...: run CMD (a shell function allowed) holding flock MODE on LOCKFILE.
               # Writer priority (flock has none): an exclusive requester leaves LOCKFILE.want.<pid> while it waits, and a
               # shared requester waits while a LIVE requester's file exists (a dead one's file is removed).
  local mode=$1 file=$2; shift 2
  (
    exec {lfd}>>"$file"
    if [ "$mode" = -x ]; then
      want="$file.want.$BASHPID"; : > "$want"
      flock -x "$lfd" || { rm -f "$want"; echo "flock -x $file failed"; exit 98; }
      rm -f "$want"
    else
      while :; do
        live=0
        for wf in "$file".want.*; do
          [ -e "$wf" ] || continue
          if kill -0 "${wf##*.}" 2>/dev/null; then live=1; else rm -f "$wf"; fi
        done
        [ $live = 0 ] && break
        sleep 10
      done
      flock -s "$lfd" || { echo "flock -s $file failed"; exit 98; }
    fi
    echo "LOCK $mode $(basename "$file") held $(date -u +%FT%TZ)"
    "$@"
  )
}
done_marker() { echo "$1 $(date -u +%FT%TZ)" > "$NIGHT/r1_confirm_${LANE}.done"; }

# ------------------------------------------------------------------ 3. testhash first (A9 item 10; no counted access)
TESTHASH_OK=0
testhash_once() {
  local d; d=$(latest ssl_r1cf_testhash_confirmatory)
  if ok "$d" && [ -f docs/notes/preregistration-data-hashes-A7-agnews-test.json ] && [ -f docs/notes/preregistration-data-hashes-A7-trec-test.json ]; then
    echo "testhash: clean run $d and both reference files exist"; return 0
  fi
  cpu_step ssl_r1cf_testhash_confirmatory 4 -- $RC --stage testhash
}
if with_lock -x "$TH_LOCK" testhash_once; then TESTHASH_OK=1; else note "testhash failed: no confirm step of this lane will run (no reference, A9 item 10)"; fi

# one configuration (arm, seed): train, census, features, probe [, fullfeat, full probe], confirm
arm_config() {
  local A=$1 S=$2 T TR F FR P PF N FULL=0
  case "$A" in A3|A2) FULL=1 ;; esac
  T=ssl_r1cf_train_${A}_seed${S}_confirmatory
  if ! gpu_step $T 16 -- $RC --stage train --arm $A --seed $S --engine cuda; then
    case "$A" in
      A3|A2) note "train $A seed $S failed: the lane continues with the other arms (A0 and A1 feed H17.4); the H17.1 to H17.3 or A2 rows of seed $S will be INCOMPLETE" ;;
      *) note "train $A seed $S failed: its configuration is skipped" ;;
    esac
    return 1
  fi
  TR=$(latest $T)
  with_lock -s "$TA_LOCK" cpu_step ssl_r1cf_census_${A}_seed${S}_confirmatory 8 -- $RC --stage census --train-run "$TR"
  F=ssl_r1cf_features_${A}_seed${S}_confirmatory
  gpu_step $F 24 -- $RC --stage features --train-run "$TR" --engine cuda || { note "NO CONFIRM for $A seed $S: features failed"; return 1; }
  FR=$(latest $F)
  P=ssl_r1cf_probe_${A}_seed${S}_confirmatory
  cpu_step $P 24 -- $RC --stage probe --features-run "$FR" || { note "NO CONFIRM for $A seed $S: the dev probe failed"; return 1; }
  ok "$(latest $P)" || { note "NO CONFIRM for $A seed $S: the dev probe run is not clean"; return 1; }
  local EXTRA=""
  if [ $FULL = 1 ]; then
    N=ssl_r1cf_fullfeat_${A}_seed${S}_confirmatory
    gpu_step $N 24 -- $RC --stage fullfeat --train-run "$TR" --features-run "$FR" --engine cuda \
      || { note "NO CONFIRM for $A seed $S: the full-pool fullfeat failed (A9 item 8 needs it before the access)"; return 1; }
    PF=ssl_r1cf_probefull_${A}_seed${S}_confirmatory
    cpu_step $PF 24 -- $RC --stage probe --features-run "$(latest $N)" \
      || { note "NO CONFIRM for $A seed $S: the full-pool probe failed (A9 item 8 needs it before the access)"; return 1; }
    ok "$(latest $PF)" || { note "NO CONFIRM for $A seed $S: the full-pool probe run is not clean"; return 1; }
    EXTRA="--probe-full-run $(latest $PF)"
  fi
  [ $TESTHASH_OK = 1 ] || { note "NO CONFIRM for $A seed $S: no test hash references"; return 1; }
  N=ssl_r1cf_confirm_${A}_seed${S}_confirmatory
  with_lock -x "$TA_LOCK" gpu_step $N 24 -- $RC --stage confirm --probe-run "$(latest $P)" $EXTRA --engine cuda
}

case "$LANE" in
  seed1|seed2|seed3)
    S=${LANE#seed}
    for A in A3 A3T A2 A2T A0 A1; do arm_config $A $S; done
    COMMON=""
    for i in $(seq 1 288); do   # the common run (PPMI-SVD for Q_con): at most 24 h, recorded
      d=$(latest ssl_r1cf_common_confirmatory); ok "$d" && { COMMON=$d; break; }; sleep 300
    done
    [ -n "$COMMON" ] || note "no clean common run after 24 h: qcon of seed $S skipped (H17.3 INCOMPLETE)"
    for A in A3 A2; do
      TR=$(latest ssl_r1cf_train_${A}_seed${S}_confirmatory); FR=$(latest ssl_r1cf_features_${A}_seed${S}_confirmatory)
      PR=$(latest ssl_r1cf_probe_${A}_seed${S}_confirmatory)
      ok "$TR" && ok "$FR" && ok "$PR" || { note "qcon and qint of $A seed $S skipped: no clean train, features or dev probe run"; continue; }
      if [ -n "$COMMON" ] && [ $TESTHASH_OK = 1 ]; then
        with_lock -x "$TA_LOCK" gpu_step ssl_r1cf_qcon_${A}_seed${S}_confirmatory 24 -- $RC --stage qcon --train-run "$TR" --probe-run "$PR" --common-run "$COMMON" --engine cuda
      elif [ -n "$COMMON" ]; then   # A9 item 10: no counted access (the MEN test pairs included) before every reference exists
        note "NO QCON for $A seed $S: no test hash references (A9 item 10: no counted access before them; H17.3 INCOMPLETE)"
      fi
      with_lock -s "$TA_LOCK" gpu_step ssl_r1cf_qint_${A}_seed${S}_confirmatory 32 -- $RC --stage qint --train-run "$TR" --features-run "$FR" --probe-run "$PR" --engine cuda
    done
    for A in A6 A3R; do arm_config $A $S; done
    for A in A3 A3T A2 A2T A0 A1 A6 A3R; do
      TR=$(latest ssl_r1cf_train_${A}_seed${S}_confirmatory); ok "$TR" || { note "qpred $A seed $S skipped: no clean train run"; continue; }
      with_lock -s "$TA_LOCK" gpu_step ssl_r1cf_qpred_${A}_seed${S}_confirmatory 24 -- $RC --stage qpred --train-run "$TR" --engine cuda
    done
    done_marker DONE ;;
  common)
    N=ssl_r1cf_common_confirmatory
    P=ssl_r1cf_probe_common_confirmatory
    CM=""
    if gpu_step $N 24 -- $RC --stage common --engine cuda; then
      CM=$(latest $N)
      cpu_step $P 24 -- $RC --stage probe --features-run "$CM" || note "NO CONFIRM for common: the dev probe failed"
    fi
    for S in 1 2 3; do
      N=ssl_r1cf_randbank_seed${S}_confirmatory
      gpu_step $N 24 -- $RC --stage randbank --seed $S --engine cuda || { note "NO CONFIRM for rand_seed$S: randbank failed"; continue; }
      PR=ssl_r1cf_probe_rand_seed${S}_confirmatory
      cpu_step $PR 24 -- $RC --stage probe --features-run "$(latest $N)" || { note "NO CONFIRM for rand_seed$S: the dev probe failed"; continue; }
      ok "$(latest $PR)" || { note "NO CONFIRM for rand_seed$S: the dev probe run is not clean"; continue; }
      [ $TESTHASH_OK = 1 ] || { note "NO CONFIRM for rand_seed$S: no test hash references"; continue; }
      with_lock -x "$TA_LOCK" gpu_step ssl_r1cf_confirm_rand_seed${S}_confirmatory 24 -- $RC --stage confirm --probe-run "$(latest $PR)" --engine cuda
    done
    if [ -n "$CM" ] && ok "$(latest $P)"; then
      NF=ssl_r1cf_fullfeat_common_confirmatory
      PF=ssl_r1cf_probefull_common_confirmatory
      if gpu_step $NF 32 -- $RC --stage fullfeat --common-run "$CM" --engine cuda \
         && cpu_step $PF 40 -- $RC --stage probe --features-run "$(latest $NF)" && ok "$(latest $PF)"; then
        if [ $TESTHASH_OK = 1 ]; then
          with_lock -x "$TA_LOCK" gpu_step ssl_r1cf_confirm_common_confirmatory 24 -- $RC --stage confirm --probe-run "$(latest $P)" --probe-full-run "$(latest $PF)" --engine cuda
        else
          note "NO CONFIRM for common: no test hash references"
        fi
      else
        note "NO CONFIRM for common: the full-pool fullfeat or probe failed (A9 item 8 needs it before the access; H17.1 and H17.2 stay INCOMPLETE until it is fixed)"
      fi
    else
      note "NO CONFIRM for common: no clean common or dev probe run"
    fi
    done_marker DONE ;;
  report)
    col() {  # the latest clean run of every distinct run name of stage $1 (a retried stage enters once)
      local seen=" " d n
      for d in $(ls -td results/*_ssl_r1cf_$1_*confirmatory 2>/dev/null); do
        n=${d#results/*Z_}; case "$seen" in *" $n "*) continue ;; esac
        ok "$d" && { seen="$seen$n "; echo "$d"; }
      done | paste -sd, -
    }
    PROBES=$(echo "$(col probe),$(col probefull)" | sed 's/^,//; s/,$//')
    cpu_step ssl_r1cf_report_confirmatory 8 -- $RC --stage report --probe-runs "$PROBES" --confirm-runs "$(col confirm)" \
      --qpred-runs "$(col qpred)" --qcon-runs "$(col qcon)" --qint-runs "$(col qint)" --census-runs "$(col census)" \
      --skipped-file "$SKIPPED"
    done_marker DONE ;;
esac
echo "CHAIN r1_confirm lane $LANE DONE $(date -u +%FT%TZ)"
