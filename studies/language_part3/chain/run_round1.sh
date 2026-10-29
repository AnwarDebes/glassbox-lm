#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# chain/run_round1.sh: the round-1 Part 3 CPU chain (PLAN_part3.md section 6, steps 2-6), one job at a time.
#
#   chain/run_round1.sh          run it; restart-safe (a step whose outputs exist and verify is skipped)
#   chain/run_round1.sh --dry    print every step with its done state and run every hash check; runs nothing and
#                                writes nothing
#
# Order (plan, section 6):
#   step 2  C Step 0 (exploratory, decides nothing)
#   step 3  A S0, the bit-list LOG.md line (A/log_bitlist.sh, not a CPU job), S1 (K1)
#   step 4  A S2a, S2b-twin (K9), S2c (K8), the 8 learners full/nf x tree/lr1/lr2/hgb (K9), S2d (K3)
#   step 5  A K3c, for each arm with go_to_K3c: pilot s 15, pilot s 40, gate; when the gate asks (exit 5), the shuffled
#           pilot at the better s, then the gate again
#   step 6  C prep, twin, learners, gates (K0 sanity, K0b), "after 5, or after 4 if no A arm passed K3"
# Before each step: the hash check (chain/round1_lib.py hashcheck: the pins written in the R1 lines of LOG.md, the
# arm's frozen record and every file in it, the R1 lines' code prefixes, the chain wrapper), the done check, the time
# box (A: K7 by A/k7_clock.py; C: C-K4), a wait while another job holds the chain lock, then the CPU gate
# (cpu_load.py 10, read by A's registered parser A/gate_check.sh: at most 4.0 cores and 0 throttled periods; checked
# again every 2 minutes while closed). Each CPU
# job runs as `nice -n 19 chain/part3_cpu_job.sh ...`: the wrapper holds chain/part3_cpu.lock, and the arm's runner
# reads the gate again inside it and runs python -I -B at nice 19 with 2 threads, caches in the arm's tmp/, no GPU,
# and listings of the imported folders before and after.
# Stops. A registered STOP (K1, K8, K9, K3 for both arms; K7) ends A's line; C step 6 still follows (plan, section 6).
# A K3c STOP ends that arm only. A C gates STOP or C-K4 ends C. Anything else (a failed job, exit 4 = a listing
# change, exit 2 = a refused start, a hash mismatch, outputs that do not verify) stops the whole chain at once.
# Records: chain/round1.log (every check, start, end, exit code and decision, times from date -u in the writing
# command); chain/round1_out/<UTC>_<step>.txt (the wrapper's output of each job); chain/round1_state/<step>.sha256
# (the files of each finished A step, checked on a restart). No GPU step runs here (plan steps 7, 9 and 11 are
# scheduled by the GPU queue). Stopping the chain (SIGTERM) lets a running job finish under the chain lock; a
# restart skips every finished step.
# Exit: 0 every line reached its end (a registered STOP is an end); 1 stopped at a failure; 2 refused to start.
set -u
TM=${REPO_ROOT}/studies/language_part3
CH=$TM/chain
SELF=$CH/run_round1.sh
# a clean environment: no variables of the calling session reach the jobs or their records
if [ "${ROUND1_ENV_CLEAN:-}" != "1" ]; then
  keep=()
  for v in ROUND1_TEST_DIR ROUND1_TEST_PINS ROUND1_TEST_SLEEP ROUND1_TEST_HASHCACHE; do
    if [ -n "${!v:-}" ]; then keep+=("$v=${!v}"); fi
  done
  exec env -i ROUND1_ENV_CLEAN=1 HOME=${HOME} USER=coder LOGNAME=coder SHELL=/bin/bash LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "${keep[@]}" \
    /bin/bash "$SELF" "$@"
fi
export CUDA_VISIBLE_DEVICES= CUDA_CACHE_DISABLE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2
PY_TM=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
PY_DL=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
CPU_LOAD=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
LIB=$CH/round1_lib.py
WRAPPER=$CH/part3_cpu_job.sh
CHAIN_LOCK=$CH/part3_cpu.lock
LOG=$CH/round1.log
OUTS=$CH/round1_out
SELF_LOCK=$CH/tmp/run_round1.lock
A_JOBS=$TM/A/out/jobs.log
BITLIST_LOG=$TM/LOG.md
CTMP=$CH/tmp/round1
SLEEP_GATE=110      # + the 10 s reading of cpu_load.py = one check every 2 minutes
SLEEP_RETRY=60
MAX_RETRY=240
TEST=0
T=""
if [ -n "${ROUND1_TEST_DIR:-}" ]; then
  T=$(realpath -m "$ROUND1_TEST_DIR")
  case "$T/" in "$CH/tmp/"*) ;; *) echo "ROUND1_TEST_DIR must be inside $CH/tmp/ (got $T)"; exit 2 ;; esac
  [ -f "$T/fake_wrapper.sh" ] || { echo "test mode needs $T/fake_wrapper.sh"; exit 2; }
  TEST=1
  WRAPPER=$T/fake_wrapper.sh; CHAIN_LOCK=$T/part3_cpu.lock; LOG=$T/round1.log; OUTS=$T/out
  SELF_LOCK=$T/run_round1.lock; A_JOBS=$T/A_out/jobs.log; BITLIST_LOG=$T/LOG.md; CTMP=$T/tmp
  SLEEP_GATE=${ROUND1_TEST_SLEEP:-0}; SLEEP_RETRY=${ROUND1_TEST_SLEEP:-0}; MAX_RETRY=3
fi
export NUMBA_CACHE_DIR=$CTMP/numba TMPDIR=$CTMP/tmpdir JOBLIB_TEMP_FOLDER=$CTMP/joblib MPLCONFIGDIR=$CTMP/mpl \
  XDG_CACHE_HOME=$CTMP/xdg
DRY=0
case "${1:-}" in
  --dry) DRY=1 ;;
  "") ;;
  *) echo "usage: run_round1.sh [--dry]"; exit 2 ;;
esac
cd "$TM" || exit 2

log() {
  if [ "$DRY" = 1 ]; then echo "$*"; return 0; fi
  echo "$(date -u +%FT%TZ) $*" >> "$LOG"
}
fail() {
  log "FAIL $*; the chain stops here"
  [ "$DRY" = 1 ] || echo "FAIL $*"
  exit 1
}
lib() { nice -n 19 "$PY_TM" -I -B "$LIB" "$@"; }
header() {
  if [ "$DRY" = 1 ]; then echo; echo "== $*"; else log "== $*"; fi
}

NOTED_A=0
NOTED_C=0
hashcheck_step() {  # <arm> <step>: exits on a mismatch
  local arm=$1 step=$2 out rc n_ok n_note
  out=$(lib hashcheck "$arm" "$step" 2>&1); rc=$?
  n_ok=$(printf '%s\n' "$out" | grep -c '^ok')
  n_note=$(printf '%s\n' "$out" | grep -c '^NOTE')
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$out" | grep -v '^ok' | sed 's/^/    /' | while IFS= read -r ln; do log "hashcheck $step: $ln"; done
    fail "$step: the hash check against the R1 lines and the frozen record failed (see the lines above)"
  fi
  if { [ "$arm" = A ] && [ "$NOTED_A" = 0 ]; } || { [ "$arm" = C ] && [ "$NOTED_C" = 0 ]; }; then
    printf '%s\n' "$out" | grep -v '^hashcheck' | while IFS= read -r ln; do log "hashcheck $step: $ln"; done
    if [ "$arm" = A ]; then NOTED_A=1; else NOTED_C=1; fi
  fi
  log "hashcheck $step: OK ($n_ok checks ok, $n_note notes; $(printf '%s\n' "$out" | grep "^ok step" | sed 's/^ok step [^:]*: //'))"
  return 0
}

gate_line() {
  if [ "$TEST" = 1 ]; then
    local f=$T/gate_lines.txt l=""
    if [ -s "$f" ]; then l=$(head -n 1 "$f"); sed -i '1d' "$f"; fi
    [ -n "$l" ] || l="$(date -u +%FT%TZ) cores 0.00 over 10.0 s; throttled 0 of 100 periods"
    echo "$l"
  else
    nice -n 19 "$PY_DL" -I -B "$CPU_LOAD" 10 2>&1
  fi
}

timebox() {  # <arm> <step>: 0 open, 6 reached (logged)
  local arm=$1 step=$2 out rc
  if [ "$arm" = A ]; then
    out=$(nice -n 19 "$PY_TM" -I -B "$TM/A/k7_clock.py" --jobs "$A_JOBS" 2>&1); rc=$?
    log "timebox $step: K7 $out"
  else
    out=$(lib ck4 2>&1); rc=$?
    log "timebox $step: C-K4 $out"
  fi
  case "$rc" in
    0) return 0 ;;
    3) return 6 ;;
    *) fail "$step: the time-box check failed ($out)" ;;
  esac
}

wait_gate() {  # <arm> <step>: 0 when open, 6 when the time box is reached while waiting
  local arm=$1 step=$2 line n=0 bad=0
  while true; do
    line=$(gate_line)
    if printf '%s\n' "$line" | bash "$TM/A/gate_check.sh"; then
      log "gate OPEN ($line) before $step"
      return 0
    fi
    if printf '%s\n' "$line" | grep -Eq ' cores [0-9.]+ over .* throttled [0-9]+ of '; then
      bad=0
    else
      bad=$((bad + 1))
      [ "$bad" -lt 5 ] || fail "$step: the CPU gate reading failed 5 times in a row (last: $line)"
    fi
    log "gate closed ($line) before $step"
    n=$((n + 1))
    if [ $((n % 30)) = 0 ]; then
      timebox "$arm" "$step" || return 6
    fi
    sleep "$SLEEP_GATE"
  done
}

wait_lock_free() {  # <step>
  local step=$1 n=0
  while bash "$TM/A/lock_held.sh" "$CHAIN_LOCK"; do
    if [ "$n" = 0 ]; then log "wait $step: another job holds the chain lock ${CHAIN_LOCK#$TM/}"; fi
    n=$((n + 1))
    sleep "$SLEEP_RETRY"
  done
  if [ "$n" != 0 ]; then log "the chain lock is free again before $step (after $n checks)"; fi
  return 0
}

dry_show() {  # <step> <status code> <status text> <command>
  local what
  case "$2" in
    0) what="done: skip" ;;
    3) what="done, a registered STOP: skip; the line ends" ;;
    10) what="not done: run" ;;
    *) what="INCONSISTENT: the chain would stop" ;;
  esac
  printf '  %-20s %s\n  %-20s   -> %s (%s)\n' "$1" "$4" "" "$what" "$3"
}

LAST_OUT=""
# run_step <step> <arm A|C> <wrapper arguments...>
# returns 0 done (its gate passed), 3 a registered STOP, 5 the K3c gate asks for the shuffled pilot, 6 the time box
# is reached; exits on a failure
run_step() {
  local step=$1 arm=$2
  shift 2
  local st rc src out wl n=0 wpid
  hashcheck_step "$arm" "$step"
  st=$(lib status "$step" 2>&1); rc=$?
  if [ "$DRY" = 1 ]; then
    dry_show "$step" "$rc" "$st" "part3_cpu_job.sh $*"
    return 0
  fi
  case "$rc" in
    0) log "skip $step: done ($st)"; return 0 ;;
    3) log "skip $step: done, a registered STOP ($st)"; return 3 ;;
    10) log "todo $step: $st" ;;
    *) fail "$step: outputs exist but do not verify ($st)" ;;
  esac
  timebox "$arm" "$step" || return 6
  while true; do
    wait_lock_free "$step"
    wait_gate "$arm" "$step" || return 6
    out="$OUTS/$(date -u +%Y%m%dT%H%M%SZ)_${step}.txt"
    log "start $step: nice -n 19 part3_cpu_job.sh $* (wrapper output ${out#$CH/})"
    nice -n 19 bash "$WRAPPER" "$@" > "$out" 2>&1 < /dev/null &
    wpid=$!
    wait "$wpid"
    rc=$?
    wl=$(tail -n 1 "$out")
    log "end $step: wrapper exit $rc ($wl)"
    LAST_OUT=$out
    [ "$rc" = 75 ] || break
    n=$((n + 1))
    [ "$n" -lt "$MAX_RETRY" ] || fail "$step: C's own lock stayed busy for $n tries"
    log "retry $step in $SLEEP_RETRY s: C's own lock was busy (try $n of $MAX_RETRY)"
    sleep "$SLEEP_RETRY"
  done
  st=$(lib status "$step" 2>&1); src=$?
  case "$rc" in
    0)
      [ "$src" = 0 ] || fail "$step: wrapper exit 0, but the outputs do not show a finished step with its gate passed (status $src: $st)"
      write_manifest "$step"
      log "done $step: $st"
      return 0 ;;
    3)
      [ "$src" = 3 ] || fail "$step: wrapper exit 3 (a registered STOP), but the outputs do not show one (status $src: $st)"
      write_manifest "$step"
      log "decision $step: registered STOP: $st"
      return 3 ;;
    5)
      case "$step" in A_k3c_gate_*) ;; *) fail "$step: unexpected exit 5" ;; esac
      [ "$src" = 10 ] || fail "$step: exit 5, but a decision folder exists ($st)"
      log "decision $step: K3c passes at the better s; K9 for the TM needs the shuffled pilot first"
      return 5 ;;
    4) fail "$step: the imported folders changed during the job (exit 4); read the listing diff named in $out" ;;
    2) fail "$step: the start was refused or the command is wrong (exit 2; nothing written); see $out" ;;
    *) fail "$step: the job failed (exit $rc); see $out and the job log it names" ;;
  esac
}

write_manifest() {
  local step=$1 out
  case "$step" in
    A_*) out=$(lib manifest write "$step" 2>&1) || fail "$step: cannot write the chain manifest ($out)"
         log "manifest $step: ${out#*: } listed in round1_state/$step.sha256" ;;
  esac
  return 0
}

bitlist_step() {  # README job 2: one LOG.md line with the frozen bit list's SHA-256 (not a CPU job)
  local st rc src out
  hashcheck_step A A_bitlist
  st=$(lib status A_bitlist 2>&1); rc=$?
  if [ "$DRY" = 1 ]; then
    dry_show A_bitlist "$rc" "$st" "bash A/log_bitlist.sh (not a CPU job: no wrapper, no CPU gate)"
    return 0
  fi
  case "$rc" in
    0) log "skip A_bitlist: done ($st)"; return 0 ;;
    10) log "todo A_bitlist: $st" ;;
    *) fail "A_bitlist: $st" ;;
  esac
  out="$OUTS/$(date -u +%Y%m%dT%H%M%SZ)_A_bitlist.txt"
  log "start A_bitlist: bash A/log_bitlist.sh (one LOG.md line; output ${out#$CH/})"
  if [ "$TEST" = 1 ]; then
    A_S0_DIR="$T/A_out/s0" nice -n 19 bash "$TM/A/log_bitlist.sh" "$BITLIST_LOG" > "$out" 2>&1 < /dev/null
  else
    nice -n 19 bash "$TM/A/log_bitlist.sh" > "$out" 2>&1 < /dev/null
  fi
  rc=$?
  log "end A_bitlist: exit $rc"
  st=$(lib status A_bitlist 2>&1); src=$?
  [ "$rc" = 0 ] && [ "$src" = 0 ] || fail "A_bitlist: exit $rc, status $src ($st); see $out"
  log "done A_bitlist: $st"
  return 0
}

A_END="not started"
C0_END="not started"
C_END="not started"
declare -A ARM_END=([full]="not reached" [nf]="not reached")

a_stop() {  # <step> <reason>
  log "A STOP at $1: $2. A's line ends here; C step 6 follows (plan, section 6)."
  A_END="STOP at $1 ($2)"
}

a_go() {  # <rc> <step> [<what a STOP means here>]: 0 go on, 1 A's line ends
  local rc=$1 step=$2 why=${3:-}
  case "$rc" in
    0) return 0 ;;
    3) [ -n "$why" ] || fail "$step: a STOP from a step without a gate"
       a_stop "$step" "$why"; return 1 ;;
    6) a_stop "$step" "K7 time box reached (72 gate-open hours or 10 days since S0): stop and report"; return 1 ;;
    *) fail "$step: unexpected return $rc" ;;
  esac
}

k3c_arm() {  # <arm>
  local arm=$1 rc s want st
  if [ "$DRY" = 1 ]; then
    echo "  (A-$arm only if A/out/s2d_gates/gates.json has go_to_K3c[$arm] true)"
  else
    st=$(lib go-k3c "$arm" 2>&1); rc=$?
    case "$rc" in
      0) log "A-$arm: $st: its CPU TM pilots (K3c) run" ;;
      3) log "A-$arm: $st: no pilot (the arm stopped at the S2 gates; no GPU job)"
         ARM_END[$arm]="stopped at the S2 gates (K3, K8 or K9)"; return 0 ;;
      *) fail "A-$arm: cannot read go_to_K3c ($st)" ;;
    esac
  fi
  run_step "A_k3c_${arm}_s15" A A dl k3c_pilot.py --arm "$arm" --s 15; rc=$?
  a_k3c_go "$rc" "$arm" "A_k3c_${arm}_s15" || return 0
  run_step "A_k3c_${arm}_s40" A A dl k3c_pilot.py --arm "$arm" --s 40; rc=$?
  a_k3c_go "$rc" "$arm" "A_k3c_${arm}_s40" || return 0
  run_step "A_k3c_gate_${arm}" A A tm k3c_gate.py --arm "$arm"; rc=$?
  if [ "$DRY" = 1 ]; then
    echo "  (when the gate asks for it, exit 5:)"
    rc=5
  fi
  case "$rc" in
    0|3) k3c_end "$arm" "$rc"; return 0 ;;
    5) ;;
    6) ARM_END[$arm]="K7 time box reached before its K3c decision"; log "A-$arm: K7 time box reached: stop and report"; return 0 ;;
    *) fail "A_k3c_gate_${arm}: unexpected return $rc" ;;
  esac
  if [ "$DRY" = 1 ]; then
    s="<the better s>"
  else
    s=$(lib better-s "$arm" 2>&1) || fail "A_k3c_gate_${arm}: the better s is not known ($s)"
    want=$(grep -o -e "--arm $arm --s [0-9]* --shuffled" "$LAST_OUT" | tail -n 1 | awk '{print $4}')
    [ "$s" = "$want" ] || fail "A_k3c_gate_${arm}: the gate asks for s '$want', but the pilots' better s is '$s'"
  fi
  run_step "A_k3c_${arm}_shuffled" A A dl k3c_pilot.py --arm "$arm" --s "$s" --shuffled; rc=$?
  a_k3c_go "$rc" "$arm" "A_k3c_${arm}_shuffled" || return 0
  run_step "A_k3c_gate_${arm}" A A tm k3c_gate.py --arm "$arm"; rc=$?
  case "$rc" in
    0|3) [ "$DRY" = 1 ] || k3c_end "$arm" "$rc"; return 0 ;;
    5) fail "A_k3c_gate_${arm}: asks for the shuffled pilot again after it ran" ;;
    6) ARM_END[$arm]="K7 time box reached before its K3c decision"; log "A-$arm: K7 time box reached: stop and report"; return 0 ;;
    *) fail "A_k3c_gate_${arm}: unexpected return $rc" ;;
  esac
}

a_k3c_go() {  # <rc> <arm> <step>: 0 go on, 1 this arm ends
  case "$1" in
    0) return 0 ;;
    6) ARM_END[$2]="K7 time box reached before its K3c decision"; log "A-$2: K7 time box reached at $3: stop and report"; return 1 ;;
    *) fail "$3: unexpected return $1 from a pilot" ;;
  esac
}

k3c_end() {  # <arm> <rc of the decided gate>
  local st
  [ "$DRY" = 1 ] && return 0
  st=$(lib status "A_k3c_gate_$1" 2>&1)
  if [ "$2" = 0 ]; then
    ARM_END[$1]="K3c pass: may go to the GPU stage (plan step 7, scheduled by the GPU queue)"
    log "A-$1: K3c and K9 (TM) pass; the arm may go to the GPU stage, plan step 7 (not run by this chain): $st"
  else
    ARM_END[$1]="K3c STOP: no GPU job"
    log "A-$1: K3c STOP (or K9 for the TM failed): no GPU job: $st"
  fi
}

a_line() {
  local rc arm fam
  header "plan step 3: A S0, the bit-list LOG.md line, S1 (K1)"
  run_step A_s0 A A tm s0_data.py; rc=$?
  a_go "$rc" A_s0 || return 0
  bitlist_step
  run_step A_s1 A A tm s1_floors.py; rc=$?
  a_go "$rc" A_s1 "K1 STOP (the in-house KN-5 differs from KenLM by more than 1e-3: stop and fix)" || return 0
  header "plan step 4: A S2 for both arms (K9, K8, K3)"
  run_step A_s2a A A tm s2a_rows.py; rc=$?
  a_go "$rc" A_s2a || return 0
  run_step A_s2b_twin A A tm s2b_twin.py; rc=$?
  a_go "$rc" A_s2b_twin "K9 STOP for the count twin (shuffled labels): stop and find the leak" || return 0
  run_step A_s2c_aa A A tm s2c_aa.py; rc=$?
  a_go "$rc" A_s2c_aa "K8 STOP (equal-size A/A outside +-0.0036 or its interval misses 0): no learner runs" || return 0
  for arm in full nf; do
    for fam in tree lr1 lr2 hgb; do
      run_step "A_s2b_${arm}_${fam}" A A tm s2b_arm.py --arm "$arm" --family "$fam"; rc=$?
      a_go "$rc" "A_s2b_${arm}_${fam}" "K9 STOP for $arm/$fam (shuffled labels): stop and find the leak" || return 0
    done
  done
  run_step A_s2d_gates A A tm s2d_gates.py; rc=$?
  if [ "$rc" = 3 ]; then
    ARM_END[full]="stopped at the S2 gates"; ARM_END[nf]="stopped at the S2 gates"
  fi
  a_go "$rc" A_s2d_gates "both arms stop at the S2 gates (K3, K8 or K9): no pilot, no GPU job" || return 0
  header "plan step 5: A CPU TM pilots (K3c), for each arm with go_to_K3c"
  for arm in full nf; do
    k3c_arm "$arm"
  done
  A_END="A-full: ${ARM_END[full]}; A-nf: ${ARM_END[nf]}"
  return 0
}

c_go() {  # <rc> <step>: 0 go on, 1 C's line ends
  case "$1" in
    0) return 0 ;;
    6) log "C STOP at $2: C-K4 time box reached (no K0/K0b decision within 72 gate-open hours or by 2026-10-31T03:09:23Z): stop and report"
       C_END="STOP at $2 (C-K4 time box)"; return 1 ;;
    *) fail "$2: unexpected return $1" ;;
  esac
}

c_line() {
  local rc st
  header "plan step 6: C codes, twin, LR, LightGBM on valid (K0 sanity, K0b, A/A, shuffled labels)"
  run_step C_prep C C prep; rc=$?
  c_go "$rc" C_prep || return 0
  run_step C_twin C C twin; rc=$?
  c_go "$rc" C_twin || return 0
  run_step C_learners C C learners; rc=$?
  c_go "$rc" C_learners || return 0
  run_step C_gates C C gates; rc=$?
  [ "$DRY" = 1 ] && return 0
  st=$(lib status C_gates 2>&1)
  case "$rc" in
    0) C_END="PASS ($st)"; log "C: K0 and K0b pass; the GPU TM step (plan step 9) may be scheduled by the GPU queue (not run here): $st" ;;
    3) C_END="STOP ($st)"; log "C STOP at the CPU gates: $st" ;;
    *) c_go "$rc" C_gates ;;
  esac
  return 0
}

main() {
  local rc
  header "plan step 2: C Step 0 (frozen T0 outputs mixed with the floor; exploratory, no gate)"
  run_step C_step0 C C step0; rc=$?
  [ "$rc" = 0 ] || [ "$rc" = 6 ] || fail "C_step0: unexpected return $rc"
  if [ "$rc" = 6 ]; then C0_END="not run (C-K4 time box)"; else C0_END="done"; fi
  a_line
  if [ "$rc" = 6 ]; then
    log "C step 6 not run: C-K4 time box reached"; C_END="not run (C-K4 time box)"
  else
    c_line
  fi
}

if [ "$DRY" = 1 ]; then
  echo "run_round1.sh --dry at $(date -u +%FT%TZ): nothing runs and nothing is written"
  echo "chain files: $(sha256sum "$SELF" | cut -c1-16) run_round1.sh, $(sha256sum "$LIB" | cut -c1-16) round1_lib.py"
  for f in "$PY_TM" "$PY_DL" "$CPU_LOAD" "$WRAPPER"; do [ -e "$f" ] || { echo "missing: $f"; exit 2; }; done
  if bash "$TM/A/lock_held.sh" "$CHAIN_LOCK"; then echo "the chain lock is held now (another Part 3 job runs)"; else echo "the chain lock is free"; fi
  line=$(gate_line)
  if printf '%s\n' "$line" | bash "$TM/A/gate_check.sh"; then echo "CPU gate now: OPEN ($line)"; else echo "CPU gate now: closed ($line)"; fi
  main
  echo
  echo "dry run done: every hash check passed; the steps above are the chain's order"
  exit 0
fi

mkdir -p "$OUTS" "$CTMP/numba" "$CTMP/tmpdir" "$CTMP/joblib" "$CTMP/mpl" "$CTMP/xdg" "$(dirname "$SELF_LOCK")"
exec 5>>"$SELF_LOCK"
if ! flock -n 5; then
  log "refused: another run_round1.sh (or a job it started) holds ${SELF_LOCK#$TM/}"
  echo "refused: another run_round1.sh (or a job it started) holds $SELF_LOCK"
  exit 2
fi
trap 'log "chain stopped by a signal (pid $$); a running job finishes under the chain lock; a restart skips finished steps"; exit 143' TERM INT HUP
log "chain start: pid $$, mode $([ "$TEST" = 1 ] && echo "test ($T)" || echo real); run_round1.sh $(sha256sum "$SELF" | cut -c1-16), round1_lib.py $(sha256sum "$LIB" | cut -c1-16), wrapper $(sha256sum "$WRAPPER" | cut -c1-16)"
main
log "chain end: A: $A_END; C step 0: $C0_END; C step 6: $C_END"
exit 0
