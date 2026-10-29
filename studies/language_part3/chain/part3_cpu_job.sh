#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# One Part 3 CPU job of A-Struct or C-core under the Part 3 chain lock (PLAN_part3.md section 6: one Part 3 CPU job
# at a time, 2 threads, nice 19; code review of 2026-10-21, issues 2 and 6).
#   chain/part3_cpu_job.sh A tm|dl <script.py> [args...]   runs  A/run_job.sh tm|dl <script.py> [args...]
#   chain/part3_cpu_job.sh C <stage> [args...]             runs  C/scripts/run_stage.sh <stage> [args...]
# - Takes an exclusive flock on chain/part3_cpu.lock, waiting while another Part 3 job holds it, and keeps it until the
#   arm's runner ends. So an A job and a C job never run together, and each runner checks the CPU gate only while it
#   holds the lock. Exports PART3_CHAIN_LOCK; A's runner refuses real jobs unless this lock is named and held.
# - Never wrap this script, or an arm's runner, in another flock on chain/part3_cpu.lock: the inner open would wait
#   for the outer one forever. A nested call of this script is refused.
# - Every wait, start, note and end goes to chain/part3_cpu_jobs.log (times from date -u in the writing command); the
#   runner's own output goes to chain/logs/<UTC>_<arm>_<name>.log, and its last 40 lines are printed at the end.
# Exit code, with the same meaning for both arms:
#   0   done: go on
#   3   a registered STOP. A: the script's own exit 3 (K1, K8, K9, K3, K3c). C: the real gates stage ran and
#       C/data/step6_decision.json "decision" starts with "STOP" (C's stages exit 0 on a gate STOP)
#   4   the imported folders changed during the job: stop and report
#   5   A's k3c_gate.py asks for the shuffled pilot first (it prints the command)
#   75  not run: C's own lock (C/tmp/c_job.lock) was busy, e.g. a C self-test; try again later
#   2   usage error or refused start; anything else: the job failed (read its log)
# C's runner exits 4 also when only the ".." (parent folder) lines of its listing changed, which other sessions cause
# by writing next to an imported folder; such a change is logged as a note and counts as 0 (A's listing.py already
# ignores ".." lines). C's runner itself is unchanged (its code is frozen in R1 [C-core]).
set -u
TM=${REPO_ROOT}/studies/language_part3
CH=$TM/chain
LOCK=$CH/part3_cpu.lock
LOG=$CH/part3_cpu_jobs.log
LOGS=$CH/logs
C_RUNNER=$TM/C/scripts/run_stage.sh
C_DECISION=$TM/C/data/step6_decision.json
PY=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
if [ -n "${PART3_CHAIN_TEST_DIR:-}" ]; then
  # self-tests only (chain/tests/): their own lock, log and logs folder; arm T runs any command; a stand-in C runner
  LOCK=$PART3_CHAIN_TEST_DIR/part3_cpu.lock; LOG=$PART3_CHAIN_TEST_DIR/part3_cpu_jobs.log; LOGS=$PART3_CHAIN_TEST_DIR/logs
  C_RUNNER=${PART3_CHAIN_TEST_C_RUNNER:-$C_RUNNER}; C_DECISION=${PART3_CHAIN_TEST_C_DECISION:-$C_DECISION}
fi
usage() { echo "usage: part3_cpu_job.sh A tm|dl <script.py> [args...]  |  part3_cpu_job.sh C <stage> [args...]"; exit 2; }
[ $# -ge 2 ] || usage
arm=$1; shift
case "$arm" in
  A) [ $# -ge 2 ] || usage; cmd=(bash "$TM/A/run_job.sh" "$@"); name="A_$(basename "$2" .py)" ;;
  C) cmd=(bash "$C_RUNNER" "$@"); name="C_$1" ;;
  T) [ -n "${PART3_CHAIN_TEST_DIR:-}" ] || { echo "arm T is for the self-tests only"; exit 2; }; cmd=("$@"); name="T" ;;
  *) usage ;;
esac
if [ -n "${PART3_CPU_LOCK:-}" ] && [ "$(realpath -m "$PART3_CPU_LOCK")" = "$(realpath -m "$LOCK")" ]; then
  echo "PART3_CPU_LOCK must not be the chain lock $LOCK (A's runner would wait for this wrapper forever)"; exit 2
fi
if [ -n "${PART3_CHAIN_LOCK:-}" ]; then
  echo "nested call refused: this process already runs under the Part 3 chain lock ($PART3_CHAIN_LOCK)"; exit 2
fi
mkdir -p "$LOGS"
desc="$arm $*"
exec 7>>"$LOCK"
echo "$(date -u +%FT%TZ) wait $desc" >> "$LOG"
flock 7
t0=$(date +%s)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$LOGS/${stamp}_${name}.log"
echo "$(date -u +%FT%TZ) start $desc (output: ${out#$TM/})" >> "$LOG"
export PART3_CHAIN_LOCK="$LOCK"
"${cmd[@]}" > "$out" 2>&1
rc=$?
code=$rc
status="exit $rc"
if [ "$arm" = "C" ]; then
  stage=$1
  dry=0
  for a in "$@"; do [ "$a" = "--dry" ] && dry=1; done
  if [ "$rc" = "3" ]; then
    if grep -q "another C job holds" "$out"; then code=75; status="not run: C's own lock was busy"
    else code=1; status="unexpected exit 3 from the C runner"; fi
  fi
  if [ "$rc" = "4" ]; then
    before=$(grep -o "listing before/after: CHANGED ([^)]*_before.txt)" "$out" | tail -n 1 | sed 's/.*(\(.*\))/\1/')
    cls=none
    [ -n "$before" ] && cls=$("$PY" -I -B "$CH/part3_status.py" diff-class "${before%_before.txt}_diff.txt")
    if [ "$cls" = "parent-only" ]; then
      code=0; status="exit 4 from a change of '..' lines only (parent folders; a note, not a stop)"
      echo "$(date -u +%FT%TZ) note $desc: only '..' (parent folder) lines changed in ${before%_before.txt}_diff.txt" >> "$LOG"
    else
      status="the imported folders changed ($cls): stop and report"
    fi
  fi
  if [ "$code" = "0" ] && [ "$stage" = "gates" ] && [ "$dry" = "0" ]; then
    dec=$("$PY" -I -B "$CH/part3_status.py" c-decision "$C_DECISION" "$t0"); drc=$?
    case "$drc" in
      0) status="$status; C gates: $dec" ;;
      3) code=3; status="C gates STOP: $dec" ;;
      *) code=1; status="C gates: no fresh decision file ($dec)" ;;
    esac
  fi
fi
echo "$(date -u +%FT%TZ) end $desc: runner exit $rc -> $code ($status)" >> "$LOG"
tail -n 40 "$out"
echo "part3_cpu_job: $desc -> exit $code ($status); output ${out}"
exit "$code"
