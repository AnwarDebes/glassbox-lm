#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# One A-Struct CPU job under the Part 3 run rules (PLAN_part3.md section 6; task rules 1, 4, 5).
#   chain/part3_cpu_job.sh A tm|dl <script.py> [args...]     (how every real job starts; amendment 1)
#   run_job.sh tm|dl <script.py> [args...]                   (what the chain wrapper calls)
# - one Part 3 CPU job at a time across A and C (amendment 1): a real job (no self-test hook) refuses to start unless
#   it runs under the Part 3 chain lock, i.e. PART3_CHAIN_LOCK names glassbox-lm/studies/language_part3/chain/part3_cpu.lock and
#   that lock is held (lock_held.sh); chain/part3_cpu_job.sh takes it for every A and C job. A's own jobs also take
#   A's lock $PART3_CPU_LOCK (default A/tmp/cpu_job.lock), which must never be the chain lock (a second open of the
#   chain lock here would wait for the wrapper forever);
# - CPU gate: decision_layer scripts/cpu_load.py 10 must show at most 4.0 cores in use and 0 throttled periods;
#   while it does not, check again every 2 minutes; every check is appended to A/out/jobs.log (gate-open hours, K7);
# - 2 threads everywhere, caches and temporary files in A/tmp/, no GPU (env.sh), nice -n 19, python -I -B;
#   A_RUNNER=1 marks a real job (the scripts refuse real outputs without it), A_RUNNER=test the self-test mode;
# - the imported folders are listed before and after the job and compared (listing.py); a change stops the work.
# Exit code: the script's (3 = a registered STOP; 5 = k3c_gate.py asks for the shuffled pilot), 4 when the listings
# differ, 2 for a usage error or a refused start (nothing written then).
set -u
source ${REPO_ROOT}/studies/language_part3/A/env.sh
if [ $# -lt 2 ]; then echo "usage: run_job.sh tm|dl <script.py> [args...]"; exit 2; fi
venv=$1; shift; script=$1; shift
case "$venv" in tm) PY=$PY_TM ;; dl) PY=$PY_DL ;; *) echo "first argument: tm or dl"; exit 2 ;; esac
CHAIN_LOCK=${REPO_ROOT}/studies/language_part3/chain/part3_cpu.lock
LOCK=${PART3_CPU_LOCK:-$A/tmp/cpu_job.lock}
JOBLOG=${A_JOBLOG:-$A/out/jobs.log}   # self-tests point this elsewhere
CPU_LOAD=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
if [ "$(realpath -m "$LOCK")" = "$(realpath -m "$CHAIN_LOCK")" ]; then
  echo "PART3_CPU_LOCK must not be the chain lock $CHAIN_LOCK (the wrapper already holds it)"; exit 2
fi
# self-tests only: A_TEST_GATE_LINES names a file of cpu_load.py-style lines, one used per check (and A_TEST_GATE_SLEEP
# the wait between checks); every such check is logged as TEST, and real jobs never set them
gate_line() {
  if [ -n "${A_TEST_GATE_LINES:-}" ]; then
    head -n 1 "$A_TEST_GATE_LINES"; sed -i '1d' "$A_TEST_GATE_LINES"
  else
    "$PY_DL" -I -B "$CPU_LOAD" 10
  fi
}
if [ -n "${A_TEST_GATE_LINES:-}" ] && [ "$JOBLOG" = "$A/out/jobs.log" ]; then
  echo "A_TEST_GATE_LINES is for self-tests only (set A_JOBLOG to a test log)"; exit 2
fi
if [ -z "${A_TEST_GATE_LINES:-}" ]; then
  # a real job: only under the Part 3 chain lock (checked before anything is written)
  if [ "${PART3_CHAIN_LOCK:-}" != "$CHAIN_LOCK" ]; then
    echo "real A jobs start only through chain/part3_cpu_job.sh A ... (PART3_CHAIN_LOCK is '${PART3_CHAIN_LOCK:-}', not $CHAIN_LOCK)"; exit 2
  fi
  if ! bash "$A/lock_held.sh" "$CHAIN_LOCK"; then
    echo "the Part 3 chain lock $CHAIN_LOCK is not held: start real A jobs through chain/part3_cpu_job.sh"; exit 2
  fi
  export A_RUNNER=1
else
  export A_RUNNER=test
fi
SLEEP=110
[ -n "${A_TEST_GATE_LINES:-}" ] && SLEEP=${A_TEST_GATE_SLEEP:-1}
src=""; [ -n "${A_TEST_GATE_LINES:-}" ] && src=" TEST"
mkdir -p "$A/out"
tag="$(basename "$script" .py) $*"
exec 9>"$LOCK"
echo "$(date -u +%FT%TZ) wait-lock $tag" >> "$JOBLOG"
flock 9
while true; do
  line=$(gate_line)
  if echo "$line" | bash "$A/gate_check.sh"; then
    echo "$(date -u +%FT%TZ) gate OPEN$src ($line) $tag" >> "$JOBLOG"
    break
  fi
  echo "$(date -u +%FT%TZ) gate closed$src ($line) $tag" >> "$JOBLOG"
  sleep "$SLEEP"
done
stamp=$(date -u +%Y%m%dT%H%M%SZ)
before="$A/tmp/listing_${stamp}_before.txt"
after="$A/tmp/listing_${stamp}_after.txt"
"$PY_TM" -I -B "$A/listing.py" snapshot "$before" || exit 2
echo "$(date -u +%FT%TZ) start $tag" >> "$JOBLOG"
nice -n 19 "$PY" -I -B "$A/$script" "$@"
rc=$?
"$PY_TM" -I -B "$A/listing.py" snapshot "$after" || exit 2
cmp=$("$PY_TM" -I -B "$A/listing.py" compare "$before" "$after"); lrc=$?
echo "$(date -u +%FT%TZ) end $tag exit $rc; $(echo "$cmp" | tr '\n' ' ')" >> "$JOBLOG"
echo "$cmp"
if [ $lrc -ne 0 ]; then exit 4; fi
exit $rc
