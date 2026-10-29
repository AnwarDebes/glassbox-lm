#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# test_bg2_sample.sh OUTDIR   CPU dry test of bg2_sample.sh with FAKE runners (this test's own processes; no GPU job).
#   idle    : runner writes the patch line into a fresh log, its python child sleeps -> exit 0, A3 true, same pid
#   busy    : the same with a spinning python child                           -> exit 0, A3 false (main >= 0.8)
#   ended   : runner exits without the line                                     -> exit 1 ("ended before the line came")
#   stale   : a log WITH the line exists before the sampler starts, the runner never writes -> exit 1 ("never came"),
#             i.e. an earlier run's log does not fire the sampler
#   refused : OUT_PREFIX files of case idle exist                               -> exit 2
# Each fake runner gets its own session (setsid) and is killed by its group at the end. Writes only into OUTDIR.
# The check strings are single-quoted on purpose (check() evaluates them later).
# shellcheck disable=SC2016
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
out="${1:?usage: test_bg2_sample.sh OUTDIR}"
[ -e "$out" ] && { echo "refused: $out exists" >&2; exit 2; }
mkdir -p "$out"
res="$out/test_bg2_sample.txt"
ok=0; fail=0; runners=()
check() { local name="$1" cond="$2"; if eval "$cond"; then ok=$((ok + 1)); echo "PASS $name" | tee -a "$res"; else fail=$((fail + 1)); echo "FAIL $name" | tee -a "$res"; fi; }
fake_runner() {  # fake_runner ROOT CODE WRITE_LINE(yes|no): a bash runner with one python child, in its own session
  local root="$1" code="$2" write="$3" line=""
  [ "$write" = yes ] && line="echo 'speed_patch R1: make_tm -> FastCoTM(fake runner of test_bg2_sample.sh)' >> '$root/logs/B1_patch_R1_fake.log';"
  setsid bash -c "sleep 1; $line '$py' -I -B -c '$code' & wait" > /dev/null 2>&1 &
  echo $!
}
SLEEP_CODE='import time; time.sleep(40)'
BUSY_CODE='import time
t = time.time()
while time.time() - t < 40:
    pass'
export SETTLE=1 N=2 WINDOW=1 GAP=1 POLL=1 WAIT_MAX=20
echo "$(date -u +%FT%TZ) start" >> "$res"
for case in idle busy; do
  root="$out/root_$case"; mkdir -p "$root/logs"
  code="$SLEEP_CODE"; [ "$case" = busy ] && code="$BUSY_CODE"
  r=$(fake_runner "$root" "$code" yes); runners+=("$r")
  bash "$here/bg2_sample.sh" "$root" "$r" "$out/$case" 'B1_patch_R1_*.log' 'speed_patch R1: make_tm'
  rc=$?
  a3=$("$py" -I -B -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['a3_main_thread_le_0.2'], d['same_pid_in_all_samples'], d['max_main_thread_cores'])" "$out/$case.sample.json" 2> /dev/null)
  echo "$case: exit $rc; a3 / same pid / max main: $a3" >> "$res"
  if [ "$case" = idle ]; then
    check "idle: exit 0, A3 true, one pid" '[ $rc -eq 0 ] && [[ "$a3" == "True True "* ]]'
  else
    check "busy: exit 0, A3 false (spinning main thread)" '[ $rc -eq 0 ] && [[ "$a3" == "False True "* ]]'
  fi
  kill -KILL -- "-$r" 2> /dev/null
done
root="$out/root_ended"; mkdir -p "$root/logs"
setsid bash -c "sleep 1" > /dev/null 2>&1 &
r=$!; runners+=("$r")
bash "$here/bg2_sample.sh" "$root" "$r" "$out/ended" 'B1_patch_R1_*.log' 'speed_patch R1: make_tm'
rc=$?
check "ended: exit 1, runner ended before the line" '[ $rc -eq 1 ] && grep -q "ended before the line came" "$out/ended.sample.log"'
root="$out/root_stale"; mkdir -p "$root/logs"
echo 'speed_patch R1: make_tm -> FastCoTM(stale log of an earlier run)' > "$root/logs/B1_patch_R1_old.log"
setsid bash -c "sleep 30" > /dev/null 2>&1 &
r=$!; runners+=("$r")
WAIT_MAX=3 bash "$here/bg2_sample.sh" "$root" "$r" "$out/stale" 'B1_patch_R1_*.log' 'speed_patch R1: make_tm'
rc=$?
check "stale: an earlier run's log does not fire (exit 1, never came)" '[ $rc -eq 1 ] && grep -q "never came" "$out/stale.sample.log"'
kill -KILL -- "-$r" 2> /dev/null
bash "$here/bg2_sample.sh" "$out/root_idle" 1 "$out/idle" 'B1_patch_R1_*.log' 'x' > /dev/null 2>&1
rc=$?
check "refused: existing OUT_PREFIX files (exit 2)" '[ $rc -eq 2 ]'
left=0
for r in "${runners[@]}"; do [ -d "/proc/$r" ] && left=$((left + 1)); done
check "no fake runner left" '[ $left -eq 0 ]'
echo "$(date -u +%FT%TZ) $ok passed, $fail failed" | tee -a "$res"
[ "$fail" -eq 0 ]
