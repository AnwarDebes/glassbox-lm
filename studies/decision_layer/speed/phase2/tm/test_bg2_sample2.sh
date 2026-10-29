#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# test_bg2_sample2.sh OUTDIR   CPU dry test of bg2_sample2.sh with FAKE runners (this test's own processes; no GPU job).
#   idle        : runner writes the patch line into a fresh log, its one python child sleeps
#                 -> exit 0; A3 true by both tools; the pgrep child = the tree-walk pid; niceness 19 logged
#   busy        : the same with a spinning python child -> exit 0; A3 false; both tools >= 0.8 cores; |difference| <= 0.1
#   ended       : runner exits without the line                     -> exit 1 ("ended before the line came")
#   stale       : a log WITH the line exists before the sampler starts, the runner never writes -> exit 1 ("never came")
#   two_pythons : the runner has two python children                -> exit 1 ("not exactly one")
#   not_python  : the runner's only child is sleep                  -> exit 1 ("not a python")
#   refused     : OUT_PREFIX files of case idle exist               -> exit 2
#   tool_pin    : PROTO_TOOL points at a changed copy of the protocol tool -> exit 2, nothing sampled
#   no fake runner is left at the end.
# Each fake runner gets its own session (setsid) and is killed by its group at the end. Writes only into OUTDIR.
# The check strings are single-quoted on purpose (check() evaluates them later).
# shellcheck disable=SC2016
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
out="${1:?usage: test_bg2_sample2.sh OUTDIR}"
[ -e "$out" ] && { echo "refused: $out exists" >&2; exit 2; }
mkdir -p "$out"
res="$out/test_bg2_sample2.txt"
ok=0; fail=0; runners=()
check() { local name="$1" cond="$2"; if eval "$cond"; then ok=$((ok + 1)); echo "PASS $name" | tee -a "$res"; else fail=$((fail + 1)); echo "FAIL $name" | tee -a "$res"; fi; }
field() { "$py" -I -B -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2" 2> /dev/null; }
LOGNAME_OK=B1_patch_R1_20261021T000000Z.log
fake_runner() {  # fake_runner ROOT CHILDREN(one|two|sleep) CODE: a bash runner in its own session; writes the line
  local root="$1" kids="$2" code="$3" line cmd
  line="echo 'speed_patch R1: make_tm -> FastCoTM(fake runner of test_bg2_sample2.sh)' >> '$root/logs/$LOGNAME_OK';"
  case "$kids" in
    one) cmd="'$py' -I -B -c '$code' &" ;;
    two) cmd="'$py' -I -B -c '$code' & '$py' -I -B -c '$code' &" ;;
    sleep) cmd="sleep 40 &" ;;
  esac
  setsid bash -c "sleep 1; $line $cmd wait" > /dev/null 2>&1 &
  echo $!
}
SLEEP_CODE='import time; time.sleep(40)'
BUSY_CODE='import time
t = time.time()
while time.time() - t < 40:
    pass'
export SETTLE=1 N=2 WINDOW=2 GAP=1 POLL=1 WAIT_MAX=20
echo "$(date -u +%FT%TZ) start" >> "$res"
for case in idle busy; do
  root="$out/root_$case"; mkdir -p "$root/logs"
  code="$SLEEP_CODE"; [ "$case" = busy ] && code="$BUSY_CODE"
  r=$(fake_runner "$root" one "$code"); runners+=("$r")
  bash "$here/bg2_sample2.sh" "$root" "$r" "$out/$case" 'B1_patch_R1_2*.log' 'speed_patch R1: make_tm'
  rc=$?
  j="$out/$case.sample.json"
  a3=$(field "$j" "(d['a3_main_thread_le_0.2'], d['a3_protocol_tool_le_0.2'], d['a3_per_thread_tool_le_0.2'])")
  agree=$(field "$j" "(d['pids_agree'], d['same_pid_in_all_samples'], d['protocol_lines_parsed'])")
  mins=$(field "$j" "min(min(r['protocol_tool']['main_thread_cores'], r['per_thread_tool']['main_thread_cores']) for r in d['samples'])")
  diff=$(field "$j" "d['max_abs_difference']")
  echo "$case: exit $rc; a3 (both, protocol, per-thread) $a3; pids agree / same pid / parsed $agree; min main $mins; max |diff| $diff" >> "$res"
  if [ "$case" = idle ]; then
    check "idle: exit 0, A3 true by both tools, pids agree" '[ $rc -eq 0 ] && [ "$a3" = "(True, True, True)" ] && [ "$agree" = "(True, True, True)" ]'
    check "idle: niceness 19 logged for the tools" 'grep -q "niceness of such a child: 19)" "$out/idle.sample.log"'
    check "idle: both tool outputs exist" '[ -s "$out/idle.proto_1.txt" ] && [ -s "$out/idle.spin_1.json" ] && [ -s "$out/idle.proto_2.txt" ]'
  else
    check "busy: exit 0, A3 false by both tools, both >= 0.8 cores, |difference| <= 0.1" \
      '[ $rc -eq 0 ] && [ "$a3" = "(False, False, False)" ] && [ "$agree" = "(True, True, True)" ] && awk -v m="$mins" -v d="$diff" "BEGIN { exit !(m >= 0.8 && d <= 0.1) }"'
  fi
  kill -KILL -- "-$r" 2> /dev/null
done
root="$out/root_ended"; mkdir -p "$root/logs"
setsid bash -c "sleep 1" > /dev/null 2>&1 &
r=$!; runners+=("$r")
bash "$here/bg2_sample2.sh" "$root" "$r" "$out/ended" 'B1_patch_R1_2*.log' 'speed_patch R1: make_tm'
rc=$?
check "ended: exit 1, runner ended before the line" '[ $rc -eq 1 ] && grep -q "ended before the line came" "$out/ended.sample.log"'
root="$out/root_stale"; mkdir -p "$root/logs"
echo 'speed_patch R1: make_tm -> FastCoTM(stale log of an earlier run)' > "$root/logs/B1_patch_R1_20261020T000000Z.log"
setsid bash -c "sleep 30" > /dev/null 2>&1 &
r=$!; runners+=("$r")
WAIT_MAX=3 bash "$here/bg2_sample2.sh" "$root" "$r" "$out/stale" 'B1_patch_R1_2*.log' 'speed_patch R1: make_tm'
rc=$?
check "stale: an earlier run's log does not fire (exit 1, never came)" '[ $rc -eq 1 ] && grep -q "never came" "$out/stale.sample.log"'
kill -KILL -- "-$r" 2> /dev/null
for case in two_pythons not_python; do
  root="$out/root_$case"; mkdir -p "$root/logs"
  kids="two"; [ "$case" = not_python ] && kids="sleep"
  r=$(fake_runner "$root" "$kids" "$SLEEP_CODE"); runners+=("$r")
  bash "$here/bg2_sample2.sh" "$root" "$r" "$out/$case" 'B1_patch_R1_2*.log' 'speed_patch R1: make_tm'
  rc=$?
  if [ "$case" = two_pythons ]; then
    check "two pythons: exit 1, not exactly one child" '[ $rc -eq 1 ] && grep -q "not exactly one" "$out/two_pythons.sample.log"'
  else
    check "not a python: exit 1" '[ $rc -eq 1 ] && grep -q "not a python" "$out/not_python.sample.log"'
  fi
  kill -KILL -- "-$r" 2> /dev/null
done
bash "$here/bg2_sample2.sh" "$out/root_idle" 1 "$out/idle" 'B1_patch_R1_2*.log' 'x' > /dev/null 2>&1
rc=$?
check "refused: existing OUT_PREFIX files (exit 2)" '[ $rc -eq 2 ]'
cp ${REPO_ROOT}/studies/decision_layer/speed/tmcode/spin_check.py "$out/changed_protocol_tool.py"
echo "# planted change" >> "$out/changed_protocol_tool.py"
PROTO_TOOL="$out/changed_protocol_tool.py" bash "$here/bg2_sample2.sh" "$out/root_idle" 1 "$out/pin" 'B1_patch_R1_2*.log' 'x' 2> "$out/pin.err"
rc=$?
check "tool pin: a changed protocol tool is refused (exit 2, nothing sampled)" '[ $rc -eq 2 ] && grep -q "not the pinned protocol tool" "$out/pin.err" && [ ! -e "$out/pin.sample.start" ]'
left=0
for r in "${runners[@]}"; do [ -d "/proc/$r" ] && left=$((left + 1)); done
check "no fake runner left" '[ $left -eq 0 ]'
echo "$(date -u +%FT%TZ) $ok passed, $fail failed" | tee -a "$res"
[ "$fail" -eq 0 ]
