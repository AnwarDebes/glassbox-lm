#!/usr/bin/env bash
# Dry test of e84_wait_start.sh with fakes: an OLD end line present before the start must not fire; a new one fires;
# a busy GPU holds the start until it is free; the plan's commands are stubs; the end line is written.
set -u
H=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); mkdir -p "$T/logs"
printf 'r3c chain end: done 2026-10-21T00:00:00Z exit 0\n' > "$T/r3c.log"          # a stale end line
echo 2 > "$T/busy"                                                                     # GPU busy for 2 polls
cat > "$T/nvsmi.sh" <<'X'
n=$(cat "$1/busy"); if [ "$n" -gt 0 ]; then echo 4242; echo $((n - 1)) > "$1/busy"; fi
X
E84_ROOT="$T" R3C_LOG="$T/r3c.log" NVSMI_CMD="bash $T/nvsmi.sh $T" QUEUE_CMD="touch $T/queue_ran" CHECK_CMD="echo 'G-ref ... MET, steady'" \
  POLL_S=1 MAX_WAIT_S=20 bash "$H/e84_wait_start.sh" > "$T/out.txt" 2>&1 &
pid=$!
sleep 3
[ ! -e "$T/queue_ran" ] || { echo "FAIL: fired on the stale end line"; exit 1; }
printf 'r3c chain end: done 2026-10-22T12:00:00Z exit 0\n' >> "$T/r3c.log"             # the real end line
wait $pid; rc=$?
[ "$rc" -eq 0 ] && [ -e "$T/queue_ran" ] && [ "$(cat "$T/busy")" -eq 0 ] || { echo "FAIL rc $rc"; cat "$T/out.txt"; exit 1; }
grep -q "^e84 end: done .* exit 0$" "$T/logs/e84_end.log" && grep -q "MET, steady" "$T/logs/e84_check.out" || { echo "FAIL end line"; exit 1; }
E84_ROOT="$T" R3C_LOG="$T/r3c.log" bash "$H/e84_wait_start.sh" > /dev/null 2>&1; [ $? -eq 1 ] || { echo "FAIL: no refusal"; exit 1; }
echo "ok: stale end line ignored; new end line fired; busy GPU waited (2 polls); stubs ran; end line written; refuses a second start"
echo PASSED; rm -rf "$T"
