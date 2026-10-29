#!/usr/bin/env bash
# Dry test of e86_chain.sh with stubs (no data, no GPU): the handoff waits for the process and for 3 empty-GPU checks
# in a row; dry, run and both write their end lines; the guards refuse.
set -u; H=$(cd "$(dirname "$0")" && pwd)
new() {
  T=$(mktemp -d); mkdir -p "$T/logs"; echo "plan stub" > "$T/logs/plan_e86_clause_drop.md"
  printf '{"sha256": "%s", "frozen_utc": "2026-10-23T01:00:00Z"}\n' "$(sha256sum "$T/logs/plan_e86_clause_drop.md" | cut -d' ' -f1)" > "$T/logs/plan_e86_clause_drop.frozen.json"
  printf 'echo "$1 $2" >> "%s/queue_calls"; touch "$2"\n' "$T" > "$T/q.sh"
  printf '[ "${1:-}" = --dry ] && { [ -e "%s/dryfail" ] && exit 1; echo "ok x"; echo "all dry checks passed"; exit 0; }; echo "E86.1 stub"\n' "$T" > "$T/c.sh"
  printf '[ -e "%s/alive" ]\n' "$T" > "$T/ps.sh"
  printf 'n=$(( $(cat "%s/gpu_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/gpu_calls"; case " ${BUSY:-} " in *" $n "*) echo 4242;; esac; true\n' "$T" "$T" > "$T/nv.sh"
}
run() { E86_ROOT="$T" QUEUE_CMD="bash $T/q.sh" CHECK_CMD="bash $T/c.sh" PS_CMD="bash $T/ps.sh" NVSMI_CMD="bash $T/nv.sh" WAIT_PID=999999 POLL_S=1 \
        bash "$H/e86_chain.sh" "$@" > "$T/out.txt" 2>&1; echo $?; }
fail() { echo "FAIL $1"; cat "$T/out.txt" 2>/dev/null; exit 1; }
# both: waits for the process, then needs 3 empty checks in a row (GPU busy on checks 1 and 3)
new; touch "$T/alive"
( E86_ROOT="$T" QUEUE_CMD="bash $T/q.sh" CHECK_CMD="bash $T/c.sh" PS_CMD="bash $T/ps.sh" NVSMI_CMD="bash $T/nv.sh" WAIT_PID=999999 POLL_S=1 BUSY="1 3" \
  bash "$H/e86_chain.sh" both > "$T/out.txt" 2>&1; echo $? > "$T/rc" ) &
sleep 3; [ ! -e "$T/queue_calls" ] && [ ! -e "$T/gpu_calls" ] || fail "did not wait for the process"
rm "$T/alive"; wait
[ "$(cat "$T/rc")" = 0 ] && [ "$(cat "$T/gpu_calls")" = 6 ] || fail "handoff (rc $(cat "$T/rc"), gpu checks $(cat "$T/gpu_calls" 2>/dev/null))"
[ "$(cut -d' ' -f1 "$T/queue_calls" | tr '\n' ' ')" = "logs/jobs_e86_dry.txt logs/jobs_e86.txt " ] || fail "queue order"
grep -q "^e86 dry end: done .* exit 0$" "$T/logs/e86_dry_end.log" && grep -q "^e86 end: done .* exit 0$" "$T/logs/e86_end.log" \
  && grep -q "E86.1 stub" "$T/logs/e86_check.out" && [ "$(tail -n 1 "$T/logs/e86_dry_check.out")" = "all dry checks passed" ] || fail "both: end lines"
[ "$(run both)" = 1 ] || fail "second start not refused"
rm -rf "$T"
# dry, then run
new; [ "$(run dry)" = 0 ] && grep -q "^e86 dry end: done .* exit 0$" "$T/logs/e86_dry_end.log" && [ ! -e "$T/logs/e86_end.log" ] || fail "dry stage"
[ "$(run run)" = 0 ] && grep -q "^e86 end: done .* exit 0$" "$T/logs/e86_end.log" || fail "run stage"
rm -rf "$T"
# guards
new; [ "$(run run)" = 1 ] && grep -q "^e86 end: refused (the dry checks have not passed) .* exit 1$" "$T/logs/e86_end.log" && [ ! -e "$T/queue_calls" ] || fail "run without dry"
rm -rf "$T"
new; touch "$T/dryfail"; [ "$(run both)" = 1 ] && grep -q "^e86 dry end: dry check failed .* exit 1$" "$T/logs/e86_dry_end.log" && [ ! -e "$T/logs/e86_queue.log" ] || fail "failing dry check"
rm -rf "$T"
new; echo changed >> "$T/logs/plan_e86_clause_drop.md"
[ "$(run both)" = 1 ] && grep -q "^e86 dry end: refused (no freeze record of the plan's text) .* exit 1$" "$T/logs/e86_dry_end.log" && [ ! -e "$T/queue_calls" ] || fail "freeze guard"
rm -rf "$T"
new; [ "$(run go)" = 1 ] || fail "bad stage"; rm -rf "$T"
echo "ok: waits for the process, then for 3 empty-GPU checks in a row (busy on checks 1 and 3: 6 checks); both runs dry then the 14 jobs;"
echo "    dry and run as separate stages; refusals: a second start, run before the dry checks passed, a failing dry check stops before the jobs,"
echo "    a changed plan, an unknown stage"
echo PASSED
