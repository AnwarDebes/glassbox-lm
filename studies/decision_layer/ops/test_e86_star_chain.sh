#!/usr/bin/env bash
# Dry test of e86_star_chain.sh with stubs (no data, no GPU): it waits for 3 empty-GPU checks in a row, runs the
# STAR job file and then the check, writes its end line, and refuses a missing or stale freeze record, a second
# start and a failing check.
set -u; H=$(cd "$(dirname "$0")" && pwd)
new() {
  T=$(mktemp -d); mkdir -p "$T/logs"; echo "note stub" > "$T/logs/plan_e86_note_2026-10-25.md"
  printf '{"sha256": "%s", "frozen_utc": "2026-10-25T11:00:00Z"}\n' "$(sha256sum "$T/logs/plan_e86_note_2026-10-25.md" | cut -d' ' -f1)" > "$T/logs/plan_e86_note_2026-10-25.frozen.json"
  printf 'echo "$1 $2 $3" >> "%s/queue_calls"; touch "$2"\n' "$T" > "$T/q.sh"
  printf '[ -e "%s/checkfail" ] && { echo "settings differ" >&2; exit 1; }; echo "E86.1 stub"\n' "$T" > "$T/c.sh"
  printf 'n=$(( $(cat "%s/gpu_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/gpu_calls"; case " ${BUSY:-} " in *" $n "*) echo 4242;; esac; true\n' "$T" "$T" > "$T/nv.sh"
}
run() { E86_ROOT="$T" QUEUE_CMD="bash $T/q.sh" CHECK_CMD="bash $T/c.sh" NVSMI_CMD="bash $T/nv.sh" POLL_S=0 BUSY="${BUSY:-}" \
        bash "$H/e86_star_chain.sh" > "$T/out.txt" 2>&1; echo $?; }
fail() { echo "FAIL $1"; cat "$T/out.txt" 2>/dev/null; exit 1; }
# normal: GPU busy on checks 1 and 3, so 3 empty checks in a row take 6 checks; then the queue, then the check
new; [ "$(BUSY="1 3" run)" = 0 ] || fail "normal run"
[ "$(cat "$T/gpu_calls")" = 6 ] || fail "handoff: $(cat "$T/gpu_calls") GPU checks, expected 6"
[ "$(cat "$T/queue_calls")" = "logs/jobs_e86_star.txt logs/e86_star_queue.log 4.5" ] || fail "queue call"
grep -q "^e86 star end: done .* exit 0$" "$T/logs/e86_star_end.log" && grep -q "E86.1 stub" "$T/logs/e86_check.out" || fail "end line or check output"
[ "$(run)" = 1 ] && [ "$(wc -l < "$T/logs/e86_star_end.log")" = 1 ] || fail "second start not refused"
rm -rf "$T"
# idle GPU: exactly 3 checks
new; [ "$(run)" = 0 ] && [ "$(cat "$T/gpu_calls")" = 3 ] || fail "idle GPU: $(cat "$T/gpu_calls" 2>/dev/null) checks"
rm -rf "$T"
# a stale freeze record (the note changed after the freeze) and a missing one
new; echo changed >> "$T/logs/plan_e86_note_2026-10-25.md"
[ "$(run)" = 1 ] && grep -q "^e86 star end: refused (no freeze record of the note's text) .* exit 1$" "$T/logs/e86_star_end.log" && [ ! -e "$T/queue_calls" ] || fail "stale record"
rm -rf "$T"
new; rm "$T/logs/plan_e86_note_2026-10-25.frozen.json"
[ "$(run)" = 1 ] && grep -q "refused (no freeze record" "$T/logs/e86_star_end.log" && [ ! -e "$T/gpu_calls" ] || fail "missing record"
rm -rf "$T"
# a failing check
new; touch "$T/checkfail"
[ "$(run)" = 1 ] && grep -q "^e86 star end: check failed .* exit 1$" "$T/logs/e86_star_end.log" && grep -q "settings differ" "$T/logs/e86_check.out" || fail "failing check"
rm -rf "$T"
# an existing queue log from an earlier start
new; touch "$T/logs/e86_star_queue.log"; [ "$(run)" = 1 ] && [ ! -e "$T/logs/e86_star_end.log" ] && [ ! -e "$T/gpu_calls" ] || fail "existing queue log"
rm -rf "$T"
echo "ok: waits for 3 empty-GPU checks in a row (busy on 1 and 3: 6 checks; idle: 3), runs the STAR job file then the check,"
echo "    writes the end line; refusals: a second start, a stale or missing freeze record, a failing check, an existing queue log"
echo PASSED
