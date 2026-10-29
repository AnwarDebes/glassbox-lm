#!/usr/bin/env bash
# Dry test of e87_a_chain.sh with stubs (no data, no GPU): a0 waits for every pid and for 3 empty-GPU checks in a row,
# runs the three datasets in order and exits; a waits for A0's done line and the CPU gate; the refusals.
set -u; H=$(cd "$(dirname "$0")" && pwd)
new() {
  T=$(mktemp -d); mkdir -p "$T/logs"; echo "plan stub" > "$T/logs/plan_e87_receipts.md"
  printf '{"sha256": "%s"}\n' "$(sha256sum "$T/logs/plan_e87_receipts.md" | cut -d' ' -f1)" > "$T/logs/plan_e87_receipts.frozen.json"
  printf 'echo "$1" >> "%s/a0_calls"; [ -e "%s/a0fail_$1" ] && exit 1; true\n' "$T" "$T" > "$T/a0.sh"
  printf 'echo "$1" >> "%s/a_calls"; true\n' "$T" > "$T/a.sh"
  printf '[ -e "%s/alive_$1" ]\n' "$T" > "$T/ps.sh"
  printf 'n=$(( $(cat "%s/gpu_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/gpu_calls"; case " ${BUSY:-} " in *" $n "*) echo 4242;; esac; true\n' "$T" "$T" > "$T/nv.sh"
  printf 'n=$(( $(cat "%s/load_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/load_calls"; [ $n -le ${HIGH:-0} ] && echo "t cores 5.10 over 10.0 s" || echo "t cores 1.00 over 10.0 s"\n' "$T" "$T" > "$T/load.sh"
}
env_run() { env E87_ROOT="$T" A0_CMD="bash $T/a0.sh" A_CMD="bash $T/a.sh" PS_CMD="bash $T/ps.sh" NVSMI_CMD="bash $T/nv.sh" LOAD_CMD="bash $T/load.sh" \
            WAIT_PIDS="11 22" POLL_S=1 "$@"; }
fail() { echo "FAIL $1"; cat "$T/out.txt" 2>/dev/null; exit 1; }
# a0: waits for both pids, then 3 empty checks in a row (busy on check 2: 5 checks); a waits for A0's done line
new; touch "$T/alive_11" "$T/alive_22"
( env_run BUSY="2" bash "$H/e87_a_chain.sh" a0 > "$T/out.txt" 2>&1; echo $? > "$T/rc0" ) &
( env_run HIGH=2 bash "$H/e87_a_chain.sh" a > "$T/out_a.txt" 2>&1; echo $? > "$T/rca" ) &
sleep 3; [ ! -e "$T/gpu_calls" ] && [ ! -e "$T/a0_calls" ] && [ ! -e "$T/a_calls" ] || fail "did not wait for the pids"
rm "$T/alive_11"; sleep 2; [ ! -e "$T/gpu_calls" ] || fail "did not wait for the second pid"
rm "$T/alive_22"; wait
[ "$(cat "$T/rc0")" = 0 ] && [ "$(cat "$T/gpu_calls")" = 5 ] && [ "$(tr '\n' ' ' < "$T/a0_calls")" = "abcd star multiwoz " ] || fail "a0 (gpu checks $(cat "$T/gpu_calls"))"
grep -q "^e87 A0 end: done .* exit 0$" "$T/logs/e87_a0_end.log" || fail "a0 end line"
[ "$(cat "$T/rca")" = 0 ] && [ "$(tr '\n' ' ' < "$T/a_calls")" = "abcd star multiwoz " ] && [ "$(cat "$T/load_calls")" = 5 ] \
  && grep -q "^e87 A end: done .* exit 0$" "$T/logs/e87_a_end.log" || fail "a (load checks $(cat "$T/load_calls" 2>/dev/null))"
[ "$(env_run bash "$H/e87_a_chain.sh" a0 > /dev/null 2>&1; echo $?)" = 1 ] || fail "second a0 start not refused"
rm -rf "$T"
# a0 stops at the first failing dataset; a then refuses
new; touch "$T/a0fail_star"; env_run bash "$H/e87_a_chain.sh" a0 > "$T/out.txt" 2>&1
grep -q "^e87 A0 end: A0 failed (star) .* exit 1$" "$T/logs/e87_a0_end.log" && [ "$(tr '\n' ' ' < "$T/a0_calls")" = "abcd star " ] || fail "a0 failure"
env_run bash "$H/e87_a_chain.sh" a > "$T/out.txt" 2>&1
grep -q "^e87 A end: refused (A0 did not end done) .* exit 1$" "$T/logs/e87_a_end.log" && [ ! -e "$T/a_calls" ] || fail "a after a failed A0"
rm -rf "$T"
# freeze guard and bad stage
new; echo changed >> "$T/logs/plan_e87_receipts.md"; env_run bash "$H/e87_a_chain.sh" a0 > "$T/out.txt" 2>&1
grep -q "^e87 A0 end: refused (no freeze record of the plan's text) .* exit 1$" "$T/logs/e87_a0_end.log" && [ ! -e "$T/gpu_calls" ] || fail "freeze guard"
[ "$(env_run bash "$H/e87_a_chain.sh" b > /dev/null 2>&1; echo $?)" = 1 ] || fail "bad stage"; rm -rf "$T"
echo "ok: a0 waits for both pids, then 3 empty-GPU checks in a row (busy on check 2: 5 checks), runs abcd, star, multiwoz and exits;"
echo "    a waits for A0's done line and the CPU gate (5.10 cores twice, then 1.00), runs the three datasets; refusals: a second start,"
echo "    a failing dataset stops A0 and A refuses, a changed plan, an unknown stage"
echo PASSED
