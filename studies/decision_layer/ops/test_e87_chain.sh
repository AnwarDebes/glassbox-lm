#!/usr/bin/env bash
# Dry test of e87_chain.sh with stubs (no data, no GPU):
# - the whole run: a0 waits for 3 empty-GPU checks in a row and runs the three datasets; a waits for a0's done line
#   and the CPU gate; bdry runs the dry queue, the gate and the dry check; b needs bdry done and the dry check passed,
#   runs the 9-job file, the gate and the bar check; bkill waits for b and a, the gate, then the full check; every
#   second start refused;
# - refusals: an existing output, a failed a0 (a and bdry refuse), a failing dry check (b refuses), a dry check that
#   did not end "all dry checks passed", a stale or missing freeze record, a job file, a pinned repository file or this
#   chain off the note's table (a finish-pass row is not checked), a file on e87's path changed since the freeze (a file
#   off the path is fine), a lock held by another chain of the same stage, an e80 stage's lock held (a stale lock file is
#   fine), an unknown stage;
# - a restart after a stage that ended before any output (a handoff timeout); none when an output exists;
# - fail closed: an nvidia-smi error counts as a busy GPU, a cpu_load.py error as a busy CPU;
# - A_PARALLEL=1 for a: it waits for b's done line, runs the three datasets at once, a failing one fails the stage,
#   a gate timeout and a TERM both stop the running jobs (no orphan);
# - signals: TERM while a job runs (the job's process group is stopped), HUP while waiting, INT (default disposition
#   restored for the test): each writes "killed by signal <name>" at once.
set -u; H=$(cd "$(dirname "$0")" && pwd); PY=${HOME}/.local/bin/python3.11
new() {
  T=$(mktemp -d); mkdir -p "$T/logs" "$T/scripts"
  echo "dry jobs stub" > "$T/logs/jobs_e87b_dry.txt"; echo "jobs stub" > "$T/logs/jobs_e87b.txt"; echo "# pinned stub" > "$T/scripts/pinned_stub.py"
  { echo "note stub"; echo "| logs/jobs_e87b.txt | $(sha256sum "$T/logs/jobs_e87b.txt" | cut -d' ' -f1) |"
    echo "| logs/jobs_e87b_dry.txt | $(sha256sum "$T/logs/jobs_e87b_dry.txt" | cut -d' ' -f1) |"
    echo "| scripts/pinned_stub.py (read 2026-10-25T12:00:00Z) | $(sha256sum "$T/scripts/pinned_stub.py" | cut -d' ' -f1) |"
    echo "| probe.out (finish-pass notes, not published) | $(printf '0%.0s' $(seq 1 64)) |"
    echo "| e87_chain.sh (project notes, not published) | ${CHAIN_SHA:-$(sha256sum "$H/e87_chain.sh" | cut -d' ' -f1)} |"; } > "$T/logs/plan_e87_note_2026-10-25.md"
  printf '{"sha256": "%s", "frozen_utc": "2026-10-25T12:00:00Z"}\n' "$(sha256sum "$T/logs/plan_e87_note_2026-10-25.md" | cut -d' ' -f1)" > "$T/logs/plan_e87_note_2026-10-25.frozen.json"
  printf 'echo "$1 $2 $3" >> "%s/queue_calls"; touch "$2"; true\n' "$T" > "$T/q.sh"
  printf 'echo "c ${1:-}" >> "%s/check_calls"\ncase "${1:-}" in\n  --dry) [ -e "%s/dryfail" ] && { echo "e87b_check: settings differ" >&2; exit 1; }; echo "ok x"; echo "all dry checks passed";;\n  --bar-only) echo "the bar: stub";;\n  *) echo "the kill rule: stub";;\nesac\n' "$T" "$T" > "$T/c.sh"
  printf 'echo "$1" >> "%s/a0_calls"; [ -n "${A0SLEEP:-}" ] && { echo $$ > "%s/a0_pid"; sleep "$A0SLEEP"; }; [ -e "%s/a0fail_$1" ] && exit 1; true\n' "$T" "$T" "$T" > "$T/a0.sh"
  printf 'echo "$1" >> "%s/a_calls"; echo "start $1 $$" >> "%s/a_events"; [ -n "${ASLEEP:-}" ] && sleep "$ASLEEP"; echo "end $1" >> "%s/a_events"; [ -e "%s/afail_$1" ] && exit 1; echo "A $1 stub"\n' "$T" "$T" "$T" "$T" > "$T/a.sh"
  printf 'n=$(( $(cat "%s/gpu_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/gpu_calls"; case " ${BUSY:-} " in *" $n "*) echo 4242;; esac; true\n' "$T" "$T" > "$T/nv.sh"
  printf 'n=$(( $(cat "%s/load_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/load_calls"; [ -n "${ONLY_FIRST:-}" ] && [ $n -gt 1 ] && { echo "t cores 5.10 over 10.0 s"; exit 0; }; [ $n -le ${HIGH:-0} ] && echo "t cores 5.10 over 10.0 s" || echo "t cores 1.00 over 10.0 s"\n' "$T" "$T" > "$T/load.sh"
  printf 'cat "%s/diff_out" 2>/dev/null; true\n' "$T" > "$T/diff.sh"
  E=(E87_ROOT="$T" A0_CMD="bash $T/a0.sh" A_CMD="bash $T/a.sh" CHECK_CMD="bash $T/c.sh" QUEUE_CMD="bash $T/q.sh" NVSMI_CMD="bash $T/nv.sh" LOAD_CMD="bash $T/load.sh" DIFF_CMD="bash $T/diff.sh")
}
run() { local st=$1; shift; env "${E[@]}" POLL_S=0 "$@" bash "$H/e87_chain.sh" "$st" > "$T/out_$st.txt" 2>&1; echo $?; }
fail() { echo "FAIL $1"; cat "$T"/out_*.txt 2>/dev/null; exit 1; }
endl() { tail -n 1 "$T/logs/e87n_$1_end.log" 2>/dev/null; }
calls() { tr '\n' '|' < "$T/$1" 2>/dev/null; }
gone() { case "$(ps -o stat= -p "$1" 2>/dev/null)" in ""|Z*) return 0;; *) return 1;; esac; }
done_line() { echo "e87n $1 end: done 2026-10-25T12:10:00Z exit 0" > "$T/logs/e87n_$1_end.log"; }

# the whole run: a waits for a0 and the gate; a0 waits for 3 empty-GPU checks (busy on 1 and 3: 6 checks)
new
( env "${E[@]}" POLL_S=1 HIGH=2 bash "$H/e87_chain.sh" a > "$T/out_a.txt" 2>&1; echo $? > "$T/rc_a" ) &
sleep 2; [ ! -e "$T/a_calls" ] && [ ! -e "$T/load_calls" ] || fail "a did not wait for a0"
[ "$(run a0 BUSY="1 3")" = 0 ] && [ "$(cat "$T/gpu_calls")" = 6 ] && [ "$(calls a0_calls)" = "abcd|star|multiwoz|" ] || fail "a0 (gpu checks $(cat "$T/gpu_calls" 2>/dev/null))"
endl a0 | grep -q "^e87n a0 end: done .* exit 0$" && grep -q "pins checked" "$T/out_a0.txt" || fail "a0 end line"
wait; [ "$(cat "$T/rc_a")" = 0 ] && [ "$(calls a_calls)" = "abcd|star|multiwoz|" ] && [ "$(cat "$T/load_calls")" = 5 ] && endl a | grep -q "^e87n a end: done .* exit 0$" || fail "a (load checks $(cat "$T/load_calls" 2>/dev/null))"
[ "$(run bdry)" = 0 ] && [ "$(calls queue_calls)" = "logs/jobs_e87b_dry.txt logs/e87b_dry_queue.log 4.5|" ] && [ "$(tail -n 1 "$T/logs/e87b_dry_check.out")" = "all dry checks passed" ] \
  && [ "$(cat "$T/load_calls")" = 6 ] && endl bdry | grep -q "^e87n bdry end: done .* exit 0$" || fail "bdry (the gate before the dry check)"
[ "$(run b)" = 0 ] && [ "$(calls queue_calls)" = "logs/jobs_e87b_dry.txt logs/e87b_dry_queue.log 4.5|logs/jobs_e87b.txt logs/e87b_queue.log 4.5|" ] \
  && grep -q "the bar: stub" "$T/logs/e87b_bar.out" && [ "$(cat "$T/load_calls")" = 7 ] && endl b | grep -q "^e87n b end: done .* exit 0$" || fail "b (the gate before the bar check)"
[ "$(run bkill)" = 0 ] && grep -q "the kill rule: stub" "$T/logs/e87b_check.out" && [ "$(calls check_calls)" = "c --dry|c --bar-only|c |" ] \
  && [ "$(cat "$T/load_calls")" = 8 ] && endl bkill | grep -q "^e87n bkill end: done .* exit 0$" || fail "bkill"
for st in a0 a bdry b bkill; do [ "$(run $st)" = 1 ] && [ "$(wc -l < "$T/logs/e87n_${st}_end.log")" = 1 ] || fail "second start of $st not refused"; done
rm -rf "$T"
echo "ok whole run: a waits for a0's done line and the CPU gate (5.10 cores twice, then 1.00); a0 waits for 3 empty-GPU checks (6 checks);"
echo "   bdry: dry queue, gate, dry check; b: the 9-job file, gate, bar check; bkill: gate, full check; every second start refused"

# an existing output refuses a first start (no end line); a failed a0 makes a and bdry refuse
new; touch "$T/logs/e87_bits.out"; [ "$(run a0)" = 1 ] && [ ! -e "$T/logs/e87n_a0_end.log" ] && [ ! -e "$T/gpu_calls" ] || fail "existing output"
rm -rf "$T"
new; touch "$T/a0fail_star"
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: A0 failed (star) .* exit 1$" && [ "$(calls a0_calls)" = "abcd|star|" ] || fail "a0 failure"
[ "$(run a)" = 1 ] && endl a | grep -q "^e87n a end: refused (a0 did not end done) .* exit 1$" && [ ! -e "$T/a_calls" ] || fail "a after a failed a0"
[ "$(run bdry)" = 1 ] && endl bdry | grep -q "^e87n bdry end: refused (a0 did not end done) .* exit 1$" && [ ! -e "$T/queue_calls" ] || fail "bdry after a failed a0"
rm -rf "$T"
echo "ok an existing output refuses a first start; a failed a0 stops at its dataset, and a and bdry refuse"

# a failing dry check; b refuses after it, and when the dry check's last line is not the pass line
new; done_line a0; touch "$T/dryfail"
[ "$(run bdry)" = 1 ] && endl bdry | grep -q "^e87n bdry end: dry check failed .* exit 1$" && grep -q "settings differ" "$T/logs/e87b_dry_check.out" || fail "failing dry check"
[ "$(run b)" = 1 ] && endl b | grep -q "^e87n b end: refused (bdry did not end done) .* exit 1$" && [ "$(calls queue_calls)" = "logs/jobs_e87b_dry.txt logs/e87b_dry_queue.log 4.5|" ] || fail "b after a failed dry check"
rm -rf "$T"
new; done_line bdry; echo "ok x" > "$T/logs/e87b_dry_check.out"
[ "$(run b)" = 1 ] && endl b | grep -q "^e87n b end: refused (the dry checks have not passed) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "b without the dry pass line"
rm -rf "$T"
echo "ok a failing dry check ends bdry and b refuses; b refuses when the dry check did not end with its pass line"

# the note and the pins: a stale or missing freeze record, a job file or this chain off the table, a changed path file
new; echo changed >> "$T/logs/plan_e87_note_2026-10-25.md"
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: refused (no freeze record of the note's text) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "stale record"
rm -rf "$T"
new; rm "$T/logs/plan_e87_note_2026-10-25.frozen.json"
[ "$(run bdry)" = 1 ] && endl bdry | grep -q "refused (no freeze record" && [ ! -e "$T/queue_calls" ] || fail "missing record"
rm -rf "$T"
new; done_line a0; echo "edited" >> "$T/logs/jobs_e87b_dry.txt"
[ "$(run bdry)" = 1 ] && endl bdry | grep -q "^e87n bdry end: refused (logs/jobs_e87b_dry.txt does not hash to the note's table) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "job file off the table"
rm -rf "$T"
CHAIN_SHA=0000 new
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: refused (this chain does not hash to the note's table) .* exit 1$" || fail "chain off the table"
rm -rf "$T"
new; echo "# edited" >> "$T/scripts/pinned_stub.py"
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: refused (scripts/pinned_stub.py does not hash to the note's table) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "pinned repository file changed"
rm -rf "$T"
new; flock "$T/logs/e80n_run.lock" sleep 3 & LP=$!; sleep 0.5
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: refused (an e80 chain stage is running or waiting: logs/e80n_run.lock) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "e80 lock held"
wait "$LP"; rm -rf "$T"
new; touch "$T/logs/e80n_dry.lock"; [ "$(run a0)" = 0 ] || fail "a stale e80 lock file must not refuse"
rm -rf "$T"
new; printf 'CHANGED scripts/e86_check.py\nNEW scripts/e80_note_check.py\n' > "$T/diff_out"
[ "$(run a0)" = 0 ] || fail "a file off e87's path must not refuse"
rm -rf "$T"
new; printf 'CHANGED scripts/e86_check.py\nCHANGED src/dl/tm.py\n' > "$T/diff_out"
[ "$(run a0)" = 1 ] && endl a0 | grep -q "^e87n a0 end: refused (changed since the note's freeze: CHANGED src/dl/tm.py ) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "path file changed"
rm -rf "$T"
new; [ "$(run go)" = 1 ] && grep -q "usage" "$T/out_go.txt" || fail "unknown stage"
rm -rf "$T"
echo "ok pins: a changed note, a missing freeze record, a job file, a pinned repository file (row with a parenthesis) or this chain off the"
echo "   note's table (a finish-pass row is not a repository row), an e80 chain stage holding its lock (a stale lock file is fine),"
echo "   src/dl/tm.py changed since the freeze"
echo "   (scripts/e86_check.py, off e87's path, is fine); an unknown stage"

# restart after a stage that ended before any output; no restart when an output exists; a held lock
new; echo "e87n a0 end: handoff: no 3 empty-GPU checks in a row within 360 min 2026-10-25T12:30:00Z exit 3" > "$T/logs/e87n_a0_end.log"
[ "$(run a0)" = 0 ] && [ "$(wc -l < "$T/logs/e87n_a0_end.log")" = 2 ] && endl a0 | grep -q "^e87n a0 end: done " && grep -q "restart: no output exists" "$T/out_a0.txt" || fail "restart after a handoff timeout"
rm -rf "$T"
new; echo "e87n a0 end: killed by signal TERM 2026-10-25T12:30:00Z exit 143" > "$T/logs/e87n_a0_end.log"; touch "$T/logs/e87_bits.out"
[ "$(run a0)" = 1 ] && [ "$(wc -l < "$T/logs/e87n_a0_end.log")" = 1 ] && [ ! -e "$T/gpu_calls" ] || fail "no restart when an output exists"
rm -rf "$T"
new; echo "e87n bkill end: check failed 2026-10-25T12:30:00Z exit 1" > "$T/logs/e87n_bkill_end.log"
[ "$(run bkill)" = 1 ] && [ "$(wc -l < "$T/logs/e87n_bkill_end.log")" = 1 ] || fail "no restart after a failed check"
rm -rf "$T"
new; flock "$T/logs/e87n_a0.lock" sleep 3 & LP=$!; sleep 0.5
[ "$(run a0)" = 1 ] && grep -q "another e87n a0 chain is running" "$T/out_a0.txt" && [ ! -e "$T/logs/e87n_a0_end.log" ] && [ ! -e "$T/gpu_calls" ] || fail "held lock"
wait "$LP"; rm -rf "$T"
echo "ok a restart after a handoff timeout (the old end line kept); none when an output exists or after a failed check; a held lock refuses"

# fail closed: an nvidia-smi error is a busy GPU, a cpu_load.py error a busy CPU
new; [ "$(run a0 NVSMI_CMD=false GPU_WAIT_S=2)" = 3 ] && endl a0 | grep -q "^e87n a0 end: handoff: no 3 empty-GPU checks in a row within 0 min .* exit 3$" && [ ! -e "$T/a0_calls" ] || fail "nvidia-smi error"
rm -rf "$T"
new; done_line a0; [ "$(run a LOAD_CMD=false GATE_S=2)" = 3 ] && endl a | grep -q "^e87n a end: CPU gate never at or below 4.5 cores for 0 min (abcd) .* exit 3$" && [ ! -e "$T/a_calls" ] || fail "cpu_load error"
rm -rf "$T"
echo "ok fail closed: an nvidia-smi error and a cpu_load.py error count as busy (a0 and a end at their time limits)"

# A_PARALLEL=1: waits for b; the three datasets at once; a failing one fails the stage; a gate timeout or a TERM stops the jobs
new; done_line a0
( env "${E[@]}" POLL_S=1 A_PARALLEL=1 bash "$H/e87_chain.sh" a > "$T/out_a.txt" 2>&1; echo $? > "$T/rc_a" ) &
sleep 2; [ ! -e "$T/a_calls" ] || fail "parallel a did not wait for b"
done_line b; wait; [ "$(cat "$T/rc_a")" = 0 ] && endl a | grep -q "^e87n a end: done .* exit 0$" || fail "parallel a after b"
rm -rf "$T"
new; done_line a0; done_line b
t0=$(date +%s); [ "$(run a A_PARALLEL=1 ASLEEP=2)" = 0 ] || fail "parallel a"
[ $(( $(date +%s) - t0 )) -lt 6 ] && [ "$(head -n 3 "$T/a_events" | cut -d' ' -f1 | tr '\n' ' ')" = "start start start " ] || fail "parallel a did not run at once"
for d in abcd star multiwoz; do grep -q "A $d stub" "$T/logs/e87_readability_$d.out" || fail "parallel output $d"; done
[ ! -e "$T/logs/e87_readability.out" ] && [ "$(cat "$T/load_calls")" = 3 ] && endl a | grep -q "^e87n a end: done .* exit 0$" || fail "parallel a end line"
rm -rf "$T"
new; done_line a0; done_line b; touch "$T/afail_star"
[ "$(run a A_PARALLEL=1)" = 1 ] && endl a | grep -q "^e87n a end: A failed (star) .* exit 1$" && [ "$(grep -c '^end' "$T/a_events")" = 3 ] || fail "parallel a with a failing dataset"
rm -rf "$T"
new; done_line a0; done_line b
[ "$(run a A_PARALLEL=1 ONLY_FIRST=1 A_WAIT_S=3 ASLEEP=60)" = 3 ] && JP=$(awk '/^start abcd/ {print $3}' "$T/a_events") && [ -n "$JP" ] || fail "parallel gate timeout"
sleep 0.5; gone "$JP" || fail "parallel job $JP survived the chain's gate timeout"
endl a | grep -q "^e87n a end: CPU gate never at or below 4.5 cores for 0 min (star); its running job(s) stopped .* exit 3$" || fail "parallel gate timeout end line"
rm -rf "$T"
new; done_line a0; done_line b
env "${E[@]}" POLL_S=0 A_PARALLEL=1 ASLEEP=60 bash "$H/e87_chain.sh" a > "$T/out_a.txt" 2>&1 & CP=$!
for i in $(seq 1 50); do [ "$(grep -c '^start' "$T/a_events" 2>/dev/null)" = 3 ] && break; sleep 0.2; done
[ "$(grep -c '^start' "$T/a_events")" = 3 ] || fail "the three parallel jobs never started"
JPS=$(awk '/^start/ {print $3}' "$T/a_events"); kill -TERM "$CP"; wait "$CP"; rc=$?; sleep 0.5
for jp in $JPS; do gone "$jp" || fail "parallel job $jp survived the chain's TERM"; done
[ "$rc" = 143 ] && endl a | grep -q "^e87n a end: killed by signal TERM; its running job(s) stopped .* exit 143$" || fail "TERM in parallel a (rc $rc)"
rm -rf "$T"
echo "ok A_PARALLEL=1: waits for b; the three datasets at once into their own outputs; a failing dataset fails the stage;"
echo "   a gate timeout and a TERM both stop the running jobs"

# signals: TERM while a job runs stops the job's process group; HUP while waiting; INT with the default disposition
new
env "${E[@]}" POLL_S=0 A0SLEEP=60 bash "$H/e87_chain.sh" a0 > "$T/out_a0.txt" 2>&1 & CP=$!
for i in $(seq 1 50); do [ -s "$T/a0_pid" ] && break; sleep 0.2; done
JP=$(cat "$T/a0_pid" 2>/dev/null); [ -n "$JP" ] || fail "the a0 job never started"
t0=$(date +%s); kill -TERM "$CP"; wait "$CP"; rc=$?
sleep 0.5; gone "$JP" || fail "the running job $JP survived the chain's TERM"
[ "$rc" = 143 ] && [ $(( $(date +%s) - t0 )) -le 5 ] && endl a0 | grep -q "^e87n a0 end: killed by signal TERM; its running job(s) stopped .* exit 143$" || fail "TERM (rc $rc)"
rm -rf "$T"; new
env "${E[@]}" POLL_S=1 bash "$H/e87_chain.sh" a > "$T/out_a.txt" 2>&1 & CP=$!
sleep 2; t0=$(date +%s); kill -HUP "$CP"; wait "$CP"; rc=$?
[ "$rc" = 129 ] && [ $(( $(date +%s) - t0 )) -le 3 ] && endl a | grep -q "^e87n a end: killed by signal HUP .* exit 129$" || fail "HUP (rc $rc)"
rm -rf "$T"; new
env "${E[@]}" POLL_S=1 "$PY" -I -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp("bash", ["bash"] + sys.argv[1:])' \
  "$H/e87_chain.sh" bkill > "$T/out_bkill.txt" 2>&1 & CP=$!
sleep 2; t0=$(date +%s); kill -INT "$CP"; wait "$CP"; rc=$?
[ "$rc" = 130 ] && [ $(( $(date +%s) - t0 )) -le 3 ] && endl bkill | grep -q "^e87n bkill end: killed by signal INT .* exit 130$" || fail "INT (rc $rc)"
rm -rf "$T"
echo "ok signals: TERM while A0 runs (its job's process group stopped too), HUP while a waits, INT while bkill waits: 'killed by signal <name>' at once"
echo PASSED
