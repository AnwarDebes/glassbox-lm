#!/usr/bin/env bash
# Dry test of e80_chain.sh with stubs (no data, no GPU):
# - the whole run: dry waits for 3 empty-GPU checks in a row (busy on checks 1 and 3: 6 checks), runs the dry job file,
#   the CPU gate and the dry check; run waits for dry's done line, needs the dry check's pass line, runs the 5-job file,
#   the CPU gate, the check and the report; every second start refused;
# - refusals: an existing output, a failing dry check (run refuses), a dry check without its pass line, a failing check
#   or report, a stale or missing freeze record, a job file, a pinned repository file (row with a parenthesis) or this
#   chain off the note's table (a finish-pass row is not a repository row), a file on e80's path changed since the
#   freeze (a file off the path is fine), its own lock held, an e87 GPU stage's lock held (a stale lock file is fine),
#   an unknown stage;
# - a restart after a handoff timeout (the old end line kept); none after a failed check;
# - fail closed: an nvidia-smi error counts as a busy GPU, a cpu_load.py error as a busy CPU;
# - signals: TERM while the dry queue runs (its process group is stopped), HUP while waiting for the GPU, INT (default
#   disposition restored for the test): each writes "killed by signal <name>" at once.
set -u; H=$(cd "$(dirname "$0")" && pwd); PY=${HOME}/.local/bin/python3.11
new() {
  T=$(mktemp -d); mkdir -p "$T/logs" "$T/scripts"
  echo "dry jobs stub" > "$T/logs/jobs_e80_dry.txt"; echo "jobs stub" > "$T/logs/jobs_e80.txt"; echo "# pinned stub" > "$T/scripts/pinned_stub.py"
  { echo "note stub"; echo "| logs/jobs_e80.txt | $(sha256sum "$T/logs/jobs_e80.txt" | cut -d' ' -f1) |"
    echo "| logs/jobs_e80_dry.txt | $(sha256sum "$T/logs/jobs_e80_dry.txt" | cut -d' ' -f1) |"
    echo "| scripts/pinned_stub.py (as published) | $(sha256sum "$T/scripts/pinned_stub.py" | cut -d' ' -f1) |"
    echo "| probe.out (finish-pass notes, not published) | $(printf '0%.0s' $(seq 1 64)) |"
    echo "| e80_chain.sh (project notes, not published) | ${CHAIN_SHA:-$(sha256sum "$H/e80_chain.sh" | cut -d' ' -f1)} |"; } > "$T/logs/plan_e80_note_2026-10-25.md"
  printf '{"sha256": "%s", "frozen_utc": "2026-10-25T12:00:00Z"}\n' "$(sha256sum "$T/logs/plan_e80_note_2026-10-25.md" | cut -d' ' -f1)" > "$T/logs/plan_e80_note_2026-10-25.frozen.json"
  printf 'echo "$1 $2 $3" >> "%s/queue_calls"; touch "$2"; [ -n "${QSLEEP:-}" ] && { echo $$ > "%s/q_pid"; sleep "$QSLEEP"; }; true\n' "$T" "$T" > "$T/q.sh"
  printf 'echo dc >> "%s/calls"; [ -e "%s/dryfail" ] && { echo "e80_dry_check: settings" >&2; exit 1; }; echo "ok x"; [ -e "%s/nopass" ] || echo "all dry checks passed"\n' "$T" "$T" "$T" > "$T/dc.sh"
  printf 'echo c >> "%s/calls"; [ -e "%s/checkfail" ] && { echo "FAIL star: settings"; exit 1; }; echo "RULE: stub"\n' "$T" "$T" > "$T/c.sh"
  printf 'echo rp >> "%s/calls"; [ -e "%s/reportfail" ] && { echo "e80_report: missing" >&2; exit 1; }; echo "readings stub"\n' "$T" "$T" > "$T/rp.sh"
  printf 'n=$(( $(cat "%s/gpu_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/gpu_calls"; case " ${BUSY:-} " in *" $n "*) echo 4242;; esac; [ -n "${ALWAYS_BUSY:-}" ] && echo 4242; true\n' "$T" "$T" > "$T/nv.sh"
  printf 'n=$(( $(cat "%s/load_calls" 2>/dev/null || echo 0) + 1 )); echo $n > "%s/load_calls"; [ $n -le ${HIGH:-0} ] && echo "t cores 5.10 over 10.0 s" || echo "t cores 1.00 over 10.0 s"\n' "$T" "$T" > "$T/load.sh"
  printf 'cat "%s/diff_out" 2>/dev/null; true\n' "$T" > "$T/diff.sh"
  E=(E80_ROOT="$T" CHECK_CMD="bash $T/c.sh" DRYCHECK_CMD="bash $T/dc.sh" REPORT_CMD="bash $T/rp.sh" QUEUE_CMD="bash $T/q.sh" NVSMI_CMD="bash $T/nv.sh" LOAD_CMD="bash $T/load.sh" DIFF_CMD="bash $T/diff.sh")
}
run() { local st=$1; shift; env "${E[@]}" POLL_S=0 "$@" bash "$H/e80_chain.sh" "$st" > "$T/out_$st.txt" 2>&1; echo $?; }
fail() { echo "FAIL $1"; cat "$T"/out_*.txt 2>/dev/null; exit 1; }
endl() { tail -n 1 "$T/logs/e80n_$1_end.log" 2>/dev/null; }
calls() { tr '\n' '|' < "$T/$1" 2>/dev/null; }
gone() { case "$(ps -o stat= -p "$1" 2>/dev/null)" in ""|Z*) return 0;; *) return 1;; esac; }

# the whole run: run waits for dry; dry waits for 3 empty-GPU checks (busy on 1 and 3: 6 checks)
new
( env "${E[@]}" POLL_S=1 bash "$H/e80_chain.sh" run > "$T/out_run.txt" 2>&1; echo $? > "$T/rc_run" ) &
sleep 2; [ ! -e "$T/queue_calls" ] || fail "run did not wait for dry"
[ "$(run dry BUSY="1 3" HIGH=1)" = 0 ] && [ "$(cat "$T/gpu_calls")" = 6 ] && [ "$(calls queue_calls)" = "logs/jobs_e80_dry.txt logs/e80_dry_queue.log 4.5|" ] \
  && [ "$(tail -n 1 "$T/logs/e80_dry_check.out")" = "all dry checks passed" ] && endl dry | grep -q "^e80n dry end: done .* exit 0$" || fail "dry"
wait; [ "$(cat "$T/rc_run")" = 0 ] && [ "$(calls queue_calls)" = "logs/jobs_e80_dry.txt logs/e80_dry_queue.log 4.5|logs/jobs_e80.txt logs/e80_queue.log 4.5|" ] \
  && [ "$(calls calls)" = "dc|c|rp|" ] && grep -q "RULE: stub" "$T/logs/e80_check.out" && grep -q "readings stub" "$T/logs/e80_report.out" \
  && [ "$(cat "$T/load_calls")" = 3 ] && endl run | grep -q "^e80n run end: done .* exit 0$" || fail "run (load checks $(cat "$T/load_calls" 2>/dev/null))"
for st in dry run; do [ "$(run $st)" = 1 ] && [ "$(wc -l < "$T/logs/e80n_${st}_end.log")" = 1 ] || fail "second start of $st not refused"; done
rm -rf "$T"
echo "ok whole run: dry (6 GPU checks, the dry job file, the gate, the dry check); run waits for dry's done line (the 5-job file, the gate,"
echo "   the check, the report); every second start refused"

new; touch "$T/logs/e80_queue.log"; [ "$(run run)" = 1 ] && [ ! -e "$T/logs/e80n_run_end.log" ] || fail "existing output"
rm -rf "$T"
new; touch "$T/dryfail"
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: dry check failed .* exit 1$" || fail "failing dry check"
[ "$(run run)" = 1 ] && endl run | grep -q "^e80n run end: refused (dry did not end done) .* exit 1$" && [ "$(calls queue_calls)" = "logs/jobs_e80_dry.txt logs/e80_dry_queue.log 4.5|" ] || fail "run after a failed dry check"
rm -rf "$T"
new; echo "e80n dry end: done 2026-10-25T12:20:00Z exit 0" > "$T/logs/e80n_dry_end.log"; echo "ok x" > "$T/logs/e80_dry_check.out"
[ "$(run run)" = 1 ] && endl run | grep -q "^e80n run end: refused (the dry checks have not passed) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "run without the dry pass line"
rm -rf "$T"
new; echo "e80n dry end: done 2026-10-25T12:20:00Z exit 0" > "$T/logs/e80n_dry_end.log"; echo "all dry checks passed" > "$T/logs/e80_dry_check.out"; touch "$T/checkfail"
[ "$(run run)" = 1 ] && endl run | grep -q "^e80n run end: check failed .* exit 1$" && grep -q "FAIL star" "$T/logs/e80_check.out" || fail "failing check"
[ "$(run run)" = 1 ] && [ "$(wc -l < "$T/logs/e80n_run_end.log")" = 1 ] || fail "restart after a failed check"
rm -rf "$T"
new; echo "e80n dry end: done 2026-10-25T12:20:00Z exit 0" > "$T/logs/e80n_dry_end.log"; echo "all dry checks passed" > "$T/logs/e80_dry_check.out"; touch "$T/reportfail"
[ "$(run run)" = 1 ] && endl run | grep -q "^e80n run end: report failed .* exit 1$" || fail "failing report"
rm -rf "$T"
echo "ok an existing output; a failing dry check (run refuses); run without the dry pass line; a failing check (no restart after it); a failing report"

new; echo changed >> "$T/logs/plan_e80_note_2026-10-25.md"
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (no freeze record of the note's text) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "stale record"
rm -rf "$T"
new; rm "$T/logs/plan_e80_note_2026-10-25.frozen.json"; [ "$(run dry)" = 1 ] && endl dry | grep -q "refused (no freeze record" || fail "missing record"
rm -rf "$T"
new; echo edited >> "$T/logs/jobs_e80_dry.txt"
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (logs/jobs_e80_dry.txt does not hash to the note's table) .* exit 1$" || fail "job file off the table"
rm -rf "$T"
new; echo "# edited" >> "$T/scripts/pinned_stub.py"
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (scripts/pinned_stub.py does not hash to the note's table) .* exit 1$" || fail "pinned repository file changed"
rm -rf "$T"
CHAIN_SHA=0000 new; [ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (this chain does not hash to the note's table) .* exit 1$" || fail "chain off the table"
rm -rf "$T"
new; printf 'CHANGED scripts/e87b_check.py\n' > "$T/diff_out"; [ "$(run dry)" = 0 ] || fail "a file off e80's path must not refuse"
rm -rf "$T"
new; printf 'CHANGED scripts/e80_dry_check.py\n' > "$T/diff_out"
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (changed since the note's freeze: CHANGED scripts/e80_dry_check.py ) .* exit 1$" || fail "path file changed"
rm -rf "$T"
new; flock "$T/logs/e80n_dry.lock" sleep 3 & LP=$!; sleep 0.5
[ "$(run dry)" = 1 ] && grep -q "another e80n dry chain is running" "$T/out_dry.txt" && [ ! -e "$T/logs/e80n_dry_end.log" ] || fail "held lock"
wait "$LP"; rm -rf "$T"
new; flock "$T/logs/e87n_b.lock" sleep 3 & LP=$!; sleep 0.5
[ "$(run dry)" = 1 ] && endl dry | grep -q "^e80n dry end: refused (an e87 GPU stage is running or waiting: logs/e87n_b.lock) .* exit 1$" && [ ! -e "$T/gpu_calls" ] || fail "e87 lock held"
wait "$LP"; rm -rf "$T"
new; touch "$T/logs/e87n_bdry.lock"; [ "$(run dry)" = 0 ] || fail "a stale e87 lock file must not refuse"
rm -rf "$T"
new; [ "$(run go)" = 1 ] && grep -q "usage" "$T/out_go.txt" || fail "unknown stage"
rm -rf "$T"
echo "ok pins and locks: a changed note, a missing freeze record, a job file, a pinned repository file (row with a parenthesis) or the chain off"
echo "   the note's table (a finish-pass row is not checked), e80_dry_check.py changed (e87b_check.py, off e80's path, is fine), its own lock held,"
echo "   an e87 GPU stage's lock held (a stale lock file is fine), an unknown stage"

new; echo "e80n dry end: handoff: no 3 empty-GPU checks in a row within 360 min 2026-10-25T12:30:00Z exit 3" > "$T/logs/e80n_dry_end.log"
[ "$(run dry)" = 0 ] && [ "$(wc -l < "$T/logs/e80n_dry_end.log")" = 2 ] && endl dry | grep -q "^e80n dry end: done " && grep -q "restart: no output exists" "$T/out_dry.txt" || fail "restart"
rm -rf "$T"
new; [ "$(run dry NVSMI_CMD=false GPU_WAIT_S=2)" = 3 ] && endl dry | grep -q "^e80n dry end: handoff: no 3 empty-GPU checks in a row within 0 min .* exit 3$" && [ ! -e "$T/queue_calls" ] || fail "nvidia-smi error"
rm -rf "$T"
new; [ "$(run dry LOAD_CMD=false GATE_S=2)" = 3 ] && endl dry | grep -q "^e80n dry end: CPU gate never at or below 4.5 cores for 0 min (dry check) .* exit 3$" && [ ! -e "$T/calls" ] || fail "cpu_load error"
rm -rf "$T"
echo "ok a restart after a handoff timeout (the old end line kept); fail closed: an nvidia-smi or cpu_load.py error counts as busy"

new
env "${E[@]}" POLL_S=0 QSLEEP=60 bash "$H/e80_chain.sh" dry > "$T/out_dry.txt" 2>&1 & CP=$!
for i in $(seq 1 50); do [ -s "$T/q_pid" ] && break; sleep 0.2; done
JP=$(cat "$T/q_pid" 2>/dev/null); [ -n "$JP" ] || fail "the dry queue never started"
t0=$(date +%s); kill -TERM "$CP"; wait "$CP"; rc=$?; sleep 0.5
gone "$JP" || fail "the dry queue $JP survived the chain's TERM"
[ "$rc" = 143 ] && [ $(( $(date +%s) - t0 )) -le 5 ] && endl dry | grep -q "^e80n dry end: killed by signal TERM; its running job stopped .* exit 143$" || fail "TERM (rc $rc)"
rm -rf "$T"; new
env "${E[@]}" POLL_S=1 ALWAYS_BUSY=1 bash "$H/e80_chain.sh" dry > "$T/out_dry.txt" 2>&1 & CP=$!
sleep 2; t0=$(date +%s); kill -HUP "$CP"; wait "$CP"; rc=$?
[ "$rc" = 129 ] && [ $(( $(date +%s) - t0 )) -le 3 ] && endl dry | grep -q "^e80n dry end: killed by signal HUP .* exit 129$" && [ ! -e "$T/queue_calls" ] || fail "HUP (rc $rc)"
rm -rf "$T"; new
env "${E[@]}" POLL_S=1 ALWAYS_BUSY=1 "$PY" -I -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp("bash", ["bash"] + sys.argv[1:])' \
  "$H/e80_chain.sh" dry > "$T/out_dry.txt" 2>&1 & CP=$!
sleep 2; t0=$(date +%s); kill -INT "$CP"; wait "$CP"; rc=$?
[ "$rc" = 130 ] && [ $(( $(date +%s) - t0 )) -le 3 ] && endl dry | grep -q "^e80n dry end: killed by signal INT .* exit 130$" || fail "INT (rc $rc)"
rm -rf "$T"
echo "ok signals: TERM while the dry queue runs (its process group stopped too), HUP and INT while waiting for the GPU: 'killed by signal <name>' at once"
echo PASSED
