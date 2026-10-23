#!/usr/bin/env bash
# e81, then e81x, back to back (plan logs/plan_e81_T_screen.md; notes logs/plan_e81_note_2026-10-21.md and
# logs/plan_e81_note2_2026-10-21.md). Started by hand after step 3c of the text-understanding line was done, in the background:
#   nohup bash logs/e81_chain.sh >> logs/e81_chain.log 2>&1
# 1. the pinned driver logs/e81_driver.sh, with its output in logs/e81_driver.log;
# 2. only if the driver's last line is "e81 GPU stages done": note2's e81x jobs through scripts/seq_queue.sh;
# 3. scripts/e81x_report.py to logs/e81x_report.out.
# It adds no logic beyond the two notes. On every way out, it appends one end line to logs/e81_chain.log:
#   "e81 chain end: <branch> <UTC> exit <code>"
# The branches are e81x done, the driver stopping, a refusal to overwrite,
# or a signal (TERM or INT).
# - Each stage runs in its own process group. On a signal the chain stops that group (processes this chain started),
#   then writes the end line.
# - E81_ROOT, E81_QUEUE and E81_PY exist only for tests/test_e81_chain.py, and are passed on to the driver; the real
#   run sets none of them.
set -uo pipefail
ROOT="${E81_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
Q="${E81_QUEUE:-bash scripts/seq_queue.sh}"
P="${E81_PY:-.venv-dl/bin/python}"
cd "$ROOT" || exit 1
LOGF="$ROOT/logs/e81_chain.log"
BRANCH="stopped before the driver"
CHILD=""
log() { echo "$(date -u +%FT%TZ) $*"; }
on_exit() {
  local rc=$?
  echo "e81 chain end: $BRANCH $(date -u +%FT%TZ) exit $rc" >> "$LOGF"
}
on_signal() {  # on_signal NAME CODE
  BRANCH="signal $1"
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM -- "-$CHILD" 2>/dev/null  # the stage's own process group, started by this chain
    wait "$CHILD" 2>/dev/null
  fi
  exit "$2"
}
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
stage() {  # stage CMD...: run CMD in its own process group and wait, so that a signal reaches the chain at once
  setsid -w "$@" &
  CHILD=$!
  wait "$CHILD"
  local rc=$?
  CHILD=""
  return $rc
}
for f in logs/e81_driver.log logs/e81x_report.out logs/e81x_queue.log; do
  if [ -e "$f" ]; then
    BRANCH="refused ($f exists)"
    log "$f exists: not overwriting; the chain stops"
    exit 1
  fi
done
log "chain start (pid $$): the e81 driver"
BRANCH="driver running"
stage bash logs/e81_driver.sh > logs/e81_driver.log 2>&1
rc=$?
last=$(tail -n 1 logs/e81_driver.log | cut -d' ' -f2-)
log "driver exit $rc; its last line: $last"
if [ "$rc" -ne 0 ] || [ "$last" != "e81 GPU stages done" ]; then
  BRANCH="driver stopped (e81x not started)"
  log "e81 did not finish: e81x not started"
  exit 1
fi
log "e81x: the 6 LR jobs"
BRANCH="e81x running"
stage $Q logs/jobs_e81x_lr_lowC.txt logs/e81x_queue.log
BRANCH="e81x report running"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $P scripts/e81x_report.py > logs/e81x_report.out 2>&1
rc=$?
log "e81x_report exit $rc; logs/e81x_report.out $(sha256sum logs/e81x_report.out | cut -c1-8)"
cat logs/e81x_report.out
log "e81x done"
BRANCH="e81x done (report exit $rc)"
exit "$rc"
