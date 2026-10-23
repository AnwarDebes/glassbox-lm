#!/usr/bin/env bash
# e83's GPU stages back to back (plan logs/plan_e83_learning_dynamics.md; this file is pinned by that plan's freeze).
# The top-level launcher, started by hand, in the background:
#   nohup bash logs/e83_driver.sh >> logs/e83_driver.log 2>&1
# It runs the job files one job at a time through scripts/seq_queue.sh and the checker scripts/e83_pick.py between
# them, and stops when a check fails:
#   smoke -> e83_pick smoke -> e83a (the screen) -> e83_pick screen (writes jobs_e83b.txt) -> e83b, if it has jobs
#   -> e83_pick mid (writes jobs_e83c.txt) -> e83c, if it has jobs -> e83_pick final.
# On every way out it appends one end line to logs/e83_end.log,
# "e83 end: <branch> <UTC> exit <code>".
# - Each queue stage runs in its own process group. On TERM or INT the driver stops that group (processes it started),
#   then writes the end line.
# - E83_ROOT, E83_QUEUE and E83_PY exist only for tests/test_e83_driver.py; the real run sets none of them.
set -uo pipefail
ROOT="${E83_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
Q="${E83_QUEUE:-bash scripts/seq_queue.sh}"
P="${E83_PY:-.venv-dl/bin/python}"
cd "$ROOT" || exit 1
ENDF="$ROOT/logs/e83_end.log"
BRANCH="stopped before the smoke"
CHILD=""
log() { echo "$(date -u +%FT%TZ) $*"; }
on_exit() {
  local rc=$?
  echo "e83 end: $BRANCH $(date -u +%FT%TZ) exit $rc" >> "$ENDF"
}
on_signal() {  # on_signal NAME CODE
  BRANCH="signal $1"
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM -- "-$CHILD" 2>/dev/null  # the stage's own process group, started by this driver
    wait "$CHILD" 2>/dev/null
  fi
  exit "$2"
}
stage() {  # stage CMD...: run CMD in its own process group and wait, so that a signal reaches the driver at once
  setsid -w "$@" &
  CHILD=$!
  wait "$CHILD"
  local rc=$?
  CHILD=""
  return $rc
}
check() {  # check STAGE OUT: the checker on the CPU; its output is kept with its sha256
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $P scripts/e83_pick.py --stage "$1" > "$2" 2>&1
  local rc=$?
  log "e83_pick --stage $1: exit $rc; $2 $(sha256sum "$2" | cut -c1-8)"
  cat "$2"
  return $rc
}
jobs_in() { [ -f "$1" ] && { grep -vc '^#' "$1" || true; } || echo missing; }
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
for f in logs/e83_end.log logs/e83_smoke_check.out logs/e83smoke_queue.log logs/jobs_e83b.txt logs/jobs_e83c.txt; do
  if [ -e "$f" ]; then
    BRANCH="refused ($f exists)"
    log "$f exists: not overwriting; e83 stops"
    exit 1
  fi
done
log "e83 start (pid $$): smoke (4 tiny jobs)"
BRANCH="smoke running"
stage $Q logs/jobs_e83_smoke.txt logs/e83smoke_queue.log
check smoke logs/e83_smoke_check.out || { BRANCH="smoke failed"; log "smoke failed: e83 stops"; exit 1; }
log "e83a: the screen (5 MultiWOZ jobs at 4k)"
BRANCH="screen running"
stage $Q logs/jobs_e83a.txt logs/e83a_queue.log
check screen logs/e83_pick_screen.out || { BRANCH="screen failed"; log "screen check failed: e83 stops"; exit 1; }
nb=$(jobs_in logs/jobs_e83b.txt)
[ "$nb" != missing ] || { BRANCH="screen wrote no jobs file"; log "no logs/jobs_e83b.txt: e83 stops"; exit 1; }
if [ "$nb" -gt 0 ]; then
  log "e83b: the pick at 16k on MultiWOZ"
  BRANCH="e83b running"
  stage $Q logs/jobs_e83b.txt logs/e83b_queue.log
else
  log "e83b: no job (no pick)"
fi
check mid logs/e83_pick_mid.out || { BRANCH="mid failed"; log "mid check failed: e83 stops"; exit 1; }
nc=$(jobs_in logs/jobs_e83c.txt)
[ "$nc" != missing ] || { BRANCH="mid wrote no jobs file"; log "no logs/jobs_e83c.txt: e83 stops"; exit 1; }
if [ "$nc" -gt 0 ]; then
  log "e83c: the pick on STAR and ABCD"
  BRANCH="e83c running"
  stage $Q logs/jobs_e83c.txt logs/e83c_queue.log
else
  log "e83c: no job (C1 not met or not tested)"
fi
check final logs/e83_pick_final.out || { BRANCH="final failed"; log "final check failed: e83 stops"; exit 1; }
BRANCH="done"
log "e83 GPU stages done"
