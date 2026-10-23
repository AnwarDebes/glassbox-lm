#!/usr/bin/env bash
# e81's GPU stages back to back (frozen plan logs/plan_e81_T_screen.md, 2026-10-21T05:47:50Z; this file is pinned by
# the dated note logs/plan_e81_note_2026-10-21.md). Started by hand after step 3c of the text-understanding line was done:
#   nohup bash logs/e81_driver.sh > logs/e81_driver.log 2>&1 &
# It runs the frozen job files one job at a time through scripts/seq_queue.sh and the frozen checker
# scripts/e81_pick.py between them, and stops when a check fails. The three overrides below exist only for
# tests/test_e81_driver.py (a stub queue on made-up rows in a temporary folder); the real run sets none of them.
set -uo pipefail
ROOT="${E81_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
Q="${E81_QUEUE:-bash scripts/seq_queue.sh}"
P="${E81_PY:-.venv-dl/bin/python}"
cd "$ROOT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*"; }
check() {  # check STAGE OUT: the frozen checker on the CPU; its output is kept with its sha256
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 $P scripts/e81_pick.py --stage "$1" > "$2" 2>&1
  local rc=$?
  log "e81_pick --stage $1: exit $rc; $2 $(sha256sum "$2" | cut -c1-8)"
  return $rc
}
log "e81 smoke (3 tiny jobs)"
$Q logs/jobs_e81_smoke.txt logs/e81smoke_queue.log
check smoke logs/e81_smoke_check.out || { tail -n 4 logs/e81_smoke_check.out; log "smoke failed: e81 stops"; exit 1; }
cat logs/e81_smoke_check.out
log "e81a: 18 screens and 5 LR runs"
$Q logs/jobs_e81a.txt logs/e81a_queue.log
check screen logs/e81_pick_screen.out || { tail -n 4 logs/e81_pick_screen.out; log "screen failed: e81 stops"; exit 1; }
cat logs/e81_pick_screen.out
if [ "$(grep -vc '^#' logs/jobs_e81b.txt)" -gt 0 ]; then
  log "e81b: the follow-ups"
  $Q logs/jobs_e81b.txt logs/e81b_queue.log
else
  log "e81b: no follow-ups (no pick moved)"
fi
check final logs/e81_pick_final.out || { tail -n 4 logs/e81_pick_final.out; log "final failed: e81 stops"; exit 1; }
cat logs/e81_pick_final.out
log "e81 GPU stages done"
