#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e86 (plan logs/plan_e86_clause_drop.md), started by hand on the queue's go:
#   e86_chain.sh dry | run | both
# Before each start: the plan's freeze record holds the plan's hash; the handoff rule: process $WAIT_PID gone
# (ps -p), then 3 empty-GPU checks in a row. dry: the dry queue, then e86_check.py --dry (end line in
# logs/e86_dry_end.log). run: needs logs/e86_dry_check.out ending "all dry checks passed"; the 14 jobs, then
# e86_check.py (end line in logs/e86_end.log). both: dry, then run without a new handoff.
set -uo pipefail
S="${E86_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; Q="${QUEUE_CMD:-bash scripts/seq_queue.sh}"
C="${CHECK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/e86_check.py}"
PSC="${PS_CMD:-ps -p}"; WAIT_PID="${WAIT_PID:-3690999}"; POLL="${POLL_S:-20}"; PLAN=logs/plan_e86_clause_drop
log() { echo "$(date -u +%FT%TZ) $*"; }
STAGE="${1:-}"; case "$STAGE" in dry|run|both) ;; *) echo "usage: e86_chain.sh dry|run|both"; exit 1;; esac
END=logs/e86_end.log; [ "$STAGE" = run ] || END=logs/e86_dry_end.log
for f in "$END" $([ "$STAGE" = run ] || echo logs/e86_dry_queue.log) $([ "$STAGE" = dry ] || echo logs/e86_end.log logs/e86_queue.log); do
  [ -e "$f" ] && { log "$f exists: not starting"; exit 1; }
done
BR="refused"; trap 'rc=$?; echo "e86 $([ "$END" = logs/e86_end.log ] && echo end || echo dry end): $BR $(date -u +%FT%TZ) exit $rc" >> "$END"' EXIT
[ -e "$PLAN.frozen.json" ] && [ "$(sha256sum "$PLAN.md" | cut -d' ' -f1)" = "$(jq -r .sha256 "$PLAN.frozen.json")" ] \
  || { BR="refused (no freeze record of the plan's text)"; exit 1; }
if [ "$STAGE" = run ]; then [ "$(tail -n 1 logs/e86_dry_check.out 2>/dev/null)" = "all dry checks passed" ] || { BR="refused (the dry checks have not passed)"; exit 1; }; fi
BR="handoff: process $WAIT_PID still there after 6 h"; t0=$(date +%s)
while $PSC "$WAIT_PID" > /dev/null 2>&1; do [ $(( $(date +%s) - t0 )) -lt 21600 ] || exit 2; sleep "$POLL"; done
BR="handoff: no 3 empty-GPU checks in a row within 30 min"; t1=$(date +%s); k=0
while [ "$k" -lt 3 ]; do
  [ $(( $(date +%s) - t1 )) -lt 1800 ] || exit 3
  if [ "$($NVSMI | grep -c .)" -eq 0 ]; then k=$((k + 1)); else k=0; fi
  [ "$k" -lt 3 ] && sleep "$POLL"
done
log "handoff done: $WAIT_PID gone, 3 empty-GPU checks"
if [ "$STAGE" != run ]; then
  BR="dry run running"; log "e86 dry run"; $Q logs/jobs_e86_dry.txt logs/e86_dry_queue.log 4.5
  $C --dry > logs/e86_dry_check.out 2>&1 || { BR="dry check failed"; tail -n 2 logs/e86_dry_check.out; exit 1; }
  tail -n 1 logs/e86_dry_check.out; BR="done"
  [ "$STAGE" = dry ] && { log "e86 dry done"; exit 0; }
  echo "e86 dry end: done $(date -u +%FT%TZ) exit 0" >> "$END"; END=logs/e86_end.log; BR="refused"
  [ -e "$END" ] && { BR="refused ($END exists)"; exit 1; }
fi
BR="queue running"; log "e86 queue"; $Q logs/jobs_e86.txt logs/e86_queue.log 4.5
$C > logs/e86_check.out 2>&1 || { BR="check failed"; tail -n 2 logs/e86_check.out; exit 1; }
cat logs/e86_check.out; BR="done"; log "e86 done; logs/e86_check.out $(sha256sum logs/e86_check.out | cut -c1-8)"
