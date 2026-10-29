#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Start e84 (plan logs/plan_e84_negatives_fair.md) on the R3c end line, as decided at 10:51Z:
# wait for a NEW "r3c chain end:" line in R3C_LOG (only bytes written after this script starts count), then for
# nvidia-smi to show no compute app, then run the plan's two commands and append the end line to logs/e84_end.log.
# The overrides exist only for the dry test (test_e84_wait_start.sh).
set -uo pipefail
S="${E84_ROOT:-${REPO_ROOT}/studies/decision_layer}"
R3C_LOG="${R3C_LOG:-${REPO_ROOT}/studies/text_understanding/tu/r3c/runs_r3c.log}"
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"
QUEUE="${QUEUE_CMD:-bash scripts/seq_queue.sh logs/jobs_e84.txt logs/e84_queue.log}"
CHECK="${CHECK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/e84_check.py}"
POLL="${POLL_S:-30}"; MAXW="${MAX_WAIT_S:-18000}"; GPUW="${MAX_GPU_WAIT_S:-1800}"
cd "$S" || exit 1
log() { echo "$(date -u +%FT%TZ) $*"; }
end() { echo "e84 end: $1 $(date -u +%FT%TZ) exit $2" >> logs/e84_end.log; exit "$2"; }
[ -e logs/e84_end.log ] || [ -e logs/e84_queue.log ] && { log "an e84 output exists: not starting"; exit 1; }
off=$( [ -f "$R3C_LOG" ] && stat -c %s "$R3C_LOG" || echo 0 )
log "waiting for a new 'r3c chain end:' line in $R3C_LOG (from byte $off)"
t0=$(date +%s)
until [ -f "$R3C_LOG" ] && tail -c +"$((off + 1))" "$R3C_LOG" | grep -q "r3c chain end:"; do
  [ $(( $(date +%s) - t0 )) -lt "$MAXW" ] || { log "no r3c end line after $MAXW s: e84 not started"; exit 2; }
  sleep "$POLL"
done
log "seen: $(tail -c +"$((off + 1))" "$R3C_LOG" | grep "r3c chain end:" | tail -n 1)"
t1=$(date +%s)
until [ "$($NVSMI | grep -c .)" -eq 0 ]; do
  [ $(( $(date +%s) - t1 )) -lt "$GPUW" ] || { log "a compute app still on the GPU after $GPUW s: e84 not started"; exit 3; }
  sleep "$POLL"
done
log "GPU free (no compute app): e84 starts: $QUEUE"
$QUEUE
log "queue finished; the checker"
$CHECK > logs/e84_check.out 2>&1
rc=$?
log "e84_check exit $rc; logs/e84_check.out $(sha256sum logs/e84_check.out | cut -c1-8)"
cat logs/e84_check.out
[ "$rc" -eq 0 ] && end done 0 || end "check failed" 1
