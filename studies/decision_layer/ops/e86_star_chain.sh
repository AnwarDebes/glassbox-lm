#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e86's six STAR jobs after the stop of 2026-10-23 (dated note logs/plan_e86_note_2026-10-25.md), started by hand:
#   e86_star_chain.sh
# Refuses unless the note's freeze record holds the note's text hash, and refuses to start twice (end log, queue log
# or check output already there). Start rule: 3 empty-GPU checks in a row (one big GPU job at a time). Then the six
# job lines of logs/jobs_e86_star.txt through scripts/seq_queue.sh (GPU lock, CPU gate 4.5), then scripts/e86_check.py
# into logs/e86_check.out. End line "e86 star end: <branch> <UTC> exit <code>" in logs/e86_star_end.log.
set -uo pipefail
S="${E86_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; Q="${QUEUE_CMD:-bash scripts/seq_queue.sh}"
C="${CHECK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/e86_check.py}"
POLL="${POLL_S:-20}"; NOTE=logs/plan_e86_note_2026-10-25; END=logs/e86_star_end.log
log() { echo "$(date -u +%FT%TZ) $*"; }
for f in "$END" logs/e86_star_queue.log logs/e86_check.out; do
  [ -e "$f" ] && { log "$f exists: not starting"; exit 1; }
done
BR="refused"; trap 'rc=$?; echo "e86 star end: $BR $(date -u +%FT%TZ) exit $rc" >> "$END"' EXIT
[ -e "$NOTE.frozen.json" ] && [ "$(sha256sum "$NOTE.md" | cut -d' ' -f1)" = "$(jq -r .sha256 "$NOTE.frozen.json")" ] \
  || { BR="refused (no freeze record of the note's text)"; exit 1; }
BR="handoff: no 3 empty-GPU checks in a row within 6 h"; t1=$(date +%s); k=0
while [ "$k" -lt 3 ]; do
  [ $(( $(date +%s) - t1 )) -lt 21600 ] || exit 3
  if [ "$($NVSMI | grep -c .)" -eq 0 ]; then k=$((k + 1)); else k=0; fi
  [ "$k" -lt 3 ] && sleep "$POLL"
done
log "handoff done: 3 empty-GPU checks"
BR="queue running"; log "e86 star queue"; $Q logs/jobs_e86_star.txt logs/e86_star_queue.log 4.5
$C > logs/e86_check.out 2>&1 || { BR="check failed"; tail -n 2 logs/e86_check.out; exit 1; }
cat logs/e86_check.out; BR="done"; log "e86 done; logs/e86_check.out $(sha256sum logs/e86_check.out | cut -c1-8)"
