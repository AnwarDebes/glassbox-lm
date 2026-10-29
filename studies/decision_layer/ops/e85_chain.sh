#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e85 (plan logs/plan_e85_s_under_random.md): after R24's dry runs (logs/R24_dry_check.out) and an empty GPU, the
# screen, the moved picks at 16k and the final stage (which writes logs/jobs_R24.txt). End line: logs/e85_end.log.
set -uo pipefail
S="${E85_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; Q="${QUEUE_CMD:-bash scripts/seq_queue.sh}"
P="${PICK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/e85_pick.py}"; POLL="${POLL_S:-30}"
log() { echo "$(date -u +%FT%TZ) $*"; }
BR="refused"
for f in logs/e85_end.log logs/e85a_queue.log; do [ -e "$f" ] && { log "$f exists: not starting"; exit 1; }; done
trap 'rc=$?; echo "e85 end: $BR $(date -u +%FT%TZ) exit $rc" >> logs/e85_end.log' EXIT
BR="no R24 dry check after 6 h"; t0=$(date +%s)
until [ -e logs/R24_dry_check.out ]; do [ $(( $(date +%s) - t0 )) -lt 21600 ] || exit 2; sleep "$POLL"; done
BR="GPU busy after 30 min"; t1=$(date +%s)
until [ "$($NVSMI | grep -c .)" -eq 0 ]; do [ $(( $(date +%s) - t1 )) -lt 1800 ] || exit 3; sleep "$POLL"; done
log "e85a starts (R24 dry check: $(tail -n 1 logs/R24_dry_check.out))"
BR="screen running"; $Q logs/jobs_e85a.txt logs/e85a_queue.log
$P --stage screen > logs/e85_pick_screen.out 2>&1 || { BR="screen failed"; tail -n 2 logs/e85_pick_screen.out; exit 1; }
cat logs/e85_pick_screen.out
if [ "$(grep -vc '^#' logs/jobs_e85b.txt || true)" -gt 0 ]; then BR="e85b running"; $Q logs/jobs_e85b.txt logs/e85b_queue.log; fi
$P --stage final > logs/e85_pick_final.out 2>&1 || { BR="final failed"; tail -n 2 logs/e85_pick_final.out; exit 1; }
cat logs/e85_pick_final.out; BR="done"; log "e85 done"
