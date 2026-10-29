#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Start R24's pinned chain (logs/R24_chain.sh) once R24 is frozen, e85 has ended "done" (it wrote logs/jobs_R24.txt),
# and the GPU shows no compute app. Any other e85 end branch: R24 does not start (exit 4).
set -uo pipefail
S=${REPO_ROOT}/studies/decision_layer; cd "$S" || exit 1
log() { echo "$(date -u +%FT%TZ) $*"; }
t0=$(date +%s)
until [ -e registrations/R24_multiwoz_random_negatives.frozen.json ] && [ -e logs/e85_end.log ]; do
  [ $(( $(date +%s) - t0 )) -lt 43200 ] || { log "no R24 freeze or e85 end after 12 h"; exit 2; }; sleep 60; done
grep -q "^e85 end: done " logs/e85_end.log || { log "e85 did not end done ($(tail -n 1 logs/e85_end.log)): R24 not started"; exit 4; }
[ -e logs/jobs_R24.txt ] || { log "no logs/jobs_R24.txt: R24 not started"; exit 4; }
t1=$(date +%s); until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; do
  [ $(( $(date +%s) - t1 )) -lt 1800 ] || { log "GPU busy after 30 min: R24 not started"; exit 3; }; sleep 20; done
log "R24 starts: $(tail -n 1 logs/e85_end.log)"
bash logs/R24_chain.sh >> logs/R24_chain.log 2>&1; rc=$?
log "R24 chain exit $rc; $(tail -n 1 logs/R24_end.log 2>/dev/null)"; exit $rc
