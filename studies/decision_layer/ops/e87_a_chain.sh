#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e87 Part A (plan logs/plan_e87_receipts.md):
#   e87_a_chain.sh a0   GPU: waits until every pid in $WAIT_PIDS is gone (default: the decision_layer e86 chain 3929234 and the
#                       E4 waiter 3952401), then 3 empty-GPU checks in a row; runs scripts/e87_bits.py for
#                       abcd, star, multiwoz (stops at the first failure); end line in logs/e87_a0_end.log; exits.
#   e87_a_chain.sh a    CPU: waits for "e87 A0 end: done", then for each dataset waits for the CPU gate (at most 4.5
#                       cores over 10 s) and runs scripts/e87_readability.py; end line in logs/e87_a_end.log.
# Both refuse without the plan's freeze record or when their end log exists.
set -uo pipefail
S="${E87_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; PSC="${PS_CMD:-ps -p}"
WAIT_PIDS="${WAIT_PIDS:-3929234 3952401}"; POLL="${POLL_S:-20}"; DSETS="${DATASETS:-abcd star multiwoz}"
A0="${A0_CMD:-env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/e87_bits.py --dataset}"
A="${A_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python scripts/e87_readability.py --dataset}"
LOAD="${LOAD_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python -I scripts/cpu_load.py 10}"
PLAN=logs/plan_e87_receipts; STAGE="${1:-}"
log() { echo "$(date -u +%FT%TZ) $*"; }
case "$STAGE" in a0) END=logs/e87_a0_end.log; NAME="e87 A0 end";; a) END=logs/e87_a_end.log; NAME="e87 A end";; *) echo "usage: e87_a_chain.sh a0|a"; exit 1;; esac
[ -e "$END" ] && { log "$END exists: not starting"; exit 1; }
BR="refused"; trap 'rc=$?; echo "$NAME: $BR $(date -u +%FT%TZ) exit $rc" >> "$END"' EXIT
[ -e "$PLAN.frozen.json" ] && [ "$(sha256sum "$PLAN.md" | cut -d' ' -f1)" = "$(jq -r .sha256 "$PLAN.frozen.json")" ] \
  || { BR="refused (no freeze record of the plan's text)"; exit 1; }
if [ "$STAGE" = a0 ]; then
  BR="handoff: a process in '$WAIT_PIDS' still there after 12 h"; t0=$(date +%s)
  for p in $WAIT_PIDS; do while $PSC "$p" > /dev/null 2>&1; do [ $(( $(date +%s) - t0 )) -lt 43200 ] || exit 2; sleep "$POLL"; done; done
  BR="handoff: no 3 empty-GPU checks in a row within 30 min"; t1=$(date +%s); k=0
  while [ "$k" -lt 3 ]; do
    [ $(( $(date +%s) - t1 )) -lt 1800 ] || exit 3
    if [ "$($NVSMI | grep -c .)" -eq 0 ]; then k=$((k + 1)); else k=0; fi
    [ "$k" -lt 3 ] && sleep "$POLL"
  done
  log "handoff done: $WAIT_PIDS gone, 3 empty-GPU checks"
  for ds in $DSETS; do
    BR="A0 running ($ds)"; log "A0 $ds"
    $A0 "$ds" >> logs/e87_bits.out 2>&1 || { BR="A0 failed ($ds)"; tail -n 2 logs/e87_bits.out; exit 1; }
  done
  BR="done"; log "A0 done; the GPU is free"; exit 0
fi
BR="no 'e87 A0 end: done' within 12 h"; t0=$(date +%s)
until tail -n 1 logs/e87_a0_end.log 2>/dev/null | grep -q '^e87 A0 end: done '; do
  tail -n 1 logs/e87_a0_end.log 2>/dev/null | grep -q '^e87 A0 end: ' && { BR="refused (A0 did not end done)"; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 43200 ] || exit 2; sleep "$POLL"
done
for ds in $DSETS; do
  BR="CPU gate never at or below 4.5 cores for 2 h ($ds)"; t1=$(date +%s)
  until awk '{exit !($3 + 0 <= 4.5)}' <<< "$($LOAD)"; do [ $(( $(date +%s) - t1 )) -lt 7200 ] || exit 3; sleep "$POLL"; done
  BR="A running ($ds)"; log "A $ds"
  $A "$ds" >> logs/e87_readability.out 2>&1 || { BR="A failed ($ds)"; tail -n 2 logs/e87_readability.out; exit 1; }
done
BR="done"; log "A done"
