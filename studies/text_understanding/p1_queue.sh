#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# p1_queue.sh: registered run P1 (REGISTRATION_P1.md) in one job slot; it starts when fewer than 2 of my jobs run
set -uo pipefail
here=${REPO_ROOT}/studies/text_understanding
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$here/runs_P1.log
busy() { pgrep -c -f -u "$(id -u)" "^\\S*/python3? -I -B ${REPO_ROOT}/studies/text_understanding/(slot_tm|slot_lr|m2v_protocol|tok_tm|flat3_tm)\\.py" || true; }
sha=$(sha256sum "$here/m2v_protocol.py" | cut -c1-16)
echo "$(date -u +%H:%M:%SZ) queued (m2v_protocol.py sha256 $sha)" >> "$log"
while [ "$(busy)" -ge 2 ]; do sleep 60; done
echo "$(date -u +%H:%M:%SZ) start" >> "$log"
for ds in massive_intent banking77 clinc_oos; do
  for d in 1 2 3 4 5; do
    if [ "$(sha256sum "$here/m2v_protocol.py" | cut -c1-16)" != "$sha" ]; then echo "$(date -u +%H:%M:%SZ) STOP: m2v_protocol.py changed" >> "$log"; exit 1; fi
    echo "$(date -u +%H:%M:%SZ) run $ds draw $d" >> "$log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
      "$gpu" shared -- "$py" -I -B "$here/m2v_protocol.py" --dataset "$ds" --draw "$d" >> "$log" 2>> "$here/runs_P1.err"
  done
done
echo "$(date -u +%H:%M:%SZ) P1 done" >> "$log"
