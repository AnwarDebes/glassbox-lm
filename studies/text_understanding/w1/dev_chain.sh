#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 engine development (no evaluation data; S1 train slice + the held-out C4 check only): does the engine learn
# with dl/tm.py's literal budget? One job at a time, shared GPU lock. Log w1/runs_w1_dev.log.
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_dev.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() { tag=$1; shift; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $tag $*" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/mo_tm.py" --slice S1 --tag "$tag" --epochs 1 --patience 0 "$@" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $tag exit $?" >> "$log"; }
run dev_S1_F12k_L4_b1024 --features 12000 --max-literals 4 --batch 1024 --eval-every 115
run dev_S1_F12k_L4_b256 --features 12000 --max-literals 4 --batch 256 --eval-every 460
run dev_S1_F12k_L8_b1024 --features 12000 --max-literals 8 --batch 1024 --eval-every 115
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dev chain done" >> "$log"
