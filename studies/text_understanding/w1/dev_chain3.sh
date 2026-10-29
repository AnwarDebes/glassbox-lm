#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 autoencoder search, variant 4 (rules fixed 00:5xZ): u 25, no budget, 1,000 rounds, then the probe.
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_dev.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start dev_S1_AE_u25_L0_r1000 (variant 4)" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/mo_tm_ae.py" --slice S1 --tag dev_S1_AE_u25_L0_r1000 --rounds 1000 --eval-every 50 --patience 0 >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end dev_S1_AE_u25_L0_r1000 exit $?" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/probe_table.py" --slice S1 --table tables/dev_S1_AE_u25_L0_r1000.npz >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dev chain 3 done" >> "$log"
