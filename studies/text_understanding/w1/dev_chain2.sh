#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 engine development 2 (no evaluation data; S1 train slice + held-out C4 only): the autoencoder task
# (mo_tm_ae.py), the count twin at S1, and the label-free table probe for all S1 tables and potion. Log runs_w1_dev.log.
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_dev.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ae() { tag=$1; shift; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $tag $*" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/mo_tm_ae.py" --slice S1 --tag "$tag" --patience 0 "$@" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $tag exit $?" >> "$log"; }
ae dev_S1_AE_u25_L0 --rounds 30 --eval-every 5
ae dev_S1_AE_u25_L8 --rounds 30 --eval-every 5 --max-literals 8
ae dev_S1_AE_u10_L8 --rounds 30 --eval-every 5 --max-literals 8 --u 10
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start dev_PP_S1" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/ppmi_svd.py" --slice S1 --tag dev_PP_S1 --dims 300,500 --eigs 0.5 --features 12000 >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start probes" >> "$log"
"$gpu" shared -- "$py" -I -B "$here/probe_table.py" --slice S1 --table potion --table tables/dev_PP_S1_d500_e0.5.npz \
  --table tables/dev_PP_S1_d300_e0.5.npz --table tables/dev_S1_F12k_L4_b256.npz --table tables/dev_S1_learncheck.npz \
  --table tables/dev_S1_AE_u25_L0.npz --table tables/dev_S1_AE_u25_L8.npz --table tables/dev_S1_AE_u10_L8.npz >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dev chain 2 done" >> "$log"
