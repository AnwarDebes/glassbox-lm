#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 engine search under the rule of 01:1xZ (LOG.md): SGNS diagnostic at S1; variant 6 = PPMI twin at S10;
# variant 5 = the best S1 Tsetlin variant at S10; probes and the neighbour sanity check. One job at a time.
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_dev.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
g() { "$gpu" shared -- "$py" -I -B "$@" >> "$log" 2>&1; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $(basename "$1") exit $?" >> "$log"; }
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start SGNS S1 (diagnostic)" >> "$log"
g "$here/sgns.py" --slice S1 --tag dev_SGNS_S1 --features 12000
g "$here/probe_table.py" --slice S1 --table tables/dev_SGNS_S1.npz
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start PPMI twin S10 (variant 6)" >> "$log"
g "$here/ppmi_svd.py" --slice S10 --tag dev_PP_S10 --dims 500 --eigs 0.5 --features 12000
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start TM S10 (variant 5)" >> "$log"
g "$here/mo_tm_ae.py" --slice S10 --tag dev_S10_AE_u25_L0 --rounds 1000 --eval-every 50 --patience 3
g "$here/probe_table.py" --slice S10 --table tables/dev_PP_S10_d500_e0.5.npz --table tables/dev_S10_AE_u25_L0.npz --table potion
"$py" -I -B "$here/sanity_check.py" dev_S10_AE_u25_L0 dev_PP_S10_d500_e0.5 potion-base-32M dev_S1_AE_u25_L0_r1000 dev_PP_S1_d500_e0.5 dev_SGNS_S1 >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dev chain 5 done" >> "$log"
