#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1 autoencoder search, variants 5 and 6 (the last two of the budget; rules fixed 00:5xZ), then the probe.
here=${REPO_ROOT}/studies/text_understanding/w1
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1_dev.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() { tag=$1; shift; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $tag $*" >> "$log"
  "$gpu" shared -- "$py" -I -B "$here/mo_tm_ae.py" --slice S1 --tag "$tag" --rounds 300 --eval-every 10 --patience 5 "$@" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $tag exit $?" >> "$log"; }
run dev_S1_AE_paper_N600 --clauses 600 --T 1200 --s 5 --u 25
run dev_S1_AE_N4000_s20_T2000 --clauses 4000 --T 2000 --s 20 --u 25
"$gpu" shared -- "$py" -I -B "$here/probe_table.py" --slice S1 --table tables/dev_S1_AE_paper_N600.npz --table tables/dev_S1_AE_N4000_s20_T2000.npz >> "$log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dev chain 4 done" >> "$log"
