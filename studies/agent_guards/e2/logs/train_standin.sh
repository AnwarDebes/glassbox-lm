#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
cd ${REPO_ROOT}/studies/agent_guards/e2 || exit 1
E="env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false nice -n 19"
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
scripts/wait_cpu.sh 3.5 360; echo "start train $(date -u +%H:%M:%SZ)"
$E $PY -I -B scripts/train_loso.py --rows data/rows_gt_v1.2.2.jsonl data/rows_standin_traces.jsonl --qbits data/query_bits \
   --out models/standin --standin-grid --tag "STAND-IN (GT rows + scripted-replay trace rows in place of Qwen D0 rows; TM grid clauses 200 only)"
echo "train exit $? $(date -u +%H:%M:%SZ)"
scripts/wait_cpu.sh 3.5 360; echo "start milp $(date -u +%H:%M:%SZ)"
$E $PY -I -B scripts/milp_check.py --models models/standin --rows data/rows_gt_v1.2.2.jsonl data/rows_standin_traces.jsonl \
   --qbits data/query_bits --out results/milp_E2.5_standin.jsonl --n-rows 500 --time-limit 600
echo "milp exit $? $(date -u +%H:%M:%SZ)"
scripts/wait_cpu.sh 3.5 360; echo "start d6 $(date -u +%H:%M:%SZ)"
$E $PY -I -B scripts/build_d6.py --proposer scripted --rows data/rows_gt_v1.2.2.jsonl data/rows_standin_traces.jsonl \
   --qbits data/query_bits --out models/standin
echo "d6 exit $? $(date -u +%H:%M:%SZ)"
