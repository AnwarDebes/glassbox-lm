#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
cd ${REPO_ROOT}/studies/agent_guards/e2 || exit 1
E="env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false nice -n 19"
DL=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
AD=../agentdojo/.venv/bin/python
scripts/wait_cpu.sh 3.5 360; echo "start twins $(date -u +%H:%M:%SZ)"
$E $DL -I -B scripts/check_twins.py --models models/standin --rows data/rows_gt_v1.2.2.jsonl data/rows_standin_traces.jsonl --qbits data/query_bits
scripts/wait_cpu.sh 3.5 360; echo "start e2e $(date -u +%H:%M:%SZ)"
$E $AD -I -B scripts/offline_e2e.py --models models/standin --qbits data/query_bits --out results/offline_e2e_standin.json
echo "e2e exit $? $(date -u +%H:%M:%SZ)"
scripts/wait_cpu.sh 3.5 360; echo "start timing $(date -u +%H:%M:%SZ)"
$E $DL -I -B scripts/time_fit.py data/rows_gt_v1.2.2.jsonl data/rows_standin_traces.jsonl
echo "done $(date -u +%H:%M:%SZ)"
