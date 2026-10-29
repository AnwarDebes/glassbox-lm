#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
cd ${REPO_ROOT}/studies/agent_guards/e2 || exit 1
E="env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false nice -n 19"
PY=../agentdojo/.venv/bin/python
scripts/wait_cpu.sh; echo "start gt $(date -u +%H:%M:%SZ)"
$E $PY -I -B scripts/make_rows.py --out data/rows_gt_v1.2.2.jsonl 2>&1 | tail -6
scripts/wait_cpu.sh; echo "start standin $(date -u +%H:%M:%SZ)"
$E $PY -I -B scripts/traces_to_rows.py --source standin_trace --out data/rows_standin_traces.jsonl ${REPO_ROOT}/studies/agent_guards/agentdojo/offline_runs/v1.2.2/local-replay data/standin_traces 2>&1 | tail -30
echo "done $(date -u +%H:%M:%SZ)"
