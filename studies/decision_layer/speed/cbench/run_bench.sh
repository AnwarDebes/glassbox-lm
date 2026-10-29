#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Runs bench_c_clause.py on the three registered seed-1 agent models, one at a time, each after the CPU gate.
set -u
C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this folder: bench_c_clause.py, tmclause.c and the outputs
D=${REPO_ROOT}/studies/decision_layer
PY=$D/.venv-dl/bin/python
for ds in star multiwoz abcd; do
  model=$(jq -r .model ${REPO_ROOT}/studies/agent_vs_llm/runs/cost/agent_${ds}_seed1_cpu.json)
  for try in $(seq 1 30); do
    g=$(cd $D && nice -n 19 $PY -I -B scripts/cpu_load.py 10)
    cores=$(sed -n 's/.* cores \([0-9.]*\) over.*/\1/p' <<< "$g")
    echo "$(date -u +%H:%M:%SZ) $ds gate: $g"
    awk -v c="$cores" 'BEGIN {exit !(c <= 4.5)}' && break
    sleep 60
  done
  nice -n 19 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY -I -B $C/bench_c_clause.py --dataset $ds --model "$model" --lib $C/libtmclause.so --out $C/bench_$ds.json \
    > $C/bench_$ds.log 2>&1
  echo "$(date -u +%H:%M:%SZ) $ds exit $? $(tail -c 300 $C/bench_$ds.log | tr '\n' ' ' | cut -c1-200)"
done
