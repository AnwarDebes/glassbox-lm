#!/usr/bin/env bash
# M5 step 1 grid (docs/prereg_M5_step1.md): the 11 runs after Gate 0 PASSED.
# s in {8, 25, 50, 100} x seeds {42, 123, 456}, sequential, without repeating the
# Gate-0 run (s 8, seed 42). Each run goes through src/m5_run.py (one runlog folder
# per run; m5_run.py refuses reruns). A crash is logged and the grid continues.
set -u
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=$STUDY_DIR/.venv-cdllm/bin/python
LL=$STUDY_DIR
BACKEND="${1:-numba}"   # numba | torch | torch_gpu (prereg amendment 21:11 UTC: torch_gpu)
for s in 8 25 50 100; do
  for seed in 42 123 456; do
    if [ "$s" = 8 ] && [ "$seed" = 42 ]; then continue; fi
    echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) start s=$s seed=$seed backend=$BACKEND"
    if "$PY" "$LL/src/m5_run.py" --s "$s" --seed "$seed" --backend "$BACKEND"; then
      echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) done s=$s seed=$seed"
    else
      echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED s=$s seed=$seed (exit $?)"
    fi
  done
done
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) grid finished"
