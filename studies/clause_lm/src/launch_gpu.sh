#!/usr/bin/env bash
# Run one GPU job for clause_lm (study rule 3: thesis first).
# Waits for the shared GPU lock, then holds it EXCLUSIVELY, so it never overlaps a thesis job:
#   timeout -> mem_guard (12 GiB) -> gpu_run.sh exclusive -> .venv-htm python
# usage: launch_gpu.sh MINUTES SCRIPT.py [args...]
set -euo pipefail
MIN="$1"; shift
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_DIR/../tm_language_model"
source env/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
exec timeout --kill-after=60 "${MIN}m" scripts/mem_guard.py --scope group --limit-gib 12 -- \
  scripts/gpu_run.sh exclusive -- .venv-htm/bin/python -u "$@"
