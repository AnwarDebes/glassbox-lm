#!/usr/bin/env bash
# Run one GPU job for clause_lm (study rule 3: thesis first).
# Waits for the shared GPU lock, then holds it EXCLUSIVELY, so it never overlaps a thesis job:
#   nice 19 -> timeout -> mem_guard (12 GiB) -> gpu_run.sh exclusive -> job python
# usage: launch_gpu.sh MINUTES SCRIPT.py [args...]
#   MINUTES    whole minutes before timeout stops the job (60 s later it is killed); the wait for the GPU lock
#              counts, so start a job when the GPU is free
#   SCRIPT.py  an existing file; a relative path is read from the caller's working directory
#
# Changed 2026-10-25 (finish pass). The first version ran `.venv-htm/bin/python` inside ../tm_language_model,
# which has no such environment, and started scripts/mem_guard.py by its `#!/usr/bin/env python3` line, which
# fails with exit 127 when no python3 is on PATH. Both interpreters are now explicit:
#   CLM_JOB_PYTHON    interpreter of the job. Default: ../decision_layer/.venv-htm/bin/python, which holds
#                     PyHierarchicalTsetlinMachineCUDA 0.3.2 at commit 5b4a223, the commit src/pair_tm.py names.
#                     It belongs to the decision_layer study and is used read-only: nothing is installed into it.
#   CLM_GUARD_PYTHON  interpreter that runs scripts/mem_guard.py (standard library only, Python >= 3.11).
#                     Default: CLM_JOB_PYTHON. It runs with -I -B (isolated, no bytecode written).
#   CLM_GPU_RUN       the GPU lock wrapper. Default: scripts/gpu_run.sh of tm_language_model. Only the tests
#                     replace it, with a stub that takes no lock.
#   CLM_DRY_RUN=1     print the command, one argument per line, and exit 0 without starting it.
# Every job runs at nice 19 with the thread pools of OpenMP, OpenBLAS, MKL, numexpr, numba and rayon at 2 (set after
# env/env.sh, which caps them at 6 only): 2 threads and nice 19 (added 2026-10-25 after review).
# Exit 2: usage error, or an interpreter, script or wrapper that does not exist.
set -euo pipefail
die() { echo "launch_gpu.sh: $*" >&2; exit 2; }
[ "$#" -ge 2 ] || die "usage: launch_gpu.sh MINUTES SCRIPT.py [args...]"
MIN="$1"; shift
case "$MIN" in
  '' | *[!0-9]*) die "MINUTES must be a whole number of minutes, got '$MIN'" ;;
esac
[ "$((10#$MIN))" -ge 1 ] || die "MINUTES must be at least 1, got '$MIN'"
MIN="$((10#$MIN))"
[ -f "$1" ] || die "job script not found: $1"
SCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMLM="$(cd "$STUDY_DIR/../tm_language_model" && pwd)"
JOB_PY="${CLM_JOB_PYTHON:-$STUDY_DIR/../decision_layer/.venv-htm/bin/python}"
GUARD_PY="${CLM_GUARD_PYTHON:-$JOB_PY}"
GPU_RUN="${CLM_GPU_RUN:-$TMLM/scripts/gpu_run.sh}"
[ -x "$JOB_PY" ] || die "job interpreter not found or not executable: $JOB_PY (set CLM_JOB_PYTHON)"
[ -x "$GUARD_PY" ] || die "guard interpreter not found or not executable: $GUARD_PY (set CLM_GUARD_PYTHON)"
[ -f "$TMLM/scripts/mem_guard.py" ] || die "memory guard not found: $TMLM/scripts/mem_guard.py"
[ -x "$GPU_RUN" ] || die "GPU lock wrapper not found or not executable: $GPU_RUN"
cd "$TMLM"
source env/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
cmd=(nice -n 19 timeout --kill-after=60 "${MIN}m" "$GUARD_PY" -I -B "$TMLM/scripts/mem_guard.py" --scope group --limit-gib 12 --
     "$GPU_RUN" exclusive -- "$JOB_PY" -u "$SCRIPT" "$@")
if [ "${CLM_DRY_RUN:-0}" = 1 ]; then
  printf '%s\n' "${cmd[@]}"
  exit 0
fi
exec "${cmd[@]}"
