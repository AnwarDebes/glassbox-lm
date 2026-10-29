REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Environment of every A-Struct job and self-test (sourced; PLAN_part3.md section 6, "Files and run rules").
# Caches and temporary files stay in A/tmp/, two threads everywhere, no GPU.
A=${REPO_ROOT}/studies/language_part3/A
export A
export CUDA_VISIBLE_DEVICES=
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="$A/tmp/numba" TMPDIR="$A/tmp/tmpdir" JOBLIB_TEMP_FOLDER="$A/tmp/joblib"
export MPLCONFIGDIR="$A/tmp/mpl" XDG_CACHE_HOME="$A/tmp/xdg"
export KENLM_BIN=${WS_ROOT}/tools/kenlm/bin
mkdir -p "$NUMBA_CACHE_DIR" "$TMPDIR" "$JOBLIB_TEMP_FOLDER" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
PY_TM=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python   # numpy, numba, sklearn, pytest
PY_DL=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python      # torch (CPU), sklearn
export PY_TM PY_DL
