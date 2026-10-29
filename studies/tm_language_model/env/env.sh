# shellcheck shell=bash
# tm_language_model environment. Source it, do not execute it:
#
#     source env/env.sh
#
# Properties:
#   - bash only (uses BASH_SOURCE).
#   - Idempotent: sourcing it twice leaves PATH and LD_LIBRARY_PATH unchanged.
#   - Safe under `set -euo pipefail`.
#
# Thread caps: this container has a cgroup quota of 6 CPUs [stated: design notes],
# although nproc reports the host core count. Every thread pool is capped at 6.
# scripts/cpu_slot.sh exports TMLM_CPU_SLOTS=N (N in 1..6) before sourcing this file,
# in which case the cap becomes N, so a job never uses more threads than the slots
# it holds. Min rule: a *_NUM_THREADS value that already is an integer in 1..cap is
# KEPT (a caller may always ask for fewer threads), anything else becomes the cap.
# src/tmlm/common/threadcaps.py is the same policy for processes started from Python.

_tmlm_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMLM_ROOT="$(cd "${_tmlm_env_dir}/.." && pwd)"
export TMLM_ROOT

# _tmlm_prepend VAR DIR: put DIR first in the colon separated list VAR,
# unless DIR is already an element of it.
_tmlm_prepend() {
    local _var="$1" _dir="$2" _cur
    _cur="${!_var:-}"
    case ":${_cur}:" in
        *":${_dir}:"*) ;;
        *) printf -v "${_var}" '%s' "${_dir}${_cur:+:${_cur}}" ;;
    esac
    export "${_var?}"
}

export CUDA_ROOT="/usr/local/cuda"
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"

_tmlm_prepend PATH "/usr/local/cuda/bin"
_tmlm_prepend PATH "${HOME}/.local/bin"
_tmlm_prepend LD_LIBRARY_PATH "/usr/local/cuda/lib64"

# The helper scripts (scripts/run_logged.py, scripts/mem_guard.py, scripts/env_check.py,
# scripts/check_numbers_have_logs.py) start with `#!/usr/bin/env python3`, and the launch lines
# start them by name. The machine of the study has no python3 of its own on PATH (added
# 2026-10-25, finish pass, SHR-1: before, every start by name failed with exit status 127), so
# the folder of the study interpreter comes first, once .venv-tm is built. This also makes
# `python` and `python3` resolve to .venv-tm for everything started after sourcing this file.
if [ -x "${TMLM_ROOT}/.venv-tm/bin/python3" ]; then
    _tmlm_prepend PATH "${TMLM_ROOT}/.venv-tm/bin"
fi

_tmlm_threads="${TMLM_CPU_SLOTS:-6}"
case "${_tmlm_threads}" in
    [1-6]) ;;
    *) _tmlm_threads=6 ;;
esac

# _tmlm_cap VAR: keep VAR when it is an integer in 1..cap, else set it to the cap.
_tmlm_cap() {
    local _cur="${!1:-}"
    case "${_cur}" in
        [1-6]) [ "${_cur}" -le "${_tmlm_threads}" ] || _cur="${_tmlm_threads}" ;;
        *) _cur="${_tmlm_threads}" ;;
    esac
    printf -v "$1" '%s' "${_cur}"
    export "${1?}"
}

_tmlm_cap OMP_NUM_THREADS
_tmlm_cap MKL_NUM_THREADS
_tmlm_cap OPENBLAS_NUM_THREADS
_tmlm_cap NUMEXPR_NUM_THREADS
_tmlm_cap NUMBA_NUM_THREADS
_tmlm_cap RAYON_NUM_THREADS
export TOKENIZERS_PARALLELISM="false"

# Reproducibility: fix str hash randomization so set and dict iteration order over
# strings is identical between runs (protocol rule 4). Respect an explicit caller value.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

unset -f _tmlm_prepend _tmlm_cap
unset _tmlm_env_dir _tmlm_threads
