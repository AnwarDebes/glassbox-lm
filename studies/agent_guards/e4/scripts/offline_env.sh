#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Run a command OFFLINE in the E4 setting: no GPU, no API keys, every outbound HTTP to a dead local proxy,
# at most 2 threads, nice 19, temp files inside the E4 folder. Usage: offline_env.sh <cmd> [args...]
# Revision 3: every variable whose name ends in _API_KEY, _API_BASE or _BASE_URL is unset (no provider is named).
export CUDA_VISIBLE_DEVICES=
for v in $(compgen -e); do case "$v" in *_API_KEY|*_API_BASE|*_BASE_URL) unset "$v" ;; esac; done
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 NO_PROXY= no_proxy=
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR=${REPO_ROOT}/studies/agent_guards/e4/.tmp
exec nice -n 19 "$@"
