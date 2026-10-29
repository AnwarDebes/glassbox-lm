#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# micromamba 2.9.0 (read-only binary in tools/) with every write kept inside this folder:
# root prefix + package cache under data_untrusted/mamba-root, HOME/XDG/TMPDIR under tmp/.
# Pinned to 2 CPUs, nice 19, no GPU, no rc files, conda-forge only (the caller passes -c conda-forge --override-channels).
set -euo pipefail
F=${REPO_ROOT}/studies/agent_guards/miniwob
MM=${WS_ROOT}/tools/micromamba/bin/micromamba
export CUDA_VISIBLE_DEVICES=
export MAMBA_ROOT_PREFIX="$F/data_untrusted/mamba-root"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
export HOME="$F/tmp/mamba-home"
export XDG_CONFIG_HOME="$HOME/.config" XDG_CACHE_HOME="$HOME/.cache" XDG_DATA_HOME="$HOME/.local/share" XDG_STATE_HOME="$HOME/.local/state"
export TMPDIR="$F/tmp"
export MAMBA_EXTRACT_THREADS=2
unset CONDARC MAMBARC CONDA_PREFIX || true
mkdir -p "$MAMBA_ROOT_PREFIX" "$HOME" "$TMPDIR"
cmd="$1"; shift
exec taskset -c 0,1 nice -n 19 "$MM" "$cmd" --no-rc "$@"
