REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Sourced by the wrappers below. Environment for the BROWSER (and its driver) process only:
# the conda-forge userspace libraries + fonts, and HOME/cache/TMPDIR inside this folder.
F=${REPO_ROOT}/studies/agent_guards/miniwob
E="$F/data_untrusted/mamba-root/envs/browser-libs"
export LD_LIBRARY_PATH="$E/lib"
export FONTCONFIG_FILE="$E/etc/fonts/fonts.conf"
export FONTCONFIG_PATH="$E/etc/fonts"
export HOME="$F/tmp/browser-home"
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
export TMPDIR="$F/tmp"
export CUDA_VISIBLE_DEVICES=
