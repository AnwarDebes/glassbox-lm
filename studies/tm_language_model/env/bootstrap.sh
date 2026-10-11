#!/usr/bin/env bash
# tm_language_model bootstrap: build both virtual environments from a clean checkout, userspace only.
#
#     env/bootstrap.sh            # refuses to touch an existing .venv-tm or .venv-dl
#     env/bootstrap.sh --fresh    # removes both environments first (asks for the word yes)
#     env/bootstrap.sh --skip-dl  # only .venv-tm (the Tsetlin stack)
#
# Design decision; research protocol, "Environment setup"; docs/12 A3 and E4. Every command and
# its output are logged to results/_bootstrap/<UTC stamp>_bootstrap/bootstrap.log, and a
# manifest with `uv pip freeze` of both environments, the sha256 of this script, and the
# toolchain versions is written next to it. Nothing here needs root. Versions are pinned to
# exactly what was measured as installed on 2026-09-19 [measured: results/_bootstrap/*.log],
# so a successful run of this script is the reproducible recipe (pre-registration H2.1).
#
# Order matters: numpy, setuptools and wheel BEFORE pycuda, and pycuda is built from source
# with --no-build-isolation against that numpy (uv venv does not seed pip/setuptools/wheel).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FRESH=0; SKIP_DL=0
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=1 ;;
        --skip-dl) SKIP_DL=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$ROOT/results/_bootstrap/${STAMP}_bootstrap"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/bootstrap.log") 2>&1
echo "# bootstrap start $STAMP in $ROOT (args: $*)"

if [[ -d .venv-tm || -d .venv-dl ]]; then
    if [[ $FRESH -eq 1 ]]; then
        read -r -p "Remove .venv-tm and .venv-dl and rebuild from scratch? type yes: " ans
        [[ "$ans" == "yes" ]] || { echo "aborted"; exit 1; }
        rm -rf .venv-tm .venv-dl
    else
        echo "ERROR: .venv-tm or .venv-dl exists. Use --fresh to rebuild, or run from a clean checkout." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------- toolchain
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv not found: installing the standalone binary into ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
# shellcheck disable=SC1091
source "$ROOT/env/env.sh"
set -x
uv --version
which nvcc && nvcc --version | tail -2
g++ --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null || true
uv python install 3.11

# ---------------------------------------------------------------- .venv-tm (Tsetlin stack)
uv venv --python 3.11 .venv-tm
PYTM="$ROOT/.venv-tm/bin/python"
# Step 1: build tools and numpy first (pycuda's --no-build-isolation needs them present).
uv pip install --python "$PYTM" "numpy==2.4.6" "setuptools==84.0.0" "wheel==0.48.0" \
    "scipy==1.17.1" "pytools==2026.1.1" "mako==1.4.1"
# Step 2: pycuda from source against that numpy (CAIR's GraphTM pin: pycuda >=2025.1.2,<2026).
uv pip install --python "$PYTM" --no-build-isolation "pycuda==2025.1.3"
# Step 3: tests.
uv pip install --python "$PYTM" "pytest==9.1.1"
# Step 4: the rest of the Tsetlin stack. tmu is pinned to the upstream commit 5605ff07 (2026-08-03), not the
# 0.8.3 wheel: the wheel (2024-03-25) calls np.uint32(~0), which NumPy 2 rejects with OverflowError, so no
# tmu classifier can be constructed with it; commit 62c0e6a6 (2024-11-22) fixed that upstream and the CPU
# paths we use are otherwise unchanged (docs/02 V2). Built from source with cffi (gcc present).
uv pip install --python "$PYTM" "tmu @ git+https://github.com/cair/tmu.git@5605ff070a18549328028c907a9acf68e063346e" \
    "cupy-cuda12x==14.2.0" "numba==0.67.0" \
    "z3-solver==5.1.0.0" "python-sat==1.8.dev17" "codecarbon==3.3.1" "nvidia-ml-py==13.610.43" \
    "tokenizers==0.23.2" "datasets==5.0.1" "nltk==3.10.3"
# Step 5: GraphTM as an EDITABLE install of the vendored master (never edit that tree), the
# hierarchical package pinned to the commit whose lines docs/02 V3 cites, and this project.
uv pip install --python "$PYTM" -e ./third_party/GraphTsetlinMachine \
    "git+https://github.com/cair/PyHierarchicalTsetlinMachineCUDA.git@c85ca4e70af90511d6b4f8e8eedccd23a7d7d0ec"
uv pip install --python "$PYTM" -e .
"$PYTM" - <<'PY'
import importlib
for m in ("numpy", "pycuda", "cupy", "numba", "tmu", "z3", "pysat", "codecarbon", "pynvml",
          "tokenizers", "datasets", "nltk", "tmlm"):
    importlib.import_module(m)
import pycuda.driver as drv
drv.init()
dev = drv.Device(0)
cc = dev.compute_capability()
print("pycuda device:", dev.name(), "cc", cc)
assert cc == (7, 0), "expected a V100 (sm_70)"
PY

# ---------------------------------------------------------------- .venv-dl (neural baselines, spaCy)
if [[ $SKIP_DL -eq 0 ]]; then
    uv venv --python 3.11 .venv-dl
    PYDL="$ROOT/.venv-dl/bin/python"
    uv pip install --python "$PYDL" "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python "$PYDL" "transformers==5.17.0" "accelerate==1.15.0" "datasets==5.0.1" \
        "codecarbon==3.3.1" "nvidia-ml-py==13.610.43" "numpy==2.4.6" "pytest==9.1.1" "spacy==3.8.16" \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
    # REQUIRED: the last torch wheel line with Volta kernels.
    "$PYDL" -c "import torch; al = torch.cuda.get_arch_list(); print('torch arch list', al); assert 'sm_70' in al"
    "$PYDL" -c "import spacy; spacy.load('en_core_web_sm'); print('spacy ok')"
fi

# ---------------------------------------------------------------- manifest
set +x
uv pip freeze --python "$PYTM" > "$LOGDIR/tm_freeze.txt"
[[ $SKIP_DL -eq 0 ]] && uv pip freeze --python "$ROOT/.venv-dl/bin/python" > "$LOGDIR/dl_freeze.txt"
cp "$LOGDIR/tm_freeze.txt" env/tm.lock
[[ $SKIP_DL -eq 0 ]] && cp "$LOGDIR/dl_freeze.txt" env/dl.lock
{
    echo "script_sha256 $(sha256sum "$0" | cut -d' ' -f1)"
    echo "uv $(uv --version)"
    echo "nvcc $(nvcc --version | tail -1)"
    echo "gxx $(g++ --version | head -1)"
    echo "python_tm $("$PYTM" -c 'import sys; print(sys.version.split()[0])')"
    echo "finished_utc $(date -u +%Y%m%dT%H%M%SZ)"
} > "$LOGDIR/MANIFEST.txt"
echo "# bootstrap OK: log and manifest in $LOGDIR; lock files env/tm.lock env/dl.lock"
