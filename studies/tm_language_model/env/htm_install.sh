#!/usr/bin/env bash
# Build the SEPARATE environment .venv-htm for the exploratory H-AO pilot on the hierarchical package at
# upstream commit 5b4a223 (docs/12 F27). .venv-tm and every earlier run stay unchanged: this script never
# touches .venv-tm (it only reads nothing from it) and refuses to run when .venv-htm already exists.
#
#     scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/run_logged.py --no-gpu-sampler --name htm_env_install \
#         --config experiments/cvtm/configs/htm_env_install.json -- bash env/htm_install.sh
#
# Recipe (the pycuda part is the one of env/bootstrap.sh): uv venv (Python 3.11); numpy, setuptools, wheel,
# scipy, pytools, mako first, pinned to the versions of env/tm.lock; pycuda 2025.1.3 built from source with
# --no-build-isolation against that numpy; scikit-learn 1.9.1 (the version of .venv-tm); the package from
# GitHub at 5b4a223075c1e621577d508bc7c3c061230e29d5 (NOT from the reference clone third_party/ref/, which is
# read and cited, never imported); this project (tmlm) editable so src/tmlm/ssl/hao_world.py imports.
# Output (in $TMLM_RUN_DIR when run under scripts/run_logged.py): htm_freeze.txt (uv pip freeze), metrics.json
# (package provenance, file hashes, device); env/htm.lock is written as a copy of the freeze.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HTM_COMMIT="5b4a223075c1e621577d508bc7c3c061230e29d5"
OUT="${TMLM_RUN_DIR:-$ROOT/results/_htm_install_unlogged}"
mkdir -p "$OUT"

if [[ -e .venv-htm ]]; then
    echo "ERROR: .venv-htm exists; this script builds it once and never overwrites it." >&2
    exit 1
fi

source "$ROOT/env/env.sh"
set -x
uv --version
which nvcc && nvcc --version | tail -2
g++ --version | head -1
uv python install 3.11
uv venv --python 3.11 .venv-htm
PYH="$ROOT/.venv-htm/bin/python"
# Step 1: build tools and numpy first (pycuda's --no-build-isolation needs them present).
uv pip install --python "$PYH" "numpy==2.4.6" "setuptools==84.0.0" "wheel==0.48.0" \
    "scipy==1.17.1" "pytools==2026.1.1" "mako==1.4.1"
# Step 2: pycuda from source against that numpy.
uv pip install --python "$PYH" --no-build-isolation "pycuda==2025.1.3"
# Step 3: scikit-learn (the .venv-tm version).
uv pip install --python "$PYH" "scikit-learn==1.9.1"
# Step 4: the hierarchical package from GitHub at the pinned commit (its setup.py also pulls scikit-image
# and networkx, unpinned there; the freeze below records what was resolved).
uv pip install --python "$PYH" \
    "PyHierarchicalTsetlinMachineCUDA @ git+https://github.com/cair/PyHierarchicalTsetlinMachineCUDA@${HTM_COMMIT}"
# Step 5: this project, editable.
uv pip install --python "$PYH" -e .
set +x

uv pip freeze --python "$PYH" > "$OUT/htm_freeze.txt"
cp "$OUT/htm_freeze.txt" "$ROOT/env/htm.lock"
echo "# freeze:"
cat "$OUT/htm_freeze.txt"

HTM_COMMIT="$HTM_COMMIT" "$PYH" - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path

import numpy
import pycuda.driver as drv
import scipy
import sklearn

import PyHierarchicalTsetlinMachineCUDA as pkg
import tmlm
from tmlm.common import runlog
from tmlm.ssl import hao_world  # noqa: F401  (must import in this environment)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


d = Path(pkg.__file__).parent
dist = md.distribution("PyHierarchicalTsetlinMachineCUDA")
direct = json.loads(dist.read_text("direct_url.json") or "null")
commit = (direct or {}).get("vcs_info", {}).get("commit_id")
ref = Path("third_party/ref/PyHierarchicalTsetlinMachineCUDA/PyHierarchicalTsetlinMachineCUDA")
files = {}
for name in ("tm.py", "kernels.py", "tools.py", "utils.py", "__init__.py"):
    files[name] = {"installed_sha256": sha(d / name),
                   "reference_clone_sha256": sha(ref / name) if (ref / name).exists() else None}
    files[name]["equal"] = files[name]["installed_sha256"] == files[name]["reference_clone_sha256"]
drv.init()
dev = drv.Device(0)
import PyHierarchicalTsetlinMachineCUDA.tm as htm  # noqa: E402  (creates the CUDA context)
import inspect  # noqa: E402

sig = str(inspect.signature(htm.TsetlinMachine.__init__))
metrics = {
    "package.version": {"value": dist.version, "label": "measured"},
    "package.direct_url": {"value": direct, "label": "measured"},
    "package.commit_matches_pin": {"value": commit == os.environ["HTM_COMMIT"], "label": "measured"},
    "package.location": {"value": str(d), "label": "measured"},
    "package.files": {"value": files, "label": "measured",
                      "note": "reference clone third_party/ref/ at HEAD 5b4a223 hashed (read, not imported)"},
    "package.all_files_equal_reference_clone": {"value": all(v["equal"] for v in files.values()), "label": "measured"},
    "package.TsetlinMachine_signature": {"value": sig, "label": "measured"},
    "env.python": {"value": os.sys.version, "label": "measured"},
    "env.numpy": {"value": numpy.__version__, "label": "measured"},
    "env.scipy": {"value": scipy.__version__, "label": "measured"},
    "env.sklearn": {"value": sklearn.__version__, "label": "measured"},
    "env.pycuda": {"value": md.version("pycuda"), "label": "measured"},
    "env.tmlm_location": {"value": str(Path(tmlm.__file__).parent), "label": "measured"},
    "env.device": {"value": {"name": dev.name(), "cc": list(dev.compute_capability())}, "label": "measured"},
}
print(json.dumps(metrics, indent=1))
assert commit == os.environ["HTM_COMMIT"], commit
assert dev.compute_capability() == (7, 0), "expected a V100 (sm_70)"
if runlog.current_run_dir() is not None:
    runlog.write_metrics(metrics, on_conflict="replace")
PY
echo "# htm_install done"
