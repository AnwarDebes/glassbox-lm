#!/usr/bin/env python
"""Tests for c3_cuda_first.py (a plain script, no pytest):

    $PY -I -B tu/tests/test_c3_cuda_first.py --cpu-only   # no GPU use: the launcher refuses when CUDA is hidden from the start
    $PY -I -B tu/tests/test_c3_cuda_first.py              # also the GPU part (a CUDA context for about a second, GPU must be free)

GPU part: (1) the bug: a probe that imports c3_common and then asks for a CUDA tensor fails with "No CUDA GPUs are
available"; (2) the same probe through the launcher gets its tensor on cuda:0; (3) the launcher passes the arguments and
runs the script as __main__. Each run has its own folder in tu/tests/runs/<UTC stamp>_cudafirst/.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import os
import subprocess
import sys
import time
from pathlib import Path

TU = Path(__file__).resolve().parent.parent
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_cudafirst")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


RUNS.mkdir(parents=True, exist_ok=True)
probe = RUNS / "probe.py"
probe.write_text(
    "import sys\n"
    f"sys.path.insert(0, {str(TU)!r})\n"
    "import torch\n"
    "import c3_common  # noqa: F401  (imports build_corpus_c4, which hides the GPU)\n"
    "print('name', __name__, 'argv', sys.argv[1:])\n"
    "x = torch.zeros(1, device='cuda')\n"
    "print('tensor on', x.device)\n")
PY = [sys.executable, "-I", "-B"]
env_hidden = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
p = subprocess.run(PY + [str(TU / "c3_cuda_first.py"), str(probe)], capture_output=True, text=True, timeout=300, env=env_hidden,
                   check=False)
check("the launcher refuses when CUDA is hidden from the start", p.returncode != 0 and "tensor on" not in p.stdout,
      (p.stdout + p.stderr)[-400:])
p = subprocess.run(PY + [str(TU / "c3_cuda_first.py")], capture_output=True, text=True, timeout=60, check=False)
check("the launcher needs a script argument", p.returncode != 0 and "usage" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-200:])
if "--cpu-only" not in sys.argv:
    p = subprocess.run(PY + [str(probe), "a1"], capture_output=True, text=True, timeout=300, check=False)
    check("the bug: importing c3_common hides the GPU", p.returncode != 0 and "No CUDA GPUs are available" in p.stderr,
          (p.stdout + p.stderr)[-400:])
    p = subprocess.run(PY + [str(TU / "c3_cuda_first.py"), str(probe), "a1", "--b", "2"], capture_output=True, text=True,
                       timeout=300, check=False)
    check("the fix: through the launcher the probe gets its CUDA tensor", p.returncode == 0 and "tensor on cuda:0" in p.stdout,
          (p.stdout + p.stderr)[-400:])
    check("the launcher runs the script as __main__ with its arguments", "name __main__ argv ['a1', '--b', '2']" in p.stdout,
          p.stdout[-300:])
    load = RUNS / "load_teacher.py"
    load.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(TU)!r})\n"
        f"sys.path.insert(0, '{_REPO_ROOT}/studies/decision_layer/src')\n"
        "import torch\n"
        "import a3_teacher as TEA\n"
        "import c3_common as C  # noqa: F401\n"
        "from dl.data import load  # noqa: F401\n"
        "from dl.linear import fit_logreg, logits  # noqa: F401\n"
        "TEA.verify_files()\n"
        "tok, model = TEA.load('cuda', torch.float16)\n"
        "q = next(model.parameters())\n"
        "print('teacher on', q.device, q.dtype)\n")
    p = subprocess.run(PY + [str(TU / "c3_cuda_first.py"), str(load)], capture_output=True, text=True, timeout=300, check=False)
    check("the filter's teacher loads on cuda in fp16 through the launcher (c3_filter.py's imports first)",
          p.returncode == 0 and "teacher on cuda:0 torch.float16" in p.stdout, (p.stdout + p.stderr)[-400:])
    for mod in ("c3_generate", "c3_filter", "c3_nearcopy", "c3_feats", "c3_exists_tm", "c3_lr", "c3_verdict"):
        code = (f"import os, sys, importlib; sys.path.insert(0, {str(TU)!r}); importlib.import_module({mod!r}); "
                "v = os.environ.get('CUDA_VISIBLE_DEVICES'); print('CUDA_VISIBLE_DEVICES after import:', 'unset' if v is None else repr(v))")
        q = subprocess.run(PY + ["-c", code], capture_output=True, text=True, timeout=300, check=False, cwd="/tmp")
        line = (q.stdout.strip().splitlines() or ["(no output) " + q.stderr[-200:]])[-1]
        print(f"static check {mod}: {line}")
        hidden = "''" in line
        expect = mod in ("c3_generate", "c3_filter", "c3_nearcopy")
        check(f"static check: {mod} {'empties' if expect else 'leaves'} CUDA_VISIBLE_DEVICES", q.returncode == 0 and hidden == expect,
              line)
else:
    print("GPU part not run (--cpu-only)")
print(f"c3_cuda_first tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
