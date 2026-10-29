"""Unit test of bc1_impls.py (CPU, seconds, no training):
  - each class (T1T3, R1, T1T3_R1) builds a FastCoTM whose switches equal its configuration in configs.py;
  - an explicit switch, as the pinned tests' per-switch variants pass one, overrides only that default;
  - the module has no attribute FastCoTM, and review/bench/identity_cpu.py run on it WITHOUT --cls fails at the pinned
    test's `from fast_tm import FastCoTM` (exit != 0, no result JSON) instead of testing a class with every switch off.

    CUDA_VISIBLE_DEVICES= ... nice -n 19 <decision_layer>/.venv-dl/bin/python -I -B test_bc1_impls.py --out OUTDIR
Exit 0 only if every check holds. Writes only into OUTDIR (must not exist).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
IDENTITY_CPU = f"{_REPO_ROOT}/studies/decision_layer/speed/review/bench/identity_cpu.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refused: {a.out} exists")
    a.out.mkdir(parents=True)
    utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    spec = importlib.util.spec_from_file_location("speed_p2tm_bc1_impls", HERE / "bc1_impls.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cf = sys.modules["speed_p2tm_configs"]
    from dl.tm import TMConfig  # the decision_layer src is on sys.path once fast_cotm.py is loaded

    cfg = TMConfig(n_clauses=8, T=8, s=8.0, epochs=1, seed=1)
    checks = {"no_FastCoTM_attribute": not hasattr(mod, "FastCoTM")}
    for n in mod.NAMES:
        cls = getattr(mod, n)
        want = dict(getattr(cf, n))
        tm = cls(3, 5, cfg, device="cpu")
        checks[f"{n}_defaults_equal_config"] = tm.opt == want and isinstance(tm, sys.modules["speed_p2tm_fast_cotm"].FastCoTM)
        tm2 = cls(3, 5, cfg, device="cpu", xonly=False, evalonce=True)
        checks[f"{n}_explicit_switch_overrides"] = tm2.opt == {**want, "xonly": False, "evalonce": True}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "RAYON_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    wd = a.out / "nocls"
    p = subprocess.run(["nice", "-n", "19", PY, "-I", "-B", IDENTITY_CPU, "--impl", str(HERE / "bc1_impls.py"),
                        "--workdir", str(wd), "--test", "test_identity_steps.py"], env=env, capture_output=True,
                       text=True, timeout=300, check=False)
    (a.out / "nocls.log").write_text(f"--- stdout\n{p.stdout}--- stderr\n{p.stderr}--- exit {p.returncode}\n")
    checks["identity_cpu_without_cls_fails"] = p.returncode != 0 and "cannot import name 'FastCoTM'" in p.stderr \
        and not (wd / "results" / "identity_steps.json").exists()
    res = {"utc_start": utc0, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "bc1_impls_sha256": hashlib.sha256((HERE / "bc1_impls.py").read_bytes()).hexdigest(),
           "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks,
           "all_pass": all(checks.values())}
    (a.out / "test_bc1_impls.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    raise SystemExit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
