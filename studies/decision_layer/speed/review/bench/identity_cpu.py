#!/usr/bin/env python
"""Benchmark BC1 (TM code identity on the CPU): run one of the TM-code profiler's identity tests, UNCHANGED, against
any candidate implementation file.

    python -I -B identity_cpu.py --impl NEW_TM.py [--cls FastCoTM] --workdir DIR --test test_identity.py -- --group unchunked --rows 1024 --out DIR/x.json
    python -I -B identity_cpu.py --impl NEW_TM.py --workdir DIR --test test_identity_bounds.py
    python -I -B identity_cpu.py --impl NEW_TM.py --workdir DIR --test test_identity_steps.py

The tests (glassbox-lm/studies/decision_layer/speed/tmcode/test_identity*.py + their data loader prof_tm.py, pinned by SHA-256 below; a changed
file is refused) train dl.tm.CoTM and the class under test from the same seed and compare exactly: automata ta, weights
w, the generator state after training (the same random numbers were drawn), every dev val_acc, best_epoch, and the
integer class sums of the final machine. They are COPIED into --workdir and run from there, because the bounds and
steps tests write their JSON next to themselves (HERE/results/...): running them in place would overwrite the
profiler's own records. They import the class as `from fast_tm import FastCoTM`; this wrapper puts the candidate file
into sys.modules under that name (--cls aliased to FastCoTM). The tests exit 0 even when a case fails, so the verdict
here is the JSON's all_pass (exit 0 ALL PASS, 1 FAIL). CPU only. A CPU pass is necessary, not sufficient:
torch.binomial maps elements to draws differently on the GPU (the GPU set BG decides).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import hashlib
import importlib.util
import json
import runpy
import shutil
import sys
from pathlib import Path

TMCODE = Path(f"{_REPO_ROOT}/studies/decision_layer/speed/tmcode")
PINNED = {  # SHA-256 prefixes of the files as the TM-code profiler ran them (2026-10-21; 68 of 68 PASS on fast_tm.py)
    # test_identity.py and prof_tm.py: the published copies; as run 9cb6f4e8bc3119c8 (differs in its first line only)
    # and e2774df7be58425e (differs in its docstring wording and in finding the repository from its own place)
    "test_identity.py": "53b7187208621aa2",
    "test_identity_bounds.py": "d89bbfb4c12fa5c5",
    "test_identity_steps.py": "ac7d3fd516588e69",
    "prof_tm.py": "71f1a483b16cf8bb",
}
FIXED_OUT = {"test_identity_bounds.py": "results/identity_bounds.json", "test_identity_steps.py": "results/identity_steps.json"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    i = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, type=Path)
    ap.add_argument("--cls", default="FastCoTM")
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--test", required=True, choices=sorted(k for k in PINNED if k.startswith("test_")))
    a = ap.parse_args(sys.argv[1:i])
    rest = sys.argv[i + 1:]
    wd = a.workdir.resolve()
    if wd == TMCODE or TMCODE in wd.parents:
        raise SystemExit("refused: --workdir must not be inside the TM-code profiler's folder")
    (wd / "results").mkdir(parents=True, exist_ok=True)
    for name, want in PINNED.items():
        got = sha(TMCODE / name)
        if not got.startswith(want):
            raise SystemExit(f"refused: {TMCODE / name} sha256 {got[:16]} != pinned {want}")
        shutil.copyfile(TMCODE / name, wd / name)
        if sha(wd / name) != got:
            raise SystemExit(f"copy of {name} differs")
    if a.test == "test_identity.py":
        if "--out" not in rest:
            raise SystemExit("test_identity.py needs -- ... --out FILE")
        out = Path(rest[rest.index("--out") + 1]).resolve()
    else:
        out = wd / FIXED_OUT[a.test]
    if out.exists():
        raise SystemExit(f"refused: {out} exists (use a fresh output)")
    sys.path.insert(0, str(wd))  # prof_tm (the data loader), as the tests expect next to themselves
    spec = importlib.util.spec_from_file_location("fast_tm", a.impl.resolve())
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fast_tm"] = mod
    spec.loader.exec_module(mod)
    if a.cls != "FastCoTM":
        mod.FastCoTM = getattr(mod, a.cls)
    print(f"identity_cpu: class {a.cls} from {a.impl.resolve()} (sha256 {sha(a.impl.resolve())[:16]}); test {a.test} "
          f"(copy in {wd})", flush=True)
    sys.argv = [str(wd / a.test)] + rest
    runpy.run_path(str(wd / a.test), run_name="__main__")
    res = json.loads(out.read_text())
    n = len(res.get("cases", []))
    print(f"identity_cpu verdict: {'ALL PASS' if res.get('all_pass') else 'FAIL'} ({n} cases) -> {out}", flush=True)
    raise SystemExit(0 if res.get("all_pass") and n > 0 else 1)


if __name__ == "__main__":
    main()
