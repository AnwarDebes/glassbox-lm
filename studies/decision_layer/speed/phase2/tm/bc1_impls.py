"""BC1 implementation file for review/bench/identity_cpu.py: one class per candidate configuration, each a FastCoTM
(fast_cotm.py of this folder) whose DEFAULT switches are that configuration (configs.py), so the pinned identity tests
test the exact configuration wherever they build a bare FastCoTM(...): variant "all" of test_identity.py, every case of
test_identity_bounds.py and test_identity_steps.py. A variant that passes its own switches (test_identity.py's
"xonly", "probs", "probs_expand", "elem", "all+expand") overrides the defaults, as with the prototype.

    identity_cpu.py --impl .../phase2/tm/bc1_impls.py --cls T1T3 --workdir W --test test_identity.py -- --group unchunked ...

Classes: T1T3, R1, T1T3_R1. On purpose this module has NO attribute named FastCoTM: identity_cpu.py without --cls fails
at the tests' `from fast_tm import FastCoTM` instead of silently testing a class with every switch off.
fast_cotm.py and configs.py are loaded by path and must match the SHA-256 pinned below (else SystemExit).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    "fast_cotm.py": "a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8",
    "configs.py": "a7205c918060f3812bd29052e2128551dda0597058a6b49ca266da13977ee5a0",
}
NAMES = ("T1T3", "R1", "T1T3_R1")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(modname: str, fname: str):
    p = HERE / fname
    got = _sha(p)
    if got != PINS[fname]:
        raise SystemExit(f"bc1_impls: refused: {p} sha256 {got} != pinned {PINS[fname]}")
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_fc = _load("speed_p2tm_fast_cotm", "fast_cotm.py")
_cf = _load("speed_p2tm_configs", "configs.py")


def _with_defaults(name: str):
    opts = dict(getattr(_cf, name))

    def __init__(self, *a, **k):
        for key, val in opts.items():
            k.setdefault(key, val)
        _fc.FastCoTM.__init__(self, *a, **k)

    return type(name, (_fc.FastCoTM,), {"__init__": __init__, "CONFIG_NAME": name, "CONFIG": dict(opts),
                                         "__module__": __name__})


T1T3 = _with_defaults("T1T3")
R1 = _with_defaults("R1")
T1T3_R1 = _with_defaults("T1T3_R1")

print(f"bc1_impls: classes {', '.join(f'{n}={getattr(_cf, n)}' for n in NAMES)}; fast_cotm.py sha256 {PINS['fast_cotm.py']}; "
      f"configs.py sha256 {PINS['configs.py']}", flush=True)
