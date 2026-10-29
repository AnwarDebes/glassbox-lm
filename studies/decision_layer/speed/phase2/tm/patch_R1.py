"""Speed phase 2a patch, configuration R1 (configs.py of this folder), for review/bench/run_gpu_job.sh ROOT JOB IMPL.

review/bench/with_patch.py runs this file (runpy) before the decision_layer UNEDITED entry script, in the same process. Contract
kept (with_patch.py): it replaces one name in an imported module and writes no file. The name: dl.tm_init.CoTM, the
class that dl.tm_init.make_tm constructs (scripts/abcd_explore.py builds its TM only through make_tm, which reads
CoTM from dl.tm_init's globals when it is called). The replacement is a FastCoTM (fast_cotm.py) built with make_tm's
own arguments (n_classes, n_features, cfg, device) unchanged plus the switches of configuration R1; nothing else is
replaced. fast_cotm.py and configs.py are loaded by path and must match the SHA-256 pinned here (else SystemExit before
anything runs). dl is imported first from the search path with_patch.py set up (the run root's src, as in an "orig"
run); the sys.path entry that fast_cotm.py adds is removed again after loading, so the entry script sees the same
search path as without the patch.
Prints at import one line: configuration, switches, the SHA-256 of fast_cotm.py, configs.py and this file. Prints at
construction one line: make_tm's arguments and the switches (before the CUDA tensors are made, so a --dry-cpu log
shows it too). Both lines go to the job log only, never into a result row.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

NAME = "R1"
HERE = Path(__file__).resolve().parent
PINS = {
    "fast_cotm.py": "a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8",
    "configs.py": "a7205c918060f3812bd29052e2128551dda0597058a6b49ca266da13977ee5a0",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(modname: str, fname: str):
    p = HERE / fname
    got = _sha(p)
    if got != PINS[fname]:
        raise SystemExit(f"speed_patch {NAME}: refused: {p} sha256 {got} != pinned {PINS[fname]}")
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


import dl.tm as _tm  # noqa: E402
import dl.tm_init as _ti  # noqa: E402

if _ti.CoTM is not _tm.CoTM:
    raise SystemExit(f"speed_patch {NAME}: refused: dl.tm_init.CoTM is already replaced ({_ti.CoTM!r})")
_search_path = list(sys.path)
_fc = _load("speed_p2tm_fast_cotm", "fast_cotm.py")
_cf = _load("speed_p2tm_configs", "configs.py")
sys.path[:] = _search_path
if _fc.CoTM is not _tm.CoTM or not issubclass(_fc.FastCoTM, _tm.CoTM):
    raise SystemExit(f"speed_patch {NAME}: refused: fast_cotm.FastCoTM does not subclass this process's dl.tm.CoTM")
OPTS = dict(getattr(_cf, NAME))


class SpeedCoTM(_fc.FastCoTM):
    """make_tm's call CoTM(n_classes, n_features, cfg, device=device) -> FastCoTM(same arguments, **OPTS)."""

    def __init__(self, n_classes, n_features, cfg, device="cuda"):
        print(f"speed_patch {NAME}: make_tm -> FastCoTM(n_classes={n_classes}, n_features={n_features}, "
              f"device={device!r}, cfg={cfg}) switches {OPTS}", flush=True)
        super().__init__(n_classes, n_features, cfg, device=device, **OPTS)


_ti.CoTM = SpeedCoTM
print(f"speed_patch {NAME}: dl.tm_init.CoTM -> FastCoTM switches {OPTS}; fast_cotm.py sha256 {PINS['fast_cotm.py']}; "
      f"configs.py sha256 {PINS['configs.py']}; {Path(__file__).name} sha256 {_sha(Path(__file__).resolve())}", flush=True)
