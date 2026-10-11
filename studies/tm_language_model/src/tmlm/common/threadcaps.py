"""Thread-pool caps and PYTHONHASHSEED for every process this project starts.

Why this exists: the container has a cgroup quota of 6 CPUs [stated: design notes],
but os.cpu_count() and sched_getaffinity report the host's cores. A thread pool that is
not capped starts one thread per reported core. env/env.sh sets the caps for shells
that source it. This module is the same policy for everything launched from Python
(scripts/run_logged.py, scripts/mem_guard.py, scripts/env_check.py, tests/conftest.py),
so a run is capped even when it was started from a shell that never sourced env.sh.

Policy, identical to env/env.sh:
  - cap = TMLM_CPU_SLOTS when that is an integer in 1..6 (set by scripts/cpu_slot.sh),
    else 6.
  - each of OMP, MKL, OPENBLAS, NUMEXPR, NUMBA, RAYON `_NUM_THREADS` keeps its value when
    that is an integer in 1..cap (a caller may always ask for FEWER threads), and is set
    to the cap otherwise (missing, invalid, zero, or above the cap).
  - TOKENIZERS_PARALLELISM=false.
  - PYTHONHASHSEED defaults to 0 (protocol rule 4). An explicit caller value is kept.

Limits:
  - A variable set in os.environ affects libraries that are loaded LATER and child
    processes. A library that is already loaded (numpy with OpenBLAS, torch) has read
    its cap already: `apply_to_process` reports those in "already_imported".
  - PYTHONHASHSEED is read at interpreter start. Setting it here fixes child processes,
    never the current one. env.json records `hash_randomization` of the live process.

Standard library only, and nothing heavy is imported.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping, MutableMapping

TOTAL_CPUS = 6  # cgroup CPU quota [stated: design notes], same constant as cpu_slot.sh

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

# Libraries that size a thread pool when they are loaded.
_POOL_MODULES = ("numpy", "scipy", "numba", "numexpr", "torch", "cupy", "sklearn", "tokenizers")

_POSITIVE_INT_RE = re.compile(r"[1-9][0-9]{0,8}")


def thread_cap(env: Mapping[str, str] | None = None) -> int:
    """TMLM_CPU_SLOTS when it is an integer in 1..6, else 6."""
    source = os.environ if env is None else env
    slots = source.get("TMLM_CPU_SLOTS", "")
    if _POSITIVE_INT_RE.fullmatch(slots) and int(slots) <= TOTAL_CPUS:
        return int(slots)
    return TOTAL_CPUS


def _capped(value: str | None, cap: int) -> str:
    if value is not None and _POSITIVE_INT_RE.fullmatch(value) and int(value) <= cap:
        return value
    return str(cap)


def capped_env(env: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """A copy of `env` (default os.environ) with the policy applied, and a report.

    The report is JSON-like: {"cap": int, "values": {name: value}, "changed": {name:
    {"from": old value or None, "to": new value}}}. "changed" is empty when `env`
    already satisfied the policy.
    """
    out = dict(os.environ if env is None else env)
    cap = thread_cap(out)
    wanted = {name: _capped(out.get(name), cap) for name in THREAD_VARS}
    wanted["TOKENIZERS_PARALLELISM"] = "false"
    wanted["PYTHONHASHSEED"] = out.get("PYTHONHASHSEED") or "0"
    changed = {}
    for name, value in wanted.items():
        if out.get(name) != value:
            changed[name] = {"from": out.get(name), "to": value}
            out[name] = value
    return out, {"cap": cap, "values": wanted, "changed": changed}


def apply_to_process(environ: MutableMapping[str, str] | None = None) -> dict[str, Any]:
    """Apply the policy to os.environ (or to `environ`) in place. Returns the report of
    `capped_env` plus "already_imported": pool libraries that were loaded before the
    call and therefore may have sized their pools from the old values."""
    target = os.environ if environ is None else environ
    capped, report = capped_env(target)
    for name in report["changed"]:
        target[name] = capped[name]
    report["already_imported"] = sorted(name for name in _POOL_MODULES if name in sys.modules)
    return report


def violations(env: Mapping[str, str] | None = None) -> list[str]:
    """Names of the thread variables that are missing, invalid, or above the cap."""
    source = os.environ if env is None else env
    cap = thread_cap(source)
    return [name for name in THREAD_VARS if source.get(name) != _capped(source.get(name), cap)]
