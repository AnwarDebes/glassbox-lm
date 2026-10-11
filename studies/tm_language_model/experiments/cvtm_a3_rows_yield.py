"""Launcher of `experiments/cvtm_a3_rows.py` (lane A1, docs/13 amendment A3 rows H8.8 (a) and H8.9) that
gives the SHARED GPU lock to a waiting exclusive job at the idle points of the run. It changes no number:
the modes, configs, checks and verdicts are those of `cvtm_a3_rows.py`, imported unchanged.

Why (resume of lane A1, 2026-09-24): a job that holds the shared GPU lock through a long CPU-only phase
blocked the exclusive energy measurement for 20 minutes. `cvtm_a3_rows.py` runs one process under
`scripts/gpu_run.sh shared`, and its CPU-only phases (the memory wait before each fit, the validation
scoring after it) sit between the GPU fits. This launcher calls `tmlm.common.locks.yield_gpu_if_requested`
(the repository's hand-over protocol, documented in `scripts/gpu_run.sh`) at those points:

- before every fit, and at every poll of the memory rule while it waits (the rule: start only when
  memory.current + expected peak < 0.80 memory.max, poll every 2 minutes);
- before every validation scoring call (`cvtm_a3_rows.score`, CPU; every GPU result of the fit has been
  copied to host memory by then, so the GPU is idle for this process).

A yield happens only when a live job waits for the EXCLUSIVE lock; this run then releases its shared lock,
waits until the exclusive job has ended, and takes the shared lock back. Every yield is logged with its
duration in the metric `gpu_lock.yields` (a yield before scoring is inside the wall time of the logged
`validation_scoring_s` of that step, and its duration is logged next to it). Under the exclusive lock
(H8.8 (c)) or without a lock the calls return at once.

Launch (the configs `experiments/cvtm/configs/cvtm_a3_H8_8a_rerun.json` and `cvtm_a3_H8_9_yield.json` repeat it):

    scripts/gpu_run.sh shared -- scripts/cpu_slot.sh 1 -- nice -n 10 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name cvtm_a3_H8_8a --config experiments/cvtm/configs/cvtm_a3_H8_8a_rerun.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_a3_rows_yield.py --config experiments/cvtm/configs/cvtm_a3_H8_8a_rerun.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cvtm_a3_rows as a3  # noqa: E402
from tmlm.common import locks  # noqa: E402

YIELDS: list[dict] = []
_ORIG_WAIT = a3.wait_for_memory
_ORIG_SCORE = a3.score


def give_gpu_if_requested(point: str) -> bool:
    """Hand the shared GPU lock to a waiting exclusive job (blocks while it runs); log the yield."""
    t0 = time.time()
    if not locks.yield_gpu_if_requested():
        return False
    dt = time.time() - t0
    YIELDS.append({"point": point, "wait_s": round(dt, 1),
                   "resumed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    a3.log(f"gave the shared GPU lock to an exclusive job at '{point}'; resumed after {dt:.0f} s")
    a3.M("gpu_lock.yields", list(YIELDS), note="points where this run released the shared GPU lock for a waiting "
                                                "exclusive job, and how long it waited to take it back")
    return True


def wait_for_memory(peak_gib: float, tag: str, poll_s: int = 120, max_wait_s: int = 8 * 3600) -> None:
    """The memory rule of `cvtm_a3_rows.wait_for_memory` (same condition, same 2-minute poll), with a GPU
    hand-over before every check. When the condition holds, the original function is called; it passes
    at once and records the `memory.before.<tag>` metric."""
    waited = 0
    peak = 2.0 if a3.SMOKE else float(peak_gib)
    while True:
        give_gpu_if_requested(f"before {tag}")
        cur, ws, mx = a3.cgroup()
        if cur + peak * a3.GIB < 0.80 * mx:
            break
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag}")
        a3.log(f"memory before {tag}: current {cur / a3.GIB:.1f} GiB (working set {ws / a3.GIB:.1f}) + {peak:.0f} GiB "
               f">= 0.80 x max; waiting {poll_s} s (launcher)")
        time.sleep(poll_s)
        waited += poll_s
    if waited:
        a3.M(f"memory.before.{tag}.launcher_waited_s", waited, unit="s")
    _ORIG_WAIT(peak_gib, tag)


def score(core, ev, lnp, ks, feats=None):
    give_gpu_if_requested("validation scoring")
    return _ORIG_SCORE(core, ev, lnp, ks, feats)


def install() -> None:
    """Replace the two module globals that the mode functions of `cvtm_a3_rows` look up at call time."""
    a3.wait_for_memory = wait_for_memory
    a3.score = score


def main() -> int:
    install()
    a3.M("launcher", "experiments/cvtm_a3_rows_yield.py", label="unverified",
         note="cvtm_a3_rows.py run through the GPU-yield launcher (no change to any number)")
    return a3.main()


if __name__ == "__main__":
    sys.exit(main())
