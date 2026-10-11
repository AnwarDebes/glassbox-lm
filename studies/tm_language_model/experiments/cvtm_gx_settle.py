"""The settle rule of lane GX's energy protocol, shared by experiments/cvtm_energy.py (.venv-tm) and
experiments/cvtm_transformer_tuned.py (.venv-dl); numpy only.

Every energy run of the lane waits `energy_protocol.settle_s` seconds with nothing running (the board power
is sampled and the last three 10 s window means are logged) before its idle measurement and before every
measured interval, so that neither the idle level nor a measured interval starts in the tail of earlier GPU
activity. The value is fixed from the exploratory device calibration `experiments/cvtm_gx_idle_probe.py`
(its config states the rule that turns the probe into `settle_s`). Reason:
`results/20260924T151319Z_cvtm_H8_5_fit_energy/` measured 86.3 W as its idle one second after another parallel
GPU job ended (both earlier attempts: 72.1 W), and the CVTM-core fit's GPU steps draw about 80 to 95 W.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def settle_protocol(cfg: dict) -> float:
    """settle_s of the config's "energy_protocol" (SystemExit when missing or not positive)."""
    ep = cfg.get("energy_protocol")
    if not ep or "settle_s" not in ep:
        raise SystemExit("the config has no energy_protocol.settle_s (fixed by the GX idle probe)")
    s = float(ep["settle_s"])
    if not s > 0:
        raise SystemExit("energy_protocol.settle_s must be positive")
    return s


def last_window_means(samples: list[tuple[float, float]], window_s: float = 10.0, n: int = 3) -> list[float]:
    """Mean power of the last n consecutive windows of window_s seconds (oldest first); only windows that lie
    entirely inside the sampled span are returned."""
    if len(samples) < 2:
        return []
    t = np.array([s[0] for s in samples], dtype=np.float64)
    p = np.array([s[1] for s in samples], dtype=np.float64)
    out = []
    for k in range(n, 0, -1):
        hi = t[-1] - (k - 1) * window_s
        m = (t > hi - window_s) & (t <= hi)
        if m.any() and hi - window_s >= t[0] - 1e-9:
            out.append(float(p[m].mean()))
    return out


def compute_processes(sampler) -> list[int] | None:
    """GPU memory (MiB) of every process with a compute context on the device (this one included), from
    NVML; None when NVML is not available. Other parallel jobs' contexts are logged, never touched."""
    try:
        procs = sampler._nvml.nvmlDeviceGetComputeRunningProcesses(sampler._handle)
        return sorted(int((getattr(x, "usedGpuMemory", 0) or 0) // (1 << 20)) for x in procs)
    except Exception:  # noqa: BLE001
        return None


def settle(seconds: float, tag: str, sampler_factory: Callable, log: Callable[[str], None] = print,
           sleep: Callable[[float], None] = time.sleep) -> dict:
    """Wait `seconds` with nothing running while a fresh sampler (`sampler_factory()`, the NvmlPowerSampler
    API: available, start, stop, samples) records the board power. Returns a diagnostic record with the
    compute processes on the device at the start and at the end."""
    s = sampler_factory()
    avail = bool(getattr(s, "available", False))
    procs0 = compute_processes(s) if avail else None
    if avail:
        s.start()
    sleep(float(seconds))
    summ = s.stop() if avail else {"available": False}
    lw = last_window_means(s.samples) if avail else []
    rec = {"seconds": float(seconds), "mean_w": summ.get("mean_w"), "last_10s_window_means_w": lw,
           "last_two_windows_within_0.5_w": bool(len(lw) >= 2 and abs(lw[-1] - lw[-2]) <= 0.5),
           "compute_processes_mib_start": procs0, "compute_processes_mib_end": compute_processes(s) if avail else None}
    log(f"settle {tag}: {seconds:.0f} s, last 10 s windows {[round(x, 2) for x in lw]} W, "
        f"compute processes on the device {rec['compute_processes_mib_end']}")
    return rec
