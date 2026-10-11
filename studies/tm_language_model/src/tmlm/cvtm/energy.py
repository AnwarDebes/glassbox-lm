"""GPU energy of a CVTM step through NVML (docs/13 A3 H8.5, A4 H9.7; docs/06 energy protocol).

The method is that of `src/tmlm/eval/baselines/transformer_small.py` (the comparator whose joules
H8.5 matches), so both sides of an equal-joule comparison are measured the same way:

- a sampler thread reads the DEVICE-WIDE board power with pynvml (`nvmlDeviceGetPowerUsage`,
  milliwatts, converted to W) every `period_s` (default 0.1 s, 10 Hz);
- energy = the trapezoid integral of the samples over their timestamps (joules, "joules_raw");
- idle power = the mean of a 10 s sample with the CUDA context created and nothing running
  (`measure_idle`); idle-subtracted joules = joules_raw - idle_w x span_s, with span_s the time between
  the first and the last sample of the measured interval.

Device-wide means every process on the GPU is included; reported numbers therefore come from runs
under the EXCLUSIVE lock (`scripts/gpu_run.sh exclusive`), and a run under the shared lock labels its
joules as indicative only. CPU energy is not readable in this container (docs/13 A3 H8.5): host CPU
seconds are reported next to the joules instead (`EnergyMeter` records process CPU time).

`pynvml` is imported lazily; without NVML (`available` False) every energy field is None and the
reason is kept in `error`. The clock and the NVML module can be injected for tests.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np


def _integrate(t: np.ndarray, p: np.ndarray) -> float:
    return float(np.trapezoid(p, t)) if hasattr(np, "trapezoid") else float(np.trapz(p, t))


class NvmlPowerSampler:
    """Samples device-wide GPU power (W) in a thread; `summary()` integrates (transformer_small.py method)."""

    def __init__(self, index: int = 0, period_s: float = 0.1, *, nvml: Any = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.period_s = float(period_s)
        self.samples: list[tuple[float, float]] = []
        self.available = False
        self.error: str | None = None
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_read_errors = 0
        try:
            if nvml is None:
                import pynvml as nvml  # noqa: PLW0127
            nvml.nvmlInit()
            self._nvml = nvml
            self._handle = nvml.nvmlDeviceGetHandleByIndex(int(index))
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)

    def read_w(self) -> float:
        """One power reading in W."""
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((self.clock(), self.read_w()))
            except Exception:  # noqa: BLE001
                self.n_read_errors += 1
            self._stop.wait(self.period_s)

    def start(self) -> None:
        if self.available:
            self._stop.clear()
            self.samples = []
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return self.summary()

    def summary(self) -> dict:
        """{"available", "n_samples", "joules", "mean_w", "span_s"} of the samples so far."""
        if not self.available or len(self.samples) < 2:
            return {"available": self.available, "n_samples": len(self.samples), "joules": None, "mean_w": None,
                    "span_s": 0.0, "error": self.error}
        t = np.array([s[0] for s in self.samples], dtype=np.float64)
        p = np.array([s[1] for s in self.samples], dtype=np.float64)
        return {"available": True, "n_samples": int(t.shape[0]), "joules": _integrate(t, p), "mean_w": float(p.mean()),
                "span_s": float(t[-1] - t[0]), "n_read_errors": int(self.n_read_errors)}

    def measure_idle(self, seconds: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> dict:
        """Mean power over `seconds` with nothing running (call it after the CUDA context exists)."""
        self.start()
        sleep(float(seconds))
        return self.stop()


def idle_subtracted(energy: dict, idle_w: float | None) -> float | None:
    """joules_raw - idle_w x span_s (None when either is unknown)."""
    if energy.get("joules") is None or idle_w is None:
        return None
    return float(energy["joules"]) - float(idle_w) * float(energy["span_s"])


class EnergyMeter:
    """Context manager around one measured step:

        meter = EnergyMeter(idle_s=10.0)        # measures the idle power once
        with meter.measure("fit"):
            ...                                 # the GPU work
        meter.results["fit"]  ->  {"joules_raw", "joules_idle_subtracted", "idle_w", "span_s", "mean_w",
                                   "n_samples", "wall_s", "cpu_s", "available", "lock"}

    `lock` records the GPU lock mode of the process (TMLM_GPU_LOCK_MODE: "exclusive" or "shared"; only
    exclusive measurements are reportable, docs/13 section 5)."""

    def __init__(self, idle_s: float = 10.0, index: int = 0, period_s: float = 0.1, *, nvml: Any = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 idle_w: float | None = None) -> None:
        import os

        self.sampler = NvmlPowerSampler(index, period_s, nvml=nvml, clock=clock)
        self.clock = clock
        self.lock = os.environ.get("TMLM_GPU_LOCK_MODE")
        self.idle: dict | None = None
        if idle_w is not None:
            self.idle_w: float | None = float(idle_w)
        elif self.sampler.available:
            self.idle = self.sampler.measure_idle(idle_s, sleep)
            self.idle_w = self.idle.get("mean_w")
        else:
            self.idle_w = None
        self.results: dict[str, dict] = {}

    def measure(self, name: str) -> "_Measurement":
        return _Measurement(self, name)


class _Measurement:
    def __init__(self, meter: EnergyMeter, name: str) -> None:
        self.meter = meter
        self.name = name

    def __enter__(self) -> "_Measurement":
        self.t0 = self.meter.clock()
        self.c0 = time.process_time()
        self.meter.sampler.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        energy = self.meter.sampler.stop()
        self.meter.results[self.name] = {
            "available": bool(energy.get("available")), "joules_raw": energy.get("joules"),
            "joules_idle_subtracted": idle_subtracted(energy, self.meter.idle_w), "idle_w": self.meter.idle_w,
            "span_s": energy.get("span_s"), "mean_w": energy.get("mean_w"), "n_samples": energy.get("n_samples"),
            "wall_s": float(self.meter.clock() - self.t0), "cpu_s": float(time.process_time() - self.c0),
            "lock": self.meter.lock, "scope": "device-wide NVML board power (all processes on the GPU)",
        }
