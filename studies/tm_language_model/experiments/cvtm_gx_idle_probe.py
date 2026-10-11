"""Lane GX, exploratory calibration: how long the V100 needs after GPU activity before its board power
is back at the idle level (docs/13 A3 H8.5 and A4 H9.7 subtract an idle power; section 5 asks for it to be
reported). Runs in .venv-tm under `scripts/gpu_run.sh exclusive` (device-wide NVML readings). No data, no
hypothesis, no seed: a measurement of the device used to fix the settle rule of the lane's energy runs.

Why: the idle power measured at the start of `results/20260924T151319Z_cvtm_H8_5_fit_energy/` was 86.3 W,
one second after another parallel GPU job ended, while both earlier attempts measured 72.1 W and the same
fit's post-repeat idle was 78.2 W; the CVTM-core fit's GPU steps run at about 80 to 95 W, so an idle
level taken during the tail of earlier activity decides the sign of their idle-subtracted joules.

Phases (config `experiments/cvtm/configs/GX_idle_probe.json`, key "phases"), CUDA context (CuPy) created
first and held throughout: "as_found" (nothing running), "load_matmul" (float32 matmul loop, high power),
"rest1" (nothing running), "load_sort" (CuPy sorts of 2^26 uint64 keys with host pauses, a CVTM-like
light load), "rest2". A sampler thread reads, at 10 Hz, board power (W), SM and memory clocks (MHz), GPU
temperature (C) and the performance state.

Reported per rest phase (label measured; the analysis is derived): 10 s window means of power and SM
clock; the reference = mean power of the last 60 s of the phase; for tol in {0.5, 1, 2} W the settle time =
the start of the first 10 s window from which every later window mean lies within tol of the reference;
the time at which the SM clock reaches and keeps its final value. The trace is saved as `trace.npz`.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[gx_idle_probe {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    rec = {"value": value, "label": label, **extra}
    if runlog.current_run_dir() is not None:
        runlog.write_metrics({name: rec}, on_conflict="replace")
    if isinstance(value, (int, float, str, bool)) or value is None:
        log(f"{name} = {value}")


class Tracer:
    """10 Hz NVML trace: (t, power W, sm MHz, mem MHz, temp C, pstate, phase index)."""

    def __init__(self, period_s: float = 0.1) -> None:
        import pynvml

        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period_s = period_s
        self.rows: list[tuple] = []
        self.phase = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> tuple:
        nv, h = self.nv, self.h
        return (time.time(), nv.nvmlDeviceGetPowerUsage(h) / 1000.0, nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                nv.nvmlDeviceGetPerformanceState(h), self.phase)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.rows.append(self._read())
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period_s)

    def start(self) -> None:
        self._th.start()

    def stop(self) -> np.ndarray:
        self._stop.set()
        self._th.join()
        return np.asarray(self.rows, dtype=np.float64)


def compute_procs(tr: "Tracer") -> list[int] | None:
    """GPU memory (MiB) of every process with a compute context on the device (this one included)."""
    try:
        return sorted(int((getattr(x, "usedGpuMemory", 0) or 0) // (1 << 20))
                      for x in tr.nv.nvmlDeviceGetComputeRunningProcesses(tr.h))
    except Exception:  # noqa: BLE001
        return None


def settle_analysis(t: np.ndarray, p: np.ndarray, sm: np.ndarray, window_s: float = 10.0, ref_s: float = 60.0,
                    tols=(0.5, 1.0, 2.0)) -> dict:
    """Window means from the phase start; settle time per tolerance (see the module docstring)."""
    t = t - t[0]
    n_win = int(t[-1] // window_s)
    means, sm_means = [], []
    for k in range(n_win):
        m = (t >= k * window_s) & (t < (k + 1) * window_s)
        means.append(float(p[m].mean()))
        sm_means.append(float(sm[m].mean()))
    ref = float(p[t >= t[-1] - ref_s].mean())
    out = {"window_s": window_s, "window_mean_w": means, "window_mean_sm_mhz": sm_means, "reference_w_last_60s": ref,
           "first_window_w": means[0] if means else None, "settle_s": {}}
    for tol in tols:
        ok = np.abs(np.asarray(means) - ref) <= tol
        # first k such that ok[k:] all true
        k = n_win
        while k > 0 and ok[k - 1]:
            k -= 1
        out["settle_s"][str(tol)] = None if k >= n_win else float(k * window_s)
    sm_final = sm[-1]
    bad = np.flatnonzero(sm != sm_final)
    out["sm_final_mhz"] = float(sm_final)
    out["sm_reaches_final_s"] = float(t[bad[-1] + 1]) if bad.shape[0] and bad[-1] + 1 < t.shape[0] else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="phases shortened to a few seconds (not logged)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    phases = cfg["phases"]
    if args.smoke:
        if runlog.current_run_dir() is not None:
            raise SystemExit("--smoke must not run under run_logged.py")
        phases = [{**ph, "seconds": min(float(ph["seconds"]), 3.0 if ph["kind"] != "rest" else 25.0)} for ph in phases]
    import os

    lock = os.environ.get("TMLM_GPU_LOCK_MODE")
    M("gpu_lock_mode", lock, note="device-wide readings are meaningful only under the exclusive lock")
    import cupy

    cupy.cuda.Device(0).use()
    cupy.zeros(1).sum()
    cupy.cuda.Device().synchronize()
    tr = Tracer()
    M("gpu_name", tr.nv.nvmlDeviceGetName(tr.h) if isinstance(tr.nv.nvmlDeviceGetName(tr.h), str) else str(tr.nv.nvmlDeviceGetName(tr.h)))
    tr.start()
    bounds = []
    a = cupy.random.default_rng(1).random((4096, 4096), dtype=cupy.float32)
    keys = cupy.random.default_rng(2).integers(0, 2 ** 62, size=1 << 26, dtype=cupy.uint64)
    for i, ph in enumerate(phases):
        tr.phase = i
        t_start = time.time()
        log(f"phase {ph['name']} ({ph['kind']}, {ph['seconds']} s)")
        if ph["kind"] == "rest":
            time.sleep(float(ph["seconds"]))
        elif ph["kind"] == "matmul":
            while time.time() - t_start < float(ph["seconds"]):
                for _ in range(20):
                    b = a @ a
                cupy.cuda.Device().synchronize()
            del b
        elif ph["kind"] == "sort":
            while time.time() - t_start < float(ph["seconds"]):
                s = cupy.sort(keys)
                cupy.cuda.Device().synchronize()
                del s
                time.sleep(float(ph.get("host_pause_s", 0.5)))
        else:
            raise ValueError(ph["kind"])
        bounds.append((ph["name"], ph["kind"], t_start, time.time(), compute_procs(tr)))
    rows = tr.stop()
    rd = runlog.current_run_dir()
    out_dir = Path(rd) if rd is not None else Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "gx_idle_probe_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "trace.npz", rows=rows, columns=np.array(["t", "power_w", "sm_mhz", "mem_mhz", "temp_c", "pstate", "phase"]))
    M("trace_file", "trace.npz", n_rows=int(rows.shape[0]))
    for i, (name, kind, t0, t1, procs) in enumerate(bounds):
        m = rows[:, 6] == i
        r = rows[m]
        summ = {"kind": kind, "seconds": t1 - t0, "n": int(r.shape[0]), "mean_w": float(r[:, 1].mean()),
                "min_w": float(r[:, 1].min()), "max_w": float(r[:, 1].max()), "sm_mhz_first_last": [float(r[0, 2]), float(r[-1, 2])],
                "mem_mhz_first_last": [float(r[0, 3]), float(r[-1, 3])], "temp_c_first_last": [float(r[0, 4]), float(r[-1, 4])],
                "pstate_values": sorted({int(x) for x in r[:, 5]}), "compute_processes_mib_at_end": procs}
        M(f"phase.{name}", summ)
        if kind == "rest" and r.shape[0] > 100:
            M(f"phase.{name}.settle", settle_analysis(r[:, 0], r[:, 1], r[:, 2]), label="derived",
              note="10 s window means from the phase start; settle_s[tol] = start of the first window after which every "
                   "window mean is within tol W of the mean of the last 60 s")
    M("wall_s", round(time.time() - T0, 1), unit="s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
