"""Lane GX: the settle analysis of experiments/cvtm_gx_idle_probe.py (synthetic traces, no GPU).

    .venv-tm/bin/python -m pytest tests/test_cvtm_gx_settle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import cvtm_gx_idle_probe as P  # noqa: E402


def test_step_drop_after_a_hold():
    t = np.arange(0, 300, 0.1) + 1000.0
    p = np.where(t - 1000.0 < 95.0, 85.0, 72.0)          # a plateau (clock hold), then idle
    sm = np.where(t - 1000.0 < 95.0, 1530.0, 135.0)
    a = P.settle_analysis(t, p, sm)
    assert a["reference_w_last_60s"] == 72.0
    assert a["settle_s"]["0.5"] == 100.0                 # the window 90..100 s still holds part of the plateau
    assert a["sm_final_mhz"] == 135.0 and abs(a["sm_reaches_final_s"] - 95.0) < 0.11


def test_already_idle_and_never_settles():
    t = np.arange(0, 200, 0.1)
    a = P.settle_analysis(t, np.full(t.shape, 71.0), np.full(t.shape, 135.0))
    assert a["settle_s"]["0.5"] == 0.0 and a["sm_reaches_final_s"] == 0.0
    p = 90.0 - 0.1 * t          # still falling: the last full window (180..190 s) is 1.5 W below the last-60 s mean
    b = P.settle_analysis(t, p, np.full(t.shape, 135.0))
    assert b["settle_s"]["0.5"] is None and b["settle_s"]["1.0"] is None and b["settle_s"]["2.0"] == 150.0


def test_exponential_decay():
    t = np.arange(0, 420, 0.1)
    p = 71.0 + 14.0 * np.exp(-t / 40.0)
    a = P.settle_analysis(t, p, np.full(t.shape, 135.0))
    # within 0.5 W of the final mean once 14 exp(-t/40) is below about 0.5: t about 133 s
    assert 110.0 <= a["settle_s"]["0.5"] <= 140.0
    assert a["settle_s"]["2.0"] < a["settle_s"]["1.0"] < a["settle_s"]["0.5"]


# ------------------------------------------------------------------ the shared settle helper
import cvtm_gx_settle as S  # noqa: E402


class _Proc:
    def __init__(self, mib):
        self.usedGpuMemory = mib << 20


class _FakeNvml:
    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [_Proc(306), _Proc(618)]


class _FakeSampler:
    available = True

    def __init__(self):
        self._nvml, self._handle, self.samples = _FakeNvml(), 0, []

    def start(self):
        t = np.arange(0.0, 45.0, 0.1)
        self.samples = [(float(x), 84.0 if x < 12.0 else 72.0) for x in t]

    def stop(self):
        return {"mean_w": float(np.mean([s[1] for s in self.samples]))}


def test_settle_record_and_protocol():
    rec = S.settle(45.0, "t", _FakeSampler, log=lambda m: None, sleep=lambda s: None)
    assert rec["seconds"] == 45.0 and rec["last_10s_window_means_w"] == [72.0, 72.0, 72.0]
    assert rec["last_two_windows_within_0.5_w"] is True
    assert rec["compute_processes_mib_start"] == [306, 618] == rec["compute_processes_mib_end"]
    assert S.settle_protocol({"energy_protocol": {"settle_s": 90}}) == 90.0
    for bad in ({}, {"energy_protocol": {}}, {"energy_protocol": {"settle_s": 0}}):
        try:
            S.settle_protocol(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_last_window_means_only_full_windows():
    samples = [(float(t), float(t)) for t in np.arange(0.0, 25.0, 1.0)]     # 0..24 s, value = time
    assert S.last_window_means(samples) == [9.5, 19.5]                       # windows (4,14], (14,24]; (-6,4] is cut
