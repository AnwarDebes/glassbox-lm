"""Lane GX (exclusive GPU): experiments/cvtm_energy.py, the measured GPU steps of the CVTM-core fit and the
verdict helpers. Small corpora, numpy backend, mocked NVML. The transformer side is in
tests/test_cvtm_gx_transformer.py (.venv-dl).

    .venv-tm/bin/python -m pytest tests/test_cvtm_gx_energy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from tmlm.cvtm import energy as E  # noqa: E402
from tmlm.cvtm import kn_gpu  # noqa: E402
from tmlm.cvtm.model import CVTMCore  # noqa: E402
from tmlm.cvtm.suffix import SuffixClauses  # noqa: E402

import cvtm_energy as CE  # noqa: E402


class FakeNvml:
    def __init__(self, watts):
        self.watts = list(watts)
        self.i = 0

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return ("handle", index)

    def nvmlDeviceGetPowerUsage(self, handle):
        w = self.watts[self.i % len(self.watts)]
        self.i += 1
        return int(w * 1000)


def corpus(seed: int, vocab: int = 30, n_docs: int = 400):
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(vocab, 0.3), size=vocab)
    cum = np.cumsum(trans, axis=1)
    docs = []
    for _ in range(n_docs):
        n = int(rng.integers(30, 160))
        t = [int(rng.integers(vocab))]
        for x in rng.random(n - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        if n > 40:
            a, L = int(rng.integers(0, n // 2 - 12)), int(rng.integers(6, 12))
            b = int(rng.integers(n // 2, n - L))
            t[b:b + L] = t[a:a + L]
        docs.append(t)
    tok = np.asarray([x for d in docs for x in d], dtype=np.uint16)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    return tok, ds


@pytest.fixture(scope="module")
def train():
    return corpus(7)


# ------------------------------------------------------------------ measured GPU steps
def test_measured_gpu_steps_leave_the_fit_unchanged(train):
    ref = CVTMCore.fit(*train, vocab=30, backend="numpy", expect_h=False)
    meter = E.EnergyMeter(nvml=FakeNvml([150.0, 170.0]), idle_w=40.0, period_s=0.001)
    orig_fit, orig_from = kn_gpu.fit_kn, SuffixClauses.__dict__["from_kn"]
    synced = []
    with CE.measured_gpu_steps(meter, "repeat1", sync=lambda: synced.append(1)) as names:
        core = CVTMCore.fit(*train, vocab=30, backend="numpy", expect_h=False)
    assert core.state_hash() == ref.state_hash()
    assert [n.split(".", 1)[1] for n in names] == list(CE.GPU_STEP_NAMES)
    assert all(n.startswith("repeat1.") for n in names) and len(synced) == 4
    for n in names:
        r = meter.results[n]
        assert r["available"] and r["joules_raw"] is not None and r["wall_s"] >= 0.0 and r["cpu_s"] >= 0.0
        assert r["joules_idle_subtracted"] == pytest.approx(r["joules_raw"] - 40.0 * r["span_s"])
    assert kn_gpu.fit_kn is orig_fit
    assert SuffixClauses.__dict__["from_kn"] is orig_from


def test_measured_gpu_steps_restore_after_an_exception(train):
    meter = E.EnergyMeter(nvml=FakeNvml([100.0]), idle_w=0.0, period_s=0.001)
    orig_fit, orig_from = kn_gpu.fit_kn, SuffixClauses.__dict__["from_kn"]
    with pytest.raises(RuntimeError):
        with CE.measured_gpu_steps(meter, "r"):
            raise RuntimeError("boom")
    assert kn_gpu.fit_kn is orig_fit and SuffixClauses.__dict__["from_kn"] is orig_from


def test_summarize_and_doc_bits():
    s = CE.summarize([3.0, 1.0, 2.0, 5.0, 4.0])
    assert s == {"median": 3.0, "min": 1.0, "max": 5.0, "n": 5, "values": [3.0, 1.0, 2.0, 5.0, 4.0]}
    bits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ds = np.array([0, 2, 2, 5])
    assert np.array_equal(CE.doc_bits_of(bits, ds), np.array([3.0, 0.0, 12.0]))


def test_registered_constants():
    assert CE.REGISTERED_POWER_W == 259.2
    assert CE.OLD_600S_JOULES / 600.0 == pytest.approx(259.2, abs=1e-3)
    assert CE.G1_THRESHOLD == 1.2209 and CE.H9_7_PREDICTION_REF == 1.2409
