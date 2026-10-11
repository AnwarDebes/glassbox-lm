"""GPU tests of the image-code read-out of experiments/ssl_r4_images.py (docs/13 A7, phase R4): on a tiny synthetic
patch-token store, the batched GPU aggregation of (fired clause, quadrant) counts equals the CPU reference built from
the public fired_clauses of every patch position, and the masked top-1 of the device argmax equals the one of
predict_scores. Also checks that the CUDA engine trains on the store (masked mode, F1 targets) and that the CPU
engine reaches the same state hash on it (docs/11 bit-exactness on the image corpus).

Skipped when CuPy or the GPU is absent. Run through the locks, CPU slot first (docs/12 F25):
    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest -q tests/test_ssl_r4_images_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.Device(0).compute_capability
except Exception as exc:  # pragma: no cover
    pytest.skip(f"no CUDA device: {exc}", allow_module_level=True)

from tmlm.engine.api import CorpusArrays  # noqa: E402
from tmlm.engine.config import EngineConfig  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.engine.cuda.cuda_engine import CudaEngine  # noqa: E402
from tmlm.ssl import patch_tokens as PT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
E = pytest.importorskip("ssl_r4_images")

pytestmark = pytest.mark.gpu


def _store():
    rng = np.random.default_rng(0)
    n = 80
    imgs = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        r0, c0 = rng.integers(0, 18, 2)
        imgs[i, r0:r0 + 10, c0:c0 + 8] = 220
        imgs[i] |= (rng.random((28, 28)) < 0.03).astype(np.uint8) * 150
    labels = rng.integers(0, 10, n).astype(np.uint8)
    dev = PT.dev_mask(n, 16, salt=3)
    arrays, _ = PT.build_store_arrays(imgs, labels, dev, n_codes=63, cluster_max_iter=3, n_stop=8)
    keep = PT.f1_keep(arrays["targets"], arrays["tok"][arrays["targets"].astype(np.int64)], t=1e-2)
    ca = CorpusArrays(tok=arrays["tok"], doc_start=arrays["doc_start"], kind_map=arrays["kind_map"], stop=arrays["stop"],
                      sym_freq=arrays["sym_freq"], targets=arrays["targets"][keep].astype(np.uint32))
    return ca, n


def test_codes_and_top1_match_public_api_and_cpu_engine_parity():
    ca, n = _store()
    cfg = EngineConfig(mode="masked", vocab=64, n_clauses=512, seed=5, n_bag_k=3, batch=256, sweep_period=4)
    g = CudaEngine(cfg, ca)
    c = CpuEngine(cfg, ca)
    for e in range(2):
        for b in range(g.n_batches()):
            dg, dc = g.train_batch(e, b), c.train_batch(e, b)
            assert dg == dc
    assert g.state_hash() == c.state_hash()
    assert int(g.state()["alive"].sum()) > 0
    codes, info, df = E.image_codes(g, n, chunk_images=17)
    assert info["positions"] == n * 49 and codes.n_doc == n
    zero = np.zeros(ca.tok.shape[0], np.uint8)
    ev_p, ev_j = [], []
    for p in PT.patch_positions(np.arange(n)).tolist():
        a = g.fired_clauses(0, p, zero)
        assert np.array_equal(a, c.fired_clauses(0, p, zero))
        ev_p.append(np.full(a.shape[0], p, np.int64))
        ev_j.append(a.astype(np.int64))
    ref = PT.image_code_from_events(np.concatenate(ev_p), np.concatenate(ev_j), np.arange(n), n, cfg.n_clauses)
    assert np.array_equal(ref.indptr, codes.indptr) and np.array_equal(ref.ids, codes.ids) and np.array_equal(ref.cnt, codes.cnt)
    assert np.array_equal(df, np.bincount(ref.ids.astype(np.int64), minlength=4 * cfg.n_clauses))
    ok_agg, ok_bf = E.codes_check(g, codes, np.arange(n), np.random.default_rng(1), n_check=3)
    assert ok_agg and ok_bf
    # a row subset and a document-frequency subset
    keep = np.zeros(n, bool)
    keep[[1, 5, 6, 40, 79]] = True
    dfr = np.arange(n) % 3 == 0
    sub, info2, df2 = E.image_codes(g, n, chunk_images=23, keep=keep, df_rows=dfr)
    assert info2["rows_kept"] == 5 and sub.n_doc == 5
    full_sub = ref.rows(np.flatnonzero(keep))
    assert np.array_equal(sub.indptr, full_sub.indptr) and np.array_equal(sub.ids, full_sub.ids)
    assert np.array_equal(df2, np.bincount(ref.rows(np.flatnonzero(dfr)).ids.astype(np.int64), minlength=4 * cfg.n_clauses))
    ok_agg, ok_bf = E.codes_check(g, sub, np.flatnonzero(keep), np.random.default_rng(2), n_check=5)
    assert ok_agg and ok_bf
    hid = PT.frozen_hidden(ca.tok.shape[0])
    pos = PT.patch_positions(np.arange(n))
    t1 = E.masked_top1(g, pos, hid, chunk=300)
    sc = g.predict_scores(0, pos, hid)
    assert t1 == pytest.approx(float(np.mean(sc.argmax(axis=1) == ca.tok[pos.astype(np.int64)])))
