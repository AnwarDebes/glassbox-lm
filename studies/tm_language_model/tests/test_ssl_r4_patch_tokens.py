"""Tests of src/tmlm/ssl/patch_tokens.py (docs/13 A7, phase R4: Fashion-MNIST patch tokens) and of the pure parts of
experiments/ssl_r4_images.py (H17.10 rule). CPU only, synthetic images, deterministic; no download."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.engine.rng import mix64
from tmlm.ssl import patch_tokens as PT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def test_mix64_np_matches_the_reference():
    z = np.array([0, 1, 2, 12345, 2**40 + 7, 2**64 - 1], dtype=np.uint64)
    assert [int(v) for v in PT.mix64_np(z)] == [mix64(int(v)) for v in z]


def test_patch_pattern_bit_layout():
    img = np.zeros((1, 28, 28), np.uint8)
    img[0, 4 * 2 + 1, 4 * 5 + 3] = 200  # patch (2, 5), dy 1, dx 3 -> bit 7
    img[0, 0, 0] = 76                   # patch (0, 0), bit 0 (> 75)
    img[0, 27, 27] = 75                 # not > 75
    p = PT.patch_patterns(PT.binarize(img))
    assert p.shape == (1, 7, 7) and p.dtype == np.uint16
    assert p[0, 2, 5] == 1 << 7 and p[0, 0, 0] == 1 and p[0, 6, 6] == 0
    assert int(np.count_nonzero(p)) == 2


def test_codebook_order_and_nearest_code_lut():
    counts = np.zeros(1 << 16, np.int64)
    counts[[0, 0b1111, 0b1, 0xF000]] = [100, 50, 50, 10]
    cb = PT.build_codebook(counts, 4)
    assert cb.tolist() == [0, 0b1, 0b1111, 0xF000]  # count desc, then pattern asc
    lut = PT.nearest_code_lut(cb)
    for tid, pat in enumerate(cb.tolist()):
        assert lut[pat] == tid
    assert lut[0b0111] == 2          # distance 1 to 0b1111, 3 to 0b1, 3 to 0
    assert lut[0b11] == 1            # distance 2 to 0 and to 0b1111, 1 to 0b1 (token 1)
    # exhaustive check of the rule on random patterns: min Hamming, ties to the lower token id
    rng = np.random.default_rng(0)
    for pat in rng.integers(0, 1 << 16, size=300):
        d = [bin(int(pat) ^ int(c)).count("1") for c in cb]
        assert lut[pat] == int(np.argmin(d))


def test_lut_tie_rule_prefers_lower_token_id():
    counts = np.zeros(1 << 16, np.int64)
    counts[[0b0001, 0b0010]] = [10, 20]
    cb = PT.build_codebook(counts, 2)
    assert cb.tolist() == [0b0010, 0b0001]
    lut = PT.nearest_code_lut(cb)
    assert lut[0b0011] == 0 and lut[0] == 0  # both at distance 1: the lower token id (the more frequent pattern)


def test_image_tokens_layout_and_geometry():
    pats = np.arange(2 * 49, dtype=np.uint16).reshape(2, 7, 7)
    lut = np.arange(1 << 16, dtype=np.uint16) % 50
    tok = PT.image_tokens(pats, lut, row_end=99)
    assert tok.shape == (112,)
    assert np.all(tok.reshape(2, 7, 8)[:, :, 7] == 99)
    assert tok[8 * 3 + 2] == lut[pats[0, 3, 2]]
    ds = PT.doc_starts(2)
    assert ds.tolist() == [0, 56, 112] and ds.dtype == np.uint32
    img, r, c = PT.position_rc(112)
    pos = PT.patch_positions(np.array([1]))
    assert pos.shape == (49,) and np.all(c[pos] < 7) and np.all(img[pos] == 1)
    assert pos[0] == 56 and pos[48] == 56 + 6 * 8 + 6
    q = PT.quadrant(r[pos], c[pos])
    assert np.bincount(q).tolist() == [16, 12, 12, 9]
    # offset -8 from a patch below the first row is the patch directly above
    assert np.all(tok[pos[7:].astype(int) - 8] == tok[pos[:-7].astype(int)])


def test_dev_mask_is_exact_and_hash_defined():
    dev = PT.dev_mask(600, 100, salt=5)
    assert dev.sum() == 100
    h = [mix64(i ^ 5) for i in range(600)]
    assert set(np.flatnonzero(dev).tolist()) == set(np.argsort(h)[:100].tolist())
    full = PT.dev_mask()
    assert full.sum() == PT.DEV_SIZE and full.shape == (PT.N_IMAGES,)


def test_f1_keep_formula_and_determinism():
    rng = np.random.default_rng(1)
    cls = rng.choice(5, size=20000, p=[0.6, 0.2, 0.1, 0.0999, 0.0001])
    pos = np.arange(20000, dtype=np.uint64) * 3 + 7
    keep = PT.f1_keep(pos, cls, t=1e-4)
    assert np.array_equal(keep, PT.f1_keep(pos, cls, t=1e-4))
    f = np.bincount(cls, minlength=5) / cls.size
    for y in range(5):
        if f[y] == 0:
            continue
        prob = min(1.0, np.sqrt(1e-4 / f[y]) + 1e-4 / f[y])
        m = cls == y
        # scalar reference of the draw on the members of class y
        ref = [(prob >= 1.0) or ((mix64(int(p)) >> 32) < min(2**32 - 1, int(np.floor(prob * 2**32)))) for p in pos[m]]
        assert keep[m].tolist() == ref
    assert keep[cls == 0].mean() < 0.05


def test_stratified_sample():
    labels = np.repeat(np.arange(10), 30)
    pool = np.arange(300)[np.arange(300) % 7 != 0]
    a = PT.stratified_sample(labels, pool, 10, key=(1, 10, 0))
    b = PT.stratified_sample(labels, pool, 10, key=(1, 10, 0))
    c = PT.stratified_sample(labels, pool, 10, key=(1, 10, 1))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.bincount(labels[a]).tolist() == [10] * 10
    assert np.all(np.isin(a, pool)) and np.all(np.diff(a) > 0)
    with pytest.raises(ValueError):
        PT.stratified_sample(labels, pool, 29, key=(1,))


def test_random_kind_map_keeps_structure():
    V = 50
    km = np.zeros((4, V), np.uint16)
    km[0] = np.arange(V)
    rng = np.random.default_rng(2)
    l1 = rng.integers(0, 16, V)
    km[1] = V + l1
    km[2] = V + 16 + l1 * 16 + rng.integers(0, 16, V)
    km[3] = V + 272 + (km[2] - V - 16).astype(np.int64) * 16 + rng.integers(0, 16, V)
    r = PT.random_kind_map(km, 3)
    assert np.array_equal(r[0], km[0])
    assert not np.array_equal(r, km)
    for k in (1, 2, 3):
        assert sorted(r[k].tolist()) == sorted(km[k].tolist())
    # token t carries the whole cluster path of one other token
    cols = {tuple(km[1:, t]) for t in range(V)}
    assert all(tuple(r[1:, t]) in cols for t in range(V))
    assert np.array_equal(PT.random_kind_map(km, 3), r)


def _synthetic_images(n=120, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        r0, c0 = rng.integers(0, 20, 2)
        imgs[i, r0:r0 + 8, c0:c0 + 6] = 200
        imgs[i] |= (rng.random((28, 28)) < 0.02).astype(np.uint8) * 150
    labels = rng.integers(0, 10, n).astype(np.uint8)
    return imgs, labels


def test_store_build_write_load_roundtrip(tmp_path):
    from tmlm.engine.config import EngineConfig

    imgs, labels = _synthetic_images()
    dev = PT.dev_mask(120, 20, salt=9)
    arrays, meta = PT.build_store_arrays(imgs, labels, dev, n_codes=49, cluster_max_iter=3, n_stop=8)
    V = 50
    assert arrays["tok"].shape == (120 * 56,) and arrays["tok"].dtype == np.uint16 and int(arrays["tok"].max()) < V
    assert arrays["kind_map"].shape == (4, V) and meta["n_pool"] == 100
    assert arrays["targets"].shape == (100 * 49,)
    tgt_img = arrays["targets"].astype(np.int64) // 56
    assert not np.any(dev[tgt_img])
    assert np.all((arrays["targets"].astype(np.int64) % 56) % 8 < 7)
    stored = PT.write_store(arrays, meta, corpus="fm_test", root=tmp_path)
    again = PT.write_store(arrays, meta, corpus="fm_test", root=tmp_path)   # existing store: verified, not rewritten
    assert again["sha256"] == stored["sha256"]
    ca, extras, m2 = PT.load_store("fm_test", V, tmp_path, check=True)
    ca.validate(EngineConfig(mode="masked", vocab=V, n_clauses=64))
    assert np.array_equal(ca.tok, arrays["tok"]) and np.array_equal(extras["dev"], dev.astype(np.uint8))
    bad = dict(arrays)
    bad["labels"] = (arrays["labels"] + 1).astype(np.uint8)
    with pytest.raises(RuntimeError):
        PT.write_store(bad, meta, corpus="fm_test", root=tmp_path)


def test_bovw_and_event_aggregation():
    imgs, labels = _synthetic_images(12, 1)
    pats = PT.patch_patterns(PT.binarize(imgs))
    counts = PT.pattern_counts(pats)
    cb = PT.build_codebook(counts, 49)
    tok = PT.image_tokens(pats, PT.nearest_code_lut(cb), 49)
    sd = PT.bovw_quadrant(tok, np.array([3, 7]), vocab=50)
    assert sd.n_doc == 2 and sd.n_cols == 200
    for row, img in enumerate((3, 7)):
        ref = np.zeros(200, np.int64)
        for r in range(7):
            for c in range(7):
                ref[int(tok[img * 56 + r * 8 + c]) * 4 + int(PT.quadrant(np.array(r), np.array(c)))] += 1
        got = np.zeros(200, np.int64)
        s = sd.rows(np.array([row]))
        got[s.ids.astype(np.int64)] = s.cnt
        assert np.array_equal(ref, got)
    pos = np.array([56 * 2 + 0, 56 * 2 + 9, 56 * 5 + 54, 56 * 5 + 54])
    cl = np.array([3, 3, 1, 1])
    ev = PT.image_code_from_events(pos, cl, np.arange(6), 6, 8)
    assert ev.rows(np.array([2])).ids.tolist() == [12] and ev.rows(np.array([2])).cnt.tolist() == [2]
    assert ev.rows(np.array([5])).ids.tolist() == [1 * 4 + 3] and ev.rows(np.array([5])).cnt.tolist() == [2]


def test_frozen_hidden_rate():
    h = PT.frozen_hidden(200000)
    assert h.dtype == np.uint8 and abs(float(h.mean()) - 0.15) < 0.005
    assert np.array_equal(h, PT.frozen_hidden(200000))


def test_h17_10_rule():
    E = pytest.importorskip("ssl_r4_images")
    cells = []
    for s in (1, 2, 3):
        for d in range(5):
            cells.append({"probe": "lr", "budget": 10, "seed": s, "draw": d, "rep": "code_main", "acc": 0.70 + 0.001 * d})
            cells.append({"probe": "lr", "budget": 10, "seed": s, "draw": d, "rep": "code_young", "acc": 0.60})
            cells.append({"probe": "lr", "budget": 10, "seed": s, "draw": d, "rep": "bovw", "acc": 0.66 + 0.002 * s})
    r = E.h17_10(cells, 10)
    assert r["reference"] == "bovw" and r["pairs"] == 15 and r["meets"] is True
    for c in cells:
        if c["rep"] == "bovw":
            c["acc"] = 0.69
    r = E.h17_10(cells, 10)
    assert r["meets"] is False and r["diff_bootstrap"]["mean"] < 0.02
    assert E.h17_10([c for c in cells if c["rep"] != "code_main"], 10)["status"] == "incomplete"
