"""Lane L0 (freeze and cache): tmlm.cvtm.protocol, tmlm.cvtm.model, tmlm.cvtm.energy and
tmlm.cvtm.cache on small corpora (numpy backend; the real data is only read, never written).

    .venv-tm/bin/python -m pytest tests/test_cvtm_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm import cache as C
from tmlm.cvtm import energy as E
from tmlm.cvtm import kn_gpu, protocol
from tmlm.cvtm.doc_clauses import doc_candidates, doc_features
from tmlm.cvtm.mixture import (CACHE_COMPONENTS, COMPONENTS, bucket_ids, counts_matrix, fit_map_em,
                               full_distribution, gold_components)
from tmlm.cvtm.model import (CVTMCore, kn_from_arrays, kn_to_arrays, load_kn, save_kn, suffix_gold, suffix_ln_rows,
                             topk_rows)
from tmlm.engine.rng import mix64
from tmlm.eval.baselines.ngram import KNModel

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "wikitext103" / "bpe8000" / "train" / "meta.json").exists(),
    reason="the V 8000 WikiText-103 store is not built (data/ is not part of a clean checkout)")


# ------------------------------------------------------------------ corpora
def corpus(seed: int, vocab: int = 40, n_docs: int = 700, lo: int = 30, hi: int = 260):
    """Markov documents with copied spans (so that document clauses and the pointer fire)."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(vocab, 0.3), size=vocab)
    cum = np.cumsum(trans, axis=1)
    docs = []
    for _ in range(n_docs):
        n = int(rng.integers(lo, hi))
        t = [int(rng.integers(vocab))]
        for x in rng.random(n - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        if n > 40 and rng.random() < 0.6:                 # repeat an earlier span later in the document
            a = int(rng.integers(0, n // 2 - 12))
            L = int(rng.integers(6, 12))
            b = int(rng.integers(n // 2, n - L))
            t[b:b + L] = t[a:a + L]
        docs.append(t)
    tok = np.asarray([x for d in docs for x in d], dtype=np.uint16)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    return tok, ds


@pytest.fixture(scope="module")
def train():
    return corpus(1)


@pytest.fixture(scope="module")
def evalset():
    return corpus(2, n_docs=12)


@pytest.fixture(scope="module")
def core(train):
    return CVTMCore.fit(*train, vocab=40, backend="numpy")


# ------------------------------------------------------------------ protocol
def test_residue_is_mix64_and_vectorised():
    idx = np.arange(5000)
    r = protocol.residues(idx)
    assert r.dtype == np.int64
    for i in list(range(0, 5000, 7)) + [2**40 + 3]:
        assert protocol.residue(i) == mix64(i ^ 0xC0FFEE) % 100
    assert all(int(r[i]) == protocol.residue(i) for i in range(0, 5000, 3))
    assert np.array_equal(protocol.residues(5000), r)
    # the first H documents logged by results/20260924T054225Z_cvtm_early_fulldata_exploratory/
    for i in (21, 195, 254, 340, 495, 839, 948, 1038, 1095, 1117):
        assert protocol.residue(i) == 0
    assert protocol.residue(0) != 0
    with pytest.raises(ValueError):
        protocol.residue(-1)


def test_select_split_prefix_and_roles():
    tok = np.arange(1000, dtype=np.uint16) % 50
    ds = np.array([0, 10, 10, 250, 400, 401, 1000], dtype=np.int64)
    keep = np.array([True, False, True, False, True, False])
    sub = protocol.select_docs(tok, ds, keep)
    assert sub.n_docs == 3 and np.array_equal(sub.doc_index, [0, 2, 4])
    assert np.array_equal(sub.doc_start, [0, 10, 250, 251])
    assert np.array_equal(sub.tok, np.concatenate([tok[0:10], tok[10:250], tok[400:401]]))
    assert np.array_equal(sub.residue, protocol.residues(np.array([0, 2, 4])))
    assert np.array_equal(sub.role_of(), np.repeat(sub.residue.astype(np.int8), [10, 240, 1]))
    n_docs = 3000
    ds2 = np.arange(0, 3 * n_docs + 1, 3, dtype=np.int64)
    tok2 = np.zeros(ds2[-1], dtype=np.uint16)
    held, rest = protocol.split_by_residue(tok2, ds2, (0, 1, 2))
    assert held.n_docs + rest.n_docs == n_docs and held.n_tokens + rest.n_tokens == ds2[-1]
    assert set(np.unique(held.residue)) <= {0, 1, 2} and not np.isin(rest.residue, [0, 1, 2]).any()
    assert np.array_equal(protocol.doc_mask(n_docs, (0, 1, 2)), np.isin(protocol.residues(n_docs), [0, 1, 2]))
    # prefix cut: the rule of experiments/m3_tiny_lm.py slice_train
    for n_tok in (1, 9, 10, 11, 250, 399, 401, 999, 1000, 5000, None):
        t, d = protocol.prefix_documents(tok, ds, n_tok)
        if n_tok is None or n_tok >= 1000:
            assert d[-1] == 1000
            continue
        dd = int(np.searchsorted(ds, n_tok, side="right")) - 1
        end = int(ds[max(1, dd)])
        want = ds[:dd + 1] if ds[dd] == end else ds[:dd + 2]
        assert np.array_equal(d, want) and t.shape[0] == end
    assert protocol.ROLES == {0: "core_em", 1: "t2_train", 2: "selection"}


def test_bootstrap_helpers():
    idx = protocol.bootstrap_index(7, n_boot=50)
    assert idx.shape == (50, 7) and np.array_equal(idx, protocol.bootstrap_index(7, n_boot=50))
    bits = np.array([10.0, 20, 30, 40, 50, 60, 70])
    by = np.array([5.0, 5, 5, 5, 5, 5, 5])
    r = protocol.bootstrap_bpb(bits, by, idx)
    assert r["point"] == bits.sum() / by.sum()
    assert r["ci95"][0] <= r["point"] <= r["ci95"][1]
    same = protocol.bootstrap_bpb(bits, by, idx, doc_bits_ref=bits)
    assert same["point"] == 0.0 and same["ci95"] == [0.0, 0.0]
    c = protocol.bootstrap_bpb(bits, by, idx, const=1.5)
    assert c["point"] == pytest.approx(r["point"] - 1.5)


def test_test_split_needs_a_purpose():
    with pytest.raises(ValueError):
        protocol.load_eval_split("test")
    with pytest.raises(ValueError):
        protocol.load_eval_split("train")


@needs_data
def test_h_and_hprime_on_the_real_train_split(monkeypatch):
    monkeypatch.setenv("TMLM_DATA_DIR", str(DATA_ROOT))          # read only
    ds = np.load(DATA_ROOT / "wikitext103" / "bpe8000" / "train" / "doc_start.npy").astype(np.int64)
    n = ds.shape[0] - 1
    s = protocol.selection_summary(n, ds, (0,))
    h = np.flatnonzero(protocol.residues(n) == 0)
    protocol.check_h(len(h), int(np.diff(ds)[h].sum()), h)
    assert s["n_docs"] == 304 and s["n_tokens"] == 1_397_702
    s3 = protocol.selection_summary(n, ds, (0, 1, 2))
    assert s3["per_residue"]["0"]["doc_index_sha256"] == protocol.EXPECTED_H["doc_index_sha256"]
    assert s3["n_tokens"] == sum(v["n_tokens"] for v in s3["per_residue"].values())
    with pytest.raises(AssertionError):
        protocol.check_h(len(h) - 1, int(np.diff(ds)[h].sum()), h[:-1])
    hs = protocol.dataset_hashes(("train", "validation"))
    meta = json.loads((DATA_ROOT / "wikitext103" / "bpe8000" / "validation" / "meta.json").read_text())
    assert hs["validation_tok_V8000"] == meta["sha256"]["tok.npy"]
    assert set(hs) == {"train_tok_V8000", "train_doc_start_V8000", "validation_tok_V8000", "validation_doc_start_V8000",
                       "validation_text_utf8", "validation_doc_bytes"}


# ------------------------------------------------------------------ model: exactness
def test_fit_matches_the_reference_pipeline(core, train):
    tok, ds = train
    held, rest = protocol.split_by_residue(tok, ds, (0,))
    assert held.n_docs >= 3
    # the KN on all documents is KNModel.fit, array for array
    kn_ref = KNModel(5, 40).fit(tok, ds)
    assert kn_gpu.compare_models(core.kn, kn_ref) == []
    # the EM inputs are KNModel-without-H probabilities on H (vote form, ln within 1e-12)
    kn_noh = KNModel(5, 40).fit(rest.tok, rest.doc_start)
    ctx, valid = kn_noh.contexts(held.tok, held.doc_start, np.arange(held.n_tokens))
    p = kn_noh.prob(ctx, held.tok.astype(np.int64), valid)
    assert np.max(np.abs(core.fit_inputs.lnp_s - np.log(p))) <= 1e-12
    assert np.array_equal(core.fit_inputs.doc_index, held.doc_index)
    # the weights are mixture.fit_map_em on those inputs
    feats = doc_features(held.tok, held.doc_start)
    pc, act = gold_components(np.exp(core.fit_inputs.lnp_s), feats, held.tok.astype(np.int64))
    b, nb = bucket_ids(core.fit_inputs.kstar.astype(np.int64), counts_matrix(feats), feats.ptr_len)
    w = fit_map_em(pc, act, b, iters=50, prior=16.0, n_buckets=nb)
    assert np.array_equal(w, core.weights)
    assert core.weights.shape == (520, 7)


def test_kn5_alone_is_exactly_kn(core, evalset):
    tok, ds = evalset
    kn5 = core.variant("S")
    assert kn5.components == ("S",)
    sc = kn5.score_stream(tok, ds, n_bytes=12345)
    ref = core.kn.logprob(tok, ds)
    assert np.max(np.abs(sc["log2p"] - ref)) <= 1e-12
    lnp, ks = suffix_gold(core.suffix, tok, ds)
    assert np.array_equal(sc["log2p"], np.log2(np.exp(lnp)))       # weight 1.0 exactly: P = P_S
    assert sc["bits_per_byte"] == sc["total_bits"] / 12345


def test_window_zero_reduces_to_kn(core, evalset):
    tok, ds = evalset
    w0 = core.variant(window=0)
    assert w0.window == 0 and w0.components == COMPONENTS
    r = w0.gold(tok, ds)
    assert not r.act[:, 1:].any()
    assert np.array_equal(r.p_gold, np.exp(r.lnp_s))
    assert w0.state_hash() != core.state_hash()


def test_core_beats_kn_on_copy_corpus_and_components(core, evalset):
    tok, ds = evalset
    full = core.score_stream(tok, ds)["bits_per_token"]
    cache = core.variant("S+D0")
    assert cache.components == CACHE_COMPONENTS and cache.weights.shape == (30, 7)
    kn = core.variant("S").score_stream(tok, ds)["bits_per_token"]
    assert full < kn                                        # the planted copies are predictable in context


def test_dense_pass_equals_mixture_full_distribution(core, evalset):
    tok, ds = evalset
    rec = core.gold(tok, ds)
    pos = np.arange(0, tok.shape[0], 3)
    res = core.dense_pass(rec, 10, positions=pos, chunk=97)
    ch = res["checks"]
    assert ch["base_gold_max_abs_diff"] == 0.0 and ch["core_gold_max_abs_diff"] == 0.0
    assert ch["base_row_sum_max_abs_err"] < 1e-12 and ch["core_row_sum_max_abs_err"] < 1e-12
    tok64 = tok.astype(np.int64)
    ctx, valid = core.kn.contexts(tok64, ds, pos)
    base = core.kn.dist(ctx, valid)
    cands = doc_candidates(tok64, ds, pos)
    for r, p in enumerate(pos[:200]):
        rows = [cands[m].row(r) for m in range(5)]
        pc = int(rec.feats.ptr_cand[p]) if rec.feats.ptr_len[p] >= 5 else -1
        full = full_distribution(base[r], rows, pc, core.weights[rec.bucket[p]])
        np.testing.assert_allclose(res["core_p"][r], full[res["core_ids"][r].astype(np.int64)], rtol=0, atol=1e-14)
        want = sorted(range(40), key=lambda j: (-full[j], j))[:10]
        top = [int(x) for x in res["core_ids"][r]]
        if top != want:                                     # only exact float ties may reorder
            assert np.allclose(full[top], full[want], rtol=0, atol=1e-15)
        np.testing.assert_allclose(res["base_p"][r], base[r][res["base_ids"][r].astype(np.int64)], rtol=0, atol=1e-14)
    ids, vals = core.topk(tok, ds, pos[:50], k=10, rec=rec)
    assert np.array_equal(ids, res["core_ids"][:50].astype(np.int64)) and np.array_equal(vals, res["core_p"][:50])
    with pytest.raises(ValueError):
        core.dense_pass(rec, 10, positions=np.array([3, 2]))


def test_suffix_ln_rows_bit_identical_to_logprob(core, evalset):
    tok, ds = evalset
    sc = core.suffix
    pos = np.arange(tok.shape[0])
    ctx, valid = sc.contexts(tok, ds, pos)
    rows = suffix_ln_rows(sc, ctx, valid)
    for w in range(40):
        assert np.array_equal(rows[:, w], sc.logprob(ctx, np.full(pos.shape[0], w), valid))
    assert np.all(np.isneginf(rows[:, 40]))


def test_topk_rows_canonical_ties():
    rng = np.random.default_rng(3)
    p = rng.integers(0, 4, size=(200, 30)).astype(np.float64)
    for k in (1, 5, 30):
        ids, vals = topk_rows(p, k)
        for i in range(200):
            want = sorted(range(30), key=lambda j: (-p[i, j], j))[:k]
            assert list(ids[i]) == want
            assert np.array_equal(vals[i], p[i, want])
    with pytest.raises(ValueError):
        topk_rows(p, 31)


def test_explain_is_readable_and_consistent(core, evalset):
    tok, ds = evalset
    rec = core.gold(tok, ds)
    p = int(np.flatnonzero(rec.act[:, 6])[0])              # a position where the pointer fires
    items, text = core.explain(tok, ds, p, token_text=lambda t: f"t{t}")
    assert items[0]["family"] == "S" and items[-1]["family"] == "total"
    assert items[-1]["p"] == pytest.approx(float(rec.p_gold[p]), rel=0, abs=0)
    assert "IF the last" in text and "BIAS" in text and "bits" in text
    assert any(it["family"] == "P" for it in items)


# ------------------------------------------------------------------ model: determinism and persistence
def test_fit_is_deterministic(core, train):
    again = CVTMCore.fit(*train, vocab=40, backend="numpy")
    assert again.state_hash() == core.state_hash()
    assert np.array_equal(again.weights, core.weights)


def test_save_load_round_trip(core, evalset, tmp_path):
    rec = core.save(tmp_path / "core.npz")
    assert rec["state_hash"] == core.state_hash()
    back = CVTMCore.load(tmp_path / "core.npz", backend="numpy")
    assert back.state_hash() == core.state_hash()
    tok, ds = evalset
    assert np.array_equal(back.score_stream(tok, ds)["log2p"], core.score_stream(tok, ds)["log2p"])
    assert kn_gpu.compare_models(back.kn, core.kn) == []
    assert np.array_equal(back.variant("S+D0").weights, core.variant("S+D0").weights)
    with np.load(tmp_path / "core.npz") as z:                 # only integer arrays besides the weights and lnp_s
        floats = sorted(k for k in z.files if z[k].dtype.kind == "f")
    assert floats == ["fit.lnp_s", "weights"]
    # a tampered weight is refused
    with np.load(tmp_path / "core.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["weights"] = arrays["weights"].copy()
    arrays["weights"][0, 0] += 1e-9
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError):
        CVTMCore.load(tmp_path / "bad.npz", backend="numpy")


def test_pruned_model_round_trip(train, evalset, tmp_path):
    tok, ds = train
    pr = CVTMCore.fit(tok, ds, vocab=40, backend="numpy", min_counts={3: 2, 4: 2, 5: 2}, components="S+D0")
    assert pr.config["min_counts"] == {3: 2, 4: 2, 5: 2}
    assert any(isinstance(lv, kn_gpu._PrunedLevel) for lv in pr.kn.raw[1:] if lv is not None)
    info = pr.save(tmp_path / "pruned.npz")
    back = CVTMCore.load(tmp_path / "pruned.npz", backend="numpy")
    assert back.state_hash() == pr.state_hash() == info["state_hash"]
    et, eds = evalset
    assert np.array_equal(back.score_stream(et, eds)["log2p"], pr.score_stream(et, eds)["log2p"])
    ref = kn_gpu.fit_kn(tok, ds, 5, 40, backend="numpy", min_counts={3: 2, 4: 2, 5: 2})
    assert kn_gpu.compare_models(pr.kn, ref) == []


def test_kn_serialisation_is_exact(train, tmp_path):
    kn = KNModel(5, 40).fit(*train)
    arrays, header = kn_to_arrays(kn)
    assert all(a.dtype.kind in "iub" for a in arrays.values())
    back = kn_from_arrays(arrays, header)
    assert kn_gpu.compare_models(back, kn) == []
    rec = save_kn(kn, tmp_path / "kn.npz", extra_meta={"variant": "test"})
    kn2, meta = load_kn(tmp_path / "kn.npz")
    assert meta["variant"] == "test" and kn_gpu.compare_models(kn2, kn) == []
    assert rec["arrays_sha256"] == header["hashes"]
    arrays["kn.raw.3.cnt"] = arrays["kn.raw.3.cnt"].copy()
    arrays["kn.raw.3.cnt"][0] += 1
    with pytest.raises(ValueError):
        kn_from_arrays(arrays, header)


# ------------------------------------------------------------------ energy (mocked NVML)
class FakeNvml:
    def __init__(self, watts, fail_init: bool = False):
        self.watts = list(watts)
        self.i = 0
        self.fail_init = fail_init

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML not found")

    def nvmlDeviceGetHandleByIndex(self, index):
        return ("handle", index)

    def nvmlDeviceGetPowerUsage(self, handle):
        w = self.watts[min(self.i, len(self.watts) - 1)]
        self.i += 1
        return int(w * 1000)


class FakeClock:
    def __init__(self, step: float):
        self.t = 1000.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def test_energy_constant_power_and_idle_subtraction():
    clock = FakeClock(0.5)
    meter = E.EnergyMeter(nvml=FakeNvml([150.0]), clock=clock, idle_w=40.0, period_s=0.001)
    with meter.measure("fit"):
        time.sleep(0.05)
    r = meter.results["fit"]
    assert r["available"] and r["n_samples"] >= 2
    assert r["mean_w"] == pytest.approx(150.0)
    assert r["joules_raw"] == pytest.approx(150.0 * r["span_s"])
    assert r["joules_idle_subtracted"] == pytest.approx(110.0 * r["span_s"])
    assert r["idle_w"] == 40.0 and r["cpu_s"] >= 0.0


def test_energy_trapezoid_and_idle_measurement():
    s = E.NvmlPowerSampler(nvml=FakeNvml([100.0, 200.0, 100.0, 200.0] * 1000), clock=FakeClock(1.0), period_s=0.001)
    idle = s.measure_idle(10.0, sleep=lambda sec: time.sleep(0.05))
    t = np.array([x[0] for x in s.samples])
    p = np.array([x[1] for x in s.samples])
    assert idle["n_samples"] == len(s.samples) >= 2
    want = float(np.sum((p[1:] + p[:-1]) / 2 * np.diff(t)))
    assert idle["joules"] == pytest.approx(want)
    assert idle["span_s"] == pytest.approx(t[-1] - t[0])
    meter = E.EnergyMeter(idle_s=10.0, nvml=FakeNvml([60.0]), clock=FakeClock(0.1), period_s=0.001,
                          sleep=lambda sec: time.sleep(0.03))
    assert meter.idle_w == pytest.approx(60.0) and meter.idle["n_samples"] >= 2
    assert E.idle_subtracted({"joules": 100.0, "span_s": 2.0}, 10.0) == 80.0
    assert E.idle_subtracted({"joules": None, "span_s": 2.0}, 10.0) is None


def test_energy_without_nvml():
    s = E.NvmlPowerSampler(nvml=FakeNvml([1.0], fail_init=True))
    assert not s.available and "NVML" in s.error
    s.start()
    out = s.stop()
    assert out["joules"] is None and out["available"] is False
    meter = E.EnergyMeter(nvml=FakeNvml([1.0], fail_init=True))
    with meter.measure("x"):
        pass
    assert meter.idle_w is None and meter.results["x"]["joules_idle_subtracted"] is None


# ------------------------------------------------------------------ cache writers and loaders
def test_cache_writers_loaders_and_parallel_dense(core, evalset, tmp_path):
    tok, ds = evalset
    cdir = tmp_path / "cache"
    g = cdir / "validation"
    rec = core.gold(tok, ds)
    files = {}
    files["gold"] = C.write_array(g, "gold", tok.astype(np.uint16))
    files["doc_start"] = C.write_array(g, "doc_start", ds.astype(np.int64))
    files["residue"] = C.write_array(g, "residue", np.array([0, 1, 2] * 4, dtype=np.int8))
    for name, arr in (("doc_n", rec.feats.n), ("doc_g", rec.feats.g), ("ptr_len", rec.feats.ptr_len),
                      ("ptr_cand", rec.feats.ptr_cand), ("ptr_pos", rec.feats.ptr_pos)):
        files[name] = C.write_array(g, name, arr.astype(np.dtype(C.ARRAYS[name][0])))
    with pytest.raises(TypeError):
        C.write_array(g, "gold", tok.astype(np.int64))
    one = C.parallel_dense(core, rec, g, k=8, workers=1, chunk=64)
    one_arrays = {n: np.load(g / f"{n}.npy").copy() for n in ("kn_top_ids", "kn_top_p", "core_top_ids", "core_top_p")}
    two = C.parallel_dense(core, rec, g, k=8, workers=2, chunk=64)
    for n, a in one_arrays.items():
        assert np.array_equal(np.load(g / f"{n}.npy"), a), n
        assert one["files"][n]["sha256"] == two["files"][n]["sha256"]
    assert two["checks"]["core_gold_max_abs_diff"] == 0.0 and two["checks"]["positions"] == tok.shape[0]
    files.update(two["files"])
    kn_rec = save_kn(core.kn, cdir / C.KN_VARIANTS["all"])
    core_rec = core.save(cdir / C.CORE_FILE)
    frecs = {f"validation/{n}.npy": {"sha256": r["sha256"]} for n, r in files.items()}
    frecs[C.KN_VARIANTS["all"]] = {"sha256": kn_rec["sha256"]}
    frecs[C.CORE_FILE] = {"sha256": core_rec["sha256"]}
    C.write_meta(cdir, {"files": frecs, "core": {"state_hash": core_rec["state_hash"]}})
    assert C.read_meta(cdir)["format"] == C.FORMAT
    assert C.verify(cdir) == []
    arr = C.load_arrays(cdir, "validation", ["gold", "doc_n", "core_top_ids"])
    assert isinstance(arr["gold"], np.memmap) and np.array_equal(arr["gold"], tok)
    assert np.array_equal(arr["core_top_ids"][:, 0], one_arrays["core_top_ids"][:, 0])
    full = C.load_arrays(cdir, "validation", ["doc_start", "residue", "doc_n", "doc_g", "ptr_len", "ptr_cand", "ptr_pos"])
    assert C.doc_features_of(full).same_values(rec.feats)
    m = C.role_mask(full, "t2_train")
    assert np.array_equal(m, np.repeat(np.array([0, 1, 2] * 4) == 1, np.diff(ds)))
    assert np.array_equal(C.role_mask(full, [0, "selection"]), np.repeat(np.array([0, 1, 2] * 4) != 1, np.diff(ds)))
    kn = C.load_kn_variant(cdir, "all")
    assert kn_gpu.compare_models(kn, core.kn) == []
    back = C.load_core(cdir, backend="numpy")
    assert back.state_hash() == core.state_hash()
    with pytest.raises(KeyError):
        C.load_arrays(cdir, "validation", ["nope"])
    with pytest.raises(ValueError):
        C.load_arrays(cdir, "test")
    # tampering is detected
    p = g / "gold.npy"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    assert C.verify(cdir) == ["validation/gold.npy: sha256 differs"]
