"""Tests for experiments/cvtm_t1_readout.py (docs/13 amendment A3, H8.1; docs/11 sections 11.5 and 11.6).

Covered: the fit-document rule (splitmix64 of docs/11 section 3, checked against an independent numpy
implementation and the standard SplitMix64 value), the clause groups (equal to the probe's buckets),
the exact streamed counting (equal to dense counting, independent of how the events are split into
batches and clause ranges), the log-linear objective's analytic gradient against finite differences,
the linear mixture fit, the bootstrap and verdict helpers, and the GPU read-out kernel against the
float64 CPU reference kernel of experiments/ckm_posthoc_probe.py (skipped when no GPU is visible).
All data are synthetic and seeded; no corpus, no results/ access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import cvtm_t1_readout as t1  # noqa: E402
from ckm_posthoc_probe import k_dense_eval, k_dense_to_csr, k_gval, k_loglin  # noqa: E402


# ------------------------------------------------------------------ helpers
def np_splitmix64(z: np.ndarray) -> np.ndarray:
    """Independent vectorised SplitMix64 (docs/11 section 3), arithmetic mod 2^64."""
    z = z.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def synthetic(C: int = 40, V: int = 30, n_rows: int = 60, seed: int = 0, max_fired: int = 8) -> dict:
    """A small count table, fired lists and gold labels; clause 0 has no counts."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((C, V), np.int32)
    for j in range(1, C):
        k = int(rng.integers(1, min(V, 60)))
        cls = rng.choice(V, size=k, replace=False)
        counts[j, cls] = rng.integers(1, 50, size=k)
    indptr, idx, val = k_dense_to_csr(counts)
    N = t1.k_row_sums(indptr, val)
    D = np.diff(indptr)
    lam, inv_n = t1.witten_bell(N, D)
    p_uni = t1.add_one_unigram(rng.integers(0, V, 500), V)
    gval = k_gval(indptr, idx, val, lam, inv_n, p_uni)
    grp = t1.clause_groups(rng.integers(0, 15, C), N)
    ptr = [0]
    ids = []
    for _ in range(n_rows):
        m = int(rng.integers(0, max_fired + 1))
        f = np.sort(rng.choice(C, size=m, replace=False)).astype(np.uint32)
        ids.append(f)
        ptr.append(ptr[-1] + m)
    ptr = np.array(ptr, np.int64)
    ids = np.concatenate(ids)
    rp, rowid, comp = t1.csr_select(ptr, ids, N > 0)
    rows = dict(n=n_rows, rp=rp, rowid=rowid, comp=comp, bkt=grp[comp].astype(np.int64), gold=rng.integers(0, V, n_rows).astype(np.int64))
    return dict(C=C, V=V, counts=counts, indptr=indptr, idx=idx, val=val, N=N, D=D, lam=lam, inv_n=inv_n, p_uni=p_uni,
                logpu=np.log(p_uni), gval=gval, grp=grp, rows=rows)


def cpu_objective(s: dict, theta: np.ndarray, used: np.ndarray) -> tuple[float, np.ndarray]:
    alpha = np.zeros(t1.N_BUCKETS)
    alpha[used] = theta[:-1]
    ll, grad, gb, _, _, _ = t1.cpu_loglin(s["rows"], alpha, float(theta[-1]), s["indptr"], s["idx"], s["gval"], s["logpu"], True)
    R = s["rows"]["n"]
    return -ll / R, -np.append(grad[used], gb) / R


# ------------------------------------------------------------------ fit documents (docs/11 11.6)
def test_splitmix64_reference_value_and_fit_rule():
    from tmlm.engine.rng import mix64
    assert mix64(0) == 0xE220A8397B1DCDAF  # first output of SplitMix64 seeded with 0
    n = 30000
    mask = t1.fit_document_mask(n)
    ref = (np_splitmix64(np.arange(n, dtype=np.uint64) ^ np.uint64(0xC0FFEE)) % np.uint64(100)) == 0
    assert np.array_equal(mask, ref)
    assert 0.005 < mask.mean() < 0.015  # about one document in a hundred
    assert np.array_equal(t1.fit_document_mask(813), mask[:813])  # a prefix of the split keeps its fit documents


# ------------------------------------------------------------------ groups, Witten-Bell, projected gradient
def test_clause_groups_equal_probe_buckets():
    rng = np.random.default_rng(1)
    n_inc = rng.integers(0, 20, 5000)
    N = np.concatenate([np.array([0, 1, 2, 3, 4, 7, 8, 1 << 20, (1 << 23) - 1, 1 << 23, 1 << 30]), rng.integers(0, 1 << 26, 4989)])
    got = t1.clause_groups(n_inc, N)
    fam = 0
    logn = np.minimum(np.floor(np.log2(np.maximum(N, 1))).astype(np.int64), 23)
    probe = (fam * 13 + np.minimum(n_inc, 12)) * 24 + logn  # experiments/ckm_posthoc_probe.py `bucket`
    assert np.array_equal(got, probe)
    assert got.max() < t1.N_BUCKETS


def test_witten_bell_vote_identity():
    # lam / (1 - lam) = N / D, so q = ln(1 + O / (D P_uni)) (docs/11 11.5)
    s = synthetic(seed=2)
    for j in range(1, s["C"]):
        lo, hi = s["indptr"][j], s["indptr"][j + 1]
        q = s["gval"][lo:hi]
        alt = np.log1p(s["val"][lo:hi] / (s["D"][j] * s["p_uni"][s["idx"][lo:hi]]))
        assert np.allclose(q, alt, rtol=1e-12, atol=0)
    assert s["lam"][0] == 0.0 and s["inv_n"][0] == 0.0


def test_projected_gradient_norm_respects_bounds():
    x = np.array([0.0, 1.0, -1.0])
    g = np.array([0.5, -2.0, 3.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    # x[1] at the upper bound with a negative gradient and x[2] at the lower bound with a positive one: no move possible
    assert t1.projected_gradient_inf_norm(x, g, lo, hi) == 0.5


# ------------------------------------------------------------------ exact streamed counting
def test_merge_sorted_counts_edge_cases():
    e_k = np.zeros(0, np.uint32)
    e_c = np.zeros(0, np.int32)
    k, c = t1.k_merge_sorted_counts(e_k, e_c, e_k, e_c)
    assert k.shape[0] == 0 and c.shape[0] == 0
    a_k = np.array([1, 5, 9], np.uint32)
    a_c = np.array([2, 3, 4], np.int32)
    k, c = t1.k_merge_sorted_counts(a_k, a_c, e_k, e_c)
    assert np.array_equal(k, a_k) and np.array_equal(c, a_c)
    b_k = np.array([0, 5, 10], np.uint32)
    b_c = np.array([1, 10, 1], np.int32)
    k, c = t1.k_merge_sorted_counts(a_k, a_c, b_k, b_c)
    assert k.tolist() == [0, 1, 5, 9, 10] and c.tolist() == [1, 2, 13, 4, 1]
    k, c = t1.k_merge_sorted_counts(a_k, a_c, a_k, a_c)
    assert k.tolist() == [1, 5, 9] and c.tolist() == [4, 6, 8]


@pytest.mark.parametrize("range_bits", [3, 5, 16])
def test_streamed_counts_equal_dense_counts_for_any_batching(range_bits):
    C, V = 300, 50
    rng = np.random.default_rng(3)
    j = rng.integers(0, C, 40000)
    y = rng.integers(0, V, 40000)
    j[:500] = C - 1  # the last clause range is partial
    dense = np.zeros((C, V), np.int32)
    np.add.at(dense, (j, y), 1)
    ref = k_dense_to_csr(dense)
    keys = j.astype(np.uint64) * np.uint64(V) + y.astype(np.uint64)
    results = []
    for split_seed in (0, 1):
        acc = t1.SortedCountAccumulator(C, V, range_bits=range_bits)
        cuts = np.sort(np.random.default_rng(split_seed).choice(np.arange(1, keys.shape[0]), size=7, replace=False))
        for part in np.split(np.random.default_rng(10 + split_seed).permutation(keys), cuts):
            acc.add_unsorted(part)
        assert acc.events == keys.shape[0]
        results.append(acc.finalize())
    one = t1.SortedCountAccumulator(C, V, range_bits=range_bits)
    one.add_unsorted(keys)
    results.append(one.finalize())
    for indptr, idx, val in results:
        assert np.array_equal(indptr, ref[0])
        assert np.array_equal(idx.astype(np.int64), ref[1].astype(np.int64))
        assert np.array_equal(val, ref[2])
        assert idx.dtype == np.uint16 and val.dtype == np.int32
    assert int(t1.k_row_sums(results[0][0], results[0][2]).sum()) == keys.shape[0]


def test_per_range_feeding_equals_global_keys():
    # the causal run downloads one clause range at a time (add_range); the result must equal add()
    C, V, rb = 100, 13, 3
    rng = np.random.default_rng(11)
    keys = np.unique(rng.integers(0, C * V, 3000).astype(np.uint64))
    cnts = rng.integers(1, 9, keys.shape[0]).astype(np.int32)
    a = t1.SortedCountAccumulator(C, V, range_bits=rb)
    a.add(keys, cnts)
    b = t1.SortedCountAccumulator(C, V, range_bits=rb)
    bounds = np.searchsorted(keys, np.arange(b.n_ranges + 1, dtype=np.uint64) * np.uint64(b.span))
    for r in range(b.n_ranges - 1, -1, -1):  # any range order
        lo, hi = bounds[r], bounds[r + 1]
        b.add_range(r, (keys[lo:hi] - np.uint64(r * b.span)).astype(np.uint32), cnts[lo:hi])
    for x, y in zip(a.finalize(), b.finalize()):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError):
        b.add_range(0, np.array([b.span], np.uint32), np.array([1], np.int32))


def _events_with_heavy_pairs(C, V, n, seed):
    rng = np.random.default_rng(seed)
    j = rng.integers(0, C, n)
    y = rng.integers(0, V, n)
    heavy = np.array([[0, 0], [C - 1, V - 1], [C // 2, 3]])
    reps = np.array([255, 70000, 254])  # at, far above and just below the uint8 overflow threshold
    j = np.concatenate([j, np.repeat(heavy[:, 0], reps)])
    y = np.concatenate([y, np.repeat(heavy[:, 1], reps)])
    return j, y


@pytest.mark.parametrize("range_bits", [2, 4, 16])
def test_compact_counts_equal_dense_counts_including_overflow(range_bits):
    C, V = 150, 40  # C is not a multiple of 2^range_bits for range_bits 2 and 4
    j, y = _events_with_heavy_pairs(C, V, 30000, 12)
    dense = np.zeros((C, V), np.int64)
    np.add.at(dense, (j, y), 1)
    ref = k_dense_to_csr(dense.astype(np.int32))
    keys = j.astype(np.uint64) * np.uint64(V) + y.astype(np.uint64)
    outs = []
    for split_seed in (0, 1):
        acc = t1.CompactCountAccumulator(C, V, range_bits=range_bits)
        cuts = np.sort(np.random.default_rng(split_seed).choice(np.arange(1, keys.shape[0]), size=9, replace=False))
        for part in np.split(np.random.default_rng(20 + split_seed).permutation(keys), cuts):
            acc.add_unsorted(part)
        assert acc.events == keys.shape[0]
        assert acc.n_overflow == int((dense >= 255).sum())
        outs.append(acc.finalize())
    ref_sorted = t1.SortedCountAccumulator(C, V, range_bits=min(range_bits, 16))
    ref_sorted.add_unsorted(keys)
    outs.append(ref_sorted.finalize())
    for indptr, idx, val in outs:
        assert np.array_equal(indptr, ref[0])
        assert np.array_equal(idx.astype(np.int64), ref[1].astype(np.int64))
        assert np.array_equal(val.astype(np.int64), ref[2].astype(np.int64))
    with pytest.raises(ValueError):
        t1.CompactCountAccumulator(C, V, range_bits=range_bits).add_range(0, np.array([0, 0], np.uint32), np.array([1, 1]))


def test_compact_counts_to_device_match_host():
    cp = _gpu_or_skip()
    C, V = 150, 40
    j, y = _events_with_heavy_pairs(C, V, 20000, 13)
    keys = j.astype(np.uint64) * np.uint64(V) + y.astype(np.uint64)
    a = t1.CompactCountAccumulator(C, V, range_bits=4)
    b = t1.CompactCountAccumulator(C, V, range_bits=4)
    for acc in (a, b):
        for part in np.array_split(keys, 5):
            acc.add_unsorted(part)
    indptr, idx, val = a.finalize()
    d_indptr, d_idx, d_val, N = b.finalize_to_device(cp)
    assert np.array_equal(d_indptr, indptr)
    assert np.array_equal(d_idx.get(), idx) and np.array_equal(d_val.get(), val)
    assert np.array_equal(N, t1.k_row_sums(indptr, val))


def test_accumulator_rejects_bad_input_and_empty_finalize():
    acc = t1.SortedCountAccumulator(10, 7, range_bits=2)
    with pytest.raises(ValueError):
        acc.add(np.array([3, 3], np.uint64), np.array([1, 1], np.int32))
    with pytest.raises(ValueError):
        acc.add(np.array([5, 4], np.uint64), np.array([1, 1], np.int32))
    with pytest.raises(ValueError):
        acc.add(np.array([70], np.uint64), np.array([1], np.int32))
    indptr, idx, val = acc.finalize()
    assert indptr.shape == (11,) and int(indptr[-1]) == 0 and idx.shape[0] == 0 and val.shape[0] == 0
    with pytest.raises(ValueError):
        t1.SortedCountAccumulator(1 << 20, 8000, range_bits=20)  # local keys would not fit in uint32


# ------------------------------------------------------------------ the log-linear objective and the linear mixture
def test_loglin_gradient_matches_finite_differences():
    s = synthetic(seed=4)
    used = np.unique(s["rows"]["bkt"])
    rng = np.random.default_rng(5)
    theta = np.append(rng.normal(0.3, 0.2, used.shape[0]), 0.8)
    f0, g0 = cpu_objective(s, theta, used)
    h = 1e-6
    for i in range(theta.shape[0]):
        e = np.zeros_like(theta)
        e[i] = h
        fd = (cpu_objective(s, theta + e, used)[0] - cpu_objective(s, theta - e, used)[0]) / (2 * h)
        assert abs(fd - g0[i]) < 1e-6 * max(1.0, abs(fd)), (i, fd, g0[i])
    assert np.isfinite(f0)


def test_loglin_rows_are_distributions_and_fit_improves():
    s = synthetic(seed=6, n_rows=120)
    used = np.unique(s["rows"]["bkt"])
    budget = {"maxiter": 200, "maxfun": 400, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 10}
    lo = np.append(np.full(used.shape[0], -20.0), -5.0)
    hi = np.append(np.full(used.shape[0], 20.0), 5.0)
    x0 = np.append(np.full(used.shape[0], 0.1), 1.0)
    res, trace = t1.lbfgsb(lambda th: cpu_objective(s, th, used), x0, lo, hi, budget)
    assert res.fun <= trace[0] + 1e-12
    alpha = np.zeros(t1.N_BUCKETS)
    alpha[used] = res.x[:-1]
    _, _, _, pg, rank, maxerr = t1.cpu_loglin(s["rows"], alpha, float(res.x[-1]), s["indptr"], s["idx"], s["gval"], s["logpu"], False)
    assert maxerr < 1e-12 and np.all(pg > 0) and np.all(rank >= 0)
    info = t1.fit_info(res, trace, res.x, cpu_objective(s, res.x, used)[1], lo, hi, s["rows"]["n"])
    assert info["converged"] and info["projected_gradient_inf_norm"] < 1e-6


def test_lin_fit_and_rows_sum_to_one():
    from ckm_posthoc_probe import k_pair_pk
    s = synthetic(seed=7, n_rows=150)
    r = s["rows"]
    pk = k_pair_pk(r["rp"], r["comp"], r["gold"], s["indptr"], s["idx"], s["val"], s["inv_n"])
    budget = {"maxiter": 300, "maxfun": 600, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 10, "log_weight_bounds": [-30, 30]}
    w_group, info = t1.fit_lin(r["rp"], pk, r["bkt"], s["p_uni"][r["gold"]], budget)
    assert info["objective_recomputed"] <= info["objective_nats_per_row"] + 1e-12
    w, wuni = t1.lin_weights(r["rowid"], r["bkt"], r["n"], w_group)
    assert np.allclose(np.bincount(r["rowid"], weights=w, minlength=r["n"]) + wuni, 1.0)
    pos = np.arange(r["n"], dtype=np.int64)
    pg, rank, maxerr = k_dense_eval(r["rp"], r["comp"], w, np.zeros(w.shape[0]), np.full(w.shape[0], -1, np.int64), s["indptr"],
                                    s["idx"], s["val"], s["inv_n"], wuni, s["p_uni"], r["gold"], pos, pos * 0, pos * 0,
                                    np.ones(1, np.uint8), 0.0, False)
    assert maxerr < 1e-12 and np.all(pg > 0)


# ------------------------------------------------------------------ rows, bootstrap, verdicts
def test_csr_select_and_subset_rows():
    ptr = np.array([0, 3, 3, 5], np.int64)
    ids = np.array([1, 2, 4, 0, 2], np.uint32)
    keep = np.array([True, False, True, True, True])
    rp, rowid, comp = t1.csr_select(ptr, ids, keep)
    assert rp.tolist() == [0, 2, 2, 4] and rowid.tolist() == [0, 0, 2, 2] and comp.tolist() == [2, 4, 0, 2]
    rp2, sub = t1.subset_rows(ptr, ids, np.array([2, 0]))
    assert rp2.tolist() == [0, 2, 5] and sub.tolist() == [0, 2, 1, 2, 4]


def test_doc_block_bootstrap():
    num = np.array([2.0, 4.0, 6.0, 0.0])
    den = np.array([1.0, 2.0, 3.0, 0.0])  # an empty document is dropped
    bs = t1.doc_block_bootstrap(num, den, 500, 1)
    assert bs["point"] == 2.0 and bs["ci95"] == [2.0, 2.0] and bs["docs"] == 3
    rng = np.random.default_rng(8)
    a = t1.doc_block_bootstrap(rng.normal(size=60), np.full(60, 3.0), 2000, 5)
    b = t1.doc_block_bootstrap(np.random.default_rng(8).normal(size=60), np.full(60, 3.0), 2000, 5)
    assert a == b and a["ci95"][0] < a["point"] < a["ci95"][1]


def test_verdicts_follow_the_preregistered_thresholds():
    assert t1.verdict_masked(0.18, 0.2301, 0.05) == "MET"
    assert t1.verdict_masked(0.18, 0.2299, 0.05) == "NOT MET"
    assert t1.verdict_causal(2.1749, 2.0248, 0.15) == "MET"
    assert t1.verdict_causal(2.1749, 2.0250, 0.15) == "NOT MET"


# ------------------------------------------------------------------ GPU read-out vs the CPU float64 reference
def _gpu_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    return cp


@pytest.mark.parametrize("V", [300, 8000])
def test_gpu_readout_matches_cpu_reference_and_is_deterministic(V):
    _gpu_or_skip()
    s = synthetic(C=50, V=V, n_rows=200, seed=9, max_fired=12)
    r = s["rows"]
    used = np.unique(r["bkt"])
    rng = np.random.default_rng(10)
    alpha = np.zeros(t1.N_BUCKETS)
    alpha[used] = rng.normal(0.4, 0.3, used.shape[0])
    beta0 = 0.9
    ll, grad, gb, pg, rank, _ = t1.cpu_loglin(r, alpha, beta0, s["indptr"], s["idx"], s["gval"], s["logpu"], True)
    g = t1.T1Gpu(V, s["indptr"], s["idx"], s["val"], s["lam"], s["inv_n"], s["p_uni"], grid=64)
    ch = g.prepare(r["rp"], r["comp"], r["bkt"], r["gold"], rows_per_chunk=37)
    o1 = g.run(ch, r["n"], int(r["comp"].shape[0]), alpha, beta0, True)
    o2 = g.run(ch, r["n"], int(r["comp"].shape[0]), alpha, beta0, True)
    for k in o1:
        assert np.array_equal(o1[k], o2[k]), f"{k} differs between two identical runs"
    assert np.abs(o1["lpg"] - np.log(pg)).max() < 1e-8
    assert np.abs(o1["sum"] - 1.0).max() < 1e-12
    assert np.array_equal(o1["rank"], rank)
    gsum = np.bincount(r["bkt"], weights=o1["gpair"], minlength=t1.N_BUCKETS)
    assert np.allclose(gsum, grad, rtol=1e-9, atol=1e-8)
    assert abs(o1["gbeta"].sum() - gb) < 1e-8
    assert abs(o1["lpg"].sum() - ll) < 1e-6
    o3 = g.run(g.prepare(r["rp"], r["comp"], r["bkt"], r["gold"]), r["n"], int(r["comp"].shape[0]), alpha, beta0, False)
    assert np.array_equal(o3["lpg"], o1["lpg"])  # chunking does not change a row
