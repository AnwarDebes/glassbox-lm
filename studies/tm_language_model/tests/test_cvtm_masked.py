"""Masked CVTM (docs/11 section 11.8; docs/13 A3 H8.11): tmlm.cvtm.masked and the pipeline of
experiments/cvtm_masked.py on small synthetic corpora (numpy backend; CuPy cases skip without a GPU).

    .venv-tm/bin/python -m pytest tests/test_cvtm_masked.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm import masked as MK
from tmlm.cvtm import protocol
from tmlm.eval.baselines.ngram import KNModel, reverse_docs, reversed_positions

ROOT = Path(__file__).resolve().parents[1]


def has_gpu() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not has_gpu(), reason="CuPy with a CUDA device is not available")


def corpus(seed: int, vocab: int = 12, n_docs: int = 40, lo: int = 3, hi: int = 70):
    """Random documents with copied spans (so that document clauses fire), a few very short ones."""
    rng = np.random.default_rng(seed)
    docs = []
    for d in range(n_docs):
        n = int(rng.integers(lo, hi))
        x = rng.integers(0, vocab, size=n)
        if n > 12 and rng.random() < 0.7:
            a = int(rng.integers(0, n - 6))
            L = int(rng.integers(3, 6))
            b = int(rng.integers(0, n - L))
            x[b:b + L] = x[a:a + L]
        docs.append(x)
    tok = np.concatenate(docs).astype(np.uint16)
    ds = np.zeros(n_docs + 1, dtype=np.int64)
    ds[1:] = np.cumsum([len(x) for x in docs])
    return tok, ds


def manual_key(tok, ds, p, offs, V, vis=None):
    d = int(np.searchsorted(ds, p, side="right") - 1)
    lo, hi = int(ds[d]), int(ds[d + 1])
    bits = MK.sym_bits(V)
    key = 0
    valid = True
    for o in offs:
        q = p + o
        if lo <= q < hi:
            s = int(tok[q])
            if vis is not None and not vis[q]:
                valid = False
        else:
            s = V
        key = (key << bits) | s
    return key, valid


# ------------------------------------------------------------------ patterns and keys
def test_patterns_equal_the_probe_script():
    src = (ROOT / "experiments" / "ckm_posthoc_probe.py").read_text()
    tree = ast.parse(src)
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NGRAM_TYPES" for t in node.targets):
            found = ast.literal_eval(node.value)
    assert found is not None
    assert tuple((n, tuple(o)) for n, o in found) == MK.PATTERNS
    cfg = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "H8_11_masked.json").read_text())
    assert tuple(cfg["patterns"]["names"]) == MK.PATTERN_NAMES
    assert [tuple(o) for o in cfg["patterns"]["offsets"]] == [o for _, o in MK.PATTERNS]


@pytest.mark.parametrize("seed", [0, 1])
def test_pattern_keys_match_a_manual_loop(seed):
    V = 12
    tok, ds = corpus(seed, V)
    rng = np.random.default_rng(seed + 10)
    vis = (rng.random(tok.shape[0]) > 0.2).astype(np.uint8)
    pos = np.arange(tok.shape[0])
    for _, offs in MK.PATTERNS:
        k, v = MK.pattern_keys(tok, ds, pos, offs, V)
        kv, vv = MK.pattern_keys(tok, ds, pos, offs, V, vis=vis)
        assert np.all(v)
        assert np.array_equal(k, kv)
        for p in range(tok.shape[0]):
            mk, mv = manual_key(tok, ds, p, offs, V, vis)
            assert int(k[p]) == mk and bool(vv[p]) == mv


def test_cut_semantics_match_knmodel_contexts():
    """Left patterns L1..L3 fire iff KNModel.contexts leaves at least k visible trailing tokens, with
    the same symbols (BOS = EDGE = V); right patterns are the left patterns of the reversed documents."""
    V = 12
    tok, ds = corpus(3, V)
    rng = np.random.default_rng(4)
    hidden = (rng.random(tok.shape[0]) < 0.25).astype(np.uint8)
    vis = 1 - hidden
    pos = np.arange(tok.shape[0])
    kn = KNModel(4, V)
    ctx, valid = kn.contexts(tok, ds, pos, hidden, 4)
    rtok, rhid = reverse_docs(tok, ds), reverse_docs(hidden, ds)
    rpos = reversed_positions(pos, ds)
    rctx, rvalid = kn.contexts(rtok, ds, rpos, rhid, 4)
    bits = MK.sym_bits(V)
    for k in (1, 2, 3):
        for side, c, vl in (("L", ctx, valid), ("R", rctx, rvalid)):
            name = f"{side}{k}"
            offs = dict(MK.PATTERNS)[name]
            key, ok = MK.pattern_keys(tok, ds, pos, offs, V, vis=vis)
            assert np.array_equal(ok, vl >= k), name
            cols = c[:, 3 - k:] if side == "L" else c[:, 3 - k:][:, ::-1]
            exp = np.zeros(pos.shape[0], dtype=np.uint64)
            for j in range(k):
                exp = (exp << np.uint64(bits)) | cols[:, j].astype(np.uint64)
            assert np.array_equal(key, exp), name


def test_stream_keys_equal_pattern_keys_numpy():
    V = 12
    tok, ds = corpus(5, V)
    st = MK.upload_stream(tok, ds, "numpy")
    for _, offs in MK.PATTERNS:
        a = MK.stream_keys(st, offs, V)
        b, _ = MK.pattern_keys(tok, ds, np.arange(tok.shape[0]), offs, V)
        assert np.array_equal(a, b)


@needs_gpu
@pytest.mark.gpu
def test_stream_keys_equal_pattern_keys_cupy():
    import cupy
    V = 12
    tok, ds = corpus(6, V)
    st = MK.upload_stream(tok, ds, "cupy")
    for _, offs in MK.PATTERNS:
        a = cupy.asnumpy(MK.stream_keys(st, offs, V))
        b, _ = MK.pattern_keys(tok, ds, np.arange(tok.shape[0]), offs, V)
        assert np.array_equal(a, b)
    st.close()


def test_key_width_guard():
    with pytest.raises(ValueError):
        MK.check_pattern((-2, -1, 1, 2, 3), 8000)        # 5 x 13 bits > 63
    MK.check_pattern((-2, -1, 1, 2), 8000)               # 52 bits


# ------------------------------------------------------------------ counting
def _queries(tok, ds, offs, V, rng):
    k, _ = MK.pattern_keys(tok, ds, np.arange(tok.shape[0]), offs, V)
    sel = rng.choice(tok.shape[0], size=min(300, tok.shape[0]), replace=False)
    gk = np.concatenate([k[sel], rng.integers(0, 2 ** 30, size=20).astype(np.uint64)])
    gt = np.concatenate([tok[sel].astype(np.int64), rng.integers(0, V, size=20)])
    gt[:50] = rng.integers(0, V, size=50)
    lk = np.unique(np.concatenate([k[sel[:100]], rng.integers(0, 2 ** 30, size=10).astype(np.uint64)]))
    return gk, gt, lk


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_count_queried_matches_bruteforce(seed):
    V = 12
    tok, ds = corpus(seed, V)
    rng = np.random.default_rng(seed)
    st = MK.upload_stream(tok, ds, "numpy")
    for _, offs in MK.PATTERNS:
        bf = MK.count_table_bruteforce(tok, ds, offs, V)
        gk, gt, lk = _queries(tok, ds, offs, V, rng)
        r = MK.count_queried(st, offs, V, gold_keys=gk, gold_tok=gt, list_keys=lk)
        assert r["n_ctx"] == len(bf)
        assert r["n_pairs"] == sum(len(v) for v in bf.values())
        assert r["max_n"] == max(sum(v.values()) for v in bf.values())
        for i in range(gk.shape[0]):
            d = bf.get(int(gk[i]), {})
            assert r["gold_n"][i] == sum(d.values())
            assert r["gold_cnt"][i] == d.get(int(gt[i]), 0)
        for i in range(lk.shape[0]):
            d = bf.get(int(lk[i]), {})
            lo, hi = r["list_ptr"][i], r["list_ptr"][i + 1]
            assert bool(r["list_found"][i]) == bool(d)
            assert r["list_n"][i] == sum(d.values())
            assert r["list_tgt"][lo:hi].tolist() == sorted(d)
            assert r["list_cnt"][lo:hi].tolist() == [d[t] for t in sorted(d)]


@needs_gpu
@pytest.mark.gpu
def test_count_queried_cupy_equals_numpy():
    V = 12
    tok, ds = corpus(9, V, n_docs=200)
    rng = np.random.default_rng(9)
    a_st = MK.upload_stream(tok, ds, "cupy")
    b_st = MK.upload_stream(tok, ds, "numpy")
    for _, offs in MK.PATTERNS:
        gk, gt, lk = _queries(tok, ds, offs, V, rng)
        a = MK.count_queried(a_st, offs, V, gold_keys=gk, gold_tok=gt, list_keys=lk)
        b = MK.count_queried(b_st, offs, V, gold_keys=gk, gold_tok=gt, list_keys=lk)
        for k in ("gold_n", "gold_cnt", "list_found", "list_n", "list_ptr", "list_tgt", "list_cnt", "n_ctx", "n_pairs", "max_n"):
            assert np.array_equal(a[k], b[k]), k
    a_st.close()


def test_counts_without_h_equal_all_minus_h():
    """The fit-row counts of the pipeline: counting on all documents minus counting on H equals
    counting on the documents without H, for the H keys and golds."""
    V = 12
    tok, ds = corpus(11, V, n_docs=60)
    rng = np.random.default_rng(11)
    keep_h = rng.random(60) < 0.2
    keep_h[0] = True
    h = protocol.select_docs(tok, ds, keep_h)
    rest = protocol.select_docs(tok, ds, ~keep_h)
    st_all = MK.upload_stream(tok, ds, "numpy")
    st_h = MK.upload_stream(h.tok, h.doc_start, "numpy")
    st_r = MK.upload_stream(rest.tok, rest.doc_start, "numpy")
    for _, offs in MK.PATTERNS:
        kh, _ = MK.pattern_keys(h.tok, h.doc_start, np.arange(h.n_tokens), offs, V)
        a = MK.count_queried(st_all, offs, V, gold_keys=kh, gold_tok=h.tok)
        b = MK.count_queried(st_h, offs, V, gold_keys=kh, gold_tok=h.tok)
        c = MK.count_queried(st_r, offs, V, gold_keys=kh, gold_tok=h.tok)
        assert np.all(b["gold_n"] >= 1) and np.all(b["gold_cnt"] >= 1)
        assert np.array_equal(a["gold_n"] - b["gold_n"], c["gold_n"])
        assert np.array_equal(a["gold_cnt"] - b["gold_cnt"], c["gold_cnt"])
    uni_all = MK.add_one_unigram(tok, V, minus=h.tok)
    assert np.array_equal(uni_all, MK.add_one_unigram(rest.tok, V))
    assert abs(uni_all.sum() - 1.0) < 1e-12


# ------------------------------------------------------------------ document clauses
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_doc_candidates_eval_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = 500
    keys = rng.integers(0, 6, size=n).astype(np.uint64)
    valid = rng.random(n) > 0.2
    vis = (rng.random(n) > 0.3).astype(np.uint8)
    doc = np.sort(rng.integers(0, 7, size=n)).astype(np.int64)
    got = MK.doc_candidates_eval(keys, valid, vis, doc)
    exp = MK.doc_candidates_bruteforce_eval(keys, valid, vis, doc)
    assert np.array_equal(got, exp)


@pytest.mark.parametrize("seed", [0, 1])
def test_doc_candidates_fit_equal_per_row_visibility(seed):
    """Fit rows hide only their own target: the fit candidate of row p equals the evaluation
    candidate of p under a mask that hides p alone (keys and cuts recomputed with that mask)."""
    V = 5
    tok, ds = corpus(seed, V, n_docs=8, lo=10, hi=60)
    n = tok.shape[0]
    doc = MK.doc_index_of(ds, n)
    for _, offs in MK.PATTERNS:
        k_all, _ = MK.pattern_keys(tok, ds, np.arange(n), offs, V)
        fit = MK.doc_candidates_fit(k_all, doc, MK.right_extent(offs))
        for p in range(n):
            vis = np.ones(n, np.uint8)
            vis[p] = 0
            k, v = MK.pattern_keys(tok, ds, np.arange(n), offs, V, vis=vis)
            assert v[p]
            ev = MK.doc_candidates_eval(k, v, vis, doc)
            assert ev[p] == fit[p], (offs, p)


def test_fit_rows_do_not_read_their_own_target():
    V = 7
    tok, ds = corpus(21, V, n_docs=10, lo=20, hi=80)
    n = tok.shape[0]
    doc = MK.doc_index_of(ds, n)
    rng = np.random.default_rng(21)
    for _, offs in MK.PATTERNS:
        k0, _ = MK.pattern_keys(tok, ds, np.arange(n), offs, V)
        c0 = MK.doc_candidates_fit(k0, doc, MK.right_extent(offs))
        for p in rng.choice(n, size=40, replace=False):
            t2 = tok.copy()
            t2[p] = (int(tok[p]) + 1 + int(rng.integers(0, V - 1))) % V
            k1, _ = MK.pattern_keys(t2, ds, np.arange(n), offs, V)
            c1 = MK.doc_candidates_fit(k1, doc, MK.right_extent(offs))
            assert k1[p] == k0[p]
            assert c1[p] == c0[p]
            if c0[p] >= 0:
                assert c0[p] != p and t2[c1[p]] == tok[c0[p]]


def test_eval_rows_do_not_read_hidden_tokens():
    V = 7
    tok, ds = corpus(22, V, n_docs=10, lo=20, hi=80)
    n = tok.shape[0]
    doc = MK.doc_index_of(ds, n)
    rng = np.random.default_rng(22)
    hidden = (rng.random(n) < 0.15).astype(np.uint8)
    vis = 1 - hidden
    targets = np.flatnonzero(hidden)
    t2 = tok.copy()
    t2[targets] = rng.integers(0, V, size=targets.shape[0])
    for _, offs in MK.PATTERNS:
        ka, va = MK.pattern_keys(tok, ds, np.arange(n), offs, V, vis=vis)
        kb, vb = MK.pattern_keys(t2, ds, np.arange(n), offs, V, vis=vis)
        assert np.array_equal(va, vb)
        assert np.array_equal(ka[va], kb[vb])
        ca = MK.doc_candidates_eval(ka, va, vis, doc)[targets]
        cb = MK.doc_candidates_eval(kb, vb, vis, doc)[targets]
        assert np.array_equal(ca, cb)
        ok = ca >= 0
        assert np.all(vis[ca[ok]] == 1)
        assert np.array_equal(tok[ca[ok]], t2[cb[ok]])


# ------------------------------------------------------------------ rows, EM, scoring
def test_s_param_levels_exact():
    n = np.array([1, 2, 3, 4, 7, 8, 2 ** 20 - 1, 2 ** 20, 2 ** 31 - 1, 2 ** 31], dtype=np.int64)
    lv = MK.s_param(0, n)
    assert lv.tolist() == [0, 1, 1, 2, 2, 3, 19, 20, 30, 31]
    assert MK.s_param(3, np.array([5]))[0] == 3 * MK.LOGN_LEVELS + 2
    with pytest.raises(ValueError):
        MK.s_param(0, np.array([0]))
    with pytest.raises(OverflowError):
        MK.s_param(0, np.array([2 ** 32]))


def random_fit_rows(seed: int, n: int = 3000, unseen_gold: float = 0.2):
    rng = np.random.default_rng(seed)
    V = 20
    gold = rng.integers(0, V, size=n)
    p_uni = rng.dirichlet(np.ones(V))
    s_n = np.where(rng.random((n, 10)) < 0.6, rng.integers(1, 40, size=(n, 10)), 0)
    s_cnt = np.minimum(s_n, rng.binomial(s_n, 0.3))
    d_tok = np.where(rng.random((n, 10)) < 0.3, np.where(rng.random((n, 10)) < 0.5, gold[:, None], rng.integers(0, V, size=(n, 10))), -1)
    if unseen_gold > 0:
        # rows whose gold token no clause predicts (rare tokens): only the unigram has mass on them
        u = rng.random(n) < unseen_gold
        s_cnt[u] = 0
        d_tok[u] = np.where(d_tok[u] >= 0, (gold[u, None] + 1) % V, -1)
    return MK.fit_rows(gold, p_uni, s_n.astype(np.int64), s_cnt.astype(np.int64), d_tok.astype(np.int64))


def test_em_never_decreases_and_reaches_the_optimum():
    from scipy.optimize import minimize

    rows = random_fit_rows(0)
    fit = MK.fit_em(rows, max_iter=3000, tol=1e-13)
    tr = fit["trace"]
    assert fit["stopped_by_tol"]
    assert np.all(np.diff(tr) >= -1e-12)
    assert fit["weights"].max() == 1.0 and fit["weights"][MK.U_PARAM] > 0.0
    used = np.flatnonzero(fit["rows_per_param"] > 0)
    used = used[used != MK.U_PARAM]
    par, pg = rows.par, rows.pg

    def nll(theta):
        w = np.ones(MK.N_PARAMS)          # the unigram weight fixed at 1 (scale free)
        w[used] = np.exp(theta)
        W = np.where(par >= 0, w[np.maximum(par, 0)], 0.0)
        return -float(np.mean(np.log((W * pg).sum(1)) - np.log(W.sum(1))))

    res = minimize(nll, np.zeros(used.shape[0]), method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
    assert -tr[-1] <= res.fun + 1e-7
    # the sparse gold formula equals the EM's own objective
    pgs = MK.gold_prob_sparse(rows, fit["weights"])
    assert abs(np.log(pgs).mean() - tr[-1]) < 1e-12


def test_em_stays_finite_when_the_supremum_is_on_the_boundary():
    """Without rows that only the unigram explains, the unigram weight tends to 0 relative to the
    others (the supremum is on the boundary): the weights stay finite and the trace non-decreasing."""
    rows = random_fit_rows(0, unseen_gold=0.0)
    fit = MK.fit_em(rows, max_iter=3000, tol=-1.0)
    assert np.all(np.isfinite(fit["trace"])) and np.all(np.isfinite(fit["weights"]))
    assert np.all(np.diff(fit["trace"]) >= -1e-12)
    assert fit["weights"].max() == 1.0 and fit["weights"][MK.U_PARAM] < 1e-3


def test_em_is_deterministic():
    rows = random_fit_rows(1)
    a = MK.fit_em(rows, 200, 1e-10)
    b = MK.fit_em(rows, 200, 1e-10)
    assert MK.weights_hash(a["weights"]) == MK.weights_hash(b["weights"])
    assert np.array_equal(a["trace"], b["trace"])


def test_resolve_weights_rule():
    w = np.arange(MK.N_PARAMS, dtype=np.float64) + 1.0
    rpp = np.zeros(MK.N_PARAMS, np.int64)
    rpp[MK.U_PARAM] = 5
    rpp[MK.D_PARAM0:MK.D_PARAM0 + MK.N_PATTERNS] = 3
    for pi in range(MK.N_PATTERNS):
        rpp[pi * MK.LOGN_LEVELS + 2] = 1
        rpp[pi * MK.LOGN_LEVELS + 6] = 1
    out, rep = MK.resolve_weights(w, rpp)
    b = MK.LOGN_LEVELS  # pattern 1
    assert out[b + 0] == w[b + 2] and out[b + 3] == w[b + 2]
    assert out[b + 4] == w[b + 2]          # tie between levels 2 and 6: the lower level
    assert out[b + 5] == w[b + 6] and out[b + 31] == w[b + 6]
    assert out[b + 2] == w[b + 2] and out[b + 6] == w[b + 6]
    assert rep["n_substituted"] == MK.N_PATTERNS * (MK.LOGN_LEVELS - 2)
    rpp[MK.D_PARAM0 + 4] = 0
    out, rep = MK.resolve_weights(w, rpp)
    assert np.isnan(out[MK.D_PARAM0 + 4]) and rep["unfitted_d"] == [MK.PATTERN_NAMES[4]]
    rpp[MK.U_PARAM] = 0
    with pytest.raises(ValueError):
        MK.resolve_weights(w, rpp)


def test_score_refuses_an_unfitted_weight():
    rows, lists, w, p_uni, gold = random_eval_setup(4)
    w = w.copy()
    col = MK.N_PATTERNS + 2
    assert np.any(rows.par[:, col] >= 0)
    w[MK.D_PARAM0 + 2] = np.nan
    with pytest.raises(ValueError):
        MK.score(rows, w, lists, p_uni, gold)


def random_eval_setup(seed: int, n: int = 400, V: int = 30):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(MK.N_PATTERNS):
        K = 25
        lens = rng.integers(0, 6, size=K)
        ptr = np.zeros(K + 1, np.int64)
        ptr[1:] = np.cumsum(lens)
        tgt = np.concatenate([np.sort(rng.choice(V, size=L, replace=False)) for L in lens]).astype(np.int32) if lens.sum() else np.zeros(0, np.int32)
        cnt = rng.integers(1, 9, size=int(lens.sum())).astype(np.int64)
        nn = np.array([cnt[ptr[k]:ptr[k + 1]].sum() for k in range(K)], np.int64)
        parts.append({"list_ptr": ptr, "list_tgt": tgt, "list_cnt": cnt, "list_n": nn, "list_found": nn > 0})
    lists, bases = MK.concat_lists(parts)
    lists.check()
    s_n = np.zeros((n, 10), np.int64)
    s_ref = np.full((n, 10), -1, np.int64)
    for pi in range(10):
        k = rng.integers(0, 25, size=n)
        fire = (rng.random(n) < 0.6) & (parts[pi]["list_n"][k] > 0)
        s_n[fire, pi] = parts[pi]["list_n"][k[fire]]
        s_ref[fire, pi] = bases[pi] + k[fire]
    d_tok = np.where(rng.random((n, 10)) < 0.3, rng.integers(0, V, size=(n, 10)), -1)
    gold = rng.integers(0, V, size=n)
    p_uni = rng.dirichlet(np.ones(V))
    w = np.exp(rng.normal(size=MK.N_PARAMS))
    w[MK.U_PARAM] = 1.0
    return MK.eval_rows(s_n, s_ref, d_tok), lists, w, p_uni, gold


def test_score_matches_a_dense_reference():
    rows, lists, w, p_uni, gold = random_eval_setup(3)
    sc = MK.score(rows, w, lists, p_uni, gold)
    V = p_uni.shape[0]
    for r in range(rows.n):
        acc = w[MK.U_PARAM] * p_uni.copy()
        Z = w[MK.U_PARAM]
        for pi in range(10):
            b = rows.par[r, pi]
            if b >= 0:
                k = rows.ref[r, pi]
                lo, hi = lists.ptr[k], lists.ptr[k + 1]
                d = np.zeros(V)
                d[lists.tgt[lo:hi]] = lists.cnt[lo:hi] / lists.n[k]
                acc += w[b] * d
                Z += w[b]
            b = rows.par[r, 10 + pi]
            if b >= 0:
                acc[rows.ref[r, 10 + pi]] += w[b]
                Z += w[b]
        P = acc / Z
        assert abs(P.sum() - 1.0) < 1e-12
        g = gold[r]
        assert sc["pred"][r] == int(np.argmax(P))
        assert abs(sc["p_gold"][r] - P[g]) < 1e-14
        assert sc["rank"][r] == int((P > P[g]).sum() + ((P == P[g]) & (np.arange(V) < g)).sum())
    assert sc["max_row_sum_error"] < 1e-12
    assert sc["max_rel_gold_diff"] < 1e-12


def test_score_ties_go_to_the_lowest_id():
    V = 6
    parts = [{"list_ptr": np.array([0, 0], np.int64), "list_tgt": np.zeros(0, np.int32), "list_cnt": np.zeros(0, np.int64),
              "list_n": np.zeros(1, np.int64), "list_found": np.zeros(1, bool)} for _ in range(10)]
    lists, _ = MK.concat_lists(parts)
    d_tok = np.full((1, 10), -1, np.int64)
    rows = MK.eval_rows(np.zeros((1, 10), np.int64), np.full((1, 10), -1, np.int64), d_tok)
    p_uni = np.full(V, 1.0 / V)
    sc = MK.score(rows, np.ones(MK.N_PARAMS), lists, p_uni, np.array([3]))
    assert sc["pred"][0] == 0 and sc["rank"][0] == 3


# ------------------------------------------------------------------ the pipeline end to end (numpy)
def load_driver():
    spec = importlib.util.spec_from_file_location("cvtm_masked_driver", ROOT / "experiments" / "cvtm_masked.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cvtm_masked_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_end_to_end_numpy():
    drv = load_driver()
    V = 10
    tok, ds = corpus(31, V, n_docs=300, lo=20, hi=90)
    ev_tok, ev_ds = corpus(32, V, n_docs=12, lo=30, hi=90)
    rng = np.random.default_rng(33)
    hidden = (rng.random(ev_tok.shape[0]) < 0.15).astype(np.uint8)
    targets = np.flatnonzero(hidden)
    data = drv.make_data(tok, ds, None, ev_tok, ev_ds, targets, hidden, V, "test")
    assert data.h.n_docs >= 1
    cfg = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "H8_11_masked.json").read_text())
    cfg["checks"] = {"prefix_tokens": 2000, "direct_noH_patterns": ["L1", "L2R2"], "key_sample": 500}
    cfg["em"]["max_iter"] = 300
    drv.FAILED.clear()
    res = drv.run_pipeline(data, cfg, "numpy", with_comparators=True)
    assert drv.FAILED == []
    feats = res["feats"]
    # fit-row counts equal a brute-force count on the documents without H
    for pi, (_, offs) in enumerate(MK.PATTERNS):
        bf = MK.count_table_bruteforce(data.rest.tok, data.rest.doc_start, offs, V)
        kh, _ = MK.pattern_keys(data.h.tok, data.h.doc_start, np.arange(data.h.n_tokens), offs, V)
        for i in range(0, data.h.n_tokens, 7):
            d = bf.get(int(kh[i]), {})
            assert feats["s_n_fit"][i, pi] == sum(d.values())
            assert feats["s_cnt_fit"][i, pi] == d.get(int(data.h.tok[i]), 0)
    main = res["main"]["summary"]
    assert 0.0 <= main["top1"] <= 1.0 and main["n"] == targets.shape[0]
    lb = res["comparators"]["left_bigram"]
    assert lb["top1_function"] == lb["top1_rows"]
    # the KN-5 both-sides comparator path equals the reference fit on the host
    from tmlm.eval.baselines.ngram import BidirectionalKN
    bi = BidirectionalKN.fit(tok, ds, 5, V)
    ref = bi.predict(ev_tok, ev_ds, targets, hidden, order=5, side="both")
    assert np.array_equal(ref, res["comparators"]["kn5_both"]["pred"])
