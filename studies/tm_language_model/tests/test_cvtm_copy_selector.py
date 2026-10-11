"""Lane B1 (docs/13 A4 H9.1, H9.2): tmlm.cvtm.copy_selector on small synthetic inputs. The pair
kernel is checked against an independent brute-force implementation of the exploratory feature
definitions, the core rows against `CVTMCore.dense_pass`, the read-out against its full distribution
and finite differences, the controls' evaluators against their reference implementations.

    .venv-tm/bin/python -m pytest tests/test_cvtm_copy_selector.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tmlm.cvtm import copy_selector as CS
from tmlm.cvtm.model import CVTMCore, topk_rows


# ------------------------------------------------------------------ helpers
def bl(x: int) -> int:
    return int(x).bit_length() - 1


def nlog2b(x: float, cap: int) -> int:
    if x >= 1.0:
        return 0
    return min(max(int(math.floor(-math.log2(x))), 0), cap)


def brute_features(tok, dl, p, Pc, Ps, kstar, corder, c1, c2, c3, fcls, ctype, cap):
    """Independent implementation of the 29 pair features (docs of the exploratory k_pairs)."""
    V = Pc.shape[0]
    hist = list(range(p - 1, dl - 1, -1))
    order = []
    for q in hist:
        if tok[q] not in order:
            order.append(tok[q])
    order = order[:cap]
    top = float(Pc.max())
    cm1 = np.bincount(c1, weights=Pc, minlength=16)
    cm2 = np.bincount(c2, weights=Pc, minlength=256)
    out = []
    for s, c in enumerate(order):
        occ = [q for q in hist if tok[q] == c]            # most recent first

        def match(q, eq, lmax):
            L = 0
            while L < lmax and q - 1 - L >= dl and eq(tok[p - 1 - L]) == eq(tok[q - 1 - L]):
                L += 1
            return L

        ident = lambda t: t  # noqa: E731
        Ls = [match(q, ident, CS.LMAX) for q in occ]
        L1 = [match(q, lambda t: c1[t], CS.LCMAX) for q in occ]
        L2 = [match(q, lambda t: c2[t], CS.LCMAX) for q in occ]
        L3 = [match(q, lambda t: c3[t], CS.LCMAX) for q in occ]
        best = max(Ls)
        q_best = occ[Ls.index(best)]                       # the most recent occurrence reaching the maximum
        n64 = sum(1 for q in occ if p - q <= 64)
        rank = int(np.sum(Pc > Pc[c]))
        rb = next((e for e, edge in enumerate(CS.RANK_EDGES) if rank <= edge), len(CS.RANK_EDGES))
        f = [Ls[0], best, L1[0], max(L1), L2[0], max(L2), L3[0], max(L3),
             min(bl(p - occ[0]), 14), min(bl(p - q_best), 14), min(bl(s + 1), 8), min(bl(len(occ)), 7),
             n64 if n64 <= 2 else (3 if n64 <= 4 else 4),
             min(bl(sum(1 for x in Ls if x >= 1) + 1), 5), min(bl(sum(1 for x in Ls if x >= 2) + 1), 4),
             min(bl(occ[-1] - dl + 1), 14), fcls[c], ctype[c], nlog2b(Pc[c], 20), rb, nlog2b(Ps[c], 20),
             nlog2b(cm2[c2[c]], 12), nlog2b(cm1[c1[c]], 8), nlog2b(Pc[c] / top, 12), kstar, nlog2b(top, 8),
             ctype[tok[p - 1]] if p - 1 >= dl else 0, min(bl(p - dl), 14) if p > dl else 0, corder]
        out.append((c, f, float(Pc[c])))
    return out


def rand_tables(rng, V):
    return {"c1": rng.integers(0, 16, V), "c2": rng.integers(0, 256, V), "c3": rng.integers(0, 4096, V),
            "fcls": rng.integers(0, 6, V), "ctype": rng.integers(0, 64, V)}


# ------------------------------------------------------------------ literals, subsets
def test_literal_construction_is_the_exploratory_set():
    lits = CS.literal_construction()
    assert lits.n == 237 and lits.LT.shape == (237, 256)
    assert int(lits.reflective.sum()) == 89          # as logged by the exploratory run (literals.n_reflective)
    assert any("P_core(candidate)" in n for n in lits.names)
    assert not any("P_B" in n or "B's" in n or "B mass" in n for n in lits.names)
    # every literal's feature value range fits the table bounds of the LR controls
    for j, (_, f, kind, t) in enumerate(lits.spec):
        assert 0 <= f < CS.NF
    m = lits.matrix(np.zeros((3, CS.NF), np.uint8))
    assert m.shape == (3, 237) and m.dtype == np.uint8


def test_s2_mask_and_pool_hash_are_deterministic():
    doc = np.repeat(np.arange(50) * 7 + 3, 400)
    off = np.tile(np.arange(400), 50)
    m1 = CS.s2_mask(doc, off)
    assert np.array_equal(m1, CS.s2_mask(doc.copy(), off.copy()))
    assert abs(m1.mean() - 1 / CS.S2_DEN) < 0.01
    h = CS._pool_hash_vec(doc.astype(np.uint64), off.astype(np.uint64), (off % 97).astype(np.uint64))
    assert np.array_equal(h, CS._pool_hash_vec(doc.astype(np.uint64), off.astype(np.uint64), (off % 97).astype(np.uint64)))
    rate = float(np.mean(h < CS.POOL_THRESH))
    assert abs(rate - CS.POOL_RATE) < 0.003


def test_candidate_counts_match_brute_force():
    rng = np.random.default_rng(3)
    tok = rng.integers(0, 30, 900).astype(np.int64)
    ds = np.array([0, 250, 251, 600, 900], np.int64)
    pos = np.sort(rng.choice(900, 300, replace=False)).astype(np.int64)
    nc, nd = CS.candidate_counts(tok, ds, pos, 7)
    for i, p in enumerate(pos):
        d = int(np.searchsorted(ds, p, side="right")) - 1
        k = len(set(tok[ds[d]:p].tolist()))
        assert nd[i] == k and nc[i] == min(k, 7)


# ------------------------------------------------------------------ pair kernel
@pytest.mark.parametrize("cap", [4, 1024])
def test_pair_kernel_matches_brute_force(cap):
    rng = np.random.default_rng(11 + cap)
    V = 40
    tabs = rand_tables(rng, V)
    docs = [rng.integers(0, 12, n).tolist() for n in (90, 1, 130)]
    docs[2][70:85] = docs[2][20:35]                     # a copied span (long suffix matches)
    tok = np.array([t for d in docs for t in d], np.int64)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    pos = np.arange(tok.shape[0], dtype=np.int64)
    q = pos.shape[0]
    Pc = rng.dirichlet(np.full(V, 0.3), size=q)
    Ps = rng.dirichlet(np.full(V, 0.5), size=q)
    kstar = rng.integers(0, 5, q)
    corder = rng.integers(0, 5, q)
    lo = ds[np.searchsorted(ds, pos, side="right") - 1]
    feat = np.zeros((q * cap, CS.NF), np.uint8)
    cand = np.zeros(q * cap, np.uint16)
    lab = np.zeros(q * cap, np.uint8)
    pc = np.zeros(q * cap)
    ncand = np.zeros(q, np.int64)
    pstat = np.zeros((q, 7))
    CS.k_pairs(pos, tok, lo, Pc, Ps, kstar, corder, tabs["c1"], tabs["c2"], tabs["c3"], tabs["fcls"], tabs["ctype"],
               CS.RANK_EDGES, cap, feat, cand, lab, pc, ncand, pstat)
    n_checked = 0
    for i, p in enumerate(pos):
        ref = brute_features(tok, int(lo[i]), int(p), Pc[i], Ps[i], int(kstar[i]), int(corder[i]), tabs["c1"], tabs["c2"],
                             tabs["c3"], tabs["fcls"], tabs["ctype"], cap)
        assert ncand[i] == len(ref)
        gold_slot = -1
        for s, (c, f, pcv) in enumerate(ref):
            r = i * cap + s
            assert cand[r] == c
            assert lab[r] == (1 if c == tok[p] else 0)
            assert pc[r] == pcv
            assert feat[r].tolist() == f, (i, s, feat[r].tolist(), f)
            if c == tok[p]:
                gold_slot = s
            n_checked += 1
        assert pstat[i, 6] == gold_slot
        assert pstat[i, 1] == (1.0 if tok[p] in tok[lo[i]:p] else 0.0)
        assert pstat[i, 0] == Pc[i, tok[p]]
        assert np.all(feat[i * cap: i * cap + ncand[i]] <= CS.FEAT_MAX[None, :])
    assert n_checked > 500


# ------------------------------------------------------------------ core rows
def corpus(seed: int, vocab: int = 40, n_docs: int = 400, lo: int = 30, hi: int = 200):
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(vocab, 0.3), size=vocab)
    cum = np.cumsum(trans, axis=1)
    docs = []
    for _ in range(n_docs):
        n = int(rng.integers(lo, hi))
        t = [int(rng.integers(vocab))]
        for x in rng.random(n - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        if n > 40 and rng.random() < 0.6:
            a = int(rng.integers(0, n // 2 - 12))
            L = int(rng.integers(6, 12))
            b = int(rng.integers(n // 2, n - L))
            t[b:b + L] = t[a:a + L]
        docs.append(t)
    tok = np.asarray([x for d in docs for x in d], dtype=np.uint16)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    return tok, ds


@pytest.fixture(scope="module")
def small_core():
    tok, ds = corpus(1)
    core = CVTMCore.fit(tok, ds, vocab=40, backend="numpy")
    etok, eds = corpus(2, n_docs=8)
    rec = core.gold(etok, eds)
    return core, rec


def test_core_rows_equal_dense_pass(small_core):
    core, rec = small_core
    pos = np.arange(3, rec.tok.shape[0], 5, dtype=np.int64)
    P_S, Pc = CS.core_rows(core, rec, pos)
    assert np.array_equal(Pc[np.arange(pos.shape[0]), rec.tok[pos]], rec.p_gold[pos])
    assert np.array_equal(P_S[np.arange(pos.shape[0]), rec.tok[pos]], np.exp(rec.lnp_s[pos]))
    dp = core.dense_pass(rec, 16, pos, want_base=True, want_core=True)
    ids, pv = topk_rows(Pc, 16)
    assert np.array_equal(ids.astype(np.uint16), dp["core_ids"]) and np.array_equal(pv, dp["core_p"])
    bids, bpv = topk_rows(P_S, 16)
    assert np.array_equal(bids.astype(np.uint16), dp["base_ids"]) and np.array_equal(bpv, dp["base_p"])
    assert np.max(np.abs(Pc.sum(1) - 1.0)) < 1e-12


def test_pairs_for_positions_consistent(small_core):
    core, rec = small_core
    lits = CS.literal_construction()
    rng = np.random.default_rng(5)
    tabs = rand_tables(rng, 40)
    corder = CS.cache_order(rec.feats.n)
    pos = np.arange(0, rec.tok.shape[0], dtype=np.int64)
    ch = CS.pairs_for_positions(core, rec, pos, tabs, corder, cap=16, topk=8)
    nc, _ = CS.candidate_counts(rec.tok, rec.doc_start, pos, 16)
    assert np.array_equal(ch.ncand, nc)
    assert ch.feat.shape[0] == int(nc.sum())
    offs = np.concatenate([[0], np.cumsum(nc)])
    for i in range(pos.shape[0]):
        g = int(ch.pstat[i, 6])
        if g >= 0:
            assert ch.lab[offs[i] + g] == 1 and int(ch.cand[offs[i] + g]) == int(rec.tok[pos[i]])
        assert ch.lab[offs[i]:offs[i + 1]].sum() == (1 if g >= 0 else 0)
    assert np.array_equal(ch.pstat[:, 0], rec.p_gold[pos])
    # pool selection keeps only negatives
    sel = CS.pool_select(ch, np.zeros(pos.shape[0], np.int64), pos)
    assert np.all(ch.lab[sel] == 0)
    cols = CS.pack_literals(ch.feat, lits)
    m = lits.matrix(ch.feat)
    r = np.arange(ch.feat.shape[0])
    for j in (0, 17, 100, 236):
        bits = (cols[r // 64, j] >> (r % 64).astype(np.uint64)) & np.uint64(1)
        assert np.array_equal(bits.astype(np.uint8), m[:, j])


# ------------------------------------------------------------------ read-out
def synthetic_readout(seed=0, n=300, V=60):
    rng = np.random.default_rng(seed)
    rows = rng.dirichlet(np.full(V, 0.2), size=n)
    offs = [0]
    cands, scores, pcs, gslot, gold = [], [], [], [], []
    for i in range(n):
        k = int(rng.integers(0, 12))
        c = rng.choice(V, size=k, replace=False)
        g = int(rng.integers(V)) if rng.random() < 0.5 or k == 0 else int(c[rng.integers(k)])
        s = rng.normal(size=k) + (2.0 * (c == g))
        cands.append(c)
        scores.append(s)
        pcs.append(rows[i, c])
        gslot.append(int(np.flatnonzero(c == g)[0]) if np.any(c == g) else -1)
        gold.append(g)
        offs.append(offs[-1] + k)
    rd = CS.ReadoutData(offs=np.array(offs, np.int64), gslot=np.array(gslot, np.int64),
                        lpgold=np.log(rows[np.arange(n), gold]), bucket=rng.integers(0, 3, n).astype(np.int64),
                        pc=np.concatenate(pcs))
    return rd, np.concatenate(scores), rows, cands, np.array(gold)


def test_readout_is_normalised_and_matches_its_full_distribution():
    rd, sc, rows, cands, gold = synthetic_readout()
    beta = np.array([0.7, -0.3, 1.9])
    alpha = np.array([0.2, 0.5, -1.0])
    lp = CS.readout_logp(sc, rd, beta, alpha)
    for i in range(rd.n):
        a, b = rd.offs[i], rd.offs[i + 1]
        bk = rd.bucket[i]
        full = CS.k_readout_row(sc[a:b], rd.pc[a:b], cands[i].astype(np.int64), rows[i], beta[bk], alpha[bk])
        assert abs(full.sum() - 1.0) < 1e-12
        assert abs(math.log(full[gold[i]]) - lp[i]) < 1e-12
    lp0 = CS.readout_logp(sc, rd, np.zeros(3), np.zeros(3))
    assert np.max(np.abs(lp0 - rd.lpgold)) < 1e-12


def test_readout_gradient_and_overflow():
    rd, sc, *_ = synthetic_readout(1)
    beta = np.array([0.4, 0.1, -0.2])
    alpha = np.array([-0.3, 0.2, 0.1])
    gb, ga = np.zeros(3), np.zeros(3)
    lp = np.empty(rd.n)
    f = CS.k_readout(sc, rd.pc, rd.offs, rd.gslot, rd.lpgold, rd.bucket, beta, alpha, True, gb, ga, lp)
    h = 1e-6
    for k in range(3):
        for arr, g in ((beta, gb), (alpha, ga)):
            e = arr.copy()
            e[k] += h
            fp = CS.k_readout(sc, rd.pc, rd.offs, rd.gslot, rd.lpgold, rd.bucket,
                              e if arr is beta else beta, e if arr is alpha else alpha, False, gb[:1], ga[:1], lp)
            e[k] -= 2 * h
            fm = CS.k_readout(sc, rd.pc, rd.offs, rd.gslot, rd.lpgold, rd.bucket,
                              e if arr is beta else beta, e if arr is alpha else alpha, False, gb[:1], ga[:1], lp)
            assert abs((fp - fm) / (2 * h) - g[k]) < 1e-5 * max(1.0, abs(g[k]))
    assert np.isfinite(f)
    big = CS.readout_logp(sc * 500.0, rd, np.full(3, 3.0), np.full(3, 40.0))
    assert np.all(np.isfinite(big)) and np.all(big <= 1e-12)


def test_fit_pooled_and_buckets():
    rd, sc, *_ = synthetic_readout(2, n=3000)
    th, info = CS.fit_pooled(sc, rd)
    assert info["converged"] and th[0] > 0.5
    assert info["bits_per_position"] < info["core_bits_per_position"]
    be, al, binfo = CS.fit_buckets(sc, rd, th, n_buckets=5)
    assert binfo["buckets_with_positions"] == 3
    # buckets without positions keep the pooled pair; fitted buckets move toward their own optimum
    assert be[3] == th[0] and be[4] == th[0] and al[3] == th[1]
    assert binfo["bits_per_position_fitted"] <= info["bits_per_position"] + 1e-9


# ------------------------------------------------------------------ controls
def test_table_lr_equals_explicit_forms_and_gradients():
    lits = CS.literal_construction()
    rng = np.random.default_rng(4)
    n = 400
    feat = np.stack([rng.integers(0, CS.FEAT_MAX[f] + 1, n) for f in range(CS.NF)], axis=1).astype(np.uint8)
    X = lits.matrix(feat, np.float64)
    for pairwise in (False, True):
        m = CS.TableLR(lits, pairwise=pairwise)
        theta = rng.normal(size=m.n_params) * 0.05
        W = m.W_of(theta)
        explicit = (np.einsum("ni,ij,nj->n", X, np.triu(W), X) if pairwise else X @ np.diag(W)) + theta[-1]
        got = m.score(feat, theta)
        assert np.max(np.abs(got - explicit)) < 1e-10
        # gradient of sum r_n s_n equals X' r (linear) or the upper triangle of X' diag(r) X (pairwise)
        r = rng.normal(size=n)
        H = np.zeros(m.n_table)
        CS.k_table_hist(feat, r, m.diag_off, m.pair_f, m.pair_g, m.pair_off, m.pair_ng, H)
        G = m.grad_from_hist(H)
        full = X.T @ (X * r[:, None])
        want = full[m.iu]
        assert np.max(np.abs(G - want)) < 1e-9
    y = (X[:, 5] * X[:, 40] + (rng.random(n) < 0.1)).clip(0, 1).astype(np.uint32)
    m = CS.TableLR(lits, pairwise=True)
    th, info = m.fit(feat, y, l2=1.0, maxiter=200)
    s = m.score(feat, th)
    assert info["loss"] < n * math.log(2)
    assert np.mean((s > 0) == (y == 1)) > 0.9


def test_tm_clause_set_evaluator_equals_tmu():
    from tmu.models.classification.vanilla_classifier import TMClassifier

    lits = CS.literal_construction()
    rng = np.random.default_rng(6)
    n = 3000
    feat = np.stack([rng.integers(0, CS.FEAT_MAX[f] + 1, n) for f in range(CS.NF)], axis=1).astype(np.uint8)
    X = lits.matrix(feat, np.uint32)
    y = ((X[:, 3] & X[:, 60]) | (X[:, 100] & (1 - X[:, 7]))).astype(np.uint32)
    tm = TMClassifier(40, T=30, s=5.0, platform="CPU", weighted_clauses=True, seed=5)
    tm.fit(X, y)
    cs = CS.tm_clause_set(tm, lits.n)
    cols = CS.pack_literals(feat, lits)
    mine = CS.tm_scores(cols, 0, n, cs)
    _, csum = tm.predict(X, return_class_sums=True)
    d = csum[:, 1].astype(np.int64) - csum[:, 0].astype(np.int64) - mine.astype(np.int64)
    ud = np.unique(d)
    assert ud.shape[0] == 1 and int(ud[0]) in (0, cs.empty_const)
    # a slice that is not block aligned
    part = CS.tm_scores(cols, 70, 1301, cs)
    assert np.array_equal(part, mine[70:1301])
    # clause statistics against brute force
    labbits = CS.pack_labels(y.astype(np.uint8))
    fire = np.zeros(cs.n, np.int64)
    firep = np.zeros(cs.n, np.int64)
    CS.k_clause_stats(cols, labbits, n, cs.ptr, cs.lits, cs.negs, fire, firep)
    for c in range(min(cs.n, 15)):
        f = np.ones(n, bool)
        for j, ng in zip(cs.lits[cs.ptr[c]:cs.ptr[c + 1]], cs.negs[cs.ptr[c]:cs.ptr[c + 1]]):
            f &= (X[:, j] == 0) if ng else (X[:, j] == 1)
        assert fire[c] == f.sum() and firep[c] == (f & (y == 1)).sum()
        assert isinstance(CS.rule_text(lits, cs.lits[cs.ptr[c]:cs.ptr[c + 1]], cs.negs[cs.ptr[c]:cs.ptr[c + 1]]), str)


def test_pack_labels_bit_order():
    lab = (np.arange(200) % 3 == 0).astype(np.uint8)
    w = CS.pack_labels(lab)
    for r in range(200):
        assert int((w[r // 64] >> np.uint64(r % 64)) & np.uint64(1)) == lab[r]


def test_hgb_evaluator_equals_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    lits = CS.literal_construction()
    rng = np.random.default_rng(8)
    n = 4000
    feat = np.stack([rng.integers(0, CS.FEAT_MAX[f] + 1, n) for f in range(CS.NF)], axis=1).astype(np.uint8)
    X = lits.matrix(feat, np.uint8)
    y = ((X[:, 3] & X[:, 60]) | (rng.random(n) < 0.05)).astype(np.int64)
    mdl = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=15, early_stopping=False, random_state=1).fit(X, y)
    ens = CS.hgb_ensemble(mdl)
    out = CS.hgb_scores(feat, lits, ens, [10, 30])
    assert np.array_equal(out[1], mdl.decision_function(X))
    staged = list(mdl._staged_raw_predict(X))
    assert np.array_equal(out[0], staged[9].ravel())


# ------------------------------------------------------------------ rules and verdicts
def test_rule_text_takes_tightest_bounds():
    lits = CS.literal_construction()
    spec = lits.spec
    idx = {s[0]: j for j, s in enumerate(spec)}
    a = idx["exact suffix match at most recent occurrence >= 2"]
    b = idx["exact suffix match at most recent occurrence >= 4"]
    c = idx["exact suffix match at most recent occurrence >= 6"]
    txt = CS.rule_text(lits, np.array([a, b, c]), np.array([0, 0, 1]))
    assert txt == ("exact suffix match at most recent occurrence >= 4 AND "
                   "NOT exact suffix match at most recent occurrence >= 6")
    pb = [j for j, s in enumerate(spec) if s[0].startswith("P_B(candidate) >")][0]
    assert "P_core(candidate)" in CS.rule_text(lits, np.array([pb]), np.array([0]))


def test_verdict_rules():
    assert CS.h91_verdict(-0.012, [-0.02, -0.005])["met"]
    assert not CS.h91_verdict(-0.012, [-0.02, 0.001])["met"]
    assert not CS.h91_verdict(-0.009, [-0.02, -0.001])["met"]
    assert CS.h92_verdict(-0.003, [-0.005, -0.001]) == "beats"
    assert CS.h92_verdict(-0.003, [-0.005, 0.001]) == "matches"
    assert CS.h92_verdict(0.0, [-0.005, 0.001]) == "matches"
    assert CS.h92_verdict(0.0001, [-0.005, 0.001]) == "trails"
    # docs/13 A5 D0.1: band B also needs the CI excluding 0 on every seed
    assert CS.g1_t2([0.011, 0.012, 0.010], [True, True, True])["band"] == "A"
    assert CS.g1_t2([0.011, 0.012, 0.010], [True, True, True])["gate"].startswith("open")
    assert CS.g1_t2([0.011, 0.012, 0.010], [True, False, True])["band"] == "C"
    assert CS.g1_t2([0.011, 0.006, 0.010], [True, True, True])["band"] == "B"
    assert CS.g1_t2([0.011, 0.006, 0.010], [True, True, True])["gate"].startswith("middle")
    assert CS.g1_t2([0.011, 0.006, 0.010], [True, False, True])["band"] == "C"
    assert CS.g1_t2([0.011, 0.004, 0.010], [True, True, True])["band"] == "C"
    assert CS.g1_t2([0.011, 0.004, 0.010], [True, True, True])["gate"].startswith("closed")
    assert CS.g1_t2([0.02, 0.015, 0.012], [True, True, True])["min_gain_bpb"] == 0.012
    with pytest.raises(ValueError):
        CS.g1_t2([0.02, 0.015], [True, True, True])
    # docs/13 A5 D0.3
    assert CS.credit_d03("A", -0.006, [-0.009, -0.002])["credit"]
    assert not CS.credit_d03("B", -0.006, [-0.009, -0.002])["credit"]
    assert not CS.credit_d03("A", -0.004, [-0.009, -0.002])["credit"]
    assert not CS.credit_d03("A", -0.006, [-0.009, 0.001])["credit"]
    # docs/13 A5 D0.2: an L-BFGS fit that stopped at its cap
    assert CS.lbfgs_hit_cap({"nit": 300, "message": "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"}, 300)
    assert CS.lbfgs_hit_cap({"nit": 12, "message": "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"}, 300)
    assert not CS.lbfgs_hit_cap({"nit": 120, "message": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"}, 300)


def test_lbfgs_cap_message_of_the_installed_scipy():
    """The cap detector recognises the message the installed scipy returns at maxiter."""
    lits = CS.literal_construction()
    rng = np.random.default_rng(21)
    n = 600
    feat = np.stack([rng.integers(0, CS.FEAT_MAX[f] + 1, n) for f in range(CS.NF)], axis=1).astype(np.uint8)
    y = (rng.random(n) < 0.3).astype(np.uint32)
    m = CS.TableLR(lits, pairwise=False)
    _, info = m.fit(feat, y, l2=1e-3, maxiter=2)
    assert CS.lbfgs_hit_cap(info, 2) and not info["converged"]
    _, info = m.fit(feat, y, l2=10.0, maxiter=2000)
    assert info["converged"] and not CS.lbfgs_hit_cap(info, 2000)


def test_fast_tm_scorer_equals_exploratory_evaluator():
    from tmu.models.classification.vanilla_classifier import TMClassifier

    lits = CS.literal_construction()
    rng = np.random.default_rng(12)
    n = 5000
    feat = np.stack([rng.integers(0, CS.FEAT_MAX[f] + 1, n) for f in range(CS.NF)], axis=1).astype(np.uint8)
    X = lits.matrix(feat, np.uint32)
    y = ((X[:, 3] & X[:, 60]) | (X[:, 100] & (1 - X[:, 7]))).astype(np.uint32)
    tm = TMClassifier(60, T=40, s=3.0, platform="CPU", weighted_clauses=True, seed=9)
    tm.fit(X, y)
    cs = CS.tm_clause_set(tm, lits.n)
    cols = CS.pack_literals(feat, lits)
    for lo, hi in ((0, n), (1, 4999), (130, 777)):
        assert np.array_equal(CS.tm_scores(cols, lo, hi, cs), CS.tm_scores_reference(cols, lo, hi, cs))


def test_s2_halves_partition_s2():
    doc = np.repeat(np.arange(40) * 11 + 1, 500)
    off = np.tile(np.arange(500), 40)
    m = CS.s2_mask(doc, off)
    h = CS.s2_half(doc, off)
    assert set(np.unique(h[m]).tolist()) == {0, 1}
    assert abs(h[m].mean() - 0.5) < 0.05
