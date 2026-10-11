"""Tests of the CR-TM pilot P1 module (src/tmlm/cvtm/crtm.py, docs/13 A5): literal tables, the pair
kernel against a plain reference, the bit-parallel TM evaluator against a plain evaluation and tmu,
negative sampling, the table LR controls, the HGB design matrix, the exact read-out, the verdict rules
and the BLiMP split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tmlm.cvtm import copy_selector as CS
from tmlm.cvtm import crtm
from tmlm.cvtm.mixture import BucketLayout

V = crtm.V


def nested_kind_map(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c3 = rng.integers(0, 243 * 16, size=V)          # 243 level-2 clusters used, as in the real map
    c2 = c3 // 16
    c1 = c2 // 16
    km = np.zeros((4, V), np.int64)
    km[0] = np.arange(V)
    km[1] = c1 + V
    km[2] = c2 + V + 16
    km[3] = c3 + V + 272
    return km


@pytest.fixture(scope="module")
def tabs():
    km = nested_kind_map()
    counts = np.random.default_rng(1).integers(0, 1000, size=V)
    t = crtm.token_tables(km, counts, None)
    t["ctype"] = np.random.default_rng(2).integers(0, 64, size=V).astype(np.int64)
    return t


@pytest.fixture(scope="module")
def lits(tabs):
    return crtm.build_literals(tabs["c2_used"])


def test_token_tables_nested_and_idx4(tabs):
    km = nested_kind_map()
    c3 = km[3] - (V + 272)
    for c in np.unique(c3)[:50]:
        members = np.flatnonzero(c3 == c)
        assert np.array_equal(np.sort(tabs["idx4"][members]), np.arange(members.shape[0]))
    assert np.array_equal(tabs["sub2"], tabs["c2"] % 16)
    assert np.all(tabs["c2"] // 16 == tabs["c1"])


def test_token_tables_rejects_non_nested():
    km = nested_kind_map()
    km[2, 5] = (km[2, 5] - (V + 16) + 16) % 256 + V + 16
    with pytest.raises(ValueError):
        crtm.token_tables(km, np.ones(V), None)


def test_literal_tables_and_sets(lits, tabs):
    for j in range(lits.n):
        k, t = lits.kind[j], int(lits.thr[j])
        v = np.arange(256)
        want = {"eq": v == t, "ge": v >= t, "le": v <= t, "bit": ((v >> t) & 1) == 1}[k]
        assert np.array_equal(lits.LT[j], want)
    sets = crtm.literal_sets(lits, int(tabs["idx4"].max()))
    fam_main = set(sets["MAIN"].family)
    assert fam_main == {"context", "candidate", "relational", "reflective", "copy"}
    assert "relational" not in set(sets["NOREL"].family)
    assert set(sets["CTXID"].family) == {"context", "candidate", "candidate_id"}
    assert sets["MAIN"].n - sets["NOREL"].n == 37
    # context: 8 offsets x (16 + 16) one-hots + 6 type bits + 11 depth thresholds
    assert sum(f == "context" for f in sets["MAIN"].family) == 8 * 32 + 6 + 11
    # every literal of every set is a function of one raw feature within RAW_MAX
    for ls in sets.values():
        assert np.all(ls.feat >= 0) and np.all(ls.feat < crtm.NRAW)


def test_bitscan_de_bruijn():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        x = np.uint64(rng.integers(1, 2 ** 63, dtype=np.uint64)) << np.uint64(rng.integers(0, 2))
        if x == 0:
            continue
        want = (int(x) & -int(x)).bit_length() - 1
        assert int(crtm._bitscan(x, crtm._DEBRUIJN)) == want
    assert int(crtm._bitscan(np.uint64(1) << np.uint64(63), crtm._DEBRUIJN)) == 63


def test_tm_evaluator_equals_plain(lits, tabs):
    ls = crtm.literal_sets(lits, int(tabs["idx4"].max()))["MAIN"]
    rng = np.random.default_rng(4)
    n = 300
    feat = np.zeros((n, crtm.NRAW), np.uint8)
    for f in range(crtm.NRAW):
        feat[:, f] = rng.integers(0, crtm.RAW_MAX[f] + 1, size=n)
    X = ls.matrix(feat).astype(bool)
    ptr, lit, neg, w = [0], [], [], []
    for c in range(60):
        k = int(rng.integers(1, 4))
        js = rng.choice(ls.n, size=k, replace=False)
        ng = rng.integers(0, 2, size=k)
        # make some clauses fire: take literal values of a random row
        r = int(rng.integers(0, n))
        ng = np.where(X[r, js], 0, 1) if c % 2 == 0 else ng
        lit.extend(js.tolist())
        neg.extend(ng.tolist())
        ptr.append(len(lit))
        w.append(int(rng.integers(-20, 21)))
    cs = CS.ClauseSet(ptr=np.array(ptr, np.int64), lits=np.array(lit, np.int64), negs=np.array(neg, np.uint8),
                      w=np.array(w, np.int64), origin=np.zeros((60, 2), np.int64))
    got = crtm.TMEval(ls, cs).score(feat)
    want = np.zeros(n)
    for c in range(60):
        fire = np.ones(n, bool)
        for e in range(ptr[c], ptr[c + 1]):
            v = X[:, lit[e]]
            fire &= ~v if neg[e] else v
        want += w[c] * fire
    assert np.array_equal(got, want)


def plain_features(tok, ds, p, top_ids, top_p, kn_ids, kn_p, kstar, bucket, ptr_len, ptr_cand, slot, t):
    """Reference implementation of one pair's raw features (module docstring of crtm)."""
    d = int(np.searchsorted(ds, p, side="right") - 1)
    a = int(ds[d])
    c = int(top_ids[slot])
    f = {}

    def nl(x, cap):
        if x <= 0:
            return cap
        if x >= 1:
            return 0
        return min(int(math.floor(-math.log2(x))), cap)

    def bl(x):
        return int(x).bit_length() - 1

    def b5(n):
        return 0 if n <= 0 else 1 if n == 1 else 2 if n <= 3 else 3 if n <= 7 else 4

    for k in range(1, 9):
        x = p - k
        f[f"ctx_c16_{k}"] = int(t["c1"][tok[x]]) if x >= a else 16
        f[f"ctx_sub_{k}"] = int(t["c2"][tok[x]] % 16) if x >= a else 16
    f["prev_type"] = int(t["ctype"][tok[p - 1]]) if p - 1 >= a else 0
    f["depth"] = min(bl(p - a), 14) if p > a else 0
    f["cand_c16"], f["cand_c2"], f["cand_type"], f["cand_fcls"] = int(t["c1"][c]), int(t["c2"][c]), int(t["ctype"][c]), int(t["fcls"][c])
    e16 = e2 = et = 0
    for k in range(8):
        x = p - 1 - k
        if x >= a:
            e16 |= (t["c1"][tok[x]] == t["c1"][c]) << k
            e2 |= (t["c2"][tok[x]] == t["c2"][c]) << k
            et |= (tok[x] == c) << k
    f["eq16"], f["eq243"], f["eqtok"] = int(e16), int(e2), int(et)
    w = tok[max(a, p - 32):p]
    f["bag16_w32"] = b5(int((t["c1"][w] == t["c1"][c]).sum()))
    f["bag243_w32"] = b5(int((t["c2"][w] == t["c2"][c]).sum()))
    nb = int((t["c2"][tok[a:p]] == t["c2"][c]).sum())
    f["bag243_doc"] = 5 if nb >= 16 else b5(nb)
    pc = float(top_p[slot])
    f["pcore"] = nl(pc, 20)
    f["rank"] = bl(slot + 1)
    f["ratio_top"] = nl(pc / top_p[0], 12)
    kr = np.flatnonzero(kn_ids == c)
    f["knrank"] = bl(int(kr[0]) + 1) if kr.shape[0] else 7
    f["ps"] = nl(float(kn_p[kr[0]]), 20) if kr.shape[0] else 20
    f["mass16"] = nl(float(top_p[t["c1"][top_ids] == t["c1"][c]].sum()), 8)
    f["mass243"] = nl(float(top_p[t["c2"][top_ids] == t["c2"][c]].sum()), 12)
    f["top1"] = nl(float(top_p[0]), 8)
    f["tail"] = nl(1.0 - float(top_p.sum()), 12)
    f["kstar"] = int(kstar)
    ms, cb, lb = crtm.bucket_parts(np.array([bucket]))
    f["mstar"], f["cbucket"], f["lbucket"] = int(ms[0]), int(cb[0]), int(lb[0])
    f["is_ptr"] = int(ptr_len >= 5 and ptr_cand == c)
    occ = [q for q in range(a, p) if tok[q] == c]
    k_occ = len(occ)
    f["nocc"] = [0, 1, 2, 2, 3, 3, 3, 3][k_occ] if k_occ < 8 else (4 if k_occ < 16 else 5 if k_occ < 32 else 6)
    Ls = []
    for q in reversed(occ[-64:]):
        L = 0
        while L < 8 and q - 1 - L >= a and tok[p - 1 - L] == tok[q - 1 - L]:
            L += 1
        Ls.append((L, p - q))
    f["dist"] = min(bl(p - occ[-1]), 14) if occ else 14
    f["lrec"] = Ls[0][0] if Ls else 0
    f["lbest"] = max([x[0] for x in Ls], default=0)
    f["nl1"] = b5(sum(x[0] >= 1 for x in Ls))
    f["nl2"] = b5(sum(x[0] >= 2 for x in Ls))
    f["nl3"] = b5(sum(x[0] >= 3 for x in Ls))
    n64 = sum(x[1] <= 64 for x in Ls)
    f["n64"] = n64 if n64 <= 2 else (3 if n64 <= 4 else 4)
    f["cand_sub2"], f["cand_sub3"], f["cand_idx4"] = int(t["c2"][c] % 16), int(t["sub3"][c]), int(t["idx4"][c])
    return np.array([f[n] for n in crtm.RAW], np.int64), int(c == tok[p])


def test_pair_kernel_equals_reference(tabs):
    rng = np.random.default_rng(5)
    # two documents over a small alphabet (many repeats), then positions and fake top-64 rows
    alpha = rng.choice(V, size=40, replace=False)
    tok = alpha[rng.integers(0, 40, size=900)].astype(np.int64)
    ds = np.array([0, 500, 900], np.int64)
    pos = np.sort(rng.choice(np.r_[np.arange(1, 500), np.arange(501, 900)], size=40, replace=False))
    q = pos.shape[0]
    top_ids = np.zeros((q, 64), np.int64)
    kn_ids = np.zeros((q, 64), np.int64)
    for i, p in enumerate(pos):
        ids = np.r_[rng.choice(alpha, size=30, replace=False),
                    rng.choice(np.setdiff1d(np.arange(V), alpha), size=34, replace=False)]
        if i % 2 == 0 and tok[p] not in ids:
            ids[int(rng.integers(0, 30))] = tok[p]
        assert np.unique(ids).shape[0] == 64
        top_ids[i] = ids
        kn_ids[i] = rng.permutation(ids)
    raw = rng.random((q, 64)) ** 3
    top_p = -np.sort(-raw, axis=1) / raw.sum(axis=1, keepdims=True) * 0.9
    kn_p = -np.sort(-rng.random((q, 64)), axis=1) * 0.01
    lay = BucketLayout.of(None)
    bucket = rng.integers(0, lay.n_buckets, size=q)
    pa = crtm.PosArrays(pos=pos, top_ids=top_ids, top_p=top_p, kn_ids=kn_ids, kn_p=kn_p,
                        kstar=rng.integers(0, 5, size=q), bucket=bucket, ptr_len=rng.integers(0, 12, size=q),
                        ptr_cand=np.where(rng.random(q) < 0.5, top_ids[:, 3], -1), p_gold=np.full(q, 0.1),
                        gold=tok[pos])
    slots = np.stack([rng.permutation(64)[:7] for _ in range(q)])
    feat, lab, cand = crtm.pair_features(tok, ds, pa, slots, tabs)
    for i in range(q):
        for s in range(slots.shape[1]):
            want, wl = plain_features(tok, ds, int(pos[i]), top_ids[i], top_p[i], kn_ids[i], kn_p[i], pa.kstar[i],
                                      bucket[i], pa.ptr_len[i], pa.ptr_cand[i], int(slots[i, s]), tabs)
            got = feat[i * 7 + s].astype(np.int64)
            bad = [crtm.RAW[k] for k in np.flatnonzero(got != want)]
            assert not bad, (i, s, bad)
            assert lab[i * 7 + s] == wl and cand[i * 7 + s] == top_ids[i, slots[i, s]]
    assert np.all(feat.max(axis=0) <= crtm.RAW_MAX)


def test_negatives_distinct_proportional_and_chunk_free():
    rng = np.random.default_rng(6)
    q = 4000
    p = np.tile(np.r_[0.4, 0.2, 0.1, 0.1, 0.05, np.full(59, 0.15 / 59)], (q, 1))
    g = np.full(q, 1, np.int64)
    doc = np.zeros(q, np.int64)
    off = np.arange(q, dtype=np.int64)
    neg = crtm.draw_negatives(p, g, doc, off, 100, 4)
    assert np.all(neg != 1)
    assert all(len(set(r)) == 4 for r in neg.tolist())
    neg2 = np.concatenate([crtm.draw_negatives(p[a:a + 700], g[a:a + 700], doc[a:a + 700], off[a:a + 700], 100, 4)
                           for a in range(0, q, 700)])
    assert np.array_equal(neg, neg2)
    assert not np.array_equal(neg, crtm.draw_negatives(p, g, doc, off, 101, 4))
    # the largest remaining class is drawn first most often (Gumbel top-K = sequential draws without replacement)
    first = np.bincount(neg[:, 0], minlength=64) / q
    want0 = 0.4 / (1 - 0.2)
    assert abs(first[0] - want0) < 0.03
    assert abs(first[2] - 0.1 / 0.8) < 0.03


def test_split_2ab_deterministic():
    doc_index = np.arange(1000, 1823)
    residue = np.array([i % 3 for i in range(823)])
    residue[residue == 2] = 2
    r2 = np.flatnonzero(residue == 2)[:261]
    residue[:] = 0
    residue[r2] = 2
    a, b = crtm.split_2ab(doc_index, residue)
    assert a.sum() == 130 and b.sum() == 131 and not np.any(a & b)
    a2, b2 = crtm.split_2ab(doc_index, residue)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_bucket_parts_match_layout():
    lay = BucketLayout.of(None)
    b = np.arange(lay.n_buckets)
    ms, cb, lb = crtm.bucket_parts(b)
    for x in range(lay.n_buckets):
        k, m, c, l = lay.decode(x)
        assert ms[x] == (0 if m is None else m + 1)
        assert cb[x] == (0 if c is None else c + 1)
        assert lb[x] == l


def test_table_lr_equals_explicit_products(lits, tabs):
    sets = crtm.literal_sets(lits, int(tabs["idx4"].max()))
    ls = sets["NOREL"].subset(np.arange(0, sets["NOREL"].n, 9))
    rng = np.random.default_rng(7)
    n = 400
    feat = np.zeros((n, crtm.NRAW), np.uint8)
    for f in range(crtm.NRAW):
        feat[:, f] = rng.integers(0, crtm.RAW_MAX[f] + 1, size=n)
    X = ls.matrix(feat).astype(np.float64)
    for pairwise in (False, True):
        m = crtm.TableLR(ls, pairwise)
        theta = rng.normal(size=m.n_params) * 0.1
        W = m.W_of(theta)
        want = np.einsum("ni,ij,nj->n", X, np.triu(W) if pairwise else np.diag(np.diag(W)), X) + theta[-1]
        assert np.allclose(m.score(feat, theta), want, atol=1e-10)
    # gradient of the fit objective by finite differences (pairwise, a few coordinates)
    m = crtm.TableLR(ls, True)
    y = rng.integers(0, 2, size=n)
    theta, info = m.fit(feat, y, 1.0, maxiter=3)
    assert info["nit"] <= 3


def test_hgb_columns_determine_every_literal(lits, tabs):
    ls = crtm.literal_sets(lits, int(tabs["idx4"].max()))["MAIN"]
    lay = crtm.hgb_layout(ls)
    rng = np.random.default_rng(8)
    n = 3000
    feat = np.zeros((n, crtm.NRAW), np.uint8)
    for f in range(crtm.NRAW):
        feat[:, f] = rng.integers(0, crtm.RAW_MAX[f] + 1, size=n)
    feat[:, crtm.R["cand_c2"]] = rng.choice(tabs["c2_used"], size=n)
    Xh = crtm.hgb_matrix(feat, lay)
    L = ls.matrix(feat)
    col_of = {}
    for k, (f, kind, bit, _) in enumerate(lay.cols):
        col_of.setdefault(f, []).append(k)
    for j in range(ls.n):
        cols = col_of[int(ls.feat[j])]
        key = [tuple(r) for r in Xh[:, cols]]
        seen = {}
        for r, v in zip(key, L[:, j]):
            assert seen.setdefault(r, v) == v, ls.names[j]
    assert lay.categorical.sum() == 8 * 2 + 2


def test_readout_exact_and_zero_score_is_core():
    rng = np.random.default_rng(9)
    q = 50
    rows = rng.random((q, V)) ** 4
    rows /= rows.sum(axis=1, keepdims=True)
    top = np.argsort(-rows, axis=1, kind="stable")[:, :64]
    top_p = np.take_along_axis(rows, top, axis=1)
    gold = np.where(rng.random(q) < 0.7, top[np.arange(q), rng.integers(0, 64, size=q)], rng.integers(0, V, size=q))
    gslot = crtm.gold_slots(top, gold)
    pa = crtm.PosArrays(pos=np.arange(q), top_ids=top, top_p=top_p, kn_ids=top, kn_p=top_p, kstar=np.zeros(q, np.int64),
                        bucket=rng.integers(0, 520, size=q), ptr_len=np.zeros(q, np.int64), ptr_cand=np.full(q, -1),
                        p_gold=rows[np.arange(q), gold], gold=gold)
    rd = crtm.readout_data(pa, gslot)
    score = rng.normal(size=q * 64)
    be = rng.normal(size=520) * 0.5
    al = rng.normal(size=520) * 0.5
    lp = CS.readout_logp(score, rd, be, al)
    for i in range(q):
        b = pa.bucket[i]
        full = CS.k_readout_row(score[i * 64:(i + 1) * 64], top_p[i], top[i], rows[i], be[b], al[b])
        assert abs(full.sum() - 1.0) < 1e-12
        assert abs(math.log(full[gold[i]]) - lp[i]) < 1e-10
    lp0 = CS.readout_logp(np.zeros(q * 64), rd, np.zeros(520), np.zeros(520))
    assert np.allclose(lp0, np.log(pa.p_gold), atol=1e-14)


def test_verdicts():
    good = {"point": -0.02, "ci95": [-0.03, -0.01]}
    ctrl = {"point": -0.004, "ci95": [-0.006, -0.001]}
    c0 = {"point": -0.006, "ci95": [-0.01, -0.002]}
    v = crtm.verdict_arm1(good, ctrl, c0)
    assert v["GO_i"] and not v["arm_i_nogo_condition"]
    v2 = crtm.verdict_arm1({"point": -0.003, "ci95": [-0.005, -0.001]}, {"point": 0.001, "ci95": [-0.001, 0.003]}, c0)
    assert not v2["GO_i"] and v2["arm_i_nogo_condition"]
    a2_no = crtm.verdict_arm2({"point": 0.5, "ci95": [-0.2, 1.2]}, {"point": -0.1, "ci95": [-1, 1]})
    a2_go = crtm.verdict_arm2({"point": 2.5, "ci95": [1.0, 4.0]}, {"point": 1.2, "ci95": [0.1, 2.0]})
    assert a2_go["GO_ii"] and a2_no["arm_ii_nogo_condition"]
    assert crtm.verdict_p1(v2, a2_no)["verdict"] == "NO-GO"
    assert crtm.verdict_p1(v2, a2_go)["verdict"] == "GO"
    assert crtm.verdict_p1(v2, None)["verdict"] == "grey"
    assert crtm.verdict_p1(v2, None)["untested_arms"] == ["arm (ii)"]
    grey = crtm.verdict_arm1({"point": -0.008, "ci95": [-0.01, -0.006]}, {"point": -0.001, "ci95": [-0.002, 0.0005]}, c0)
    assert crtm.verdict_p1(grey, a2_no)["verdict"] == "grey"


def test_blimp_split_and_normalisation():
    uid = np.repeat(np.array(["a", "b", "c"]), 1000)
    pid = np.tile(np.arange(1000), 3)
    dev = crtm.blimp_dev_mask(uid, pid)
    for u in "abc":
        assert dev[uid == u].sum() == 500
    assert np.array_equal(dev, crtm.blimp_dev_mask(uid, pid))
    assert not np.array_equal(dev[:1000], dev[1000:2000])
    assert crtm.wikitext_style("The cat isn't here.") == " The cat is n't here ."
    assert crtm.wikitext_style("Paul's dogs' bowls, yes?") == " Paul 's dogs ' bowls , yes ?"


def test_tmu_training_is_deterministic_and_matches_evaluator(lits, tabs):
    pytest.importorskip("tmu")
    from tmu.models.classification.vanilla_classifier import TMClassifier

    ls = crtm.literal_sets(lits, int(tabs["idx4"].max()))["CTXID"]
    rng = np.random.default_rng(10)
    n = 600
    feat = np.zeros((n, crtm.NRAW), np.uint8)
    for f in range(crtm.NRAW):
        feat[:, f] = rng.integers(0, crtm.RAW_MAX[f] + 1, size=n)
    y = (feat[:, crtm.R["ctx_c16_1"]] == feat[:, crtm.R["cand_c16"]]).astype(np.uint32)
    feat[: n // 2, crtm.R["cand_c16"]] = feat[: n // 2, crtm.R["ctx_c16_1"]] % 16
    y = (feat[:, crtm.R["ctx_c16_1"]] == feat[:, crtm.R["cand_c16"]]).astype(np.uint32)
    sets = []
    for _ in range(2):
        tm = TMClassifier(20, T=15, s=5.0, platform="CPU", weighted_clauses=True, seed=123)
        crtm.train_tm_epoch(tm, feat, y, ls, 123, 1, chunk=256)
        cs = CS.tm_clause_set(tm, ls.n)
        sets.append(cs)
        _, csum = tm.predict(np.ascontiguousarray(ls.matrix(feat, np.uint32)), return_class_sums=True)
        d = csum[:, 1].astype(np.int64) - csum[:, 0].astype(np.int64) - crtm.TMEval(ls, cs).score(feat).astype(np.int64)
        assert np.unique(d).shape[0] == 1 and int(np.unique(d)[0]) in (0, cs.empty_const)
    assert np.array_equal(sets[0].lits, sets[1].lits) and np.array_equal(sets[0].w, sets[1].w)


def test_hgb_walker_equals_sklearn(lits, tabs):
    from sklearn.ensemble import HistGradientBoostingClassifier

    ls = crtm.literal_sets(lits, int(tabs["idx4"].max()))["MAIN"]
    lay = crtm.hgb_layout(ls)
    rng = np.random.default_rng(11)
    n = 4000
    feat = np.zeros((n, crtm.NRAW), np.uint8)
    for f in range(crtm.NRAW):
        feat[:, f] = rng.integers(0, crtm.RAW_MAX[f] + 1, size=n)
    feat[:, crtm.R["cand_c2"]] = rng.choice(tabs["c2_used"][:60], size=n)
    y = ((feat[:, crtm.R["cand_c2"]] % 3 == 0) ^ (feat[:, crtm.R["eq16"]] & 1 == 1) ^ (feat[:, crtm.R["pcore"]] < 5)).astype(np.int64)
    mdl = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=15, categorical_features=lay.categorical,
                                         early_stopping=False, random_state=0).fit(crtm.hgb_matrix(feat, lay), y)
    trees = crtm.hgb_trees(mdl, lay)
    test = feat.copy()
    test[:500, crtm.R["cand_c2"]] = rng.choice(tabs["c2_used"][60:], size=500)      # categories unseen in training
    got = crtm.hgb_staged_scores(trees, test, [10, 30])
    X = crtm.hgb_matrix(test, lay)
    assert np.array_equal(got[1], mdl.decision_function(X))
    staged = list(mdl.staged_decision_function(X))
    assert np.array_equal(got[0], staged[9])
