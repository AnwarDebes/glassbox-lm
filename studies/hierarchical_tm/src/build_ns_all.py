"""Synonym-test pairs, all-pairs version (data/ns_all): "next paragraph" pairs with every shared word removed.

Why not just remove the shared words? Same-article neighbours share many words, so removal would leave the
positives shorter and stripped of their topic words, and one paragraph alone would give the answer.
So both classes go through the SAME removal rule. Every paragraph X loses:
  (1) the words it shares with its neighbour (the other paragraph of its consecutive pair), and
  (2) the words it shares with one paragraph from another article.
Positives: A, B = paragraphs i and i+1 of one article. (1) is the partner, (2) a random other-article paragraph.
Negatives: A, B from two different articles, each one side of a random consecutive pair in its article.
           (1) is A's own neighbour (not shown), (2) is the partner.
So a single paragraph has the same distribution in both classes, and A and B never share a word.
Unlike data/ns (the "swap" design), every next-paragraph pair can be used, not only those sharing 0-1 words.
Article first everywhere: article uniformly among those with at least 2 paragraphs, then a consecutive pair uniformly.
Then lengths are matched: both classes keep the same number of pairs in every (shorter, longer) length cell,
because neighbouring paragraphs have similar lengths and lengths alone would otherwise hint at the class.
Removal is done on the 3,072-word vocabulary; checks use the first 1,024 words. Same file format as pairdata.py.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairdata import DATA, csr_to_ids, ids_to_csr  # noqa: E402

F_MAIN, F_BIG = 1024, 3072
SIZES = {"train": 40000, "dev": 20000, "test": 10000}
CAP = 45  # length cells: lengths above 45 words (F=1024) share one bin


def side_of_pair(arts, npar, n, rng, exclude=None):
    """Article uniformly (not equal to `exclude`), a consecutive pair (i, i+1) uniformly, a side uniformly.
    Returns article, the paragraph shown, and its neighbour."""
    a = arts[rng.integers(len(arts), size=n)]
    if exclude is not None:
        bad = a == exclude
        while bad.any():
            a[bad] = arts[rng.integers(len(arts), size=int(bad.sum()))]
            bad = a == exclude
    i = rng.integers(npar[a] - 1)
    s = rng.integers(2, size=n)
    return a, i + s, i + 1 - s


def build_split(M, M1, start, npar, arts, n, rng):
    rows = lambda a, p, m=M: m[start[a] + p]
    k = n // 2
    out, st = {}, {}
    # pools of 2.5x the target per class: the length matching below keeps only part of them
    m = int(k * 2.5)
    a = arts[rng.integers(len(arts), size=m)]
    i = rng.integers(npar[a] - 1)
    ca, cp, _ = side_of_pair(arts, npar, m, rng, exclude=a)
    da, dp, _ = side_of_pair(arts, npar, m, rng, exclude=a)
    A, B, Cx, Dx = rows(a, i), rows(a, i + 1), rows(ca, cp), rows(da, dp)
    S = A.multiply(B)
    Ap, Bp = A - S - A.multiply(Cx) + A.multiply(S).multiply(Cx), B - S - B.multiply(Dx) + B.multiply(S).multiply(Dx)
    pos = dict(art_a=a, par_a=i, art_b=a, par_b=i + 1, nb_a=i + 1, nb_b=i, x_art_a=ca, x_par_a=cp, x_art_b=da, x_par_b=dp)
    # negatives
    a1, p1, q1 = side_of_pair(arts, npar, m, rng)
    a2, p2, q2 = side_of_pair(arts, npar, m, rng, exclude=a1)
    A, B, NA, NB = rows(a1, p1), rows(a2, p2), rows(a1, q1), rows(a2, q2)
    S = A.multiply(B)
    An = A - A.multiply(NA) - S + S.multiply(NA)
    Bn = B - B.multiply(NB) - S + S.multiply(NB)
    neg = dict(art_a=a1, par_a=p1, art_b=a2, par_b=p2, nb_a=q1, nb_b=q2, x_art_a=a2, x_par_a=p2, x_art_b=a1, x_par_b=p1)
    pools = []
    for name, (Xa, Xb, meta) in (("pos", (Ap, Bp, pos)), ("neg", (An, Bn, neg))):
        Xa, Xb = sparse.csr_matrix(Xa), sparse.csr_matrix(Xb)
        Xa.eliminate_zeros(), Xb.eliminate_zeros()
        assert Xa.max() <= 1 and Xb.max() <= 1 and (Xa.data > 0).all() and (Xb.data > 0).all()
        la, lb = np.diff(Xa[:, :F_MAIN].tocsr().indptr), np.diff(Xb[:, :F_MAIN].tocsr().indptr)
        ok = (la > 0) & (lb > 0)
        st[f"{name}_dropped_empty"] = int((~ok).sum())
        cell = np.minimum(np.minimum(la, lb), CAP) * (CAP + 1) + np.minimum(np.maximum(la, lb), CAP)
        pools.append((Xa, Xb, meta, np.where(ok, cell, -1)))
    # Length matching (gates.md, Check 1b): neighbouring paragraphs have similar lengths, so lengths alone
    # would hint at the class. Keep the same number of positives and negatives in every
    # (shorter length, longer length) cell, so both classes have the same joint length distribution.
    ncell = (CAP + 1) ** 2
    cp, cn = pools[0][3], pools[1][3]
    avail = np.minimum(np.bincount(cp[cp >= 0], minlength=ncell), np.bincount(cn[cn >= 0], minlength=ncell))
    st["pool_per_class"], st["matched_available"] = m, int(avail.sum())
    assert avail.sum() >= k, "not enough length-matched pairs"
    share = avail * k / avail.sum()
    q = np.floor(share).astype(np.int64)
    for c in np.argsort(-(share - q)):  # largest remainders, only where a cell still has room
        if q.sum() == k:
            break
        if q[c] < avail[c]:
            q[c] += 1
    assert q.sum() == k
    sel = []
    for Xa, Xb, meta, cell in pools:
        idx = np.concatenate([rng.permutation(np.flatnonzero(cell == c))[:q[c]] for c in np.flatnonzero(q)])
        sel.append((Xa[idx], Xb[idx], {kk: np.asarray(v)[idx] for kk, v in meta.items()}))
    (Pa, Pb, pm), (Na, Nb, nm) = sel
    A, B = sparse.vstack([Pa, Na]).tocsr(), sparse.vstack([Pb, Nb]).tocsr()
    meta = {kk: np.r_[pm[kk], nm[kk]] for kk in pm}
    y = np.r_[np.ones(k, np.uint8), np.zeros(k, np.uint8)]
    swap = rng.random(2 * k) < 0.5  # random A/B order
    AB, j = sparse.vstack([A, B]).tocsr(), np.arange(2 * k)
    A2, B2 = AB[np.where(swap, j + 2 * k, j)], AB[np.where(swap, j, j + 2 * k)]
    for x, z in (("art_a", "art_b"), ("par_a", "par_b"), ("nb_a", "nb_b"), ("x_art_a", "x_art_b"), ("x_par_a", "x_par_b")):
        meta[x], meta[z] = np.where(swap, meta[z], meta[x]), np.where(swap, meta[x], meta[z])
    perm = rng.permutation(2 * k)
    A2, B2, y = A2[perm], B2[perm], y[perm]
    meta = {kk: v[perm].astype(np.int32) for kk, v in meta.items()}
    assert A2.multiply(B2).nnz == 0, "overlap not 0"
    assert 2 * int(y.sum()) == len(y)
    assert (meta["art_a"][y == 1] == meta["art_b"][y == 1]).all() and (meta["art_a"][y == 0] != meta["art_b"][y == 0]).all()
    assert (np.abs(meta["par_a"][y == 1] - meta["par_b"][y == 1]) == 1).all()
    assert np.isin(meta["art_a"], arts).all() and np.isin(meta["art_b"], arts).all()
    a_ids, a_off = csr_to_ids(A2)
    b_ids, b_off = csr_to_ids(B2)
    out = dict(a_ids=a_ids, a_off=a_off, b_ids=b_ids, b_off=b_off, y=y, **meta)
    st["mean_len_F1024"] = {c: [round(float(np.diff(X[y == v][:, :F_MAIN].tocsr().indptr).mean()), 2)
                                for X in (A2, B2)] for c, v in (("pos", 1), ("neg", 0))}
    return out, st


def main():
    t0 = time.time()
    seed = 20261001
    z = np.load(os.path.join(DATA, "paragraphs_F3072.npz"))
    npar = z["npar"].astype(np.int64)
    M = ids_to_csr(z["ids"], z["off"], F_BIG).astype(np.int16)  # int16: no wrap-around in the set arithmetic
    M1 = M[:, :F_MAIN].tocsr()
    start = np.concatenate([[0], np.cumsum(npar)])
    splits = json.load(open(os.path.join(DATA, "splits.json")))
    os.makedirs(os.path.join(DATA, "ns_all"), exist_ok=True)
    stats = {"seed": seed, "design": "symmetric removal, then length matching (see the docstring)", "splits": {}}
    for si, (split, n) in enumerate(SIZES.items()):
        arts = np.array(splits[split])
        arts = arts[npar[arts] >= 2]
        rng = np.random.default_rng([seed, 7, si])
        d, st = build_split(M, M1, start, npar, arts, n, rng)
        np.savez_compressed(os.path.join(DATA, "ns_all", f"{split}.npz"), **d)
        stats["splits"][split] = st
        print(f"ns_all/{split}: {len(d['y'])} pairs  {st}  ({time.time() - t0:.0f}s)", flush=True)
    stats["wall_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(DATA, "ns_all", "build_stats.json"), "w"), indent=1)
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        import shutil
        shutil.copy(os.path.abspath(__file__), run_dir)
        json.dump(stats, open(os.path.join(run_dir, "build_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
