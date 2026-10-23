"""Correctness tests of tally.floor on small synthetic streams (no PTB data).

    python scripts/a1_selftest.py

Each array implementation is compared with a plain dictionary implementation of the same
definition: Kneser-Ney (both modes, orders 2 to 5), the decayed cache (both reset rules), the
duplicate mask, EM (global and bucketed) and the block bootstrap.
"""
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally.floor import cache, mix, stats  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

BOS = "<s>"


def ref_kn(train, ev, V, eos, N, mode):
    """Dictionary Kneser-Ney: lmplz conventions, uniform floor over V."""
    def seqs(x):
        x = list(x)
        if mode == "stream":
            return [[BOS] + x]
        out, cur = [], []
        for w in x:
            cur.append(w)
            if w == eos:
                out.append([BOS] + cur)
                cur = []
        if cur:
            out.append([BOS] + cur)
        return out
    raw = [collections.Counter() for _ in range(N + 1)]
    for s in seqs(train):
        for i in range(1, len(s)):
            for k in range(1, N + 1):
                if i - k + 1 < 0:
                    break
                raw[k][tuple(s[i - k + 1:i + 1])] += 1
    adj = [None] * (N + 1)
    adj[N] = raw[N]
    for k in range(N - 1, 0, -1):
        a = collections.Counter()
        for g in raw[k + 1]:
            a[g[1:]] += 1
        for g, c in raw[k].items():
            if g[0] == BOS or g not in a:
                a[g] = c
        adj[k] = a
    D = [None]
    for k in range(1, N + 1):
        coc = collections.Counter(v for v in adj[k].values() if v <= 4)
        n1, n2, n3, n4 = (coc[i] for i in (1, 2, 3, 4))
        Y = n1 / (n1 + 2 * n2)
        D.append((0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3))
    tot, gam = [None] * (N + 1), [None] * (N + 1)
    for k in range(1, N + 1):
        t, dm = collections.Counter(), collections.Counter()
        for g, c in adj[k].items():
            t[g[:-1]] += c
            dm[g[:-1]] += D[k][min(c, 3)]
        tot[k] = t
        gam[k] = {h: dm[h] / t[h] for h in t}

    def prob(w, ctx):
        a = adj[1].get((w,), 0)
        p = max(a - D[1][min(a, 3)], 0) / tot[1][()] + gam[1][()] / V
        for k in range(2, min(len(ctx) + 1, N) + 1):
            h = tuple(ctx[len(ctx) - (k - 1):])
            if not tot[k].get(h):
                continue
            a = adj[k].get(h + (w,), 0)
            p = max(a - D[k][min(a, 3)], 0) / tot[k][h] + gam[k][h] * p
        return p
    out = []
    for s in seqs(ev):
        for i in range(1, len(s)):
            out.append(math.log(prob(s[i], s[max(0, i - N + 1):i])))
    return np.array(out)


def ref_cache(x, blocks, decay, skip, reset, V):
    out = np.full(len(x), np.nan)
    hist = []
    for i, w in enumerate(x):
        if i == 0 or (reset == "block" and blocks[i] != blocks[i - 1]):
            hist = []
        c = np.zeros(V)
        for j, u in hist:
            if u not in skip:
                c[u] += decay ** (i - 1 - j)
        if c.sum() > 0:
            out[i] = c[w] / c.sum()
        hist.append((i, w))
    return out


def ref_dup(tr, tb, ev, L):
    grams = set()
    for a in range(len(tr) - L + 1):
        if tb[a + L - 1] - tb[a] <= 1:
            grams.add(tuple(tr[a:a + L]))
    m = np.zeros(len(ev), dtype=bool)
    for a in range(len(ev) - L + 1):
        if tuple(ev[a:a + L]) in grams:
            m[a:a + L] = True
    return m


def synthetic(rng, n, V, eos):
    # Zipf draws mixed with three preferred successors per word; sentences end in eos
    z = 1.0 / np.arange(1, V + 1)
    z /= z.sum()
    succ = rng.choice(V, size=(V, 3), p=z)
    draws = rng.choice(V, size=n, p=z)
    x = np.empty(n, dtype=np.int64)
    x[0] = draws[0]
    for i in range(1, n):
        x[i] = succ[x[i - 1], rng.integers(3)] if rng.random() < 0.5 else draws[i]
    x[rng.random(n) < 0.06] = eos
    x[-1] = eos
    return x


def main():
    rng = np.random.default_rng(1)
    V, eos = 3000, 3
    tr = synthetic(rng, 30000, V - 2, eos)         # words V-2, V-1 never seen in train
    ev = synthetic(rng, 1500, V, eos)
    ok = True
    for mode in ("sent", "stream"):
        for N in (2, 3, 4, 5):
            m = KN(tr, V, eos, order=N, mode=mode)
            a = m.logprob(ev)
            r = ref_kn(tr, ev, V, eos, N, mode)
            err = np.abs(a - r).max()
            D = m.dists(ev, np.arange(len(ev)))
            serr = np.abs(D.sum(1) - 1).max()
            ferr = np.abs(np.log(D[np.arange(len(ev)), ev]) - a).max()
            # dist(context) for a few positions equals the batch rows
            cerr = 0.0
            for i in (0, 1, 7, 100, 777, len(ev) - 1):
                cerr = max(cerr, np.abs(m.dist(ev[:i], at_start=True) - D[i]).max())
                if mode == "stream" and i >= N:
                    # a window of the last N - 1 words, not at the start, gives the same row
                    cerr = max(cerr, np.abs(m.dist(ev[i - N + 1:i], at_start=False) - D[i]).max())
            good = err < 1e-10 and serr < 1e-12 and ferr < 1e-12 and cerr < 1e-15 and D.min() > 0
            ok &= good
            print(f"kn {mode} N={N}: max|logp-ref| {err:.2e} sum err {serr:.2e} fast-vs-vector {ferr:.2e} "
                  f"dist(context) {cerr:.2e} min {D.min():.2e} {'ok' if good else 'FAIL'}")
    blocks = np.repeat(np.arange(15), 100)
    for reset in cache.RESETS:
        for hl, skip in ((np.inf, ()), (30.0, (eos,)), (5.0, ())):
            c = cache.Cache(V, hl, reset, skip)
            fb = np.full(len(ev), -1.0)
            p, empty = c.prob(ev, blocks, fb)
            r = ref_cache(ev, blocks, c.decay, set(skip), reset, V)
            same_empty = np.array_equal(empty, np.isnan(r))
            err = np.abs(p[~empty] - r[~empty]).max()
            pos = np.array([5, 99, 100, 101, 1499, 3])
            Dv = c.dists(ev, blocks, pos, np.full((len(pos), V), 1.0 / V))
            verr = np.abs(Dv[np.arange(len(pos)), ev[pos]] - np.where(empty[pos], 1.0 / V, p[pos])).max()
            good = same_empty and err < 1e-12 and verr < 1e-12 and np.allclose(Dv.sum(1), 1)
            ok &= good
            print(f"cache {reset} hl={hl} skip={skip}: max err {err:.2e} vector {verr:.2e} "
                  f"empty {int(empty.sum())} {'ok' if good else 'FAIL'}")
    tb = np.repeat(np.arange(60), 100)
    tb[3000:] += 1                                   # one non-neighbour join at 3000
    trd = rng.integers(0, 5, 6000)
    evd = np.concatenate([trd[2995:3005], rng.integers(0, 5, 300), trd[100:120]])
    for L in (3, 8):
        md = stats.duplicate_mask(trd, tb, evd, L)
        rd = ref_dup(trd, tb, evd, L)
        good = np.array_equal(md, rd)
        ok &= good
        print(f"duplicates L={L}: share {md.mean():.3f} equal to reference {good}")
    # EM
    P = np.stack([rng.random(2000), rng.random(2000) ** 3, np.full(2000, 0.3)], 1)
    lam, trace = mix.em(P)
    grad = (P / (P @ lam)[:, None]).mean(0)
    good = np.all(np.diff(trace) >= -1e-9) and abs(lam.sum() - 1) < 1e-12 and np.allclose(grad[lam > 1e-6], 1, atol=1e-4)
    ok &= good
    print(f"em: lam {lam.round(4)} monotone and stationary {good}")
    b = rng.integers(0, 6, 2000)
    lb, tb2 = mix.em_bucketed(P, b, 8, lam, tau=10)
    good = np.all(np.diff(tb2) >= -1e-9) and np.allclose(lb.sum(1), 1) and np.allclose(lb[6:], lam)
    ok &= good
    print(f"em bucketed: monotone, rows sum to 1, empty buckets at prior {good}")
    # bootstrap: a constant difference gives a point interval; coverage of a true mean 0 (printed)
    blk = np.repeat(np.arange(40), 50)
    r0 = stats.block_bootstrap(np.full(2000, 0.25), blk)
    cover = 0
    for s in range(200):
        d = np.random.default_rng(100 + s).normal(size=40)[blk] + np.random.default_rng(500 + s).normal(size=2000)
        r = stats.block_bootstrap(d, blk, n_boot=300, seed=s)
        cover += r["lo"] <= 0 <= r["hi"]
    good = abs(r0["lo"] - 0.25) < 1e-12 and abs(r0["hi"] - 0.25) < 1e-12
    ok &= good
    print(f"bootstrap: constant case {good}; intervals covering the true mean 0: {cover}/200")
    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
