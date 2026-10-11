"""Baselines every clause language model must beat, on PTB word level (validation split).

Models: unigram (maximum likelihood), bigram and trigram interpolated Kneser-Ney with one
discount per order (D = n1 / (n1 + 2 n2)); the middle order of the trigram uses continuation
counts, as in standard interpolated KN.
Metrics on the validation stream (every token, <eos> included): perplexity, and next-word
top-1 / top-5 accuracy (the argmax over the full vocabulary given the context).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptb
from runlog import Run


def discount(counts):
    n1, n2 = int((counts == 1).sum()), int((counts == 2).sum())
    return n1 / (n1 + 2 * n2)


def group(keys, vals, V):
    """CSR over the first id of 2-part keys k = a*V + b: returns start[a], b-array, val-array."""
    a, b = keys // V, keys % V
    order = np.argsort(a, kind="stable")
    a, b, vals = a[order], b[order], vals[order]
    start = np.searchsorted(a, np.arange(V + 1))
    return start, b, vals


def topk_hits(dist, target, k=5):
    top = np.argpartition(-dist, k)[:k]
    best = top[np.argmax(dist[top])]
    return int(best == target), int(target in top)


def main():
    vocab, w2i, ids = ptb.load(("train", "valid"))
    V, eos = len(vocab), w2i["<eos>"]
    tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
    run = Run("ngram_baselines", {"data": "PTB word level", "V": V, "n_train": len(tr), "n_valid": len(va),
                                  "eval": "validation stream, <eos> included"},
              data_files=[ptb.path("train"), ptb.path("valid")])
    stream = np.concatenate([[eos, eos], tr])           # two leading <eos> as the start context
    vprev2 = np.concatenate([[eos, eos], va])[:-2]       # context t for each validation token
    vprev1 = np.concatenate([[eos], va])[:-1]            # context u for each validation token
    out = {}

    # ---- unigram
    c1 = np.bincount(tr, minlength=V).astype(np.float64)
    p1 = c1 / c1.sum()
    top = np.argsort(-p1)[:5]
    out["unigram"] = dict(ppl=float(np.exp(-np.log(p1[va]).mean())), top1=float((va == top[0]).mean()),
                          top5=float(np.isin(va, top).mean()))
    run.print("unigram", out["unigram"])

    # ---- counts
    k2, c2 = np.unique(stream[1:-1] * V + stream[2:], return_counts=True)
    k3, c3 = np.unique((stream[:-2] * V + stream[1:-1]) * V + stream[2:], return_counts=True)
    D2, D3 = discount(c2), discount(c3)
    cont1 = np.bincount(k2 % V, minlength=V).astype(np.float64)       # N1+(. w)
    pcont = (cont1 + 1e-3) / (cont1.sum() + 1e-3 * V)
    kmid, cmid = np.unique(k3 % (V * V), return_counts=True)           # N1+(. u w)
    Dm = discount(cmid)

    # bigram tables keyed by u
    b_start, b_w, b_c = group(k2, c2.astype(np.float64), V)
    cu = np.add.reduceat(np.r_[b_c, 0], b_start[:-1])[:V] * (np.diff(b_start) > 0)
    nu = np.diff(b_start).astype(np.float64)
    # middle (continuation) tables keyed by u
    m_start, m_w, m_c = group(kmid, cmid.astype(np.float64), V)
    su = np.add.reduceat(np.r_[m_c, 0], m_start[:-1])[:V] * (np.diff(m_start) > 0)
    mu = np.diff(m_start).astype(np.float64)

    def bigram_dist(u):
        if cu[u] == 0:
            return pcont.copy()
        d = (D2 * nu[u] / cu[u]) * pcont
        s, e = b_start[u], b_start[u + 1]
        d[b_w[s:e]] += np.maximum(b_c[s:e] - D2, 0) / cu[u]
        return d

    def mid_dist(u):
        if su[u] == 0:
            return pcont.copy()
        d = (Dm * mu[u] / su[u]) * pcont
        s, e = m_start[u], m_start[u + 1]
        d[m_w[s:e]] += np.maximum(m_c[s:e] - Dm, 0) / su[u]
        return d

    # ---- bigram KN: group validation positions by u
    lp, h1, h5 = np.zeros(len(va)), 0, 0
    order = np.argsort(vprev1, kind="stable")
    bounds = np.flatnonzero(np.diff(np.r_[-1, vprev1[order], V]))
    for gi in range(len(bounds) - 1):
        pos = order[bounds[gi]:bounds[gi + 1]]
        d = bigram_dist(int(vprev1[pos[0]]))
        lp[pos] = np.log(d[va[pos]])
        top = np.argpartition(-d, 5)[:5]
        best = top[np.argmax(d[top])]
        h1 += int((va[pos] == best).sum()); h5 += int(np.isin(va[pos], top).sum())
    out["bigram_kn"] = dict(ppl=float(np.exp(-lp.mean())), top1=h1 / len(va), top5=h5 / len(va), D=D2)
    run.print("bigram_kn", out["bigram_kn"])

    # ---- trigram KN: contexts (t,u); group by u to reuse the middle distribution
    t_ctx = vprev2 * V + vprev1
    k3ctx = k3 // V
    c3ctx_keys, c3ctx_start = np.unique(k3ctx, return_index=True)    # k3 is sorted, so contexts are contiguous
    c3ctx_end = np.r_[c3ctx_start[1:], len(k3)]
    lp, h1, h5 = np.zeros(len(va)), 0, 0
    order = np.lexsort((vprev2, vprev1))
    uo = vprev1[order]
    ubounds = np.flatnonzero(np.diff(np.r_[-1, uo, V]))
    for gi in range(len(ubounds) - 1):
        upos = order[ubounds[gi]:ubounds[gi + 1]]
        base = mid_dist(int(vprev1[upos[0]]))
        tvals = vprev2[upos]
        tb = np.flatnonzero(np.diff(np.r_[-1, tvals, V + 1]))
        for ti in range(len(tb) - 1):
            pos = upos[tb[ti]:tb[ti + 1]]
            ctx = t_ctx[pos[0]]
            j = np.searchsorted(c3ctx_keys, ctx)
            if j < len(c3ctx_keys) and c3ctx_keys[j] == ctx:
                s, e = c3ctx_start[j], c3ctx_end[j]
                ws, cs = k3[s:e] % V, c3[s:e].astype(np.float64)
                tot, n = cs.sum(), len(cs)
                d = (D3 * n / tot) * base
                d[ws] += np.maximum(cs - D3, 0) / tot
            else:
                d = base
            lp[pos] = np.log(d[va[pos]])
            top = np.argpartition(-d, 5)[:5]
            best = top[np.argmax(d[top])]
            h1 += int((va[pos] == best).sum()); h5 += int(np.isin(va[pos], top).sum())
    out["trigram_kn"] = dict(ppl=float(np.exp(-lp.mean())), top1=h1 / len(va), top5=h5 / len(va), D=D3, D_mid=Dm)
    run.print("trigram_kn", out["trigram_kn"])
    run.finish(out)


if __name__ == "__main__":
    main()
