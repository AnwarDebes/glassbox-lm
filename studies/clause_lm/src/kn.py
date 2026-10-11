"""Interpolated Kneser-Ney trigram (one discount per order; continuation counts in the middle
order) with an evaluator that walks the validation stream grouped by context (t, u). This lets
any extra component be mixed into the full next-word distribution per context, and gives exact
perplexity and top-1/top-5 over the whole vocabulary. Matches ngram_baselines.py (trigram_kn)."""
import numpy as np


def discount(counts):
    n1, n2 = int((counts == 1).sum()), int((counts == 2).sum())
    return n1 / (n1 + 2 * n2)


def _group(keys, vals, V):
    a, b = keys // V, keys % V
    order = np.argsort(a, kind="stable")
    a, b, vals = a[order], b[order], vals[order]
    return np.searchsorted(a, np.arange(V + 1)), b, vals


class KN3:
    def __init__(self, train_ids, V, eos):
        self.V, self.eos = V, eos
        stream = np.concatenate([[eos, eos], train_ids.astype(np.int64)])
        self.k3, self.c3 = np.unique((stream[:-2] * V + stream[1:-1]) * V + stream[2:], return_counts=True)
        k2 = np.unique(stream[1:-1] * V + stream[2:])
        self.D3 = discount(self.c3)
        cont1 = np.bincount(k2 % V, minlength=V).astype(np.float64)
        self.pcont = (cont1 + 1e-3) / (cont1.sum() + 1e-3 * V)
        kmid, cmid = np.unique(self.k3 % (V * V), return_counts=True)
        self.Dm = discount(cmid)
        self.m_start, self.m_w, self.m_c = _group(kmid, cmid.astype(np.float64), V)
        self.su = np.add.reduceat(np.r_[self.m_c, 0], self.m_start[:-1])[:V] * (np.diff(self.m_start) > 0)
        self.mu = np.diff(self.m_start).astype(np.float64)
        k3ctx = self.k3 // V
        self.ctx_keys, self.ctx_start = np.unique(k3ctx, return_index=True)
        self.ctx_end = np.r_[self.ctx_start[1:], len(self.k3)]

    def mid(self, u):
        if self.su[u] == 0:
            return self.pcont.copy()
        d = (self.Dm * self.mu[u] / self.su[u]) * self.pcont
        s, e = self.m_start[u], self.m_start[u + 1]
        d[self.m_w[s:e]] += np.maximum(self.m_c[s:e] - self.Dm, 0) / self.su[u]
        return d

    def tri(self, t, u, base):
        """P(. | t, u) given base = mid(u); also returns whether the trigram context was seen."""
        ctx = t * self.V + u
        j = np.searchsorted(self.ctx_keys, ctx)
        if j < len(self.ctx_keys) and self.ctx_keys[j] == ctx:
            s, e = self.ctx_start[j], self.ctx_end[j]
            ws, cs = self.k3[s:e] % self.V, self.c3[s:e].astype(np.float64)
            tot = cs.sum()
            d = (self.D3 * len(cs) / tot) * base
            d[ws] += np.maximum(cs - self.D3, 0) / tot
            return d, True
        return base, False

    def walk(self, ids):
        """Yields (positions, t, u, kn_dist, trigram_seen) for every distinct context in the stream."""
        ids = ids.astype(np.int64)
        prev2 = np.concatenate([[self.eos, self.eos], ids])[:-2]
        prev1 = np.concatenate([[self.eos], ids])[:-1]
        order = np.lexsort((prev2, prev1))
        uo = prev1[order]
        ub = np.flatnonzero(np.diff(np.r_[-1, uo, self.V]))
        for gi in range(len(ub) - 1):
            upos = order[ub[gi]:ub[gi + 1]]
            u = int(prev1[upos[0]])
            base = self.mid(u)
            tv = prev2[upos]
            tb = np.flatnonzero(np.diff(np.r_[-1, tv, self.V + 1]))
            for ti in range(len(tb) - 1):
                pos = upos[tb[ti]:tb[ti + 1]]
                t = int(prev2[pos[0]])
                d, seen = self.tri(t, u, base)
                yield pos, t, u, d, seen
