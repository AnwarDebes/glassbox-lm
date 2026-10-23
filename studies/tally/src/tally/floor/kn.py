"""Interpolated modified Kneser-Ney n-gram model (Chen and Goodman 1999), three discounts per order.

The model follows the defaults of KenLM lmplz (interpolated, no pruning, unigrams interpolated
with the uniform distribution), so that it can be checked against it:

- Sentence mode ("sent"): every sentence (ending with <eos>) is its own sequence with a start
  symbol BOS in front. BOS has id V; it is a context only and is never predicted. <eos> is
  predicted as an ordinary word (it plays the role of KenLM's </s>). Contexts never cross a
  sentence end.
- Stream mode ("stream"): the whole split is one sequence with one BOS in front; contexts cross
  sentence ends.
- Adjusted counts: raw counts at the top order; below it, the number of distinct words seen to
  the left. N-grams that start with BOS (no left word possible) keep raw counts.
- Discounts per order from the count-of-counts n1..n4 of the adjusted counts:
  Y = n1 / (n1 + 2 n2), D1 = 1 - 2Y n2/n1, D2 = 2 - 3Y n3/n2, D3+ = 3 - 4Y n4/n3.
- p_1(w) = max(a(w) - D, 0) / T + gamma / V, with V the fixed vocabulary (sums to 1 over V).
  KenLM spreads gamma over the words of its training text plus </s> and <unk> instead.
- p_k(w | h) = max(a(hw) - D, 0) / T(h) + gamma(h) p_{k-1}(w | h'), where h' drops the oldest
  word; a context h never seen with a continuation falls through with weight 1.

Tables (orders k = 2..N, flattened into one array per field with offsets per order):
  keys   sorted int64, key = id of the (k-1)-word prefix * B + last word, B = V + 1.
         The id of a k-gram is its index inside its order. The id of a 1-gram is the word id
         (BOS = V). Because keys sort by prefix first, the continuations of one context form
         one contiguous range [cstart[c], cstart[c + 1]).
  num    discounted part max(a - D, 0) / T(context) of every k-gram.
  gam    gamma(context) per context id; ctot raw count of the context with a continuation.
"""
import numpy as np
from numba import njit

MODES = ("sent", "stream")


def with_bos(stream, eos, bos, mode):
    """Insert BOS before every sequence. Returns (s, target), target = positions of the input."""
    stream = np.asarray(stream, dtype=np.int64)
    if mode == "stream":
        starts = np.zeros(1, dtype=np.int64)
    elif mode == "sent":
        starts = np.concatenate([[0], np.flatnonzero(stream == eos) + 1])
        starts = starts[starts < len(stream)]
    else:
        raise ValueError(mode)
    s = np.insert(stream, starts, bos)
    target = np.flatnonzero(s != bos)
    return s, target


FALLBACK = (0.5, 1.0, 1.5)   # KenLM lmplz --discount_fallback defaults


def discounts(adj, fallback=None):
    """Modified KN discounts (D0 = 0, D1, D2, D3+) from adjusted counts.

    When the count-of-counts give no valid discounts (small vocabularies, e.g. class streams),
    fallback (D1, D2, D3+) is used if given, as KenLM does with --discount_fallback."""
    n = np.bincount(np.minimum(adj[adj > 0], 5), minlength=6)
    n1, n2, n3, n4 = (np.float64(n[i]) for i in (1, 2, 3, 4))
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = n1 / (n1 + 2 * n2)
        D = np.array([0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3])
    if not (0 < D[1] < 1 and 0 < D[2] < 2 and 0 < D[3] < 3):
        if fallback is None:
            raise ValueError(f"discounts out of range: {D}")
        return np.array([0.0, *fallback])
    return D


@njit(cache=True)
def _find(keys, lo, hi, key):
    """Index of key in keys[lo:hi] (sorted) or -1."""
    while lo < hi:
        mid = (lo + hi) >> 1
        v = keys[mid]
        if v < key:
            lo = mid + 1
        elif v > key:
            hi = mid
        else:
            return mid
    return -1


@njit(cache=True)
def _gram_ids(s, bos, N, B, keys, koff, cstart, coff):
    """ids[k, i] = id of the k-gram ending at position i of s (k = 1..N), -1 if not in the tables."""
    n = len(s)
    ids = np.full((N + 1, n), -1, dtype=np.int64)
    for i in range(n):
        ids[1, i] = s[i]
        if s[i] == bos or i == 0:
            continue
        for k in range(2, N + 1):
            c = ids[k - 1, i - 1]
            if c < 0:
                break
            r0 = cstart[coff[k] + c]
            r1 = cstart[coff[k] + c + 1]
            j = _find(keys, koff[k] + r0, koff[k] + r1, c * B + s[i])
            if j < 0:
                break
            ids[k, i] = j - koff[k]
    return ids


@njit(cache=True)
def _score(s, target, ids, N, B, P1, keys, koff, num, cstart, coff, gam):
    """Natural log probability of s[i] for every i in target (the fast path)."""
    out = np.empty(len(target), dtype=np.float64)
    for t in range(len(target)):
        i = target[t]
        w = s[i]
        p = P1[w]
        for k in range(2, N + 1):
            c = ids[k - 1, i - 1]
            if c < 0:
                break
            r0 = koff[k] + cstart[coff[k] + c]
            r1 = koff[k] + cstart[coff[k] + c + 1]
            if r1 == r0:
                continue
            j = _find(keys, r0, r1, c * B + w)
            p = gam[coff[k] + c] * p
            if j >= 0:
                p += num[j]
        out[t] = np.log(p)
    return out


@njit(cache=True)
def _dists(s, pos, ids, N, B, P1, keys, koff, num, cstart, coff, gam):
    """Full distribution over V for every position in pos (the context is s[:i])."""
    V = len(P1)
    out = np.empty((len(pos), V), dtype=np.float64)
    for t in range(len(pos)):
        i = pos[t]
        p = out[t]
        p[:] = P1
        for k in range(2, N + 1):
            c = ids[k - 1, i - 1]
            if c < 0:
                break
            r0 = koff[k] + cstart[coff[k] + c]
            r1 = koff[k] + cstart[coff[k] + c + 1]
            if r1 == r0:
                continue
            g = gam[coff[k] + c]
            for v in range(V):
                p[v] *= g
            for j in range(r0, r1):
                p[keys[j] % B] += num[j]
    return out


@njit(cache=True)
def _match(s, target, ids, N, koff, cstart, coff, ctot):
    """Longest context length with a continuation in train (0..N-1) and its raw count."""
    L = np.zeros(len(target), dtype=np.int64)
    cnt = np.zeros(len(target), dtype=np.int64)
    for t in range(len(target)):
        i = target[t]
        for k in range(2, N + 1):
            c = ids[k - 1, i - 1]
            if c < 0:
                break
            if cstart[coff[k] + c + 1] > cstart[coff[k] + c]:
                L[t] = k - 1
                cnt[t] = ctot[coff[k] + c]
    return L, cnt


class KN:
    """Interpolated modified Kneser-Ney model of order N over the fixed vocabulary 0..V-1."""

    def __init__(self, train, V, eos, order=5, mode="sent", n_uniform=None, discount_fallback=None):
        """n_uniform: size of the uniform floor of the unigram level (default V, so that the
        vector sums to 1 over V; another value is only for diagnosis against KenLM).
        discount_fallback: (D1, D2, D3+) for orders whose count-of-counts give no valid discounts
        (default: raise); the orders that used it are in self.fallback_orders."""
        assert mode in MODES and order >= 2
        self.V, self.eos, self.N, self.mode = V, eos, order, mode
        self.bos = V
        self.B = V + 1
        N, B, bos = order, self.B, self.bos
        s, target = with_bos(train, eos, bos, mode)
        n = len(s)
        # k-gram ids at every position, raw counts and one occurrence per k-gram type
        pid = [None, s.copy()]
        keys, raw, occ = [None, None], [None, np.bincount(s[target], minlength=V)], [None, None]
        for k in range(2, N + 1):
            prev = np.full(n, -1, dtype=np.int64)
            prev[1:] = pid[k - 1][:-1]
            ok = np.flatnonzero((prev >= 0) & (s != bos))
            uk, first, inv, cnt = np.unique(prev[ok] * B + s[ok], return_index=True,
                                            return_inverse=True, return_counts=True)
            p = np.full(n, -1, dtype=np.int64)
            p[ok] = inv
            pid.append(p)
            keys.append(uk)
            raw.append(cnt)
            occ.append(ok[first])
        # adjusted counts
        adj = [None] * (N + 1)
        adj[N] = raw[N].astype(np.int64)
        for k in range(N - 1, 0, -1):
            nk = V if k == 1 else len(keys[k])
            a = np.bincount(pid[k][occ[k + 1]], minlength=nk).astype(np.int64)
            if k >= 2:
                bos_first = s[occ[k] - k + 1] == bos
                a = np.where(bos_first | (a == 0), raw[k], a)
            else:
                a = np.where(a == 0, raw[1], a)
            adj[k] = a
        self.adj = adj
        self.D = [None] + [discounts(adj[k], discount_fallback) for k in range(1, N + 1)]
        self.fallback_orders = [k for k in range(1, N + 1)
                                if discount_fallback is not None
                                and np.allclose(self.D[k][1:], discount_fallback)]
        # order 1
        a1 = adj[1]
        d1 = self.D[1][np.minimum(a1, 3)]
        T1 = a1.sum()
        self.gamma1 = d1.sum() / T1
        self.P1 = np.maximum(a1 - d1, 0) / T1 + self.gamma1 / (n_uniform or V)
        # orders 2..N, flattened
        koff, coff = np.zeros(N + 2, dtype=np.int64), np.zeros(N + 2, dtype=np.int64)
        K, NUM, CS, GAM, CTOT = [], [], [], [], []
        for k in range(2, N + 1):
            nctx = (V + 1) if k == 2 else len(keys[k - 1])
            ctx = keys[k] // B
            a = adj[k]
            d = self.D[k][np.minimum(a, 3)]
            tot = np.bincount(ctx, weights=a, minlength=nctx)
            dm = np.bincount(ctx, weights=d, minlength=nctx)
            ctot = np.bincount(ctx, weights=raw[k], minlength=nctx).astype(np.int64)
            gam = np.where(tot > 0, dm / np.maximum(tot, 1), 1.0)
            num = np.maximum(a - d, 0) / tot[ctx]
            cs = np.searchsorted(ctx, np.arange(nctx + 1), side="left").astype(np.int64)
            koff[k + 1] = koff[k] + len(keys[k])
            coff[k + 1] = coff[k] + nctx + 1
            K.append(keys[k]); NUM.append(num); CS.append(cs); GAM.append(np.append(gam, 1.0))
            CTOT.append(np.append(ctot, 0))
        self.keys = np.concatenate(K)
        self.num = np.concatenate(NUM)
        self.cstart = np.concatenate(CS)
        self.gam = np.concatenate(GAM)
        self.ctot = np.concatenate(CTOT)
        self.koff, self.coff = koff, coff
        self.n_grams = [V] + [len(keys[k]) for k in range(2, N + 1)]
        self.train_tokens = len(target)

    # internal: the stream with BOS and the k-gram ids of every position
    def _prep(self, stream):
        s, target = with_bos(stream, self.eos, self.bos, self.mode)
        ids = _gram_ids(s, self.bos, self.N, self.B, self.keys, self.koff, self.cstart, self.coff)
        return s, target, ids

    def logprob(self, stream):
        """Natural log P(w_i | context) of every token of the stream (fast path)."""
        s, target, ids = self._prep(stream)
        return _score(s, target, ids, self.N, self.B, self.P1, self.keys, self.koff, self.num,
                      self.cstart, self.coff, self.gam)

    def dists(self, stream, positions):
        """Full distributions over V, shape (len(positions), V), for positions of the stream."""
        s, target, ids = self._prep(stream)
        pos = target[np.asarray(positions, dtype=np.int64)]
        return _dists(s, pos, ids, self.N, self.B, self.P1, self.keys, self.koff, self.num,
                      self.cstart, self.coff, self.gam)

    def dist(self, context, at_start=True):
        """Full distribution over V after a context (word ids, oldest first).

        at_start: the context begins a sentence (sent mode) or the stream (stream mode).
        In sent mode only the words after the last <eos> of the context count.
        """
        ctx = np.asarray(context, dtype=np.int64)
        if self.mode == "sent":
            e = np.flatnonzero(ctx == self.eos)
            if len(e):
                ctx, at_start = ctx[e[-1] + 1:], True
        if at_start:
            return self.dists(np.append(ctx, 0), [len(ctx)])[0]
        if len(ctx) == 0:
            return self.P1.copy()
        s = np.append(ctx, 0)
        ids = _gram_ids(s, self.bos, self.N, self.B, self.keys, self.koff, self.cstart, self.coff)
        return _dists(s, np.array([len(ctx)]), ids, self.N, self.B, self.P1, self.keys, self.koff,
                      self.num, self.cstart, self.coff, self.gam)[0]

    def match(self, stream):
        """Per token: longest context length with a train continuation (0..N-1), its raw count."""
        s, target, ids = self._prep(stream)
        return _match(s, target, ids, self.N, self.koff, self.cstart, self.coff, self.ctot)

    def summary(self):
        return {"order": self.N, "mode": self.mode, "V": self.V, "train_tokens": int(self.train_tokens),
                "n_grams": [int(x) for x in self.n_grams],
                "discounts": {k: [round(float(x), 4) for x in self.D[k][1:]] for k in range(1, self.N + 1)},
                "fallback_orders": self.fallback_orders}
