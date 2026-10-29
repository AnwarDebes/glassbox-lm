"""Counted class chains of the count twin (PLAN_part3.md section 4, "Count twin"; review 7).

Component values at position t-k (k = 1..4), read from the same 60 input bits the TM sees
(layout.decode): E6 (the 6-bit E path, PAD 64 before the sentence start), B8 (the 8-bit B path,
PAD 256), B6 = B8 >> 2 (PAD 64) and B4 = B8 >> 4 (PAD 16).

T-B, T-EB, T-B4 and T-EB2 are interpolated modified absolute discounting down their key chains, ending
in the empty key (the class unigram, which is interpolated with the uniform distribution over the 65
classes). Raw counts at every level; discounts D1, D2, D3+ per level from the count-of-counts of that
level (tally floor/kn.py discounts(), with KenLM's fallback 0.5 / 1 / 1.5 when they are out of range):
    P_k(c | h) = max(N(h, c) - D(N(h, c)), 0) / N(h) + gamma(h) P_{k-1}(c | h'),
    gamma(h) = sum_c D(N(h, c)) / N(h);  an unseen key falls through with weight 1.
T-E is interpolated modified Kneser-Ney on the class stream (tally floor/kn.py, sentence mode,
65 classes, <eos> class 64, fallback discounts), orders 3, 4, 5.
PairKN is the same Kneser-Ney written over (context, target) pairs, so that targets can be permuted
across positions with the contexts kept (the shuffled-label run). On true targets it equals tally's KN
(checked in the self-tests and again in the twin stage before it is used)."""
import numpy as np

from . import layout, paths

paths.setup()
from tally.floor.kn import FALLBACK, KN, discounts  # noqa: E402

K = layout.N_CLASSES
RADIX = {"E6": layout.PAD_E + 1, "B8": layout.PAD_B + 1, "B6": 2 ** 6 + 1, "B4": 2 ** 4 + 1}

# key chains, finest level first; the empty key () is the last level of every chain
SPECS = {
    "TB": [[("B8", 1), ("B6", 2), ("B4", 3)], [("B8", 1), ("B6", 2)], [("B8", 1)], [("B6", 1)]],
    "TEB": [[("E6", 1), ("B8", 1), ("E6", 2)], [("E6", 1), ("B8", 1)], [("E6", 1), ("B6", 1)], [("E6", 1)]],
    "TB4": [[("B8", 1), ("B8", 2), ("B8", 3), ("B8", 4)], [("B8", 1), ("B8", 2), ("B8", 3)],
            [("B8", 1), ("B8", 2)], [("B8", 1)]],
    "TEB2": [[("E6", 1), ("B8", 1), ("E6", 2), ("B8", 2)], [("E6", 1), ("B8", 1), ("E6", 2)],
             [("E6", 1), ("B8", 1)], [("E6", 1)]],
}


def te_spec(order):
    """T-E as a chain description, for the coverage test: E6 at t-1 .. t-(order-1)."""
    return [[("E6", k) for k in range(1, m + 1)] for m in range(order - 1, 0, -1)]


def components(X):
    E6, B8, pad = layout.decode(X)
    B6 = np.where(pad, 2 ** 6, B8 >> 2)
    B4 = np.where(pad, 2 ** 4, B8 >> 4)
    return {"E6": E6, "B8": B8, "B6": B6, "B4": B4, "pad": pad}


def level_key(comp, level):
    key = np.zeros(comp["E6"].shape[0], dtype=np.int64)
    for name, k in level:
        key = key * RADIX[name] + comp[name][:, k - 1]
    return key


def chain_keys(comp, spec):
    return [level_key(comp, lv) for lv in spec]


class ADChain:
    """Interpolated modified absolute discounting over one key chain (levels finest first)."""

    def __init__(self, name, keys_train, y):
        y = np.asarray(y, dtype=np.int64)
        self.name = name
        n0 = np.bincount(y, minlength=K).astype(np.int64)
        self.D0 = discounts(n0, FALLBACK)
        d0 = self.D0[np.minimum(n0, 3)]
        N0 = n0.sum()
        self.P0 = np.maximum(n0 - d0, 0) / N0 + d0.sum() / N0 / K
        self.levels = []
        for kk in keys_train[::-1]:                 # coarse to fine
            pk = kk * K + y
            u, cnt = np.unique(pk, return_counts=True)
            ctx = u // K
            cl = (u % K).astype(np.int64)
            D = discounts(cnt, FALLBACK)
            d = D[np.minimum(cnt, 3)]
            uc, start, per = np.unique(ctx, return_index=True, return_counts=True)
            tot = np.add.reduceat(cnt.astype(np.float64), start)
            dm = np.add.reduceat(d, start)
            num = np.maximum(cnt - d, 0) / np.repeat(tot, per)
            self.levels.append({"ukey": uc, "ptr": np.append(start, len(u)), "cls": cl, "num": num,
                                "gam": dm / tot, "D": D, "contexts": int(len(uc)), "pairs": int(len(u))})

    def dist(self, keys_eval):
        n = len(keys_eval[0])
        P = np.tile(self.P0, (n, 1))
        for lv, kk in zip(self.levels, keys_eval[::-1]):
            idx = np.searchsorted(lv["ukey"], kk)
            idx = np.minimum(idx, len(lv["ukey"]) - 1)
            seen = lv["ukey"][idx] == kk
            rows = np.flatnonzero(seen)
            h = idx[rows]
            P[rows] *= lv["gam"][h][:, None]
            a, b = lv["ptr"][h], lv["ptr"][h + 1]
            lens = b - a
            r = np.repeat(rows, lens)
            e = np.repeat(a - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
            P[r, lv["cls"][e]] += lv["num"][e]
        return P

    def summary(self):
        return {"name": self.name, "levels": [{"contexts": lv["contexts"], "pairs": lv["pairs"],
                                               "D": [round(float(x), 4) for x in lv["D"][1:]]}
                                              for lv in self.levels[::-1]],
                "D0": [round(float(x), 4) for x in self.D0[1:]]}


class TEModel:
    """T-E: tally KN on the class stream (sentence mode, 65 classes, <eos> class 64)."""

    def __init__(self, cls_train, order):
        self.order = order
        self.kn = KN(np.asarray(cls_train, dtype=np.int64), K, K - 1, order=order, mode="sent",
                     discount_fallback=FALLBACK)

    def dist(self, cls_stream):
        return self.kn.dists(np.asarray(cls_stream, dtype=np.int64), np.arange(len(cls_stream)))

    def summary(self):
        return {"order": self.order, **self.kn.summary()}


BOS = K            # 65: the sentence-start marker inside a class context
BEYOND = -1        # before the sentence start marker
CRADIX = K + 1     # context items 0..65


def class_contexts(cls_stream, eos_class, m):
    """(n, m): item at t-1-j (j = 0..m-1) of the class stream inside the sentence of t: its class, BOS
    at the slot just before the sentence start, BEYOND further back."""
    x = np.asarray(cls_stream, dtype=np.int64)
    n = len(x)
    start_flag = np.zeros(n, dtype=bool)
    start_flag[0] = True
    start_flag[1:] = x[:-1] == eos_class
    start = np.maximum.accumulate(np.where(start_flag, np.arange(n), 0))
    ctx = np.full((n, m), BEYOND, dtype=np.int64)
    for j in range(m):
        idx = np.arange(n) - 1 - j
        inside = idx >= start
        ctx[inside, j] = x[idx[inside]]
        ctx[idx == start - 1, j] = BOS
    return ctx


def _ctx_key(ctx, k):
    """Key of the (k-1)-item context of order k (items t-(k-1) .. t-1, oldest first)."""
    key = np.zeros(ctx.shape[0], dtype=np.int64)
    for j in range(k - 2, -1, -1):
        key = key * CRADIX + ctx[:, j]
    return key


class PairKN:
    """Interpolated modified Kneser-Ney over (context, target) pairs, with tally KN's sentence-mode
    rules: adjusted counts below the top order are the numbers of distinct left extensions; k-grams that
    start with BOS (or are never extended) keep raw counts; p_1 is interpolated with the uniform."""

    def __init__(self, ctx, y, order):
        y = np.asarray(y, dtype=np.int64)
        N = order
        self.N = N
        valid = {k: (ctx[:, k - 2] != BEYOND) if k >= 2 else np.ones(len(y), dtype=bool) for k in range(1, N + 1)}
        gkey = {}                                   # k-gram key per valid position
        for k in range(1, N + 1):
            ck = _ctx_key(ctx[valid[k]], k) if k >= 2 else np.zeros(int(valid[k].sum()), dtype=np.int64)
            gkey[k] = ck * K + y[valid[k]]
        raw, types = {}, {}
        for k in range(1, N + 1):
            types[k], raw[k] = np.unique(gkey[k], return_counts=True)
        adj = {N: raw[N].astype(np.int64)}
        for k in range(N - 1, 0, -1):
            hi = types[k + 1]                       # (k+1)-gram types; their suffix k-gram:
            ctx_hi, tgt = hi // K, hi % K
            # drop the oldest item of the k-item context of the (k+1)-gram
            suffix_ctx = ctx_hi % (CRADIX ** (k - 1)) if k >= 2 else np.zeros_like(ctx_hi)
            suf = suffix_ctx * K + tgt
            pos = np.searchsorted(types[k], suf)
            assert (types[k][pos] == suf).all()
            a = np.bincount(pos, minlength=len(types[k])).astype(np.int64)
            if k >= 2:
                oldest = (types[k] // K) // (CRADIX ** (k - 2))
                bos_first = oldest == BOS
                a = np.where(bos_first | (a == 0), raw[k], a)
            else:
                a = np.where(a == 0, raw[1], a)
            adj[k] = a
        self.D = {k: discounts(adj[k], FALLBACK) for k in range(1, N + 1)}
        a1 = np.zeros(K, dtype=np.int64)
        a1[types[1].astype(np.int64)] = adj[1]
        d1 = self.D[1][np.minimum(a1, 3)]
        T1 = a1.sum()
        self.P1 = np.maximum(a1 - d1, 0) / T1 + d1.sum() / T1 / K
        self.tables = {}
        for k in range(2, N + 1):
            a = adj[k]
            d = self.D[k][np.minimum(a, 3)]
            c = types[k] // K
            uc, start, per = np.unique(c, return_index=True, return_counts=True)
            tot = np.add.reduceat(a.astype(np.float64), start)
            dm = np.add.reduceat(d, start)
            self.tables[k] = {"ukey": uc, "ptr": np.append(start, len(c)), "cls": (types[k] % K).astype(np.int64),
                              "num": np.maximum(a - d, 0) / np.repeat(tot, per), "gam": dm / tot}

    def dist(self, ctx):
        n = ctx.shape[0]
        P = np.tile(self.P1, (n, 1))
        alive = np.ones(n, dtype=bool)
        for k in range(2, self.N + 1):
            tb = self.tables[k]
            alive &= ctx[:, k - 2] != BEYOND
            key = _ctx_key(ctx, k)
            idx = np.minimum(np.searchsorted(tb["ukey"], key), len(tb["ukey"]) - 1)
            seen = alive & (tb["ukey"][idx] == key)
            alive &= seen                           # an unseen context stops the higher orders
            rows = np.flatnonzero(seen)
            h = idx[rows]
            P[rows] *= tb["gam"][h][:, None]
            a, b = tb["ptr"][h], tb["ptr"][h + 1]
            lens = b - a
            r = np.repeat(rows, lens)
            e = np.repeat(a - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
            P[r, tb["cls"][e]] += tb["num"][e]
        return P


def covered_bits(specs):
    """Which of the 60 input bits are in at least one counted chain level at full depth.

    E bits of position k: a level holding E6 at k. B bits of position k: a level holding B8 at k.
    The pad flag of position k: a level holding any component of position k (its PAD value)."""
    cov = np.zeros(layout.N_BITS, dtype=bool)
    for spec in specs:
        for lv in spec:
            for name, k in lv:
                o = (k - 1) * layout.BITS_PER_WORD
                if name == "E6":
                    cov[o:o + layout.DEPTH_E] = True
                if name == "B8":
                    cov[o + layout.DEPTH_E:o + layout.DEPTH_E + layout.DEPTH_B] = True
                cov[o + layout.BITS_PER_WORD - 1] = True
    return cov
