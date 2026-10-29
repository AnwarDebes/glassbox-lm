"""Rows (one per position and candidate), the 364 readable bits, labels and the count-twin keys.

Per position (226 bits): domain-act present (one per act name) | the top 64 (act, slot) pairs present | due
placeholders >= 1, 2, 3, 4 | previous token = top-32 word k | previous token's concept (32 concepts) | previous token
is a placeholder | cue word k (40) said before | position >= 1, 2, 3, 5, 8, 12, 17, 25 | [count-derived] floor
top-1 probability >= .1, .2, .4, .6, .8 | [count-derived] floor entropy (bits) >= .5, 1, 2, 3, 4.
Per candidate (138 bits): [count-derived] rank 1..8 or not in the top 8 | [count-derived] log2 floor probability
>= -1, -2, -3, -4, -6, -8, -11 | [count-derived] floor probability / top-1 >= .5, .25, .1, .05, .02, .01 | type:
due placeholder, other placeholder, already emitted placeholder, end, punctuation, already said word, equal to
the previous token, in the top-32 list | top-32 identity (32) | concept (64 concepts) | [count-derived] rate under
the domain-acts (max) >= .01, .03, .1, .3, .6, .9 | [count-derived] rate under the act set >= the same.
The count-derived bits (44) are absent from arm A-nf.
"""

from __future__ import annotations

import numpy as np

from .data import Corpus
from .positions import Positions
from .rates import Rates
from .store import expand_csr

T_POS = (1, 2, 3, 5, 8, 12, 17, 25)
T_TOP1 = (0.1, 0.2, 0.4, 0.6, 0.8)
T_ENT = (0.5, 1, 2, 3, 4)
T_LP = (-1, -2, -3, -4, -6, -8, -11)
T_RATIO = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
T_RATE = (0.01, 0.03, 0.1, 0.3, 0.6, 0.9)
T_PB = (-1, -2, -3, -4, -6, -8, -11)  # count twin: 8 floor-probability buckets
TYPES8 = ("due_placeholder", "other_placeholder", "emitted_placeholder", "end", "punct", "said_word", "eq_prev", "in_top32")
K = 8


def bit_names(c: Corpus) -> tuple[list, np.ndarray, int]:
    """(names, count-derived flags, number of per-position bits) in column order."""
    n, cd = [], []

    def add(names, derived=False):
        n.extend(names)
        cd.extend([derived] * len(names))
    add([f"da={a}" for a in c.meta["act_names"]])
    add([f"pair={c.meta['pair_names'][p][0]}:{c.meta['pair_names'][p][1]}" for p in c.pairs64])
    add([f"due>={k}" for k in (1, 2, 3, 4)])
    add([f"prev={c.vocab[w]}" for w in c.top32])
    add([f"prev_concept32={k}" for k in range(32)])
    add(["prev_is_placeholder"])
    add([f"said_cue={c.vocab[w]}" for w in c.cue40])
    add([f"pos>={x}" for x in T_POS])
    add([f"floor_top1>={x}" for x in T_TOP1], True)
    add([f"floor_entropy>={x}" for x in T_ENT], True)
    n_pos_bits = len(n)
    add([f"rank={k}" for k in range(1, K + 1)] + ["rank=extra"], True)
    add([f"log2p>={x}" for x in T_LP], True)
    add([f"p/top1>={x}" for x in T_RATIO], True)
    add([f"type={t}" for t in TYPES8])
    add([f"cand={c.vocab[w]}" for w in c.top32])
    add([f"cand_concept64={k}" for k in range(64)])
    add([f"rate_da>={x}" for x in T_RATE], True)
    add([f"rate_set>={x}" for x in T_RATE], True)
    return n, np.array(cd), n_pos_bits


def _ge(x, th):
    return np.stack([np.asarray(x) >= t for t in th], 1)


def _onehot(x, n):
    return np.asarray(x)[:, None] == np.arange(n)[None, :]


def _ident(c: Corpus, w: np.ndarray) -> np.ndarray:
    """Index in the top-32 list, or len(top32) for every other token (BOS and PAD included)."""
    lut = np.full(c.V + 2, len(c.top32), np.int64)
    lut[c.top32] = np.arange(len(c.top32))
    return lut[np.asarray(w, np.int64)]


def _concept(c: Corpus, w: np.ndarray, k: int) -> np.ndarray:
    lut = np.full(c.V + 2, -1, np.int64)
    lut[: c.V] = c.a[f"concept{k}"]
    return lut[np.asarray(w, np.int64)]


def position_bits(c: Corpus, P: Positions, summ: dict) -> np.ndarray:
    n = P.n
    ao, po = c.a["act_off"], c.a["pair_off"]
    da = np.zeros((n, len(c.meta["act_names"])), bool)
    pr = np.zeros((n, len(c.pairs64)), bool)
    p64 = np.full(max(len(c.meta["pair_names"]), 1), -1, np.int64)
    p64[c.pairs64] = np.arange(len(c.pairs64))
    for i, r in enumerate(P.replies):  # per reply, broadcast to its positions
        s0, s1 = P.reply_start[i], P.reply_start[i + 1]
        da[s0:s1, c.a["act_ids"][ao[r]:ao[r + 1]]] = True
        k = p64[c.a["pair_ids"][po[r]:po[r + 1]]]
        pr[s0:s1, k[k >= 0]] = True
    H0 = P.H[:, 0]
    ident = _ident(c, H0)
    cue = np.stack([P.said_before(np.arange(n), np.full(n, w)) for w in c.cue40], 1) if len(c.cue40) else np.zeros((n, 0), bool)
    prev_ph = (H0 < c.V) & c.is_ph[np.minimum(H0, c.V - 1)]
    B = np.concatenate([da, pr, _ge(P.due_n, (1, 2, 3, 4)), _onehot(ident, len(c.top32)) & (ident < len(c.top32))[:, None],
                        _onehot(_concept(c, H0, 32), 32), prev_ph[:, None], cue, _ge(P.t, T_POS),
                        _ge(summ["top1"], T_TOP1), _ge(summ["ent"], T_ENT)], 1)
    return B


def build_rows(c: Corpus, P: Positions, summ: dict, rates: Rates, bits: bool = True) -> dict:
    """Rows of every position of P, in position order; candidates as summarize() gave them."""
    off = summ["cand_off"]
    n_rows = int(off[-1])
    pos = np.repeat(np.arange(P.n), np.diff(off))
    cand = summ["cand"]
    cp = summ["cand_p"]
    rank = summ["cand_rank"].astype(np.int64)
    y = cand == P.x[pos]
    if (np.bincount(pos, weights=y, minlength=P.n) > 1).any():
        raise ValueError("a position with two true candidates")
    W = c.V + 2
    due_key = np.repeat(np.arange(P.n), np.diff(P.due_off)) * W + P.due_ids
    due_key.sort()
    rk = pos * W + cand
    j = np.minimum(np.searchsorted(due_key, rk), max(len(due_key) - 1, 0))
    is_due = (due_key[j] == rk) if len(due_key) else np.zeros(n_rows, bool)
    isph = c.is_ph[cand]
    said = P.said_before(pos, cand)
    H0 = P.H[pos, 0]
    ident = _ident(c, cand)
    type5 = np.where(isph & is_due, 0, np.where(isph, 1, np.where(cand == c.END, 2, np.where(c.is_punct[cand], 3, 4))))
    logp = np.log2(cp)
    pb = np.zeros(n_rows, np.int64)
    for thr in T_PB:
        pb += logp < thr
    out = {"pos": pos.astype(np.int64), "cand": cand.astype(np.int64), "y": y, "cand_p": cp, "rank": rank,
           "pb": pb, "type5": type5.astype(np.int64), "due3": np.minimum(P.due_n[pos], 2).astype(np.int64),
           "said": said, "ident": ident, "dset": P.A[pos], "n_pos": P.n, "pos_off": off}
    if not bits:
        return out
    rate_da = rates.rate_da(P.reply[pos], cand, corpus=c)
    rate_set = rates.rate_set(P.S[pos], P.A[pos], cand)
    out["rate_da"], out["rate_set"] = rate_da, rate_set
    top1 = summ["top1"][pos]
    T8 = np.stack([isph & is_due, isph & ~is_due, isph & said, cand == c.END, c.is_punct[cand],
                   ~isph & (cand != c.END) & said, cand == H0, ident < len(c.top32)], 1)
    Cb = np.concatenate([_onehot(rank, K + 1), _ge(logp, T_LP), _ge(cp / top1, T_RATIO), T8,
                         _onehot(ident, len(c.top32)) & (ident < len(c.top32))[:, None], _onehot(_concept(c, cand, 64), 64),
                         _ge(rate_da, T_RATE), _ge(rate_set, T_RATE)], 1)
    Pb = position_bits(c, P, summ)
    out["bits"] = np.concatenate([Pb[pos], Cb], 1).astype(np.uint8)
    names, _, _ = bit_names(c)
    if out["bits"].shape[1] != len(names):
        raise ValueError(f"{out['bits'].shape[1]} bits, {len(names)} names")
    return out


def true_index(rows: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per position: the row of the true candidate (-1 if none) and whether the true token is in C."""
    n = rows["n_pos"]
    t = np.full(n, -1, np.int64)
    yi = np.flatnonzero(rows["y"])
    t[rows["pos"][yi]] = yi
    return t, t >= 0


def shuffled_labels(rows: dict, seed: int) -> np.ndarray:
    """K9: at every position whose true token is a candidate, the positive is re-drawn uniformly among its
    candidates (positions without a true candidate keep no positive)."""
    rng = np.random.default_rng(seed)
    off = rows["pos_off"]
    _, inC = true_index(rows)
    y = np.zeros(len(rows["y"]), bool)
    p = np.flatnonzero(inC)
    sizes = off[p + 1] - off[p]
    y[off[p] + (rng.random(len(p)) * sizes).astype(np.int64)] = True
    return y


def pair_columns(names: list, n_pos_bits: int, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns (within the kept bits) of the position bits and of the candidate type and identity bits."""
    kept_idx = np.flatnonzero(keep)
    pos_cols = np.flatnonzero(kept_idx < n_pos_bits)
    q_cols = np.flatnonzero([names[i].startswith(("type=", "cand=")) for i in kept_idx])
    return pos_cols, q_cols


def pair_features(rows_bits: np.ndarray, names: list, n_pos_bits: int, keep: np.ndarray, chunk: int = 200_000) -> object:
    """LR2: the arm's plain bits plus every product (position bit x candidate type or identity bit), sparse CSR,
    columns: the kept bits, then pair (i, j) at kept-bits + i * n_q + j. Built in chunks of rows (the same matrix;
    it bounds the temporary memory on 2.1M rows)."""
    import scipy.sparse as sp
    n = rows_bits.shape[0]
    if n > chunk:
        return sp.vstack([pair_features(rows_bits[s:s + chunk], names, n_pos_bits, keep, chunk)
                          for s in range(0, n, chunk)], format="csr")
    X = rows_bits[:, keep]
    pos_cols, q_cols = pair_columns(names, n_pos_bits, keep)
    Pm = sp.csr_matrix(X[:, pos_cols])
    Qm = sp.csr_matrix(X[:, q_cols])
    n = X.shape[0]
    np_, nq = len(pos_cols), len(q_cols)
    lp = np.diff(Pm.indptr)
    lq = np.diff(Qm.indptr)
    m = lp * lq
    rows = np.repeat(np.arange(n), m)
    # for each row, all (i, j) pairs of its active position bits i and type/identity bits j
    first = np.repeat(np.cumsum(m) - m, m)
    k = np.arange(int(m.sum())) - first
    lq_r = np.repeat(lq, m)
    pi = Pm.indices[np.repeat(Pm.indptr[:-1], m) + k // np.maximum(lq_r, 1)]
    qj = Qm.indices[np.repeat(Qm.indptr[:-1], m) + k % np.maximum(lq_r, 1)]
    pairs = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, pi * nq + qj)), shape=(n, np_ * nq))
    return sp.hstack([sp.csr_matrix(X.astype(np.float32)), pairs], format="csr")


def expand_positions(off: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Row indices of the given positions (all their candidates)."""
    positions = np.asarray(positions, np.int64)
    return expand_csr(off[positions], off[positions + 1] - off[positions])
