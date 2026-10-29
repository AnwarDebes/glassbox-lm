"""Interpolated discounted chains over context levels, finest first, ending in a base distribution.

p_j(w | h_j) = max(a_j(h_j, w) - d, 0) / T_j(h_j) + gamma_j(h_j) p_{j+1}(w | h_{j+1}), with p_L = the base
(uniform, or any distribution given per position). A context never counted falls through with weight 1.

Two count modes:
- "kn" (interpolated modified Kneser-Ney, Chen and Goodman 1999, as tally floor/kn.py and KenLM lmplz): raw counts
  at the finest level and for contexts that start at the reply start (they contain BOS); otherwise the adjusted
  count of (h_j, w) is the number of distinct finer contexts h_{j-1} that back off to h_j and were seen with w.
  Discounts D1, D2, D3+ per level from the count-of-counts of its adjusted counts (tally's discounts(), with KenLM's
  fallback 0.5 / 1.0 / 1.5 when the counts give no valid discounts).
- "abs" (absolute discounting): raw counts at every level, one discount per level D = n1 / (n1 + 2 n2) (0.5 when
  that is not in (0, 1)).
A level "exists" at a position only when its context is defined there (kn: the history of k words needs t + 1 >= k);
elsewhere it falls through.

Contexts are integer tuples. Trained contexts are kept sorted by a 64-bit hash with their exact tuples, and every
lookup checks the exact tuple, so a trained chain scores any later set of positions without re-counting.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .paths import add_study_paths

add_study_paths(tally=True)
from tally.floor.kn import FALLBACK as KN_FALLBACK
from tally.floor.kn import discounts as kn_discounts

ABS_FALLBACK = 0.5
_C1, _C2, _C3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x + _C1
    x = (x ^ (x >> np.uint64(30))) * _C2
    x = (x ^ (x >> np.uint64(27))) * _C3
    return x ^ (x >> np.uint64(31))


def tuple_hash(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.int64)
    h = np.full(T.shape[0], np.uint64(0x5EED + T.shape[1]), np.uint64)
    with np.errstate(over="ignore"):
        for c in range(T.shape[1]):
            h = _mix(h ^ T[:, c].astype(np.uint64))
    return h


def abs_discount(a: np.ndarray) -> float:
    n1, n2 = float((a == 1).sum()), float((a == 2).sum())
    D = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.0
    return D if 0 < D < 1 else ABS_FALLBACK


class Level:
    """One trained level: contexts sorted by hash, CSR of (word, discounted share), back-off weights."""

    def __init__(self, tuples, h, gam, start, w, num, D, fallback):
        self.tuples, self.h, self.gam, self.start, self.w, self.num = tuples, h, gam, start, w, num
        self.D, self.fallback = D, fallback

    @property
    def n_ctx(self) -> int:
        return len(self.h)

    def lookup(self, T: np.ndarray, exists: np.ndarray) -> np.ndarray:
        out = np.full(len(exists), -1, np.int64)
        e = np.flatnonzero(exists)
        if len(e) == 0 or self.n_ctx == 0:
            return out
        Tq = np.asarray(T, np.int64)[e]
        hq = tuple_hash(Tq)
        i = np.minimum(np.searchsorted(self.h, hq), self.n_ctx - 1)
        ok = (self.h[i] == hq) & (self.tuples[i] == Tq).all(1)
        out[e[ok]] = i[ok]
        return out


def _dense(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if T.shape[0] == 0:
        return np.zeros((0, T.shape[1]), np.int64), np.zeros(0, np.int64)
    u, inv = np.unique(np.asarray(T, np.int64), axis=0, return_inverse=True)
    return u, inv.reshape(-1).astype(np.int64)


def fit_chain(Ts: list, exists: list, raw: list | None, w: np.ndarray, V: int, mode: str) -> list:
    """Train the levels on events (positions). Ts[j]: (n, k_j) tuples; exists[j]: (n,) bool; raw[j]: (n,) bool
    (kn mode: the context contains BOS; ignored at level 0 and in abs mode); w: the true token of each event."""
    assert mode in ("kn", "abs")
    w = np.asarray(w, np.int64)
    W = np.int64(V + 2)
    levels, prev_ctx = [], None
    for j, (T, ex) in enumerate(zip(Ts, exists)):
        ex = np.asarray(ex, bool)
        uq, inv = _dense(np.asarray(T)[ex])
        ctx = np.full(len(w), -1, np.int64)
        ctx[ex] = inv
        cur = ctx * W + w
        if mode == "abs" or j == 0:
            keys, a = np.unique(cur[ex], return_counts=True)
        else:
            rw = np.asarray(raw[j], bool) & ex
            ct = ex & ~rw
            if (prev_ctx[ct] < 0).any():
                raise ValueError(f"level {j}: a continuation event has no finer context")
            kr, ar = np.unique(cur[rw], return_counts=True)
            prev = prev_ctx[ct] * W + w[ct]
            _, first = np.unique(prev, return_index=True)
            kc, ac = np.unique(cur[ct][first], return_counts=True)
            if np.intersect1d(kr, kc).size:
                raise ValueError(f"level {j}: a context is both BOS-anchored and not")
            keys = np.concatenate([kr, kc])
            a = np.concatenate([ar, ac])
            o = np.argsort(keys, kind="stable")
            keys, a = keys[o], a[o]
        a = a.astype(np.int64)
        cidx, wv = keys // W, keys % W
        if mode == "kn":
            D = kn_discounts(a, KN_FALLBACK)
            fb = bool(np.allclose(D[1:], KN_FALLBACK))
            d = D[np.minimum(a, 3)]
        else:
            Dv = abs_discount(a)
            fb = Dv == ABS_FALLBACK
            D = np.array([0.0, Dv])
            d = np.minimum(a.astype(np.float64), Dv)
        n_ctx = uq.shape[0]
        T_h = np.bincount(cidx, weights=a, minlength=n_ctx)
        gam = np.bincount(cidx, weights=d, minlength=n_ctx) / T_h
        num = (a - d) / T_h[cidx]
        h = tuple_hash(uq)
        order = np.argsort(h, kind="stable")
        if len(h) and (np.diff(h[order]) == 0).any():
            raise ValueError(f"level {j}: context hash collision")
        rank = np.empty(n_ctx, np.int64)
        rank[order] = np.arange(n_ctx)
        nc = rank[cidx]
        o2 = np.lexsort((wv, nc))
        nc, wv, num = nc[o2], wv[o2], num[o2]
        start = np.searchsorted(nc, np.arange(n_ctx + 1)).astype(np.int64)
        levels.append(Level(uq[order], h[order], gam[order], start, wv.astype(np.int64), num, D, fb))
        prev_ctx = np.where(ctx >= 0, rank[np.maximum(ctx, 0)], -1)
    return levels


def lookup_all(levels: list, Ts: list, exists: list) -> np.ndarray:
    return np.stack([lv.lookup(T, ex) for lv, T, ex in zip(levels, Ts, exists)]) if levels else np.zeros((0, 0), np.int64)


@njit(cache=True)
def _apply(P, cidx, gam, gam_off, start, start_off, ws, nums):
    n, V = P.shape
    L = cidx.shape[0]
    for p in range(n):
        for j in range(L - 1, -1, -1):
            c = cidx[j, p]
            if c < 0:
                continue
            g = gam[gam_off[j] + c]
            for v in range(V):
                P[p, v] *= g
            s0 = start[start_off[j] + c]
            s1 = start[start_off[j] + c + 1]
            for e in range(s0, s1):
                P[p, ws[e]] += nums[e]


class Packed:
    """The levels flattened for the kernel."""

    def __init__(self, levels: list):
        self.L = len(levels)
        self.gam = np.concatenate([lv.gam for lv in levels]) if levels else np.zeros(0)
        self.gam_off = np.concatenate([[0], np.cumsum([lv.n_ctx for lv in levels])]).astype(np.int64)
        base, starts = 0, []
        for lv in levels:
            starts.append(lv.start + base)
            base += len(lv.w)
        self.start = np.concatenate(starts) if levels else np.zeros(1, np.int64)
        self.start_off = np.concatenate([[0], np.cumsum([lv.n_ctx + 1 for lv in levels])]).astype(np.int64)
        self.ws = np.concatenate([lv.w for lv in levels]) if levels else np.zeros(0, np.int64)
        self.nums = np.concatenate([lv.num for lv in levels]) if levels else np.zeros(0)


def apply_chain(P: np.ndarray, cidx: np.ndarray, packed: Packed) -> np.ndarray:
    """P (n, V): the base vectors, replaced in place by the chain's vectors; cidx (L, n) from lookup_all."""
    P = np.ascontiguousarray(P, np.float64)
    _apply(P, np.ascontiguousarray(cidx, np.int64), packed.gam, packed.gam_off, packed.start, packed.start_off,
           packed.ws, packed.nums)
    return P
