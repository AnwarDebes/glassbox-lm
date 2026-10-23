"""The TM decision on the CPU from a saved model: packed literal masks, AND, and an integer vote.

``pack(ta, w)`` keeps the included literals (automaton state above ``n_states``), drops the empty clauses
(silent at inference, as in CoTM.class_sums), and packs each clause's plain and negated literals into
64-bit words. ``class_sums(model, X)``: a clause fires when all its plain literals are 1 and all its
negated literals are 0, i.e. (x & plain) == plain and (x & negated) == 0 word by word; the class sums are
the integer weights of the firing clauses, summed. No GPU and no floating point. Equal to
CoTM.class_sums (tests/test_cpu_tm.py). Added 2026-10-17 for the no-network agent (decision cost on the CPU).
"""

from __future__ import annotations

import numpy as np


def _words(bits: np.ndarray) -> np.ndarray:
    """uint8 0/1 [n, f] -> uint64 [n, ceil(f / 64)] (big-endian bit order within bytes, as np.packbits)."""
    b = np.packbits(bits.astype(np.uint8), axis=1)
    pad = (-b.shape[1]) % 8
    if pad:
        b = np.concatenate([b, np.zeros((b.shape[0], pad), np.uint8)], 1)
    return np.ascontiguousarray(b).view(np.uint64)


def pack(ta: np.ndarray, w: np.ndarray, n_states: int = 256, negation: bool = True) -> dict:
    ta = np.asarray(ta)
    incl = ta > n_states
    f = incl.shape[1] // 2 if negation else incl.shape[1]
    keep = incl.any(1)
    plain = incl[keep, :f]
    neg = incl[keep, f:] if negation else np.zeros_like(plain)
    return {"plain": _words(plain), "neg": _words(neg), "w": np.asarray(w)[keep].astype(np.int64), "n_features": f,
            "n_clauses_kept": int(keep.sum()), "n_clauses": int(incl.shape[0])}


def class_sums(model: dict, X: np.ndarray, budget: int = 4_000_000) -> np.ndarray:
    Xw = _words(np.asarray(X)[:, : model["n_features"]])
    P, Q, W = model["plain"], model["neg"], model["w"]
    n = Xw.shape[0]
    chunk = max(1, budget // max(1, P.shape[0] * P.shape[1]))
    out = np.zeros((n, W.shape[1]), np.int64)
    for s in range(0, n, chunk):
        x = Xw[s : s + chunk, None, :]
        fired = ((x & P[None]) == P[None]).all(2) & ((x & Q[None]) == 0).all(2)
        out[s : s + chunk] = fired.astype(np.int64) @ W
    return out
