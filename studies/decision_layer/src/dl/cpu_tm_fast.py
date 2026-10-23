"""A faster exact TM decision on the CPU: each kept clause is a list of literal indices.

``pack(ta, w)`` keeps the included literals (automaton state above ``n_states``) of the non-empty clauses,
as dl.cpu_tm.pack does, and stores them clause by clause (CSR: literal indices, clause starts) over the
literal vector L = [x, 1 - x] (plain literals, then negated ones). ``class_sums(model, X)``: one gather of
the clauses' literals from L, the minimum per clause (np.minimum.reduceat: a clause fires when all its
literals are 1), then the integer weights of the firing clauses, summed. No GPU and no floating point in
the decision; integer results equal to dl.cpu_tm.class_sums and CoTM.class_sums (tests/test_cpu_tm_fast.py).
Added 2026-10-18 after R19 (the TM decision was 97% of the word-table agent's time in R19's speed test).
"""

from __future__ import annotations

import numpy as np


def pack(ta: np.ndarray, w: np.ndarray, n_states: int = 256, negation: bool = True) -> dict:
    ta = np.asarray(ta)
    incl = ta > n_states
    f = incl.shape[1] // 2 if negation else incl.shape[1]
    keep = np.flatnonzero(incl.any(1))
    rows, cols = np.nonzero(incl[keep])  # C order: clause by clause, literals ascending
    counts = np.bincount(rows, minlength=len(keep))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
    return {"lits": cols.astype(np.intp), "starts": starts, "w": np.ascontiguousarray(np.asarray(w)[keep].astype(np.int64)),
            "n_features": f, "negation": negation, "n_clauses_kept": int(len(keep)), "n_clauses": int(incl.shape[0]),
            "n_literals": int(len(cols))}


def literals(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.uint8)[:, : model["n_features"]]
    return np.concatenate([X, 1 - X], 1) if model["negation"] else X


def fired(model: dict, X: np.ndarray) -> np.ndarray:
    """[n, n_clauses_kept] uint8: 1 where the clause fires."""
    L = literals(model, X)
    return np.minimum.reduceat(L[:, model["lits"]], model["starts"], axis=1)


def class_sums(model: dict, X: np.ndarray, budget: int = 8_000_000) -> np.ndarray:
    X = np.asarray(X)
    n, W = X.shape[0], model["w"]
    chunk = max(1, budget // max(1, model["n_literals"]))
    out = np.zeros((n, W.shape[1]), np.int64)
    for s in range(0, n, chunk):
        F = fired(model, X[s : s + chunk]).astype(bool)
        for i in range(F.shape[0]):
            out[s + i] = W[F[i]].sum(0)
    return out


def decide(model: dict, x: np.ndarray) -> int:
    """One decision (batch 1): the class with the largest sum (ties: the lowest index, as argmax)."""
    L = literals(model, x[None])[0]
    F = np.minimum.reduceat(L[model["lits"]], model["starts"]).astype(bool)
    return int(model["w"][F].sum(0).argmax())
