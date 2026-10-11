"""count_probe (docs/05 s4, docs/08 KB2): what plain counting achieves in Track B's literal space.

The engine's exact literals are position-bound tokens `tok[p + o] == t` for small offsets o. The
count probe predicts the token at p from the SUM over offsets o of P(t | tok[p + o]) estimated by
counting on the training stream (a naive-Bayes style vote of position-bound literals), using only
context tokens that are inside the stream and VISIBLE (not hidden by the eval mask). It is a
diagnostic ceiling for the M3 gate, not a language model: the vote is not normalised and the
tables ignore document boundaries (an offset that crosses a boundary sees the neighbouring
document, exactly as a position-bound literal would).

Moved here from experiments/engine_common.py in M2; the experiments import it from here.
"""

from __future__ import annotations

import numpy as np


def count_probe_tables(train_tok: np.ndarray, vocab: int, offsets: tuple[int, ...]) -> dict[int, np.ndarray]:
    """P(t | t[o]) tables from a training token stream, one [vocab, vocab] float64 table per offset
    (rows: the context token at offset o; columns: the token at the target position). Document
    boundaries are ignored."""
    tok = np.asarray(train_tok).astype(np.int64)
    n = tok.shape[0]
    tables = {}
    for o in offsets:
        tab = np.zeros((vocab, vocab), dtype=np.float64)
        p = np.arange(max(0, -o), min(n, n - o))
        np.add.at(tab, (tok[p + o], tok[p]), 1.0)
        tab /= np.maximum(1.0, tab.sum(axis=1, keepdims=True))
        tables[o] = tab
    return tables


def count_probe_scores(eval_tok: np.ndarray, pos: np.ndarray, tables: dict[int, np.ndarray],
                       hidden: np.ndarray | None = None) -> np.ndarray:
    """[len(pos), vocab] summed votes for the positions `pos` of `eval_tok`; a context position
    that is outside the stream or hidden contributes nothing."""
    tok = np.asarray(eval_tok).astype(np.int64)
    n = tok.shape[0]
    vocab = next(iter(tables.values())).shape[0]
    pos = np.asarray(pos).astype(np.int64)
    scores = np.zeros((pos.shape[0], vocab), dtype=np.float64)
    for o, tab in tables.items():
        q = pos + o
        ok = (q >= 0) & (q < n)
        if hidden is not None:
            ok &= np.asarray(hidden)[np.clip(q, 0, n - 1)] == 0
        scores[ok] += tab[tok[q[ok]]]
    return scores


def count_probe_predict(eval_tok: np.ndarray, eval_pos: np.ndarray, tables: dict[int, np.ndarray],
                        hidden: np.ndarray | None = None, chunk: int = 20000) -> np.ndarray:
    """Argmax prediction (ties toward the lower id) for every position of `eval_pos`."""
    eval_pos = np.asarray(eval_pos)
    out = np.zeros(eval_pos.shape[0], dtype=np.int64)
    for i in range(0, eval_pos.shape[0], chunk):
        out[i:i + chunk] = count_probe_scores(eval_tok, eval_pos[i:i + chunk], tables, hidden).argmax(axis=1)
    return out


def count_probe_top1(eval_tok: np.ndarray, eval_pos: np.ndarray, tables: dict[int, np.ndarray],
                     hidden: np.ndarray | None = None, chunk: int = 20000) -> float:
    """Top-1 accuracy of `count_probe_predict` against the true tokens at `eval_pos`."""
    eval_pos = np.asarray(eval_pos)
    if eval_pos.shape[0] == 0:
        return 0.0
    pred = count_probe_predict(eval_tok, eval_pos, tables, hidden, chunk)
    gold = np.asarray(eval_tok).astype(np.int64)[eval_pos.astype(np.int64)]
    return float((pred == gold).mean())
