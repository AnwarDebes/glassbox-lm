#!/usr/bin/env python
"""Per-word input bits for the test 4 slot filler (massive_slots.py output), built on the CPU.

Parts (any order, joined by "+"):
- S: the word's meaning bits (OR of its tokens' bits: "cos(token, concept k) >= t");
- P, N: S of the previous / next word; P2, N2: two words back / ahead (all 0 past the utterance edge);
- G: the utterance's OR of S (its existsT vector, the same for each of its words);
- W: the word itself (one bit per training word; unseen words have none); PW, NW: the previous / next word.
"""

import numpy as np


def load(z):
    n_sym = int(z["n_sym"])
    n_w = int(max(z["train__wid"].max(), z["val__wid"].max(), z["test__wid"].max())) + 1
    return n_sym, n_w


def build(z, sp, parts, n_sym, n_w):
    off = z[f"{sp}__off"]
    n = int(off[-1])
    S = np.unpackbits(z[f"{sp}__bits"], axis=1)[:, :n_sym].astype(np.uint8)
    wid = z[f"{sp}__wid"]
    sent = np.repeat(np.arange(len(off) - 1), np.diff(off))
    pos = np.arange(n) - off[sent]
    length = np.diff(off)[sent]

    def shift(A, k):  # row i gets row i + k of the same utterance, zeros past its edge
        out = np.zeros_like(A)
        ok = (pos + k >= 0) & (pos + k < length)
        out[ok] = A[np.flatnonzero(ok) + k]
        return out

    def onehot(ids):
        out = np.zeros((n, n_w), np.uint8)
        hit = ids >= 0
        out[np.flatnonzero(hit), ids[hit]] = 1
        return out

    cols = []
    for p in parts.split("+"):
        if p == "S":
            cols.append(S)
        elif p in ("P", "N", "P2", "N2"):
            cols.append(shift(S, {"P": -1, "N": 1, "P2": -2, "N2": 2}[p]))
        elif p == "G":
            assert (np.diff(off) > 0).all()  # every MASSIVE utterance has a word
            cols.append(np.maximum.reduceat(S, off[:-1], axis=0)[sent])
        elif p == "W":
            cols.append(onehot(wid))
        elif p in ("PW", "NW"):
            k = -1 if p == "PW" else 1
            ids = np.full(n, -1, np.int64)
            ok = (pos + k >= 0) & (pos + k < length)
            ids[ok] = wid[np.flatnonzero(ok) + k]
            cols.append(onehot(ids))
        else:
            raise ValueError(p)
    return np.ascontiguousarray(np.concatenate(cols, 1)), z[f"{sp}__y"], off


def label_seqs(y, off, names):
    return [[names[int(t)] for t in y[off[i] : off[i + 1]]] for i in range(len(off) - 1)]


def names(parts, graph_npz, graph_meta, vocab):
    """Readable name of every input bit, in the column order of build()."""
    th, off = graph_npz["th_flat"], graph_npz["th_off"]
    concepts = graph_meta["concepts"]
    sym = [f"{{{concepts[k]}}} >= {th[j]:.3f}" for k in range(len(concepts)) for j in range(off[k], off[k + 1])]
    where = {"S": "this word ~ ", "P": "previous word ~ ", "N": "next word ~ ", "P2": "word -2 ~ ", "N2": "word +2 ~ ",
             "G": "some word ~ "}
    out = []
    for p in parts.split("+"):
        if p in where:
            out += [where[p] + s for s in sym]
        elif p in ("W", "PW", "NW"):
            lead = {"W": "word = ", "PW": "previous word = ", "NW": "next word = "}[p]
            out += [f"{lead}'{w}'" for w in vocab]
    return out
