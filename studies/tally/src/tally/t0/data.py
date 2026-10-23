"""T0 inputs and labels: next word on Penn Treebank (five splits of tally.ptb).

Output side (decision of 2026-10-15, option a): K = 1,001 classes, the 1,000 most
frequent words of the train split plus one class "other". A word outside the 1,000 gets
P(w) = P(other) * u(w), where u is the add-one unigram distribution over the 9,000 other
words counted on train. u is the same for every arm.

Input code: the previous 3 words, each one-hot over the same 1,000 words plus one "other" bit
(3 x 1,001 = 3,003 bits). Positions before the start of a split stream see "other". Built from
train only (rule 3).
"""
import numpy as np

from tally import ptb

K_TOP = 1000
CONTEXT = 3


def top_words(train_ids, V):
    counts = np.bincount(train_ids, minlength=V)
    order = np.lexsort((np.arange(V), -counts))  # by count, ties by word id
    return order[:K_TOP], counts


def build():
    vocab, w2i, ids, blocks = ptb.load()
    V = len(vocab)
    top, counts = top_words(ids["train"], V)
    cls = np.full(V, K_TOP, dtype=np.int32)          # word id -> class (K_TOP = other)
    cls[top] = np.arange(K_TOP, dtype=np.int32)
    other = cls == K_TOP
    u = np.where(other, counts + 1.0, 0.0)
    u /= u.sum()                                      # P(w | other), zero for top words
    return dict(vocab=vocab, ids=ids, blocks=blocks, cls=cls, top=top, u=u, V=V)


def context_codes(stream_cls):
    """(n, CONTEXT) int array: class of the word i-1, i-2, i-3 for each position i."""
    n = len(stream_cls)
    out = np.full((n, CONTEXT), K_TOP, dtype=np.int32)
    for k in range(1, CONTEXT + 1):
        out[k:, k - 1] = stream_cls[:-k]
    return out


def to_bits(codes, rows=None):
    """Dense uint32 bit matrix (n, CONTEXT * 1001) for the selected rows."""
    c = codes if rows is None else codes[rows]
    X = np.zeros((len(c), CONTEXT * (K_TOP + 1)), dtype=np.uint32)
    r = np.arange(len(c))
    for k in range(CONTEXT):
        X[r, k * (K_TOP + 1) + c[:, k]] = 1
    return X


def split_xy(d, split):
    c = d["cls"][d["ids"][split]]
    return context_codes(c), c
