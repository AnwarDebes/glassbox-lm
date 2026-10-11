"""The in-house Kneser-Ney (single-discount mode) against nltk.lm.KneserNeyInterpolated on a
toy corpus (docs/06 "Causal perplexity", docs/12 F12).

NLTK's model is single-discount interpolated KN (discount set explicitly here) whose lower-order
gamma counts the distinct followers of a context in the RAW counts while its alpha uses
continuation counts. On a CYCLIC token sequence every n-gram occurrence has a predecessor, so
raw and continuation follower sets coincide and NLTK's numbers are the textbook single-discount
KN; the in-house model is fitted on the sequence extended by its first n-1 tokens with
`pad=False` (it counts exactly the N cyclic n-grams) and NLTK receives the same cyclic n-grams
of every order. Scores must agree to 1e-9 on every (context, word) pair of the sequence, for
orders 2 to 4 and two discounts.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.eval.baselines.ngram import KNModel

nltk_lm = pytest.importorskip("nltk.lm")


def _cyclic_corpus(seed: int, n: int, vocab: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** 0.9
    return rng.choice(vocab, size=n, p=p / p.sum()).astype(np.int64)


def _s(x) -> str:
    """NLTK vocabularies hold strings: token i is 'w<i>'."""
    return f"w{int(x)}"


def _cyclic_ngrams(seq: np.ndarray, k: int) -> list[tuple[str, ...]]:
    n = seq.shape[0]
    ext = np.concatenate([seq, seq[:k - 1]]) if k > 1 else seq
    return [tuple(_s(x) for x in ext[i:i + k]) for i in range(n)]


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("discount", [0.1, 0.75])
def test_single_discount_matches_nltk(order, discount):
    vocab = 12
    seq = _cyclic_corpus(seed=order * 10 + int(discount * 100), n=400, vocab=vocab)
    n = seq.shape[0]
    # NLTK: every cyclic k-gram of every order as one "sentence" of n-gram tuples
    from nltk.lm import KneserNeyInterpolated

    lm = KneserNeyInterpolated(order, discount=discount)
    grams = [g for k in range(1, order + 1) for g in _cyclic_ngrams(seq, k)]
    lm.fit([grams], vocabulary_text=[_s(x) for x in seq])
    # in-house: one document = the sequence extended by its first order-1 tokens, no padding
    ext = np.concatenate([seq, seq[:order - 1]]).astype(np.uint16)
    doc_start = np.array([0, ext.shape[0]], dtype=np.uint32)
    kn = KNModel(order, vocab, discount_mode="single", discount=discount, unk_interp=False, pad=False)
    kn.fit(ext, doc_start)
    assert kn.n_train_tokens == n
    # every cyclic (context, word) pair, plus words never seen after that context
    ctx_rows = []
    words = []
    for i in range(n):
        ctx = [int(seq[(i - order + 1 + j) % n]) for j in range(order - 1)]
        ctx_rows.append(ctx)
        words.append(int(seq[i]))
        ctx_rows.append(ctx)
        words.append(int((seq[i] + 1) % vocab))
    ctx_arr = np.array(ctx_rows, dtype=np.int64).reshape(len(words), order - 1)
    ours = kn.prob(ctx_arr, np.array(words, dtype=np.int64))
    theirs = np.array([lm.score(_s(w), tuple(_s(x) for x in c)) for c, w in zip(ctx_rows, words)])
    assert np.allclose(ours, theirs, atol=1e-9, rtol=0), np.abs(ours - theirs).max()
    # the full distribution agrees too, on a few contexts
    for c in ctx_rows[:5]:
        row = kn.dist(np.array([c], dtype=np.int64))[0]
        ref = np.array([lm.score(_s(w), tuple(_s(x) for x in c)) for w in range(vocab)])
        assert np.allclose(row, ref, atol=1e-9, rtol=0)


def test_unseen_context_backs_off_like_nltk():
    """A context absent from training: both models fall back to the lower order (gamma = 1)."""
    from nltk.lm import KneserNeyInterpolated

    vocab, order = 8, 3
    seq = _cyclic_corpus(seed=7, n=120, vocab=vocab)
    lm = KneserNeyInterpolated(order, discount=0.4)
    lm.fit([[g for k in range(1, order + 1) for g in _cyclic_ngrams(seq, k)]], vocabulary_text=[_s(x) for x in seq])
    ext = np.concatenate([seq, seq[:order - 1]]).astype(np.uint16)
    kn = KNModel(order, vocab, discount_mode="single", discount=0.4, unk_interp=False, pad=False)
    kn.fit(ext, np.array([0, ext.shape[0]], dtype=np.uint32))
    seen = set(zip(seq[:-1].tolist(), seq[1:].tolist())) | {(int(seq[-1]), int(seq[0]))}
    unseen = next((a, b) for a in range(vocab) for b in range(vocab) if (a, b) not in seen)
    for w in range(vocab):
        assert kn.prob(np.array([unseen]), np.array([w]))[0] == pytest.approx(lm.score(_s(w), tuple(_s(x) for x in unseen)), abs=1e-9)
