"""Modified Kneser-Ney (three discounts) and single-discount KN: every conditional distribution
sums to one, `prob` agrees with `dist`, the log-probability stream is finite, and a tiny corpus
matches a hand computation of the Chen and Goodman 1999 formulas.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.eval.baselines.ngram import (
    BidirectionalKN,
    KNModel,
    bits_per_byte,
    modified_discounts,
    reverse_docs,
    reversed_positions,
    word_accuracy,
)


def _corpus(seed: int, n_doc: int, vocab: int, mean_len: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 2 * mean_len, size=n_doc)
    p = 1.0 / np.arange(1, vocab + 1) ** 1.1
    tok = rng.choice(vocab, size=int(lengths.sum()), p=p / p.sum()).astype(np.uint16)
    doc_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
    return tok, doc_start


@pytest.mark.parametrize("mode", ["modified", "single"])
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_distributions_sum_to_one(mode, order):
    vocab = 30
    tok, doc_start = _corpus(seed=order, n_doc=25, vocab=vocab)
    kn = KNModel(order, vocab, discount_mode=mode, discount=0.6).fit(tok, doc_start)
    rng = np.random.default_rng(1)
    m = max(order - 1, 1)
    # seen contexts (from the stream), random contexts (mostly unseen), BOS contexts, cut contexts
    pos = rng.choice(tok.shape[0], size=40, replace=False)
    ctx_seen, valid_seen = kn.contexts(tok, doc_start, pos, None, order)
    ctx_rand = rng.integers(0, vocab, size=(40, m))
    ctx_bos = np.full((5, m), vocab, dtype=np.int64)
    ctx = np.concatenate([ctx_seen, ctx_rand, ctx_bos]) if order > 1 else np.zeros((85, 1), dtype=np.int64)
    valid = np.concatenate([valid_seen, rng.integers(0, m + 1, size=40), np.full(5, m)])
    for k in range(1, order + 1):
        d = kn.dist(ctx, valid, order=k)
        assert d.shape == (85, vocab)
        assert np.all(d >= 0)
        assert np.allclose(d.sum(axis=1), 1.0, atol=1e-9)
        w = rng.integers(0, vocab, size=85)
        assert np.allclose(kn.prob(ctx, w, valid, order=k), d[np.arange(85), w], atol=1e-12)
    lp = kn.logprob(tok, doc_start)
    assert lp.shape == tok.shape and np.all(np.isfinite(lp)) and np.all(lp <= 0)
    assert bits_per_byte(lp, 1000) > 0


def test_modified_discounts_formula_and_clamping():
    # counts of counts n1=4, n2=2, n3=1, n4=1: Y = 4/8 = 0.5
    c = np.array([1, 1, 1, 1, 2, 2, 3, 4, 7, 0, 0])
    d1, d2, d3 = modified_discounts(c)
    y = 0.5
    assert d1 == pytest.approx(1 - 2 * y * 2 / 4)
    assert d2 == pytest.approx(2 - 3 * y * 1 / 2)
    assert d3 == pytest.approx(3 - 4 * y * 1 / 1)
    # degenerate: n3 = 0 makes the ratio 0 -> D3 = 3; negative values are clamped to 0
    assert modified_discounts(np.array([1, 1, 2, 2, 2]))[2] == 3.0
    assert modified_discounts(np.array([1, 2, 3, 3, 3, 3, 3]))[1] == 0.0   # 2 - 3*(1/3)*(5/1) < 0
    assert modified_discounts(np.array([0, 0])) == (1.0, 2.0, 3.0)


def test_hand_computed_bigram_example():
    """One document 'a b a b a c' (a=0, b=1, c=2, V=3), order 2, modified KN with BOS padding.

    Raw bigrams: (BOS a):1, (a b):2, (b a):2, (a c):1 -> n1=2, n2=2, Y=1/3, D1=1/3, D2=2, D3=3.
    Context a: followers b:2, c:1, T=3, N1=1, N2=1 -> gamma = (1/3 + 2)/3 = 7/9,
    alpha(b) = 0, alpha(c) = (1 - 1/3)/3 = 2/9.
    Continuation unigrams: a:2 (after BOS and b), b:1, c:1 -> n1=2, n2=1, Y=1/2, D1=1/2, D2=2;
    T=4, N1=2, N2=1 -> gamma1 = (1 + 2)/4 = 3/4; alpha1 = (0, 1/8, 1/8); uniform 1/3:
    P1 = (1/4, 3/8, 3/8).
    P(. | a) = (7/36, 7/24, 37/72); P(. | c) = P1 (c never precedes anything: gamma = 1).
    """
    tok = np.array([0, 1, 0, 1, 0, 2], dtype=np.uint16)
    kn = KNModel(2, 3).fit(tok, np.array([0, 6], dtype=np.uint32))
    assert kn.discounts() == {1: (0.5, 2.0, 3.0), 2: (pytest.approx(1 / 3), 2.0, 3.0)}
    p1 = kn.dist(np.array([[2]]))[0]
    assert np.allclose(p1, [1 / 4, 3 / 8, 3 / 8])
    pa = kn.dist(np.array([[0]]))[0]
    assert np.allclose(pa, [7 / 36, 7 / 24, 37 / 72])
    assert pa.sum() == pytest.approx(1.0)
    # the first token is predicted from BOS: alpha(a | BOS) = (1 - 1/3)/1, gamma = 1/3
    pbos = kn.dist(np.array([[3]]))[0]
    assert np.allclose(pbos, [2 / 3 + 1 / 3 * 1 / 4, 1 / 3 * 3 / 8, 1 / 3 * 3 / 8])
    lp = kn.logprob(tok, np.array([0, 6], dtype=np.uint32))
    assert lp[0] == pytest.approx(np.log2(2 / 3 + 1 / 12))
    assert lp[1] == pytest.approx(np.log2(7 / 24))


def test_hidden_context_is_cut_at_the_hidden_token():
    vocab = 20
    tok, doc_start = _corpus(seed=3, n_doc=10, vocab=vocab)
    kn = KNModel(3, vocab).fit(tok, doc_start)
    pos = np.array([10, 11, 12])
    hidden = np.zeros(tok.shape[0], dtype=np.uint8)
    hidden[10] = 1
    ctx, valid = kn.contexts(tok, doc_start, pos, hidden)
    # pos 11 has the hidden token as its nearest context token, pos 12 as its farthest
    assert valid.tolist() == [2, 0, 1]
    # cut to none: the model's own unigram level (continuation counts, not the raw order-1 model)
    assert np.allclose(kn.dist(ctx[1:2], valid[1:2]), kn._unigram_full(3)[None, :vocab])
    # cut to one token: identical to a trigram context that was never seen with the same last token
    ids = kn.context_ids(ctx[2:3], valid[2:3])
    assert ids[3][0] == -1 and ids[2][0] >= 0
    unseen = None
    for x in range(vocab):
        probe = np.array([[x, ctx[2, 1]]])
        pid = kn.context_ids(probe)[3][0]
        if pid < 0 or kn.raw[3].total[pid] == 0:
            unseen = probe
            break
    assert unseen is not None
    assert np.allclose(kn.dist(ctx[2:3], valid[2:3]), kn.dist(unseen))


def test_reverse_helpers_and_word_accuracy():
    tok = np.arange(10, dtype=np.uint16)
    ds = np.array([0, 4, 10], dtype=np.uint32)
    assert reverse_docs(tok, ds).tolist() == [3, 2, 1, 0, 9, 8, 7, 6, 5, 4]
    assert reversed_positions(np.array([0, 3, 4, 9]), ds).tolist() == [3, 0, 9, 4]
    assert word_accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 4]), np.array([0, 0, 1, 1])) == 0.5
    bi = BidirectionalKN.fit(tok, ds, order=2, vocab=10)
    assert bi.predict(tok, ds, np.array([1, 5])).shape == (2,)
