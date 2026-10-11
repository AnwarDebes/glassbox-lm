"""Synthetic corpora for engine tests and the M3 gates (docs/04 M3: planted rules, tiny LM).

Everything here is deterministic given its seed and uses only numpy's Generator, so a corpus
can be rebuilt bit for bit and hashed (protocol rule 5). Normative context: docs/11 section 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.api import CorpusArrays, TabularData
from ..engine.config import CLUSTER_SIZES, EngineConfig, n_sym, sym_cluster_base, sym_edge, sym_mask


def make_kind_map(vocab: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """kind_map u16 [4, V]: kind 0 is the identity; kinds 1 to 3 map tokens to cluster symbols.
    Without an rng the mapping is `t mod size` (cheap, deterministic); with one it is a random
    surjection so cluster tests see irregular groups."""
    km = np.zeros((4, vocab), dtype=np.uint16)
    km[0] = np.arange(vocab, dtype=np.uint16)
    for kind, size in zip((1, 2, 3), CLUSTER_SIZES):
        base = sym_cluster_base(vocab, kind)
        if rng is None:
            km[kind] = base + (np.arange(vocab) % size)
        else:
            km[kind] = base + rng.integers(0, size, size=vocab)
    return km


def sym_freq_of(tok: np.ndarray, kind_map: np.ndarray, vocab: int, n_pol: int = 0) -> np.ndarray:
    """Training frequency of every symbol (docs/11 section 2); MASK and EDGE hold 0xFFFFFFFF."""
    freq = np.zeros(n_sym(vocab, n_pol), dtype=np.uint64)
    counts = np.bincount(tok, minlength=vocab).astype(np.uint64)
    for kind in range(4):
        np.add.at(freq, kind_map[kind].astype(np.int64), counts)
    freq = np.minimum(freq, np.uint64(0xFFFFFFFF)).astype(np.uint32)
    freq[sym_mask(vocab)] = 0xFFFFFFFF
    freq[sym_edge(vocab)] = 0xFFFFFFFF
    return freq


def stop_of(tok: np.ndarray, vocab: int, n_stop: int) -> np.ndarray:
    """1 for the n_stop most frequent tokens (ties toward the lower id), else 0."""
    counts = np.bincount(tok, minlength=vocab)
    order = np.lexsort((np.arange(vocab), -counts))
    stop = np.zeros(vocab, dtype=np.uint8)
    stop[order[:n_stop]] = 1
    return stop


def corpus_from_tokens(tok: np.ndarray, doc_start: np.ndarray, vocab: int, n_stop: int = 64,
                       kind_rng: np.random.Generator | None = None, targets: np.ndarray | None = None) -> CorpusArrays:
    tok = np.asarray(tok, dtype=np.uint16)
    doc_start = np.asarray(doc_start, dtype=np.uint32)
    km = make_kind_map(vocab, kind_rng)
    if targets is None:
        targets = np.arange(tok.shape[0], dtype=np.uint32)
    return CorpusArrays(tok=tok, doc_start=doc_start, kind_map=km, stop=stop_of(tok, vocab, n_stop),
                        sym_freq=sym_freq_of(tok, km, vocab), targets=np.asarray(targets, dtype=np.uint32))


def toy_corpus(vocab: int = 50, n_docs: int = 4, doc_len: int = 60, seed: int = 1, n_stop: int = 2,
               zipf: float = 1.2) -> CorpusArrays:
    """Random Zipf-distributed tokens in equal-length documents. For unit tests."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** zipf
    p /= p.sum()
    tok = rng.choice(vocab, size=n_docs * doc_len, p=p).astype(np.uint16)
    doc_start = np.arange(0, n_docs * doc_len + 1, doc_len, dtype=np.uint32)
    return corpus_from_tokens(tok, doc_start, vocab, n_stop=n_stop, kind_rng=rng)


@dataclass(frozen=True)
class PlantedRule:
    """`IF t[o1] = a AND t[o2] = b THEN target = y` with offsets in -8..-1 (left context only)."""

    o1: int
    a: int
    o2: int
    b: int
    y: int


def planted_rules_corpus(vocab: int = 500, n_rules: int = 200, n_tokens: int = 200_000, seed: int = 1,
                         n_docs: int = 20, rule_rate: float = 0.5, noise: float = 0.0,
                         n_stop: int = 8) -> tuple[CorpusArrays, list[PlantedRule]]:
    """A corpus in which `n_rules` two-literal rules are planted (docs/04 M3, planted rules).

    Generation, document by document, left to right. At every position that is still free a
    rule INSTANCE is started with probability `rule_rate`: a rule r is drawn uniformly and its
    three tokens are written at p (= a), p + (r.o2 - r.o1) (= b) and p - r.o1 (= y, the
    consequent), provided all three cells are free and inside the document; the consequent is
    flipped to a random other token with probability `noise`. Cells that stay free receive a
    Zipf background token. Rules use distinct (offset, token) premise pairs, so each consequent
    is identifiable. Because the two premises are only 1 to 7 positions apart and the background
    is Zipf, some background positions also satisfy a rule's premises by chance; the recovery
    test counts positions where the premises hold, whichever way they arose.

    The recovery test (docs/04 M3 ACCEPT: >= 80% of planted rules recovered) asks whether a
    trained engine has, for at least 80% of the rules, a clause whose included literals are the
    two premises (plus nothing that is ever false when they hold) with its top vote on y."""
    rng = np.random.default_rng(seed)
    rules: list[PlantedRule] = []
    used: set[tuple[int, int]] = set()
    while len(rules) < n_rules:
        o1, o2 = sorted(rng.choice(np.arange(-8, 0), size=2, replace=False).tolist())
        a, b, y = rng.integers(0, vocab, size=3).tolist()
        if (o1, a) in used or (o2, b) in used or a == b:
            continue
        used.update({(o1, a), (o2, b)})
        rules.append(PlantedRule(o1, a, o2, b, y))
    p = 1.0 / np.arange(1, vocab + 1) ** 1.1
    p /= p.sum()
    doc_len = n_tokens // n_docs
    tok = np.zeros(n_docs * doc_len, dtype=np.uint16)
    assigned = np.zeros(n_docs * doc_len, dtype=bool)
    for d in range(n_docs):
        lo, hi = d * doc_len, (d + 1) * doc_len
        for pos in range(lo, hi):
            if not assigned[pos] and rng.random() < rule_rate:
                r = rules[int(rng.integers(0, n_rules))]
                q_b, q_y = pos + (r.o2 - r.o1), pos - r.o1
                if q_y < hi and not assigned[q_b] and not assigned[q_y]:
                    tok[pos], tok[q_b] = r.a, r.b
                    tok[q_y] = r.y if rng.random() >= noise else int((r.y + 1 + rng.integers(0, vocab - 1)) % vocab)
                    assigned[pos] = assigned[q_b] = assigned[q_y] = True
            if not assigned[pos]:
                tok[pos] = rng.choice(vocab, p=p)
                assigned[pos] = True
    doc_start = np.arange(0, n_docs * doc_len + 1, doc_len, dtype=np.uint32)
    return corpus_from_tokens(tok, doc_start, vocab, n_stop=n_stop, kind_rng=rng), rules


def noisy_xor(n: int, n_features: int = 12, noise: float = 0.0, seed: int = 1) -> TabularData:
    """NoisyXOR: y = x0 XOR x1, the other features are irrelevant; labels flipped w.p. noise."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, n_features)).astype(np.uint8)
    y = (x[:, 0] ^ x[:, 1]).astype(np.uint16)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return TabularData(x=x, y=y)


def two_xor_multiclass(n: int, n_features: int = 12, noise: float = 0.0, seed: int = 1) -> TabularData:
    """Four classes: y = 2 * (x0 XOR x1) + (x2 XOR x3); the other features are irrelevant."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, n_features)).astype(np.uint8)
    y = (2 * (x[:, 0] ^ x[:, 1]) + (x[:, 2] ^ x[:, 3])).astype(np.uint16)
    flip = rng.random(n) < noise
    y[flip] = (y[flip] + rng.integers(1, 4, size=int(flip.sum()))) % 4
    return TabularData(x=x, y=y)


def dense_config(n_features: int, n_classes: int, n_clauses: int, T: int, s: float, seed: int = 1) -> EngineConfig:
    """Dense-equivalent configuration (docs/11 section 6) for parity tests against tmu."""
    return EngineConfig(mode="tabular", vocab=n_classes, n_features=n_features, n_clauses=n_clauses,
                        seed=seed, T=T, s=s, l_slots=max(32, 2 * n_features), v_slots=max(16, n_classes),
                        batch=1, neg_mode="uniform_binomial", decay_mode="eager_exact", dense_equivalent=True)
