"""Reports on category codes that carry no verdict in A2.

- similarity: Spearman correlation between human ratings (SimLex-999, WS-353 combined) and the
  code similarity of the two words = the number of levels at which they share a class (0 to the
  number of levels; 0/1 for a one-level code). Pairs with a word outside the vocabulary or equal
  to <unk> are left out; the number of pairs used is reported.
- sense test (synonym pairs): pairs of distinct vocabulary words, both with at least min_count
  train tokens, that share a WordNet 3.0 synset (any part of speech, single-word lemmas). Per
  level: the share of these pairs in the same class, the same share over all pairs of the words
  that occur in a synonym pair (the base rate), and their ratio (lift).
- n-gram growth: distinct n-grams (n = 1..5, inside sentences) in growing prefixes of the train
  split, for the word stream and for a class stream; and the share of eval n-gram tokens whose
  n-gram is not in train.
- unseen_bigram_mask: eval positions whose bigram (previous word or sentence start, word) does
  not occur in train (sentence mode).
"""
import csv
import io
import os
import zipfile

import numpy as np

from .codes import EXTERNAL, _nltk_paths


def load_simlex(path=os.path.join(EXTERNAL, "simlex999", "SimLex-999.zip")):
    with zipfile.ZipFile(path) as z:
        txt = z.read("SimLex-999/SimLex-999.txt").decode("utf8")
    rows = list(csv.DictReader(io.StringIO(txt), delimiter="\t"))
    return [(r["word1"].lower(), r["word2"].lower(), float(r["SimLex999"])) for r in rows]


def load_ws353(path=os.path.join(EXTERNAL, "wordsim353", "wordsim353.zip")):
    with zipfile.ZipFile(path) as z:
        txt = z.read("combined.csv").decode("utf8")
    rows = list(csv.reader(io.StringIO(txt)))[1:]
    return [(a.lower(), b.lower(), float(s)) for a, b, s in rows]


def similarity(C, w2i, pairs):
    from scipy.stats import spearmanr
    unk = w2i.get("<unk>")
    h, s = [], []
    for a, b, score in pairs:
        if a in w2i and b in w2i and w2i[a] != unk and w2i[b] != unk:
            h.append(score)
            s.append(int((C[w2i[a]] == C[w2i[b]]).sum()))
    if len(set(s)) < 2:
        return {"pairs": len(h), "spearman": 0.0, "note": "code similarity constant"}
    r = spearmanr(h, s)
    return {"pairs": len(h), "spearman": float(r.statistic), "p": float(r.pvalue)}


def synonym_pairs(vocab, counts, min_count=10):
    _nltk_paths()
    from nltk.corpus import wordnet as wn
    w2i = {w: i for i, w in enumerate(vocab)}
    ok = {w for w in vocab if counts[w2i[w]] >= min_count and w.isalpha()}
    pairs = set()
    for w in sorted(ok):
        for ss in wn.synsets(w):
            for l in ss.lemma_names():
                l = l.lower()
                if l != w and l in ok:
                    pairs.add((min(w, l), max(w, l)))
    return sorted((w2i[a], w2i[b]) for a, b in pairs)


def sense_test(C, pairs):
    if not pairs:
        return []
    P = np.array(pairs)
    words = np.unique(P)
    out = []
    for l in range(C.shape[1]):
        same = float((C[P[:, 0], l] == C[P[:, 1], l]).mean())
        _, n = np.unique(C[words, l], return_counts=True)
        base = float((n * (n - 1)).sum() / (len(words) * (len(words) - 1)))
        out.append({"level": l, "pairs": len(P), "words": int(len(words)), "same_class": same,
                    "base_rate": base, "lift": same / base if base > 0 else None})
    return out


def _ngram_keys(x, eos, n, B):
    """uint64 hash keys of the n-grams that lie inside one sentence."""
    x = np.asarray(x, dtype=np.uint64)
    m = len(x) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    key = np.zeros(m, dtype=np.uint64)
    ok = np.ones(m, dtype=bool)
    for k in range(n):
        key = key * np.uint64(B) + x[k:k + m] + np.uint64(1)
        if k < n - 1:
            ok &= x[k:k + m] != np.uint64(eos)
    return key[ok]


def ngram_growth(train, ev, eos, B, fractions=(0.125, 0.25, 0.5, 1.0), orders=(1, 2, 3, 4, 5)):
    out = {"distinct": {}, "eval_unseen_share": {}}
    for n in orders:
        out["distinct"][n] = [int(len(np.unique(_ngram_keys(train[:int(f * len(train))], eos, n, B))))
                              for f in fractions]
        tk = np.unique(_ngram_keys(train, eos, n, B))
        ek = _ngram_keys(ev, eos, n, B)
        out["eval_unseen_share"][n] = float((~np.isin(ek, tk)).mean())
    out["fractions"] = list(fractions)
    return out


def unseen_bigram_mask(train, ev, eos, V):
    def keys(x):
        x = np.asarray(x, dtype=np.int64)
        prev = np.empty_like(x)
        prev[0] = V
        prev[1:] = x[:-1]
        prev[1:][x[:-1] == eos] = V
        return prev * (V + 1) + x
    return ~np.isin(keys(ev), np.unique(keys(train)))
