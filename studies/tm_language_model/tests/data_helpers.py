"""Tiny synthetic WikiText-style rows for the data tests (never the real corpus)."""

from __future__ import annotations

import numpy as np

WORDS = ["alpha", "beta", "gamma", "delta", "the", "of", "and", "Media.Vision", "don't", "@-@", "Senjō",
         "戦場", "x", "1998", "(", ")", ",", ".", "role", "playing", "video", "game", "an", "to", "was"]


def synthetic_rows(seed: int, n_articles: int = 6, n_paras: int = 5, words_per_para: int = 60) -> list[str]:
    """Rows in the WikiText format: '', ' = Title = \\n', '', paragraphs, section headings."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, len(WORDS) + 1) ** 0.8
    p /= p.sum()
    rows = [""]
    for a in range(n_articles):
        rows += [f" = Article {seed}-{a} = \n", ""]
        for k in range(n_paras):
            if k == 2:
                rows += [" = = Section = = \n", ""]
            words = rng.choice(WORDS, size=words_per_para, p=p)
            rows += [" " + " ".join(words) + " \n", ""]
    return rows
