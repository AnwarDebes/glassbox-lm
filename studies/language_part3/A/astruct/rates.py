"""Counted rates of a candidate word under the turn's conditions (count-derived bits, review 2).

rate under a condition = the share of the counting replies with that condition whose reply contains the word.
- rate_da: the maximum over the turn's domain-acts d of rate_d (a domain-act never counted is skipped; none
  counted: the global rate);
- rate_set: under the act-slot set S; S never counted: under the domain-act set A; A never counted: the global
  rate (the share of all counting replies that contain the word).
Training rows get rates counted on the other four fifths of fit (cross-fitting by dialogue fold); tune and final
rows get rates counted on all of fit.
"""

from __future__ import annotations

import numpy as np

from .data import Corpus
from .store import expand_csr


class Rates:
    def __init__(self, c: Corpus, count_replies: np.ndarray):
        self.c = c
        R = np.asarray(count_replies, np.int64)
        self.W = np.int64(c.V + 2)
        lens = c.a["tok_off"][R + 1] - c.a["tok_off"][R]
        rep = np.repeat(np.arange(len(R)), lens)
        toks = np.concatenate([c.tokens(r) for r in R]).astype(np.int64) if len(R) else np.zeros(0, np.int64)
        rw = np.unique(rep * self.W + toks)  # distinct (reply, word)
        r_of, w_of = rw // self.W, rw % self.W
        self.n_all = len(R)
        self.n_w = np.bincount(w_of, minlength=c.V).astype(np.float64)
        # domain-acts: dense (n_acts x V)
        ao = c.a["act_off"]
        n_acts = max(c.n_acts, int(c.a["act_ids"].max()) + 1 if len(c.a["act_ids"]) else 1)
        self.n_d = np.zeros(n_acts)
        self.n_dw = np.zeros((n_acts, c.V))
        acts_of = [c.a["act_ids"][ao[r]:ao[r + 1]].astype(np.int64) for r in R]
        for a in acts_of:
            self.n_d[a] += 1
        # (act, word) pairs: for each distinct (reply, word), every act of the reply
        start = np.searchsorted(r_of, np.arange(len(R) + 1))
        for i in range(len(R)):
            ws = w_of[start[i]:start[i + 1]]
            for a in acts_of[i]:
                self.n_dw[a, ws] += 1
        self.S = self._cond(c.a["S_hash"][R], r_of, w_of)
        self.A = self._cond(c.a["A_hash"][R], r_of, w_of)

    def _cond(self, hashes: np.ndarray, r_of: np.ndarray, w_of: np.ndarray) -> dict:
        u, inv, n = np.unique(hashes, return_inverse=True, return_counts=True)
        keys, cnt = np.unique(inv.reshape(-1)[r_of] * self.W + w_of, return_counts=True)
        return {"u": u, "n": n.astype(np.float64), "keys": keys, "cnt": cnt.astype(np.float64)}

    @staticmethod
    def _find(tab: dict, h: np.ndarray) -> np.ndarray:
        u = tab["u"]
        if len(u) == 0:
            return np.full(len(h), -1, np.int64)
        i = np.minimum(np.searchsorted(u, h), len(u) - 1)
        return np.where(u[i] == h, i, -1)

    def _rate(self, tab: dict, h: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rate, seen) for condition hashes h and words w (rate 0 where the condition was never counted)."""
        ci = self._find(tab, h)
        seen = ci >= 0
        key = np.where(seen, ci, 0) * self.W + w
        j = np.minimum(np.searchsorted(tab["keys"], key), max(len(tab["keys"]) - 1, 0))
        hit = seen & (len(tab["keys"]) > 0) & (tab["keys"][j] == key) if len(tab["keys"]) else np.zeros(len(h), bool)
        num = np.where(hit, tab["cnt"][j] if len(tab["keys"]) else 0.0, 0.0)
        den = np.where(seen, tab["n"][np.maximum(ci, 0)], 1.0)
        return np.where(seen, num / den, 0.0), seen

    def global_rate(self, w: np.ndarray) -> np.ndarray:
        return self.n_w[w] / max(self.n_all, 1)

    def rate_set(self, S: np.ndarray, A: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, np.int64)
        rs, s_seen = self._rate(self.S, np.asarray(S, np.int64), w)
        ra, a_seen = self._rate(self.A, np.asarray(A, np.int64), w)
        return np.where(s_seen, rs, np.where(a_seen, ra, self.global_rate(w)))

    def backoff_of_set(self, A: np.ndarray, w: np.ndarray) -> np.ndarray:
        """The value rate_set falls back to when S was never counted."""
        w = np.asarray(w, np.int64)
        ra, a_seen = self._rate(self.A, np.asarray(A, np.int64), w)
        return np.where(a_seen, ra, self.global_rate(w))

    def s_count(self, S: np.ndarray) -> np.ndarray:
        ci = self._find(self.S, np.asarray(S, np.int64))
        return np.where(ci >= 0, self.S["n"][np.maximum(ci, 0)], 0.0)

    def rate_da(self, replies: np.ndarray, w: np.ndarray, corpus: Corpus | None = None) -> np.ndarray:
        """Per row (reply index, word): max over the reply's domain-acts that were counted; else the global rate.
        corpus: where the reply indices point (default: the counting corpus)."""
        c = corpus if corpus is not None else self.c
        replies = np.asarray(replies, np.int64)
        w = np.asarray(w, np.int64)
        if len(replies) == 0:
            return np.zeros(0)
        ao = c.a["act_off"]
        n_a = ao[replies + 1] - ao[replies]
        if (n_a == 0).any():
            raise ValueError("a reply without domain-acts")
        idx = expand_csr(ao[replies], n_a)
        row = np.repeat(np.arange(len(replies)), n_a)
        d = c.a["act_ids"][idx].astype(np.int64)
        seen = self.n_d[d] > 0
        val = np.where(seen, self.n_dw[d, w[row]] / np.maximum(self.n_d[d], 1), -1.0)
        starts = np.concatenate([[0], np.cumsum(n_a)[:-1]])
        out = np.maximum.reduceat(val, starts)
        return np.where(out < 0, self.global_rate(w), out)
