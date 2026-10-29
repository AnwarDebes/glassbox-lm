"""Every position of a set of replies, with only what is known before the token at that position.

For position t of a reply (t = 0 is the first token; the last position predicts the end token):
- x: the true token (the label only; nothing else here reads it);
- H[:, m]: the token at t - 1 - m (m = 0 is the previous token), BOS at t - 1 - m = -1, PAD before that;
- S, A: the act-slot set and domain-act set hashes of the turn (from dialog_act);
- the due placeholders: the expected multiset minus the placeholders emitted at positions < t (R = its hash,
  due_n = its size, due types = its distinct members);
- first-occurrence lookups answer "did token w occur at a position < t of this reply?".
"""

from __future__ import annotations

import numpy as np

from .data import Corpus, hash64

NH = 5  # history slots kept (enough for KN-6)


class Positions:
    def __init__(self, corpus: Corpus, replies: np.ndarray):
        c = corpus
        self.c = c
        self.replies = np.asarray(replies, np.int64)
        lens = np.array([c.a["tok_off"][r + 1] - c.a["tok_off"][r] for r in self.replies], np.int64)
        self.reply_start = np.zeros(len(self.replies) + 1, np.int64)
        self.reply_start[1:] = np.cumsum(lens)
        n = int(self.reply_start[-1])
        self.n = n
        self.local = np.repeat(np.arange(len(self.replies)), lens)  # index into self.replies
        self.reply = self.replies[self.local]
        self.t = np.arange(n) - self.reply_start[self.local]
        self.stream = np.concatenate([c.tokens(r) for r in self.replies]).astype(np.int64) if n else np.zeros(0, np.int64)
        self.x = self.stream
        self.dlg = c.a["r_dlg"][self.reply].astype(np.int64)
        self.S = c.a["S_hash"][self.reply]
        self.A = c.a["A_hash"][self.reply]
        H = np.full((n, NH), c.PAD, np.int64)
        for m in range(NH):
            src = np.arange(n) - 1 - m
            ok = self.t - 1 - m >= 0
            H[ok, m] = self.stream[src[ok]]
            H[self.t - 1 - m == -1, m] = c.BOS
        self.H = H
        self._due()
        self._first()

    def _due(self) -> None:
        c = self.c
        R = np.empty(self.n, np.int64)
        due_n = np.zeros(self.n, np.int64)
        due_lists = []
        cache = {}
        isph = c.is_ph
        for i, r in enumerate(self.replies):
            s0, s1 = self.reply_start[i], self.reply_start[i + 1]
            due = {}
            for p in c.expected(r):
                due[int(p)] = due.get(int(p), 0) + 1
            for k in range(s0, s1):
                key = tuple(sorted((p, m) for p, m in due.items() if m > 0))
                if key not in cache:
                    cache[key] = hash64("R:" + ",".join(f"{p}x{m}" for p, m in key))
                R[k] = cache[key]
                due_n[k] = sum(m for _, m in key)
                due_lists.append([p for p, _ in key])
                x = int(self.stream[k])
                if isph[x] and due.get(x, 0) > 0:
                    due[x] -= 1
        self.R, self.due_n = R, due_n
        off = np.zeros(self.n + 1, np.int64)
        off[1:] = np.cumsum([len(x) for x in due_lists])
        self.due_off = off
        self.due_ids = np.fromiter((p for x in due_lists for p in x), dtype=np.int64, count=int(off[-1]))
        if len(np.unique(np.array(list(cache.values()), np.int64))) != len(cache):
            raise SystemExit("due-multiset hash collision")

    def _first(self) -> None:
        """Sorted (local reply, token) keys of first occurrences, with their position in the reply."""
        W = self.c.V + 2
        key = self.local * W + self.stream
        order = np.lexsort((self.t, key))
        k_sorted = key[order]
        first = np.ones(len(k_sorted), bool)
        first[1:] = k_sorted[1:] != k_sorted[:-1]
        self._fk = k_sorted[first]
        self._ft = self.t[order][first]
        self._W = W

    def said_before(self, pos: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """True where token tok occurred in the same reply at a position < t(pos)."""
        pos = np.asarray(pos, np.int64)
        q = self.local[pos] * self._W + np.asarray(tok, np.int64)
        i = np.searchsorted(self._fk, q)
        i = np.minimum(i, max(len(self._fk) - 1, 0))
        found = (len(self._fk) > 0) & (self._fk[i] == q) if len(self._fk) else np.zeros(len(q), bool)
        return found & (self._ft[i] < self.t[pos])

    def due_types(self, k: int) -> np.ndarray:
        return self.due_ids[self.due_off[k]:self.due_off[k + 1]]
