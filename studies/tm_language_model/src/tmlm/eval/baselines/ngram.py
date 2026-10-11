"""n-gram baselines on identical tokens (docs/06 "Baselines", docs/05 protocol CP).

One `KNModel` of order n holds every interpolated Kneser-Ney model of order 1..n over the same
training stream (an order-k query uses the raw k-gram counts at its top level and the continuation
counts below it, exactly what a separate order-k fit would hold). Two discount modes:

- "modified": interpolated modified Kneser-Ney (Chen and Goodman 1999) with three discounts per
  level, D1, D2, D3+, estimated from the counts-of-counts of that level (raw counts at the top
  level, continuation counts below), Y = n1 / (n1 + 2 n2), D1 = 1 - 2Y n2/n1,
  D2 = 2 - 3Y n3/n2, D3+ = 3 - 4Y n4/n3; a zero denominator makes the ratio 0 (so the discount
  is 1, 2 or 3) and every Dk is clamped to [0, k] so no probability is negative. The unigram
  level is discounted the same way and interpolated with the uniform distribution over the V
  real tokens (`unk_interp=True`), so an unseen token has a positive probability and every
  distribution sums to one. `unk_interp=False` makes the unigram level the plain continuation
  distribution without discount (NLTK's `unigram_score`); it still sums to one over seen tokens.
- "single": one discount D for every count, as NLTK's `nltk.lm.KneserNeyInterpolated`
  (docs/12 F12: single-discount, default 0.1). With `unk_interp=False` the unigram level is the
  plain continuation distribution, which is NLTK's `unigram_score`; tests/test_kn_vs_nltk.py
  checks this mode against NLTK to 1e-9 on a cyclic toy corpus, where NLTK's raw-count
  gamma equals the continuation-count gamma.

Counting is in-document (no n-gram crosses a document boundary): every document is padded with
n-1 BOS tokens (id V, never a target) so the first token is predicted from a BOS context, and
only the N real positions carry counts. `pad=False` (tests) drops the padding and counts only
the positions with a full in-document history.

Keys: a k-gram is keyed by the index of its (k-1)-gram prefix in
the sorted unique table of level k-1, shifted left by 16 bits, OR its last token (tokens and BOS
are < 65536 because V <= 32768). Prefix indices are below 2^48, so the uint64 key is exact for
every order; unique tables and per-position ids come from `numpy.unique(..., return_inverse=True)`
(a sort), and every lookup is `numpy.searchsorted` chained level by level. Nothing is hashed.

Also here: the unigram, left-bigram and right-bigram argmax predictors on raw counts (the
"left bigram" of the M3 gates, moved from experiments/engine_common.py), the bits-per-byte helper
of protocol CP, and the bidirectional fill-in scorer: a documented approximation that scores a
candidate t at a hidden position by log P_fwd(t | left context) + log P_bwd(t | right context),
the product of the forward model and a second model fitted on the reversed documents (a product
of experts; unnormalised, argmax only). The exact joint P(left, t, right) under one model is not
computed: it needs V chained lookups per position and order, which is too slow at V = 8000 on
the frozen sets (docs/06 reports the approximation as "bidirectional n-gram").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK16 = np.uint64(0xFFFF)
MISSING = -1


# ------------------------------------------------------------------ raw-count predictors
def bigram_table(train_tok: np.ndarray, vocab: int) -> np.ndarray:
    """[vocab, vocab] int64 counts of (left, right) token pairs; document boundaries ignored."""
    tr = np.asarray(train_tok).astype(np.int64)
    big = np.zeros((vocab, vocab), dtype=np.int64)
    np.add.at(big, (tr[:-1], tr[1:]), 1)
    return big


def unigram_top1(train_tok: np.ndarray, vocab: int, eval_pos: np.ndarray, eval_tok: np.ndarray) -> float:
    """The most frequent training token everywhere."""
    counts = np.bincount(np.asarray(train_tok).astype(np.int64), minlength=vocab)
    pred = int(counts.argmax())
    return float((np.asarray(eval_tok)[np.asarray(eval_pos)] == pred).mean())


def left_bigram_top1(train_tok: np.ndarray, vocab: int, eval_pos: np.ndarray, eval_tok: np.ndarray,
                     table: np.ndarray | None = None) -> float:
    """Left bigram argmax on the evaluation positions (counts from `train_tok`, ties toward the
    lower id). Uses the TRUE left token, hidden or not; a position without a left token is wrong."""
    big = bigram_table(train_tok, vocab) if table is None else table
    eval_pos = np.asarray(eval_pos).astype(np.int64)
    eval_tok = np.asarray(eval_tok).astype(np.int64)
    ok = eval_pos >= 1
    pred = np.full(eval_pos.shape[0], -1, dtype=np.int64)
    pred[ok] = big[eval_tok[eval_pos[ok] - 1]].argmax(axis=1)
    return float((pred == eval_tok[eval_pos]).mean())


def right_bigram_top1(train_tok: np.ndarray, vocab: int, eval_pos: np.ndarray, eval_tok: np.ndarray,
                      table: np.ndarray | None = None) -> float:
    """Right bigram argmax: argmax_t count(t, right token). Uses the TRUE right token."""
    big = bigram_table(train_tok, vocab) if table is None else table
    eval_pos = np.asarray(eval_pos).astype(np.int64)
    eval_tok = np.asarray(eval_tok).astype(np.int64)
    n = eval_tok.shape[0]
    ok = eval_pos + 1 < n
    pred = np.full(eval_pos.shape[0], -1, dtype=np.int64)
    pred[ok] = big[:, eval_tok[eval_pos[ok] + 1]].argmax(axis=0)
    return float((pred == eval_tok[eval_pos]).mean())


# ------------------------------------------------------------------ discounts
def modified_discounts(counts: np.ndarray) -> tuple[float, float, float]:
    """(D1, D2, D3+) of Chen and Goodman 1999 from the counts of one level (zeros ignored), with
    the zero-denominator rule and the clamping described in the module docstring."""
    c = counts[counts > 0]
    n1 = float((c == 1).sum())
    n2 = float((c == 2).sum())
    n3 = float((c == 3).sum())
    n4 = float((c == 4).sum())
    y = n1 / (n1 + 2.0 * n2) if (n1 + 2.0 * n2) > 0 else 0.0

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    d1 = 1.0 - 2.0 * y * ratio(n2, n1)
    d2 = 2.0 - 3.0 * y * ratio(n3, n2)
    d3 = 3.0 - 4.0 * y * ratio(n4, n3)
    return (min(max(d1, 0.0), 1.0), min(max(d2, 0.0), 2.0), min(max(d3, 0.0), 3.0))


@dataclass
class _Level:
    """Statistics of one level of one count variant (raw or continuation)."""

    cnt: np.ndarray          # int32 [n_types] counts used at this level (0 for garbage types)
    total: np.ndarray        # float64 [n_ctx] sum of cnt over the followers of every context
    n1: np.ndarray           # int32 [n_ctx] followers with cnt == 1
    n2: np.ndarray           # followers with cnt == 2
    n3: np.ndarray           # followers with cnt >= 3
    n_any: np.ndarray        # followers with cnt >= 1 (single-discount gamma)
    d: tuple[float, float, float]

    def discount_of(self, cnt: np.ndarray, single: float | None) -> np.ndarray:
        if single is not None:
            return np.where(cnt > 0, single, 0.0)
        d1, d2, d3 = self.d
        return np.where(cnt >= 3, d3, np.where(cnt == 2, d2, np.where(cnt == 1, d1, 0.0)))

    def gamma_mass(self, ctx: np.ndarray, single: float | None) -> np.ndarray:
        """D1 N1 + D2 N2 + D3 N3+ (or D N1+) for the contexts `ctx` (index MISSING gives 0)."""
        c = np.where(ctx >= 0, ctx, 0)
        if single is not None:
            return np.where(ctx >= 0, single * self.n_any[c].astype(np.float64), 0.0)
        d1, d2, d3 = self.d
        return np.where(ctx >= 0, d1 * self.n1[c].astype(np.float64) + d2 * self.n2[c].astype(np.float64)
                        + d3 * self.n3[c].astype(np.float64), 0.0)


def _lookup(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Index of every query key in the sorted unique `keys`, MISSING when absent."""
    i = np.searchsorted(keys, q)
    j = np.minimum(i, keys.shape[0] - 1)
    return np.where((i < keys.shape[0]) & (keys[j] == q), i, MISSING).astype(np.int64)


class KNModel:
    """Interpolated Kneser-Ney models of orders 1..`order` over one training stream."""

    def __init__(self, order: int, vocab: int, discount_mode: str = "modified", discount: float = 0.75,
                 unk_interp: bool = True, pad: bool = True) -> None:
        if order < 1:
            raise ValueError("order >= 1")
        if vocab > 32768:
            raise ValueError("token ids must fit 15 bits (V <= 32768)")
        if discount_mode not in ("modified", "single"):
            raise ValueError(discount_mode)
        self.order = order
        self.vocab = vocab
        self.bos = vocab
        self.discount_mode = discount_mode
        self.discount = float(discount)
        self.unk_interp = unk_interp
        self.pad = pad
        self.keys: list[np.ndarray | None] = [None] * (order + 1)
        self.raw: list[_Level | None] = [None] * (order + 1)
        self.cont: list[_Level | None] = [None] * (order + 1)
        self.n_train_tokens = 0
        self.n_types: list[int] = [0] * (order + 1)

    # -------------------------------------------------------------- fitting
    def _padded_stream(self, tok: np.ndarray, doc_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(stream, real): the token stream with n-1 global BOS in front and, when `pad`, n-1 BOS
        before every document; `real` marks the positions that carry counts."""
        n = self.order
        tok = np.asarray(tok).astype(np.int64)
        doc_start = np.asarray(doc_start).astype(np.int64)
        n_tok = tok.shape[0]
        n_doc = doc_start.shape[0] - 1
        lengths = np.diff(doc_start)
        doc_of = np.repeat(np.arange(n_doc, dtype=np.int64), lengths)
        if self.pad:
            length = (n - 1) + n_tok + (n - 1) * n_doc
            pos = np.arange(n_tok, dtype=np.int64) + (n - 1) * (doc_of + 2)
            real_mask = np.ones(n_tok, dtype=bool)
        else:
            length = (n - 1) + n_tok
            pos = np.arange(n_tok, dtype=np.int64) + (n - 1)
            real_mask = (np.arange(n_tok, dtype=np.int64) - doc_start[doc_of]) >= (n - 1)
        stream = np.full(length, self.bos, dtype=np.int64)
        stream[pos] = tok
        real = np.zeros(length, dtype=bool)
        real[pos[real_mask]] = True
        return stream, real

    def fit(self, tok: np.ndarray, doc_start: np.ndarray) -> "KNModel":
        n = self.order
        stream, real = self._padded_stream(tok, doc_start)
        self.n_train_tokens = int(real.sum())
        # per-position ids of the k-gram ending at every position, for k = 1..n
        ids: list[np.ndarray | None] = [None] * (n + 1)
        first: list[np.ndarray | None] = [None] * (n + 1)   # first position of every type
        raw_cnt: list[np.ndarray | None] = [None] * (n + 1)
        self.keys[1] = np.arange(self.vocab + 1, dtype=np.uint64)
        ids[1] = stream
        raw_cnt[1] = np.bincount(stream[real], minlength=self.vocab + 1).astype(np.int64)
        for k in range(2, n + 1):
            key = np.empty(stream.shape[0], dtype=np.uint64)
            key[0] = np.uint64(stream[0])
            key[1:] = (ids[k - 1][:-1].astype(np.uint64) << np.uint64(16)) | stream[1:].astype(np.uint64)
            keys_k, first_k, inv = np.unique(key, return_index=True, return_inverse=True)
            del key
            self.keys[k] = keys_k
            first[k] = first_k.astype(np.int64)
            ids[k] = inv.reshape(-1).astype(np.int64)
            raw_cnt[k] = np.bincount(ids[k][real], minlength=keys_k.shape[0]).astype(np.int64)
        # continuation counts: cont_k[g] = number of level-(k+1) types with count > 0 whose suffix is g
        cont_cnt: list[np.ndarray | None] = [None] * (n + 1)
        for k in range(n - 1, 0, -1):
            upper = raw_cnt[k + 1] if k + 1 == n else cont_cnt[k + 1]
            suffix = ids[k][first[k + 1]]
            live = upper > 0
            cont_cnt[k] = np.bincount(suffix[live], minlength=self.keys[k].shape[0]).astype(np.int64)
        del ids
        for k in range(1, n + 1):
            self.n_types[k] = int(self.keys[k].shape[0])
            self.raw[k] = self._level_stats(k, raw_cnt[k])
            self.cont[k] = self._level_stats(k, cont_cnt[k]) if k < n else None
        return self

    def _level_stats(self, k: int, cnt: np.ndarray) -> _Level:
        if k == 1:
            n_ctx = 1
            ctx = np.zeros(cnt.shape[0], dtype=np.int64)
        else:
            n_ctx = self.keys[k - 1].shape[0]
            ctx = (self.keys[k] >> np.uint64(16)).astype(np.int64)
        total = np.bincount(ctx, weights=cnt.astype(np.float64), minlength=n_ctx)
        n1 = np.bincount(ctx[cnt == 1], minlength=n_ctx).astype(np.int32)
        n2 = np.bincount(ctx[cnt == 2], minlength=n_ctx).astype(np.int32)
        n3 = np.bincount(ctx[cnt >= 3], minlength=n_ctx).astype(np.int32)
        n_any = (n1 + n2 + n3).astype(np.int32)
        d = modified_discounts(cnt) if self.discount_mode == "modified" else (self.discount,) * 3
        # int32 counts: the largest count is the unigram frequency of the most frequent token
        if int(cnt.max()) >= 2**31 - 1:
            raise OverflowError("counts do not fit int32")
        return _Level(cnt=cnt.astype(np.int32), total=total, n1=n1, n2=n2, n3=n3, n_any=n_any, d=d)

    def set_discount_mode(self, discount_mode: str, discount: float | None = None) -> "KNModel":
        """Switch between "modified" and "single" (and the single discount) on a fitted model:
        the counts stay, only the per-level discounts are recomputed."""
        if discount_mode not in ("modified", "single"):
            raise ValueError(discount_mode)
        self.discount_mode = discount_mode
        if discount is not None:
            self.discount = float(discount)
        for k in range(1, self.order + 1):
            for levels in (self.raw, self.cont):
                lv = levels[k]
                if lv is not None:
                    lv.d = modified_discounts(lv.cnt) if discount_mode == "modified" else (self.discount,) * 3
        return self

    # -------------------------------------------------------------- level access
    def level(self, k: int, order: int) -> _Level:
        """The statistics used at level k by the order-`order` model."""
        return self.raw[k] if k == order else self.cont[k]

    def _single(self) -> float | None:
        return self.discount if self.discount_mode == "single" else None

    def discounts(self, order: int | None = None) -> dict[int, tuple[float, float, float]]:
        order = order or self.order
        return {k: self.level(k, order).d for k in range(1, order + 1)}

    # -------------------------------------------------------------- context ids
    def context_ids(self, ctx: np.ndarray, valid: np.ndarray | None = None, order: int | None = None) -> list[np.ndarray | None]:
        """ctx: int64 [Q, m] context tokens, the LAST column nearest to the target; BOS where the
        document starts earlier. valid[q] (default m) is the number of trailing context tokens that
        may be used (a hidden token cuts the context). Returns per level k = 2..order the index of
        the (k-1)-gram context in keys[k-1], MISSING when absent or cut."""
        order = order or self.order
        ctx = np.asarray(ctx).astype(np.int64)
        q, m = ctx.shape
        if m < order - 1:
            raise ValueError("context shorter than order - 1")
        if valid is None:
            valid = np.full(q, m, dtype=np.int64)
        out: list[np.ndarray | None] = [None] * (order + 1)
        for k in range(2, order + 1):
            length = k - 1
            idx = ctx[:, m - length].copy()
            for j in range(m - length + 1, m):
                key = (idx.astype(np.uint64) << np.uint64(16)) | ctx[:, j].astype(np.uint64)
                depth = j - (m - length) + 1
                found = _lookup(self.keys[depth], key)
                idx = np.where(idx == MISSING, MISSING, found)
            idx = np.where(valid >= length, idx, MISSING)
            out[k] = idx
        return out

    # -------------------------------------------------------------- probabilities
    def _unigram_full(self, order: int) -> np.ndarray:
        """float64 [V + 1]: P_1(w) over real tokens (the BOS column is 0)."""
        lv = self.level(1, order)
        single = self._single()
        cnt = lv.cnt.astype(np.float64)
        total = float(lv.total[0])
        if not self.unk_interp:
            # NLTK's unigram_score: the plain continuation distribution, no discount
            p = cnt / total if total > 0 else np.zeros_like(cnt)
        else:
            gamma = float(lv.gamma_mass(np.zeros(1, dtype=np.int64), single)[0]) / total if total > 0 else 1.0
            alpha = np.maximum(cnt - lv.discount_of(lv.cnt, single), 0.0) / total if total > 0 else np.zeros_like(cnt)
            p = alpha + gamma / self.vocab
        p[self.bos] = 0.0
        return p

    def prob(self, ctx: np.ndarray, w: np.ndarray, valid: np.ndarray | None = None, order: int | None = None) -> np.ndarray:
        """P(w[q] | ctx[q]) for every query q, float64 [Q]."""
        order = order or self.order
        w = np.asarray(w).astype(np.int64)
        single = self._single()
        cids = self.context_ids(ctx, valid, order)
        p = self._unigram_full(order)[w]
        for k in range(2, order + 1):
            lv = self.level(k, order)
            cid = cids[k]
            total = np.where(cid >= 0, lv.total[np.where(cid >= 0, cid, 0)], 0.0)
            seen = total > 0
            key = (np.where(cid >= 0, cid, 0).astype(np.uint64) << np.uint64(16)) | w.astype(np.uint64)
            tid = _lookup(self.keys[k], key)
            cnt = np.where((tid >= 0) & seen, lv.cnt[np.where(tid >= 0, tid, 0)], 0).astype(np.float64)
            safe_total = np.where(seen, total, 1.0)
            alpha = np.where(seen, np.maximum(cnt - lv.discount_of(cnt, single), 0.0) / safe_total, 0.0)
            gamma = np.where(seen, lv.gamma_mass(cid, single) / safe_total, 1.0)
            p = alpha + gamma * p
        return p

    def dist(self, ctx: np.ndarray, valid: np.ndarray | None = None, order: int | None = None) -> np.ndarray:
        """Full next-token distributions, float64 [Q, V] (BOS excluded)."""
        order = order or self.order
        single = self._single()
        ctx = np.asarray(ctx).astype(np.int64)
        q = ctx.shape[0]
        cids = self.context_ids(ctx, valid, order)
        p = np.tile(self._unigram_full(order), (q, 1))
        for k in range(2, order + 1):
            lv = self.level(k, order)
            cid = cids[k]
            total = np.where(cid >= 0, lv.total[np.where(cid >= 0, cid, 0)], 0.0)
            seen = total > 0
            safe_total = np.where(seen, total, 1.0)
            gamma = np.where(seen, lv.gamma_mass(cid, single) / safe_total, 1.0)
            p *= gamma[:, None]
            rows = np.flatnonzero(seen)
            if rows.shape[0] == 0:
                continue
            c = cid[rows].astype(np.uint64)
            lo = np.searchsorted(self.keys[k], c << np.uint64(16))
            hi = np.searchsorted(self.keys[k], (c + np.uint64(1)) << np.uint64(16))
            lengths = hi - lo
            row_rep = np.repeat(rows, lengths)
            offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
            tid = np.repeat(lo, lengths) + offsets
            cnt = lv.cnt[tid].astype(np.float64)
            alpha = np.maximum(cnt - lv.discount_of(cnt, single), 0.0) / safe_total[row_rep]
            wcol = (self.keys[k][tid] & MASK16).astype(np.int64)
            p[row_rep, wcol] += alpha   # (row, w) pairs are unique within a level
        return p[:, : self.vocab]

    # -------------------------------------------------------------- streams
    def contexts(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray,
                 hidden: np.ndarray | None = None, order: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(ctx [Q, order-1], valid [Q]) of the LEFT contexts of the positions `pos` in an
        evaluation stream: BOS before the document start; with `hidden`, valid[q] counts the
        trailing context tokens up to (not including) the nearest hidden one."""
        order = order or self.order
        m = order - 1
        tok = np.asarray(tok).astype(np.int64)
        doc_start = np.asarray(doc_start).astype(np.int64)
        pos = np.asarray(pos).astype(np.int64)
        dstart = doc_start[np.searchsorted(doc_start, pos, side="right") - 1]
        ctx = np.full((pos.shape[0], m), self.bos, dtype=np.int64)
        valid = np.full(pos.shape[0], m, dtype=np.int64)
        for j in range(m):
            q = pos - m + j
            ok = q >= dstart
            ctx[ok, j] = tok[q[ok]]
            if hidden is not None:
                h = ok & (np.asarray(hidden)[np.clip(q, 0, tok.shape[0] - 1)] != 0)
                valid = np.where(h, m - j - 1, valid)
        return ctx, valid

    def logprob(self, tok: np.ndarray, doc_start: np.ndarray, order: int | None = None,
                chunk: int = 65536) -> np.ndarray:
        """Per-token log2 P(tok[i] | left in-document context) over the whole stream, float64 [N].
        Bits per byte = -sum / byte count of the same text (protocol CP)."""
        order = order or self.order
        tok = np.asarray(tok).astype(np.int64)
        out = np.zeros(tok.shape[0], dtype=np.float64)
        for i in range(0, tok.shape[0], chunk):
            pos = np.arange(i, min(i + chunk, tok.shape[0]))
            ctx, valid = self.contexts(tok, doc_start, pos, None, order)
            p = self.prob(ctx, tok[pos], valid, order)
            out[pos] = np.log2(np.maximum(p, np.finfo(np.float64).tiny))
        return out

    def predict(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray, hidden: np.ndarray | None = None,
                order: int | None = None, chunk: int = 2048) -> np.ndarray:
        """Argmax next token from the left context of every position of `pos` (int64 [Q])."""
        order = order or self.order
        pos = np.asarray(pos).astype(np.int64)
        out = np.zeros(pos.shape[0], dtype=np.int64)
        for i in range(0, pos.shape[0], chunk):
            p = pos[i:i + chunk]
            ctx, valid = self.contexts(tok, doc_start, p, hidden, order)
            out[i:i + chunk] = self.dist(ctx, valid, order).argmax(axis=1)
        return out


def bits_per_byte(logprob2: np.ndarray, n_bytes: int) -> float:
    """Total bits of a token stream (from per-token log2 probabilities) over its UTF-8 byte count."""
    return float(-np.sum(logprob2) / n_bytes)


def reverse_docs(tok: np.ndarray, doc_start: np.ndarray) -> np.ndarray:
    """The token stream with every document reversed in place (document order unchanged)."""
    tok = np.asarray(tok)
    out = np.empty_like(tok)
    ds = np.asarray(doc_start).astype(np.int64)
    for d in range(ds.shape[0] - 1):
        out[ds[d]:ds[d + 1]] = tok[ds[d]:ds[d + 1]][::-1]
    return out


def reversed_positions(pos: np.ndarray, doc_start: np.ndarray) -> np.ndarray:
    """Position of every `pos` in the stream returned by `reverse_docs`."""
    ds = np.asarray(doc_start).astype(np.int64)
    pos = np.asarray(pos).astype(np.int64)
    d = np.searchsorted(ds, pos, side="right") - 1
    return ds[d] + ds[d + 1] - 1 - pos


class BidirectionalKN:
    """Forward and backward KN models over the same training documents; the fill-in score of a
    candidate t at a hidden position p is log P_fwd(t | left) + log P_bwd(t | right) (see the
    module docstring: a product of experts, not the joint probability)."""

    def __init__(self, fwd: KNModel, bwd: KNModel) -> None:
        if fwd.order != bwd.order or fwd.vocab != bwd.vocab:
            raise ValueError("forward and backward models must match")
        self.fwd = fwd
        self.bwd = bwd

    @classmethod
    def fit(cls, tok: np.ndarray, doc_start: np.ndarray, order: int, vocab: int, **kw) -> "BidirectionalKN":
        fwd = KNModel(order, vocab, **kw).fit(tok, doc_start)
        bwd = KNModel(order, vocab, **kw).fit(reverse_docs(tok, doc_start), doc_start)
        return cls(fwd, bwd)

    def predict(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray, hidden: np.ndarray | None = None,
                order: int | None = None, chunk: int = 2048, side: str = "both") -> np.ndarray:
        """Argmax fill-in for the positions `pos`; side in {"both", "left", "right"}."""
        order = order or self.fwd.order
        pos = np.asarray(pos).astype(np.int64)
        rtok = reverse_docs(tok, doc_start)
        rhidden = None if hidden is None else reverse_docs(hidden, doc_start)
        rpos = reversed_positions(pos, doc_start)
        out = np.zeros(pos.shape[0], dtype=np.int64)
        tiny = np.finfo(np.float64).tiny
        for i in range(0, pos.shape[0], chunk):
            score = None
            if side in ("both", "left"):
                ctx, valid = self.fwd.contexts(tok, doc_start, pos[i:i + chunk], hidden, order)
                score = np.log(np.maximum(self.fwd.dist(ctx, valid, order), tiny))
            if side in ("both", "right"):
                ctx, valid = self.bwd.contexts(rtok, doc_start, rpos[i:i + chunk], rhidden, order)
                s2 = np.log(np.maximum(self.bwd.dist(ctx, valid, order), tiny))
                score = s2 if score is None else score + s2
            out[i:i + chunk] = score.argmax(axis=1)
        return out


def word_accuracy(pred: np.ndarray, gold: np.ndarray, word_of_target: np.ndarray) -> float:
    """Fraction of words (groups of targets sharing word_of_target) whose every piece is right."""
    ok = np.asarray(pred) == np.asarray(gold)
    words, inv = np.unique(np.asarray(word_of_target), return_inverse=True)
    all_ok = np.ones(words.shape[0], dtype=bool)
    np.logical_and.at(all_ok, inv, ok)
    return float(all_ok.mean()) if words.shape[0] else 0.0
