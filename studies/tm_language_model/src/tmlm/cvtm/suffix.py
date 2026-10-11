"""Family S of the CVTM: suffix clauses, the exact vote form of Kneser-Ney (docs/11 section 11.2).

Clauses. For every context u of length k (1 <= k <= K-1) that is a type of the fitted
`KNModel` (a key of level k) and has a positive total at level k+1 of the order-K model
(`kn.level(k+1, K)`: continuation statistics below the top order, raw counts at the top), clause
S(u) fires at a position iff the k tokens left of it (BOS-padded at the document start, as in
`KNModel.contexts`; `valid` cuts a context exactly as in `KNModel.context_ids`) equal u. A
context that is absent, cut, or has total 0 fires nothing: KNModel's gamma = 1 case.

Votes (natural log). With alpha(u, w) = max(c - D(c), 0) / total(u) and
gamma(u) = (D1 N1 + D2 N2 + D3 N3+)(u) / total(u) at level k+1, and p_k(w | u') the level-k
probability of w after the suffix u' of u (length k-1):

- bias clause (always fires): b(w) = ln p_1(w), p_1 the unigram of `KNModel._unigram_full`
  (with `unk_interp`, interpolated with the uniform distribution over the V real tokens);
- clause constant d(u) = ln gamma(u), added to every class when S(u) fires;
- follower vote v(u, w) = ln(1 + alpha(u, w) / (gamma(u) p_k(w | u'))) for every w with a
  positive count after u at level k+1 (computed with log1p), 0 for every other class.

Then ln P_KN(w | h) = b(w) + sum over fired S(u) of [d(u) + v(u, w)] (the interpolation
recursion telescopes). `logprob` evaluates exactly this sum from the stored votes; it never calls
`KNModel.prob`. The p_k(w | u') inside the votes come from a dynamic program over every type of
every level, p_j(t) = alpha_j(t) + gamma_j(ctx(t)) * p_{j-1}(suffix(t)), with the float64
operations of `KNModel.prob` in the same order, so the p of a top-level type is bit-identical to
`KNModel.prob` of its context (tests/test_cvtm_suffix.py checks this). Suffix links come from the
key encoding (prefix_id << 16) | token: the suffix of level-2 type (a, w) is the level-1 type w;
the suffix of a level-j type (c, w) is the level-(j-1) type with key (suffix(c) << 16) | w, found
by binary search (on the GPU with CuPy when available). The vote form needs gamma(u) > 0 for every
firing clause and a present suffix for every follower; `from_kn` raises ValueError otherwise (both
hold for `KNModel.fit` and for `tmlm.cvtm.kn_gpu.fit_kn`, pruned or not, whenever the discounts of
the counts that occur are positive).

Integer form (receipts): b, d and v as int32 in units of 2^-16 nats, rint(x * 65536) (round to
nearest, ties to even). The bias of BOS (p_1 = 0, never a target) is stored as INT32_MIN.
`logprob_q` sums the int32 votes in int64; its error against `logprob` is at most
0.5 * 2^-16 nats per rounded term (the bias, every fired clause constant, every non-zero vote).

Clause ids. `fired(ctx)` returns int64 [Q, K-1]: column k-1 holds the index of the length-k
context in `keys[k]` (the clause id within its level) when S(u) fires, else -1.

State hash (`state_hash`): sha256 over the bytes
b"tmlm.cvtm.suffix.v1\\0" + JSON {"order", "vocab", "n_types"} (sorted keys, no spaces) + b"\\0",
then for j = 2..K, in this order: the name and `tmlm.common.hashing.sha256_array` digest of
keys[j] (uint64), fires[j] (uint8, per context of level j-1), dq[j] (int32, per context),
vq[j] (int32, per type of level j), and finally bq (int32 [V+1]); each entry is written as
"<name>=<hex digest>\\n" in ASCII. Only integer arrays enter the hash (the float64 votes can differ
in the last bit between platforms; the integer votes are the receipt form).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Sequence

import numpy as np

from ..common import hashing
from ..eval.baselines.ngram import KNModel

Q16 = 65536.0
INT32_MIN = np.iinfo(np.int32).min
MISSING = -1
_SHIFT = np.uint64(16)
_MASK16 = np.uint64(0xFFFF)


def _lookup(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Index of every query key in the sorted unique `keys`, MISSING when absent."""
    if keys.shape[0] == 0:
        return np.full(q.shape[0], MISSING, dtype=np.int64)
    i = np.searchsorted(keys, q)
    j = np.minimum(i, keys.shape[0] - 1)
    return np.where((i < keys.shape[0]) & (keys[j] == q), i, MISSING).astype(np.int64)


def _q16(x: np.ndarray, what: str, chunk: int = 1 << 22) -> np.ndarray:
    """int32 rint(x * 2^16) (ties to even), in chunks; OverflowError when a value does not fit."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.int32)
    for lo in range(0, x.shape[0], chunk):
        y = np.rint(x[lo:lo + chunk] * Q16)
        if y.size and (not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > 2**31 - 1):
            raise OverflowError(f"{what}: a vote does not fit int32 in units of 2^-16 nats")
        out[lo:lo + chunk] = y
    return out


def _cupy_or_none(backend: str):
    if backend == "numpy":
        return None
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        if backend == "cupy":
            raise
        return None


def unigram_full(kn: KNModel, order: int) -> np.ndarray:
    """float64 [V + 1]: p_1(w) of the order-`order` model, the operations of
    `KNModel._unigram_full` (BOS column 0)."""
    lv = kn.level(1, order)
    single = kn.discount if kn.discount_mode == "single" else None
    cnt = lv.cnt.astype(np.float64)
    total = float(lv.total[0])
    if not kn.unk_interp:
        p = cnt / total if total > 0 else np.zeros_like(cnt)
    else:
        gamma = float(lv.gamma_mass(np.zeros(1, dtype=np.int64), single)[0]) / total if total > 0 else 1.0
        alpha = np.maximum(cnt - lv.discount_of(lv.cnt, single), 0.0) / total if total > 0 else np.zeros_like(cnt)
        p = alpha + gamma / kn.vocab
    p[kn.bos] = 0.0
    return p


class SuffixClauses:
    """The suffix clauses of a fitted KNModel as bias, clause constants and follower votes."""

    def __init__(self, *, order: int, vocab: int, keys: list, fires: list, d: list, v: list, dq: list, vq: list,
                 b: np.ndarray, bq: np.ndarray, p1: np.ndarray, cnt: list, total: list, stat: list,
                 p: list | None = None, build_info: dict | None = None) -> None:
        self.order = int(order)
        self.vocab = int(vocab)
        self.bos = int(vocab)
        self.keys = keys          # [K+1]: keys[j] uint64 [n_types_j] (shared with the KNModel)
        self.fires = fires        # [K+1]: fires[j] bool [n_types_{j-1}] (clause S(u) of level j exists)
        self.d = d                # [K+1]: d[j] float64 [n_types_{j-1}] = ln gamma (0 where it never fires)
        self.v = v                # [K+1]: v[j] float64 [n_types_j] follower votes (0 for non-followers)
        self.dq = dq              # int32 forms (units of 2^-16 nats)
        self.vq = vq
        self.b = b                # float64 [V+1] bias ln p_1 (-inf for BOS)
        self.bq = bq              # int32 [V+1] (INT32_MIN for BOS)
        self.p1 = p1              # float64 [V+1]
        self.cnt = cnt            # [K+1]: cnt[j] int32 [n_types_j] level-j counts (shared, explanations only)
        self.total = total        # [K+1]: total[j] float64 [n_types_{j-1}] (shared, explanations only)
        self.stat = stat          # [K+1]: "raw" or "continuation"
        self.p = p                # optional [K+1]: p_j per type (from_kn(keep_p=True), tests)
        self.build_info = build_info or {}

    # -------------------------------------------------------------- construction
    @classmethod
    def from_kn(cls, kn: KNModel, order: int | None = None, backend: str = "auto", chunk: int = 1 << 22,
                keep_p: bool = False) -> "SuffixClauses":
        """Build the clauses of the order-`order` model (default kn.order) of a fitted KNModel.
        backend: "auto" (CuPy for the suffix-link search when a GPU is usable), "cupy" or "numpy";
        every float64 operation runs on the host with numpy in either case."""
        import time

        t0 = time.perf_counter()
        K = int(order or kn.order)
        if not 1 <= K <= kn.order:
            raise ValueError(f"order must lie in 1..{kn.order}")
        if kn.keys[1] is None:
            raise ValueError("the KNModel is not fitted")
        cp = _cupy_or_none(backend)
        single = kn.discount if kn.discount_mode == "single" else None
        V = kn.vocab
        p1 = unigram_full(kn, K)
        b = np.full(V + 1, -np.inf)
        pos = p1 > 0
        b[pos] = np.log(p1[pos])
        bq = np.full(V + 1, INT32_MIN, dtype=np.int32)
        bq[pos] = _q16(b[pos], "bias")
        keys: list = [None] * (K + 1)
        fires: list = [None] * (K + 1)
        d: list = [None] * (K + 1)
        v: list = [None] * (K + 1)
        dq: list = [None] * (K + 1)
        vq: list = [None] * (K + 1)
        cnt_l: list = [None] * (K + 1)
        tot_l: list = [None] * (K + 1)
        stat: list = [None] * (K + 1)
        p_keep: list | None = [None] * (K + 1) if keep_p else None
        keys[1] = kn.keys[1]
        cnt_l[1] = kn.level(1, K).cnt
        stat[1] = "raw" if K == 1 else "continuation"
        if keep_p:
            p_keep[1] = p1
        p_prev = p1
        s_prev: np.ndarray | None = None       # suffix links of the level j-1 types (ids in keys[j-2])
        timings: dict[str, float] = {}
        for j in range(2, K + 1):
            tj = time.perf_counter()
            lv = kn.level(j, K)
            keys_j = kn.keys[j]
            n_t = int(keys_j.shape[0])
            n_ctx = int(kn.keys[j - 1].shape[0])
            # ---- per context: fires, ln gamma (KNModel.prob's total, seen, gamma, element by element)
            seen = lv.total > 0
            dj = np.zeros(n_ctx, dtype=np.float64)
            n_bad = 0
            for lo in range(0, n_ctx, chunk):
                hi = min(lo + chunk, n_ctx)
                st = seen[lo:hi]
                safe = np.where(st, lv.total[lo:hi], 1.0)
                g = np.where(st, lv.gamma_mass(np.arange(lo, hi, dtype=np.int64), single) / safe, 1.0)
                bad = st & ~(g > 0)
                n_bad += int(bad.sum())
                dj[lo:hi] = np.where(st & ~bad, np.log(np.where(st & ~bad, g, 1.0)), 0.0)
            if n_bad:
                raise ValueError(f"level {j}: {n_bad} firing contexts have gamma = 0 (all discounts of "
                                 "their follower counts are 0); ln gamma is not finite, the vote form does not exist")
            dq[j] = _q16(dj, f"level {j} clause constants")
            # ---- suffix links of the level-j types (the level j-1 links are not needed afterwards)
            s_j = cls._suffix_links(keys_j, kn.keys[j - 1], s_prev, j, cp, chunk)
            s_prev = None
            # ---- per type: alpha, p_j (only where a later level or keep_p needs it), follower votes
            vj = np.zeros(n_t, dtype=np.float64)
            pj = np.empty(n_t, dtype=np.float64) if (j < K or keep_p) else None
            n_follow = 0
            for lo in range(0, n_t, chunk):
                hi = min(lo + chunk, n_t)
                kk = keys_j[lo:hi]
                pre = (kk >> _SHIFT).astype(np.int64)
                tot = lv.total[pre]
                st = tot > 0
                safe = np.where(st, tot, 1.0)
                craw = lv.cnt[lo:hi]
                c = np.where(st, craw, 0).astype(np.float64)
                alpha = np.where(st, np.maximum(c - lv.discount_of(c, single), 0.0) / safe, 0.0)
                g = np.where(st, lv.gamma_mass(pre, single) / safe, 1.0)
                s = s_j[lo:hi]
                pp = np.where(s >= 0, p_prev[np.maximum(s, 0)], np.nan)
                if pj is not None:
                    pj[lo:hi] = alpha + g * pp
                fol = st & (craw > 0)
                n_follow += int(fol.sum())
                vv = np.zeros(hi - lo, dtype=np.float64)
                vv[fol] = np.log1p(alpha[fol] / (g[fol] * pp[fol]))
                if not np.all(np.isfinite(vv[fol])):
                    nb = int((~np.isfinite(vv[fol])).sum())
                    raise ValueError(f"level {j}: {nb} follower votes are not finite (a follower without a suffix "
                                     "type, or p_k = 0 below it); the type set is not suffix-closed")
                vj[lo:hi] = vv
            keys[j] = keys_j
            fires[j] = seen
            d[j] = dj
            v[j] = vj
            vq[j] = _q16(vj, f"level {j} follower votes")
            cnt_l[j] = lv.cnt
            tot_l[j] = lv.total
            stat[j] = "raw" if j == K else "continuation"
            if keep_p:
                p_keep[j] = pj
            p_prev = pj
            s_prev = s_j
            del s_j
            timings[f"level{j}_s"] = time.perf_counter() - tj
            timings[f"level{j}_followers"] = n_follow
        info = {"backend": "cupy" if cp is not None else "numpy", "total_s": time.perf_counter() - t0, **timings}
        return cls(order=K, vocab=V, keys=keys, fires=fires, d=d, v=v, dq=dq, vq=vq, b=b, bq=bq, p1=p1,
                   cnt=cnt_l, total=tot_l, stat=stat, p=p_keep, build_info=info)

    @staticmethod
    def _suffix_links(keys_j: np.ndarray, keys_prev: np.ndarray, s_prev: np.ndarray | None, j: int, cp,
                      chunk: int) -> np.ndarray:
        """int32 [n_types_j]: index in keys[j-1] of the suffix of every level-j type, MISSING if absent."""
        if keys_prev.shape[0] >= 2**31 - 1:
            raise ValueError("more than 2**31 - 2 types at one level")
        if j == 2:
            return (keys_j & _MASK16).astype(np.int32)
        n_t = int(keys_j.shape[0])
        out = np.empty(n_t, dtype=np.int32)
        if cp is not None:
            dk = cp.asarray(keys_prev)
            ds = cp.asarray(s_prev)
            nk = int(dk.shape[0])
            for lo in range(0, n_t, chunk * 4):
                hi = min(lo + chunk * 4, n_t)
                kk = cp.asarray(keys_j[lo:hi])
                sp = ds[(kk >> _SHIFT).astype(cp.int64)]
                q = (cp.maximum(sp, 0).astype(cp.uint64) << _SHIFT) | (kk & _MASK16)
                i = cp.searchsorted(dk, q)
                jj = cp.minimum(i, nk - 1)
                ok = (sp >= 0) & (i < nk) & (dk[jj] == q)
                out[lo:hi] = cp.asnumpy(cp.where(ok, i, MISSING).astype(cp.int32))
                del kk, sp, q, i, jj, ok
            del dk, ds
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()   # host staging buffers of the uploads
            return out
        for lo in range(0, n_t, chunk):
            hi = min(lo + chunk, n_t)
            kk = keys_j[lo:hi]
            sp = s_prev[(kk >> _SHIFT).astype(np.int64)].astype(np.int64)
            q = (np.maximum(sp, 0).astype(np.uint64) << _SHIFT) | (kk & _MASK16)
            found = _lookup(keys_prev, q)
            out[lo:hi] = np.where(sp >= 0, found, MISSING)
        return out

    # -------------------------------------------------------------- contexts
    def contexts(self, tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray,
                 hidden: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(ctx int64 [Q, K-1], valid int64 [Q]) of the left contexts of `pos`, BOS before the
        document start; with `hidden`, valid counts the trailing tokens up to the nearest hidden one
        (the semantics of KNModel.contexts)."""
        m = self.order - 1
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

    def _prep(self, ctx: np.ndarray, valid: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        ctx = np.asarray(ctx).astype(np.int64)
        if ctx.ndim != 2 or ctx.shape[1] < self.order - 1:
            raise ValueError(f"ctx must be [Q, m] with m >= order - 1 = {self.order - 1}")
        if valid is None:
            valid = np.full(ctx.shape[0], ctx.shape[1], dtype=np.int64)
        return ctx, np.asarray(valid).astype(np.int64)

    def fired(self, ctx: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """int64 [Q, K-1]: column k-1 is the index in keys[k] of the length-k context when its clause
        fires (context present, not cut by `valid`, positive total at level k+1), else -1."""
        ctx, valid = self._prep(ctx, valid)
        q, m = ctx.shape
        out = np.full((q, max(self.order - 1, 0)), MISSING, dtype=np.int64)
        for k in range(1, self.order):
            idx = ctx[:, m - k].copy()
            for jcol in range(m - k + 1, m):
                depth = jcol - (m - k) + 1
                key = (np.maximum(idx, 0).astype(np.uint64) << _SHIFT) | ctx[:, jcol].astype(np.uint64)
                found = _lookup(self.keys[depth], key)
                idx = np.where(idx == MISSING, MISSING, found)
            idx = np.where(valid >= k, idx, MISSING)
            ok = idx >= 0
            ok[ok] = self.fires[k + 1][idx[ok]]
            out[:, k - 1] = np.where(ok, idx, MISSING)
        return out

    def _check_w(self, w: np.ndarray, q: int) -> np.ndarray:
        w = np.asarray(w).astype(np.int64).reshape(-1)
        if w.shape[0] != q:
            raise ValueError("w must have one entry per context row")
        if w.shape[0] and (int(w.min()) < 0 or int(w.max()) >= self.vocab):
            raise ValueError("targets must lie in [0, vocab) (BOS is never a target)")
        return w

    def _votes(self, fired: np.ndarray, w: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
        """(row mask of the fired clause at `level`, index of the follower type or -1)."""
        cid = fired[:, level - 2]
        f = cid >= 0
        tid = np.full(cid.shape[0], MISSING, dtype=np.int64)
        if f.any():
            key = (cid[f].astype(np.uint64) << _SHIFT) | w[f].astype(np.uint64)
            tid[f] = _lookup(self.keys[level], key)
        return f, tid

    # -------------------------------------------------------------- scoring
    def logprob(self, ctx: np.ndarray, w: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """float64 [Q]: ln P(w | ctx) = b(w) + sum over fired clauses of (d(u) + v(u, w)),
        from the stored float64 votes only."""
        ctx, valid = self._prep(ctx, valid)
        w = self._check_w(w, ctx.shape[0])
        fired = self.fired(ctx, valid)
        lp = self.b[w].copy()
        for j in range(2, self.order + 1):
            f, tid = self._votes(fired, w, j)
            lp[f] = lp[f] + self.d[j][fired[f, j - 2]]
            has = tid >= 0
            lp[has] = lp[has] + self.v[j][tid[has]]
        return lp

    def logprob_q(self, ctx: np.ndarray, w: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """int64 [Q]: the same sum over the int32 votes, in units of 2^-16 nats."""
        ctx, valid = self._prep(ctx, valid)
        w = self._check_w(w, ctx.shape[0])
        fired = self.fired(ctx, valid)
        lp = self.bq[w].astype(np.int64)
        for j in range(2, self.order + 1):
            f, tid = self._votes(fired, w, j)
            lp[f] += self.dq[j][fired[f, j - 2]].astype(np.int64)
            has = tid >= 0
            lp[has] += self.vq[j][tid[has]].astype(np.int64)
        return lp

    def rounding_bound(self, ctx: np.ndarray, w: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """float64 [Q]: 0.5 * 2^-16 nats times the number of rounded terms of every query (the bias,
        each fired clause constant, each non-zero follower vote)."""
        ctx, valid = self._prep(ctx, valid)
        w = self._check_w(w, ctx.shape[0])
        fired = self.fired(ctx, valid)
        n = np.ones(ctx.shape[0], dtype=np.int64)
        for j in range(2, self.order + 1):
            f, tid = self._votes(fired, w, j)
            n += f
            has = tid >= 0
            n[has] += self.v[j][tid[has]] != 0
        return n * (0.5 / Q16)

    def logprob_stream(self, tok: np.ndarray, doc_start: np.ndarray, quantized: bool = False,
                       chunk: int = 1 << 16) -> np.ndarray:
        """Per-token ln P(tok[i] | left in-document context) over a whole stream: float64 [N]
        (natural log), or int64 [N] in 2^-16 nats with `quantized`."""
        tok = np.asarray(tok).astype(np.int64)
        out = np.zeros(tok.shape[0], dtype=np.int64 if quantized else np.float64)
        for i in range(0, tok.shape[0], chunk):
            pos = np.arange(i, min(i + chunk, tok.shape[0]))
            ctx, valid = self.contexts(tok, doc_start, pos)
            out[pos] = (self.logprob_q if quantized else self.logprob)(ctx, tok[pos], valid)
        return out

    # -------------------------------------------------------------- explanations
    def _text(self, token_text: Callable[[int], str] | Sequence[str] | None, t: int) -> str:
        if t == self.bos:
            return "<bos>"
        if token_text is None:
            return str(t)
        if callable(token_text):
            return str(token_text(t))
        return str(token_text[t])

    def explain(self, ctx: np.ndarray, w: int,
                token_text: Callable[[int], str] | Sequence[str] | None = None,
                valid: int | None = None) -> tuple[list[dict], str]:
        """The clauses behind ln P(w | ctx) for ONE query (ctx: 1-D, the last entry nearest to the
        target). Returns (items, text): items[0] is the bias clause, then one dict per fired suffix
        clause, shortest context first; text is the readable rendering, one line per item and a
        final line with the total."""
        c = np.asarray(ctx).astype(np.int64).reshape(1, -1)
        vv = None if valid is None else np.asarray([valid], dtype=np.int64)
        wa = np.asarray([int(w)], dtype=np.int64)
        fired = self.fired(c, vv)
        total = float(self.logprob(c, wa, vv)[0])
        total_q = int(self.logprob_q(c, wa, vv)[0])
        wn = self._text(token_text, int(w))
        items: list[dict] = [{
            "clause": "bias", "k": 0, "vote": float(self.b[w]), "vote_q": int(self.bq[w]), "p1": float(self.p1[w]),
            "text": f"BIAS (always fires): ln p_1({wn}) = {float(self.b[w]):.3f} nats",
        }]
        m = c.shape[1]
        for k in range(1, self.order):
            cid = int(fired[0, k - 1])
            if cid < 0:
                continue
            j = k + 1
            ctx_tok = [int(x) for x in c[0, m - k:]]
            key = (np.uint64(cid) << _SHIFT) | np.uint64(int(w))
            tid = int(_lookup(self.keys[j], np.asarray([key], dtype=np.uint64))[0])
            count = int(self.cnt[j][tid]) if tid >= 0 else 0
            vote = float(self.v[j][tid]) if tid >= 0 else 0.0
            vote_q = int(self.vq[j][tid]) if tid >= 0 else 0
            span = "tok[-1]" if k == 1 else f"tok[-{k}..-1]"
            words = ", ".join(self._text(token_text, t) for t in ctx_tok)
            cname = "count" if self.stat[j] == "raw" else "continuation count"
            dval = float(self.d[j][cid])
            items.append({
                "clause": "suffix", "k": k, "level": j, "context_id": cid, "context": ctx_tok,
                "stat": self.stat[j], "count": count, "total": float(self.total[j][cid]),
                "follower": bool(count > 0), "gamma": float(np.exp(dval)), "d": dval, "d_q": int(self.dq[j][cid]),
                "vote": vote, "vote_q": vote_q,
                "text": (f"IF {span} = ({words}) THEN {vote:+.3f} nats for {wn} ({cname} {count}); "
                         f"otherwise ln gamma = {dval:.3f}"),
            })
        lines = [it["text"] for it in items]
        lines.append(f"ln P({wn} | context) = bias + sum over the {len(items) - 1} fired clauses of "
                     f"(ln gamma + vote) = {total:.6f} nats (integer form {total_q} x 2^-16 = {total_q / Q16:.6f})")
        return items, "\n".join(lines)

    # -------------------------------------------------------------- accounting
    def nbytes_breakdown(self) -> dict[str, int]:
        """Bytes held: own arrays (fires, d, v, dq, vq, b, bq, p1) and arrays shared with the KNModel
        (keys; cnt and total, used by explanations only)."""
        own = {"fires": 0, "d": 0, "v": 0, "dq": 0, "vq": 0}
        for j in range(2, self.order + 1):
            for name in own:
                own[name] += int(getattr(self, name)[j].nbytes)
        own["bias"] = int(self.b.nbytes + self.bq.nbytes + self.p1.nbytes)
        shared = {"keys": sum(int(self.keys[j].nbytes) for j in range(1, self.order + 1)),
                  "cnt": sum(int(self.cnt[j].nbytes) for j in range(1, self.order + 1)),
                  "total": sum(int(self.total[j].nbytes) for j in range(2, self.order + 1))}
        out = {f"own.{k}": v for k, v in own.items()}
        out.update({f"shared.{k}": v for k, v in shared.items()})
        out["own"] = sum(own.values())
        out["scoring"] = out["own"] + shared["keys"]            # what logprob and logprob_q read
        out["total"] = out["own"] + sum(shared.values())
        return out

    def nbytes(self) -> int:
        """Bytes needed for scoring: the own arrays plus the keys (see nbytes_breakdown)."""
        return self.nbytes_breakdown()["scoring"]

    def state_hash(self) -> str:
        """sha256 over the integer arrays in the order documented in the module docstring."""
        h = hashlib.sha256()
        head = {"order": self.order, "vocab": self.vocab,
                "n_types": [int(self.keys[j].shape[0]) for j in range(1, self.order + 1)]}
        h.update(b"tmlm.cvtm.suffix.v1\x00")
        h.update(json.dumps(head, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\x00")
        for j in range(2, self.order + 1):
            for name, arr in ((f"keys[{j}]", self.keys[j]), (f"fires[{j}]", self.fires[j].astype(np.uint8)),
                              (f"dq[{j}]", self.dq[j]), (f"vq[{j}]", self.vq[j])):
                h.update(f"{name}={hashing.sha256_array(np.asarray(arr))}\n".encode("ascii"))
        h.update(f"bq={hashing.sha256_array(self.bq)}\n".encode("ascii"))
        return h.hexdigest()
