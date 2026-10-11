"""Masked CVTM (normative text: docs/11 section 11.8; hypothesis: docs/13 amendment A3, H8.11).

The masked CVTM-core predicts a hidden token from both sides with two enumerated clause families
keyed on the SAME ten two-sided contiguous patterns (`PATTERNS`, identical to `NGRAM_TYPES` of
`experiments/ckm_posthoc_probe.py`; a test asserts it):

    L1 (-1)   L2 (-2,-1)   L3 (-3,-2,-1)   R1 (+1)   R2 (+1,+2)   R3 (+1,+2,+3)
    L1R1 (-1,+1)   L2R1 (-2,-1,+1)   L1R2 (-1,+1,+2)   L2R2 (-2,-1,+1,+2)

Pattern key of position p: the symbols at p + o for the pattern's offsets o, where a position
outside p's document reads EDGE = V (the analogue of KNModel's BOS padding). With a visibility
vector, a pattern is CUT (does not fire) when any of its positions inside the document is hidden:
a pattern is all or nothing, as the context levels of `KNModel.contexts`, where a hidden token
removes every context that contains it. Keys pack the symbols with `sym_bits(V)` bits each
(13 bits at V 8000, 52 bits for the 4-symbol pattern), most significant symbol first.

Family S (enumerated count clauses). Clause S(pi, u) fires at p iff pattern pi is not cut at p and
its key equals u and u occurred at a counting position. Its read-out is the EXACT successor
distribution P_u(c) = O(u, c) / N(u) over the counting positions (every position of the counting
documents, only the target itself hidden). Counting is sort based: all keys of the stream, one
argsort, run starts give the contexts; (context id << bits | target) sorted once more gives the
(context, class) counts. `count_queried` runs this on the GPU (CuPy) or the host (numpy) with the
same integer operations; it returns only what the queries need (successor lists of the queried
contexts, or N and the gold count), plus the size of the full table.

Family D (document clauses keyed on the same patterns). D(pi) fires at p iff an EARLIER position
q < p of the same document has the same key under pattern pi, with q itself visible and pi not cut
at q; the most recent such q votes for tok[q] (a point mass). Whole-document window.

Combination (the normalised linear mixture `lin` of the probe script):

    P(c | p) = [ a0 P_uni(c) + sum over fired S(pi, u) of alpha[pi, l(N(u))] P_u(c)
                 + sum over fired D(pi) of beta[pi] [c = tok[q_pi]] ] / (a0 + sum alpha + sum beta)

with l(N) = floor(log2 N) (the (pattern, floor(log2 N)) buckets of docs/11 11.8) and P_uni the add-one
unigram of the counting positions. The mixture is scale free (one common factor of all weights does
not change it); the fitted weights are reported scaled so that the largest is 1.
beta[pi] is the confidence of the document clause of pattern pi, fitted on the fit documents by the
same EM as the alpha weights (one fitting procedure for every weight of the mixture).

Fitting (docs/11 11.6; docs/12 F20 b, c): fit documents H = training documents with
mix64(i XOR 0xC0FFEE) mod 100 == 0 (`tmlm.cvtm.protocol`). Fit rows = every position of H, only the
target hidden. Their S read-outs use the counts WITHOUT H (computed exactly as all counts minus the
H counts; the identity is tested and checked in the run), their unigram the add-one unigram without
H; the final model uses the counts over ALL training documents. The EM is the plain
minorise-maximise step of `tmlm.cvtm.mixture` pooled EM (docs/12 F20 c): w_b <- n_b / c_b with n_b
the summed responsibilities of weight b and c_b the sum over rows where b is active of 1 / (sum of
the row's active weights), then all weights divided by the largest (scale free; this keeps them
finite even when the supremum of the likelihood lies on the boundary, e.g. a useless unigram whose
weight tends to 0). The log-likelihood never decreases (Jensen on
the numerator, the tangent of the concave log on the normaliser); iteration stops when the mean
gain per fit row falls below `tol` nats or at `max_iter`. A weight whose bucket has no fit row keeps
no fitted value: `resolve_weights` gives an unfitted (pattern, level) bucket the weight of the
nearest fitted level of the same pattern (the lower level on a tie); an unfitted unigram weight is
an error, and an unfitted D weight is undefined (NaN): scoring a row that uses it is an error.

In a fit row the target p is hidden, so a pattern at an earlier q whose right span covers p
(1 <= p - q <= right extent) is cut at q: `doc_candidates_fit` skips those q. Nothing of a fit row
reads tok[p] except the gold count itself (tested by perturbing tok[p]).

Every count is an exact integer; the EM and the scoring are float64 in a fixed order (single
thread), so a run is deterministic on one platform.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

PATTERNS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("L1", (-1,)), ("L2", (-2, -1)), ("L3", (-3, -2, -1)), ("R1", (1,)), ("R2", (1, 2)),
    ("R3", (1, 2, 3)), ("L1R1", (-1, 1)), ("L2R1", (-2, -1, 1)), ("L1R2", (-1, 1, 2)),
    ("L2R2", (-2, -1, 1, 2)))
PATTERN_NAMES: tuple[str, ...] = tuple(n for n, _ in PATTERNS)
N_PATTERNS = len(PATTERNS)
LOGN_LEVELS = 32                     # floor(log2 N) in 0..31 (N < 2**32 is asserted)
N_S_PARAMS = N_PATTERNS * LOGN_LEVELS
D_PARAM0 = N_S_PARAMS                # beta[pi] = weight D_PARAM0 + pi
U_PARAM = N_S_PARAMS + N_PATTERNS    # a0 (unigram)
N_PARAMS = U_PARAM + 1
ROW_WIDTH = 2 * N_PATTERNS + 1       # columns: S of pattern 0..9, D of pattern 0..9, unigram
U_COL = 2 * N_PATTERNS
DIST_CAP = 255                       # doc distances are stored capped (patterns reach 3)


# ------------------------------------------------------------------ keys
def sym_bits(vocab: int) -> int:
    """Bits per symbol: the smallest b with vocab (the EDGE symbol) < 2**b."""
    return int(vocab).bit_length()


def right_extent(offsets) -> int:
    return max(0, max(int(o) for o in offsets))


def check_pattern(offsets, vocab: int) -> None:
    if len(offsets) == 0 or any(int(o) == 0 for o in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("a pattern is a non-empty set of distinct non-zero offsets")
    if max(abs(int(o)) for o in offsets) > DIST_CAP:
        raise ValueError("offsets beyond the distance cap")
    if sym_bits(vocab) * len(offsets) > 63:
        raise ValueError("the pattern key does not fit 63 bits")


def pattern_keys(tok: np.ndarray, doc_start: np.ndarray, pos: np.ndarray, offsets, vocab: int,
                 vis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(key uint64, valid bool) of pattern `offsets` at the positions `pos` of a stream.
    Out-of-document positions read EDGE = vocab; with `vis`, valid is False when any in-document
    position of the pattern is hidden (vis == 0). Without `vis` every key is valid."""
    check_pattern(offsets, vocab)
    tok = np.asarray(tok)
    ds = np.asarray(doc_start).astype(np.int64)
    pos = np.asarray(pos).astype(np.int64)
    n = tok.shape[0]
    bits = np.uint64(sym_bits(vocab))
    d = np.searchsorted(ds, pos, side="right") - 1
    lo, hi = ds[d], ds[d + 1]
    key = np.zeros(pos.shape[0], dtype=np.uint64)
    valid = np.ones(pos.shape[0], dtype=bool)
    for o in offsets:
        q = pos + int(o)
        inside = (q >= lo) & (q < hi)
        qc = np.clip(q, 0, max(0, n - 1))
        s = np.where(inside, tok[qc].astype(np.int64), int(vocab)).astype(np.uint64)
        if vis is not None:
            valid &= (~inside) | (np.asarray(vis)[qc] != 0)
        key = (key << bits) | s
    return key, valid


def doc_distances(doc_start: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(dl, dr) uint8 [n]: distance of every position to its document's first and last position,
    capped at DIST_CAP."""
    ds = np.asarray(doc_start).astype(np.int64)
    if int(ds[0]) != 0 or int(ds[-1]) != int(n) or np.any(np.diff(ds) < 0):
        raise ValueError("doc_start must start at 0, end at n and be non-decreasing")
    lengths = np.diff(ds)
    start = np.repeat(ds[:-1], lengths)
    idx = np.arange(int(n), dtype=np.int64)
    dl = np.minimum(idx - start, DIST_CAP).astype(np.uint8)
    del start
    end = np.repeat(ds[1:] - 1, lengths)
    dr = np.minimum(end - idx, DIST_CAP).astype(np.uint8)
    return dl, dr


def doc_index_of(doc_start: np.ndarray, n: int) -> np.ndarray:
    ds = np.asarray(doc_start).astype(np.int64)
    return np.repeat(np.arange(ds.shape[0] - 1, dtype=np.int64), np.diff(ds))


# ------------------------------------------------------------------ counting (numpy or CuPy)
@dataclass
class Stream:
    """A token stream on an array backend: tok int32, dl/dr uint8 (capped document distances)."""

    xp: object
    backend: str
    tok: object
    dl: object
    dr: object
    n: int

    def close(self) -> None:
        self.tok = self.dl = self.dr = None
        if self.backend == "cupy":
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()


def upload_stream(tok: np.ndarray, doc_start: np.ndarray, backend: str = "numpy") -> Stream:
    if backend not in ("numpy", "cupy"):
        raise ValueError("backend is 'numpy' or 'cupy'")
    tok = np.asarray(tok)
    n = int(tok.shape[0])
    dl, dr = doc_distances(doc_start, n)
    if backend == "cupy":
        import cupy as xp
    else:
        xp = np
    return Stream(xp=xp, backend=backend, tok=xp.asarray(tok.astype(np.int32)), dl=xp.asarray(dl),
                  dr=xp.asarray(dr), n=n)


def stream_keys(st: Stream, offsets, vocab: int):
    """Keys of every position of the stream (backend array, uint64); equal to `pattern_keys` at
    every position (tested). No visibility: counting positions see every token but the target."""
    check_pattern(offsets, vocab)
    xp = st.xp
    n = st.n
    bits = xp.uint64(sym_bits(vocab))
    key = xp.zeros(n, dtype=xp.uint64)
    for o in offsets:
        o = int(o)
        s = xp.full(n, vocab, dtype=xp.uint64)
        if o < 0:
            if n + o > 0:
                s[-o:] = st.tok[:n + o].astype(xp.uint64)
            inside = st.dl >= -o
        else:
            if n - o > 0:
                s[:n - o] = st.tok[o:].astype(xp.uint64)
            inside = st.dr >= o
        s[~inside] = xp.uint64(vocab)
        del inside
        key <<= bits
        key |= s
        del s
    return key


def _sync(st: Stream) -> None:
    if st.backend == "cupy":
        st.xp.cuda.Device().synchronize()


def _host(st: Stream, a) -> np.ndarray:
    if st.backend == "cupy":
        return st.xp.asnumpy(a)
    return np.asarray(a)


def count_queried(st: Stream, offsets, vocab: int, gold_keys: np.ndarray | None = None,
                  gold_tok: np.ndarray | None = None, list_keys: np.ndarray | None = None) -> dict:
    """Exact counts of pattern `offsets` over every position of the stream (target = the token at
    the position), answered for the queries:

    - gold mode (gold_keys, gold_tok, same length, any order, duplicates allowed): per query the
      context count N(u) (0 when u never occurs) and the count O(u, gold) (int64 arrays);
    - list mode (list_keys: SORTED UNIQUE uint64): per key found, N, and the successor list as CSR
      (ptr int64 [q + 1], tgt int32 ascending, cnt int64); a key that never occurs has an empty list.

    Also returned: n_ctx and n_pairs of the full table (distinct contexts and distinct
    (context, class) pairs over the stream), max_n (the largest context count), seconds."""
    xp = st.xp
    t0 = time.perf_counter()
    bits = sym_bits(vocab)
    ubits = xp.uint64(bits)
    tmask = xp.uint64((1 << bits) - 1)
    key = stream_keys(st, offsets, vocab)
    n = st.n
    if n == 0:
        raise ValueError("empty stream")
    perm = xp.argsort(key) if st.backend == "cupy" else np.argsort(key, kind="stable")
    skey = key[perm]
    del key
    tgt = st.tok[perm].astype(xp.uint64)
    del perm
    new = xp.empty(n, dtype=bool)
    new[0] = True
    new[1:] = skey[1:] != skey[:-1]
    ctx_keys = skey[new]
    del skey
    cid = xp.cumsum(new, dtype=xp.int64)
    cid -= 1
    del new
    n_ctx = int(ctx_keys.shape[0])
    ctx_n = xp.bincount(cid, minlength=n_ctx).astype(xp.int64)
    comb = cid.astype(xp.uint64)
    del cid
    comb <<= ubits
    comb |= tgt
    del tgt
    comb.sort()
    newp = xp.empty(n, dtype=bool)
    newp[0] = True
    newp[1:] = comb[1:] != comb[:-1]
    pair_keys = comb[newp]
    del comb
    starts = xp.flatnonzero(newp)
    del newp
    n_pairs = int(pair_keys.shape[0])
    pair_cnt = xp.empty(n_pairs, dtype=xp.int64)
    pair_cnt[:-1] = starts[1:] - starts[:-1]
    pair_cnt[-1] = n - starts[-1]
    del starts
    max_n = int(ctx_n.max())
    if max_n >= 2 ** LOGN_LEVELS:
        raise OverflowError("a context count reaches 2**LOGN_LEVELS")
    out: dict = {"n_ctx": n_ctx, "n_pairs": n_pairs, "max_n": max_n, "n_positions": n}
    if gold_keys is not None:
        gk = xp.asarray(np.asarray(gold_keys, dtype=np.uint64))
        gt = xp.asarray(np.asarray(gold_tok).astype(np.int64)).astype(xp.uint64)
        c = xp.searchsorted(ctx_keys, gk)
        cc = xp.minimum(c, n_ctx - 1)
        found = ctx_keys[cc] == gk
        nq = xp.where(found, ctx_n[cc], 0)
        pk = (cc.astype(xp.uint64) << ubits) | gt
        j = xp.searchsorted(pair_keys, pk)
        jc = xp.minimum(j, n_pairs - 1)
        hit = found & (pair_keys[jc] == pk)
        cq = xp.where(hit, pair_cnt[jc], 0)
        out["gold_n"] = _host(st, nq).astype(np.int64)
        out["gold_cnt"] = _host(st, cq).astype(np.int64)
        del gk, gt, c, cc, found, nq, pk, j, jc, hit, cq
    if list_keys is not None:
        lk_h = np.asarray(list_keys, dtype=np.uint64)
        if lk_h.shape[0] > 1 and not np.all(lk_h[1:] > lk_h[:-1]):
            raise ValueError("list_keys must be sorted and unique")
        lk = xp.asarray(lk_h)
        c = xp.searchsorted(ctx_keys, lk)
        cc = xp.minimum(c, n_ctx - 1)
        found = ctx_keys[cc] == lk
        nq = xp.where(found, ctx_n[cc], 0)
        # pair range of context cc: [searchsorted(pair_keys, cc << bits), searchsorted(pair_keys, (cc + 1) << bits))
        lo = xp.searchsorted(pair_keys, cc.astype(xp.uint64) << ubits)
        hi = xp.searchsorted(pair_keys, (cc.astype(xp.uint64) + xp.uint64(1)) << ubits)
        length = xp.where(found, hi - lo, 0).astype(xp.int64)
        ptr = xp.zeros(lk.shape[0] + 1, dtype=xp.int64)
        ptr[1:] = xp.cumsum(length)
        total = int(ptr[-1])
        if total:
            # gather index without repeat: segment id by boundary marks (every found list is non-empty)
            nz = xp.flatnonzero(length > 0)
            mark = xp.zeros(total, dtype=xp.int64)
            mark[ptr[nz[1:]]] = 1
            seg = nz[xp.cumsum(mark)]
            del mark
            g = xp.arange(total, dtype=xp.int64) - ptr[seg] + lo[seg]
            del seg
            ltgt = (pair_keys[g] & tmask).astype(xp.int32)
            lcnt = pair_cnt[g]
            del g
        else:
            ltgt = xp.zeros(0, dtype=xp.int32)
            lcnt = xp.zeros(0, dtype=xp.int64)
        out["list_found"] = _host(st, found).astype(bool)
        out["list_n"] = _host(st, nq).astype(np.int64)
        out["list_ptr"] = _host(st, ptr).astype(np.int64)
        out["list_tgt"] = _host(st, ltgt).astype(np.int32)
        out["list_cnt"] = _host(st, lcnt).astype(np.int64)
        del lk, c, cc, found, nq, lo, hi, length, ptr, ltgt, lcnt
    del ctx_keys, ctx_n, pair_keys, pair_cnt
    _sync(st)
    if st.backend == "cupy":
        out["pool_bytes_before_release"] = int(xp.get_default_memory_pool().total_bytes())
        xp.get_default_memory_pool().free_all_blocks()
    out["seconds"] = time.perf_counter() - t0
    return out


def count_table_bruteforce(tok: np.ndarray, doc_start: np.ndarray, offsets, vocab: int) -> dict:
    """Reference for tests: {key: {target: count}} by a Python loop over every position."""
    tok = np.asarray(tok)
    n = tok.shape[0]
    key, _ = pattern_keys(tok, doc_start, np.arange(n), offsets, vocab)
    out: dict = {}
    for p in range(n):
        d = out.setdefault(int(key[p]), {})
        t = int(tok[p])
        d[t] = d.get(t, 0) + 1
    return out


# ------------------------------------------------------------------ document clauses
def doc_prev_same_key(keys: np.ndarray, doc_of: np.ndarray) -> np.ndarray:
    """prev[p] = the largest q < p with doc_of[q] == doc_of[p] and keys[q] == keys[p], else -1."""
    n = keys.shape[0]
    pos = np.arange(n, dtype=np.int64)
    order = np.lexsort((pos, keys, doc_of))
    prev_sorted = np.full(n, -1, dtype=np.int64)
    if n > 1:
        same = (doc_of[order[1:]] == doc_of[order[:-1]]) & (keys[order[1:]] == keys[order[:-1]])
        prev_sorted[1:] = np.where(same, order[:-1], -1)
    prev = np.empty(n, dtype=np.int64)
    prev[order] = prev_sorted
    return prev


def doc_candidates_fit(keys: np.ndarray, doc_of: np.ndarray, r: int) -> np.ndarray:
    """Fit rows (every position a row, only its own target hidden, every key valid): the most
    recent earlier q of the same document with the same key whose pattern does not cover p,
    i.e. p - q > r (r = the pattern's right extent); -1 if none."""
    prev = doc_prev_same_key(keys, doc_of)
    n = keys.shape[0]
    pos = np.arange(n, dtype=np.int64)
    cand = prev.copy()
    for _ in range(max(0, int(r))):
        bad = (cand >= 0) & (pos - cand <= r)
        if not bad.any():
            break
        cand[bad] = prev[cand[bad]]
    if np.any((cand >= 0) & (pos - cand <= r)):
        raise AssertionError("candidate inside the excluded span")
    return cand


def doc_candidates_eval(keys: np.ndarray, valid: np.ndarray, vis: np.ndarray, doc_of: np.ndarray) -> np.ndarray:
    """Evaluation stream under one fixed hidden mask: for every position p with a valid key, the
    most recent q < p of the same document with vis[q], valid[q] and keys[q] == keys[p]; -1 if none
    (and -1 at positions whose key is not valid). A position with vis 1 is its own writer only for
    later positions (strictly earlier q)."""
    n = keys.shape[0]
    pos = np.arange(n, dtype=np.int64)
    el = np.flatnonzero(valid)
    out = np.full(n, -1, dtype=np.int64)
    if el.shape[0] == 0:
        return out
    order = el[np.lexsort((el, keys[el], doc_of[el]))]
    m = order.shape[0]
    newg = np.ones(m, dtype=bool)
    newg[1:] = (doc_of[order[1:]] != doc_of[order[:-1]]) | (keys[order[1:]] != keys[order[:-1]])
    g = np.cumsum(newg) - 1
    is_w = np.asarray(vis)[order] != 0
    val = np.where(is_w, pos[order] + 1, 0).astype(np.int64)
    base = g.astype(np.int64) << np.int64(32)
    run = np.maximum.accumulate(base + val)
    latest_incl = run - base - 1                       # latest writer at or before this element
    # strictly earlier: the value before this element within the group
    prev_run = np.empty(m, dtype=np.int64)
    prev_run[0] = -1
    prev_run[1:] = latest_incl[:-1]
    prev_run[newg] = -1
    out[order] = prev_run
    return out


def doc_candidates_bruteforce_eval(keys, valid, vis, doc_of) -> np.ndarray:
    n = keys.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    for p in range(n):
        if not valid[p]:
            continue
        for q in range(p - 1, -1, -1):
            if doc_of[q] != doc_of[p]:
                break
            if vis[q] and valid[q] and keys[q] == keys[p]:
                out[p] = q
                break
    return out


# ------------------------------------------------------------------ rows
def s_param(pattern: int, n: np.ndarray) -> np.ndarray:
    """Weight index of S clauses of `pattern` with context count n >= 1: pattern * 32 + floor(log2 n)."""
    n = np.asarray(n).astype(np.int64)
    if np.any(n < 1):
        raise ValueError("S weights need n >= 1")
    lvl = np.floor(np.log2(n.astype(np.float64))).astype(np.int64)
    # exact integer floor(log2): correct float rounding at powers of two
    lvl = np.where((np.int64(1) << (lvl + 1)) <= n, lvl + 1, lvl)
    lvl = np.where((np.int64(1) << lvl) > n, lvl - 1, lvl)
    if np.any(lvl >= LOGN_LEVELS):
        raise OverflowError("count level beyond LOGN_LEVELS")
    return (int(pattern) * LOGN_LEVELS + lvl).astype(np.int32)


def add_one_unigram(tok: np.ndarray, vocab: int, minus: np.ndarray | None = None) -> np.ndarray:
    """(count + 1) / (total + V) over tok (minus the tokens of `minus`, for the without-H unigram)."""
    c = np.bincount(np.asarray(tok).astype(np.int64), minlength=vocab).astype(np.int64)
    if minus is not None:
        c = c - np.bincount(np.asarray(minus).astype(np.int64), minlength=vocab).astype(np.int64)
        if np.any(c < 0):
            raise AssertionError("negative unigram count")
    return (c.astype(np.float64) + 1.0) / (float(c.sum()) + float(vocab))


@dataclass
class Rows:
    """Rows of the mixture in the fixed column layout (ROW_WIDTH columns: S of pattern 0..9, D of
    pattern 0..9, unigram). par: weight index or -1 (inactive); pg: component probability of the
    gold token (fit rows); ref: S -> global context index, D -> voted token (evaluation rows)."""

    par: np.ndarray
    pg: np.ndarray | None = None
    ref: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.par.shape[0])


def empty_rows(n: int, with_pg: bool, with_ref: bool) -> Rows:
    par = np.full((n, ROW_WIDTH), -1, dtype=np.int32)
    par[:, U_COL] = U_PARAM
    return Rows(par=par, pg=np.zeros((n, ROW_WIDTH)) if with_pg else None,
                ref=np.full((n, ROW_WIDTH), -1, dtype=np.int64) if with_ref else None)


# ------------------------------------------------------------------ EM (minorise-maximise)
@njit(cache=False)
def _em_mm(par, pg, n_params, u_param, max_iter, tol, w, trace):
    n = par.shape[0]
    W = par.shape[1]
    it_done = 0
    for it in range(max_iter + 1):
        num = np.zeros(n_params)
        den = np.zeros(n_params)
        ll = 0.0
        for r in range(n):
            A = 0.0
            Z = 0.0
            for c in range(W):
                b = par[r, c]
                if b >= 0:
                    A += w[b] * pg[r, c]
                    Z += w[b]
            ll += np.log(A) - np.log(Z)
            iA = 1.0 / A
            iZ = 1.0 / Z
            for c in range(W):
                b = par[r, c]
                if b >= 0:
                    num[b] += w[b] * pg[r, c] * iA
                    den[b] += iZ
        trace[it] = ll / n
        it_done = it
        if it == max_iter:
            break
        if it > 0 and trace[it] - trace[it - 1] < tol:
            break
        for b in range(n_params):
            if den[b] > 0.0:
                w[b] = num[b] / den[b]
        s = 0.0
        for b in range(n_params):
            if w[b] > s:
                s = w[b]
        for b in range(n_params):
            w[b] = w[b] / s
    return it_done


def rows_per_param(rows: Rows) -> np.ndarray:
    p = rows.par[rows.par >= 0].astype(np.int64)
    return np.bincount(p, minlength=N_PARAMS).astype(np.int64)


def fit_em(rows: Rows, max_iter: int, tol: float) -> dict:
    """MM fit of every weight on the fit rows, from uniform (all weights 1). Returns weights
    (largest = 1), the mean log-likelihood trace (nats per row, at the weights BEFORE each update; the
    last entry is the returned weights'), iterations, whether the tolerance stopped it, and the
    number of fit rows per weight."""
    if rows.pg is None:
        raise ValueError("fit rows need pg")
    if np.any(rows.pg[:, U_COL] <= 0.0):
        raise ValueError("the unigram component must be positive in every row")
    w = np.ones(N_PARAMS)
    trace = np.full(int(max_iter) + 1, np.nan)
    t0 = time.perf_counter()
    it = int(_em_mm(np.ascontiguousarray(rows.par), np.ascontiguousarray(rows.pg), N_PARAMS, U_PARAM,
                    int(max_iter), float(tol), w, trace))
    tr = trace[:it + 1].copy()
    diffs = np.diff(tr)
    return {"weights": w, "trace": tr, "iterations": it, "stopped_by_tol": bool(it < int(max_iter)),
            "min_step_gain": float(diffs.min()) if diffs.shape[0] else 0.0,
            "last_step_gain": float(diffs[-1]) if diffs.shape[0] else 0.0,
            "rows_per_param": rows_per_param(rows), "seconds": time.perf_counter() - t0}


def resolve_weights(weights: np.ndarray, rows_per_param_fit: np.ndarray) -> tuple[np.ndarray, dict]:
    """Final weights: fitted where the weight had fit rows; an unfitted S bucket (pattern, level)
    takes the weight of the nearest fitted level of the same pattern (lower level on a tie). An
    unfitted document-clause weight has no rule: it is set to NaN, and `score` refuses rows that
    use it. Returns (weights, report with the source level of every substituted bucket)."""
    w = np.asarray(weights, dtype=np.float64).copy()
    fitted = np.asarray(rows_per_param_fit) > 0
    if not fitted[U_PARAM]:
        raise ValueError("the unigram weight has no fit row")
    unfitted_d = []
    for pi in range(N_PATTERNS):
        if not fitted[D_PARAM0 + pi]:
            w[D_PARAM0 + pi] = np.nan
            unfitted_d.append(PATTERN_NAMES[pi])
    subst: dict = {}
    for pi in range(N_PATTERNS):
        levels = np.flatnonzero(fitted[pi * LOGN_LEVELS:(pi + 1) * LOGN_LEVELS])
        if levels.shape[0] == 0:
            raise ValueError(f"pattern {PATTERN_NAMES[pi]} has no fitted level")
        for lv in range(LOGN_LEVELS):
            b = pi * LOGN_LEVELS + lv
            if fitted[b]:
                continue
            d = np.abs(levels - lv)
            src = int(levels[np.flatnonzero(d == d.min())[0]])   # levels ascending: the lower level on a tie
            w[b] = w[pi * LOGN_LEVELS + src]
            subst[f"{PATTERN_NAMES[pi]}.{lv}"] = src
    return w, {"substituted": subst, "n_substituted": len(subst), "unfitted_d": unfitted_d}


def weights_hash(weights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(weights, dtype="<f8")).tobytes()).hexdigest()


# ------------------------------------------------------------------ scoring
@njit(cache=False)
def _gold_sparse(par, pg, w):
    n = par.shape[0]
    W = par.shape[1]
    out = np.empty(n)
    for r in range(n):
        A = 0.0
        Z = 0.0
        for c in range(W):
            b = par[r, c]
            if b >= 0:
                A += w[b] * pg[r, c]
                Z += w[b]
        out[r] = A / Z
    return out


def gold_prob_sparse(rows: Rows, weights: np.ndarray) -> np.ndarray:
    return _gold_sparse(np.ascontiguousarray(rows.par), np.ascontiguousarray(rows.pg), np.asarray(weights, np.float64))


@njit(cache=False)
def _pg_from_lists(par, ref, gptr, gtgt, gcnt, gN, gold, p_uni, n_pat, u_col):
    """Gold-token component probabilities of evaluation rows (binary search in the sorted lists)."""
    n = par.shape[0]
    W = par.shape[1]
    pg = np.zeros((n, W))
    for r in range(n):
        g = gold[r]
        for j in range(n_pat):
            if par[r, j] >= 0:
                k = ref[r, j]
                lo = gptr[k]
                hi = gptr[k + 1]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if gtgt[mid] < g:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < gptr[k + 1] and gtgt[lo] == g:
                    pg[r, j] = gcnt[lo] / gN[k]
            if par[r, n_pat + j] >= 0:
                pg[r, n_pat + j] = 1.0 if ref[r, n_pat + j] == g else 0.0
        pg[r, u_col] = p_uni[g]
    return pg


@njit(cache=False)
def _score_dense(par, ref, w, gptr, gtgt, gcnt, gN, p_uni, gold, n_pat, u_col):
    n = par.shape[0]
    V = p_uni.shape[0]
    acc = np.empty(V)
    pred = np.empty(n, np.int64)
    rank = np.empty(n, np.int64)
    pgold = np.empty(n)
    maxerr = 0.0
    for r in range(n):
        a0 = w[par[r, u_col]]
        Z = a0
        for c in range(V):
            acc[c] = a0 * p_uni[c]
        for j in range(n_pat):
            b = par[r, j]
            if b >= 0:
                a = w[b]
                Z += a
                k = ref[r, j]
                s = a / gN[k]
                for e in range(gptr[k], gptr[k + 1]):
                    acc[gtgt[e]] += s * gcnt[e]
        for j in range(n_pat):
            b = par[r, n_pat + j]
            if b >= 0:
                a = w[b]
                Z += a
                acc[ref[r, n_pat + j]] += a
        iz = 1.0 / Z
        tot = 0.0
        best = 0
        bv = -1.0
        for c in range(V):
            acc[c] = acc[c] * iz
            tot += acc[c]
            if acc[c] > bv:          # strict: ties keep the lowest class id
                bv = acc[c]
                best = c
        e1 = abs(tot - 1.0)
        if e1 > maxerr:
            maxerr = e1
        g = gold[r]
        pgr = acc[g]
        rk = 0
        for c in range(V):
            if acc[c] > pgr or (acc[c] == pgr and c < g):
                rk += 1
        pred[r] = best
        rank[r] = rk
        pgold[r] = pgr
    return pred, rank, pgold, maxerr


@dataclass
class ContextLists:
    """Successor lists of the evaluation contexts, all patterns concatenated (global index). A
    queried context that never occurs in the counting stream has n = 0 and an empty list."""

    ptr: np.ndarray      # int64 [K + 1]
    tgt: np.ndarray      # int32, ascending within a context
    cnt: np.ndarray      # int64
    n: np.ndarray        # int64 [K], context counts (sum of the list)

    def check(self) -> None:
        lens = np.diff(self.ptr)
        if np.any((lens > 0) != (self.n > 0)):
            raise AssertionError("a context count and its list disagree on emptiness")
        csum = np.zeros(self.cnt.shape[0] + 1, dtype=np.int64)
        csum[1:] = np.cumsum(self.cnt)
        if not np.array_equal(csum[self.ptr[1:]] - csum[self.ptr[:-1]], self.n):
            raise AssertionError("context counts differ from the sums of their lists")
        if np.any(self.cnt <= 0):
            raise AssertionError("a non-positive count in a list")
        inner = np.ones(self.tgt.shape[0], dtype=bool)
        inner[self.ptr[:-1][lens > 0]] = False          # first entry of every list
        if self.tgt.shape[0] > 1 and np.any(inner[1:] & (self.tgt[1:] <= self.tgt[:-1])):
            raise AssertionError("list targets are not strictly ascending")


def concat_lists(parts: list[dict]) -> tuple[ContextLists, list[int]]:
    """Concatenate the list-mode outputs of `count_queried` (one per pattern); returns the lists
    and the global base index of every part."""
    bases, ptrs, tgts, cnts, ns = [], [], [], [], []
    off_k = 0
    off_e = 0
    for part in parts:
        bases.append(off_k)
        p = part["list_ptr"]
        ptrs.append(p[:-1] + off_e)
        tgts.append(part["list_tgt"])
        cnts.append(part["list_cnt"])
        ns.append(part["list_n"])
        off_k += p.shape[0] - 1
        off_e += int(p[-1])
    ptr = np.concatenate(ptrs + [np.array([off_e], dtype=np.int64)]).astype(np.int64)
    return ContextLists(ptr=ptr, tgt=np.concatenate(tgts).astype(np.int32), cnt=np.concatenate(cnts).astype(np.int64),
                        n=np.concatenate(ns).astype(np.int64)), bases


def fit_rows(gold: np.ndarray, p_uni: np.ndarray, s_n: np.ndarray, s_cnt: np.ndarray, d_tok: np.ndarray,
             use_d: bool = True) -> Rows:
    """Fit rows: gold [n]; p_uni [V]; s_n, s_cnt int64 [n, 10]: context count and gold count of the
    pattern's context (s_n = 0: the S clause does not fire); d_tok int64 [n, 10]: the token voted by
    the document clause (-1: it does not fire)."""
    gold = np.asarray(gold).astype(np.int64)
    n = gold.shape[0]
    rows = empty_rows(n, with_pg=True, with_ref=False)
    if np.any(s_cnt < 0) or np.any(s_cnt > s_n):
        raise ValueError("gold counts must lie in [0, n]")
    for pi in range(N_PATTERNS):
        m = s_n[:, pi] > 0
        rows.par[m, pi] = s_param(pi, s_n[m, pi])
        rows.pg[m, pi] = s_cnt[m, pi] / s_n[m, pi]
        if use_d:
            md = d_tok[:, pi] >= 0
            rows.par[md, N_PATTERNS + pi] = D_PARAM0 + pi
            rows.pg[md, N_PATTERNS + pi] = (d_tok[md, pi] == gold[md]).astype(np.float64)
    rows.pg[:, U_COL] = np.asarray(p_uni, dtype=np.float64)[gold]
    return rows


def eval_rows(s_n: np.ndarray, s_ref: np.ndarray, d_tok: np.ndarray, use_d: bool = True) -> Rows:
    """Evaluation rows: s_n int64 [n, 10] (0: not fired), s_ref the global context index of the
    fired S clauses, d_tok the voted token of the document clauses (-1: not fired)."""
    n = s_n.shape[0]
    rows = empty_rows(n, with_pg=False, with_ref=True)
    for pi in range(N_PATTERNS):
        m = s_n[:, pi] > 0
        rows.par[m, pi] = s_param(pi, s_n[m, pi])
        rows.ref[m, pi] = s_ref[m, pi]
        if use_d:
            md = d_tok[:, pi] >= 0
            rows.par[md, N_PATTERNS + pi] = D_PARAM0 + pi
            rows.ref[md, N_PATTERNS + pi] = d_tok[md, pi]
    return rows


def score(rows: Rows, weights: np.ndarray, lists: ContextLists, p_uni: np.ndarray, gold: np.ndarray) -> dict:
    """Full-distribution scoring of evaluation rows: top-1 prediction (ties to the lowest id), rank
    of the gold token, P(gold), and the largest |row sum - 1|; also P(gold) by the sparse formula
    (the EM's operations) for the equality check."""
    par = np.ascontiguousarray(rows.par)
    ref = np.ascontiguousarray(rows.ref)
    w = np.asarray(weights, dtype=np.float64)
    used = np.unique(par[par >= 0])
    if not np.all(np.isfinite(w[used])) or np.any(w[used] < 0):
        raise ValueError("a row uses a weight without a fitted value")
    gold = np.asarray(gold).astype(np.int64)
    pu = np.asarray(p_uni, dtype=np.float64)
    t0 = time.perf_counter()
    pred, rank, pgold, maxerr = _score_dense(par, ref, w, lists.ptr, lists.tgt, lists.cnt, lists.n, pu, gold,
                                             N_PATTERNS, U_COL)
    pg = _pg_from_lists(par, ref, lists.ptr, lists.tgt, lists.cnt, lists.n, gold, pu, N_PATTERNS, U_COL)
    sparse = _gold_sparse(par, pg, w)
    return {"pred": pred, "rank": rank, "p_gold": pgold, "p_gold_sparse": sparse, "pg": pg,
            "max_row_sum_error": float(maxerr),
            "max_rel_gold_diff": float(np.max(np.abs(sparse - pgold) / pgold)) if pgold.shape[0] else 0.0,
            "seconds": time.perf_counter() - t0}


def topk_summary(rank: np.ndarray, p_gold: np.ndarray) -> dict:
    rank = np.asarray(rank)
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "bits_per_token": float(-np.log2(np.asarray(p_gold)).mean()), "n": int(rank.shape[0])}
