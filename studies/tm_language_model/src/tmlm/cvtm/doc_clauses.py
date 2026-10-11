"""Document clauses of the Count-Vote Tsetlin Machine (normative text: docs/11 section 11.3).

Definitions (positions are indices into one token stream split into documents by `doc_start`,
document d covering doc_start[d] <= p < doc_start[d + 1]; doc_lo(p) is the start of p's document):

- Window. For a window N (None = the whole document so far) the window of p is
  W(p) = {q : max(doc_lo(p), p - N) <= q < p}. Only q has to lie in the window; the left context
  of q may reach further back inside the document. N = 0 gives an empty window.
- D_m, m in 0..4. c_m(p) is the m-token left context of p, padded at the document start with the
  symbol EDGE = 0xFFFF exactly as `KNModel.contexts` pads with BOS (m = 0 is the empty context).
  n_m(p) = #{q in W(p) : c_m(q) = c_m(p)} and g_m(p) = #{q in W(p) : c_m(q) = c_m(p), tok[q] = tok[p]}.
  D_m fires iff n_m(p) >= 1 and reads out P_Dm(w | p) = #{q : c_m(q) = c_m(p), tok[q] = w} / n_m(p).
  A padded context names its own offset in the document (the number of EDGE symbols), so it
  equals no other context of the same document: padded contexts never fire. This is the same as
  "a D_m context needs m tokens of in-document history".
- P, the pointer clause. L(p) is the largest L in ptr_min..ptr_max such that
  tok[p - L .. p - 1] = tok[q - L .. q - 1] for some q in W(p) with q - L >= doc_lo(p) (the whole
  match inside the document; then p - L >= doc_lo(p) too), and 0 if there is none. Equivalently
  L(p) = max over q in W(p) of min(lcs(p, q), ptr_max) when that is at least ptr_min, with lcs the
  common suffix length of the two in-document prefixes ending at p - 1 and q - 1. The candidate is
  tok[q] of the MOST RECENT q whose match reaches that L (ptr_pos = q, ptr_cand = tok[q]; -1 when
  L(p) = 0). P fires iff L(p) >= ptr_min.
  Structural facts used by `tmlm.cvtm.mixture`: D_m fires => D_(m-1) fires (the same q works),
  and P fires => D_4 fires (a match of length >= 5 contains the 4-token context).

Keys. The D_m keys are EXACT: the m <= 4 context symbols (16 bits each, tokens < 0xFFFF, which
covers every V <= 32768) are packed into one uint64, the document is a separate sort key (batch)
or implicit (streaming, one fresh table per document). Nothing about n_m or g_m is hashed.

Pointer hashes. H_L(p) = sum_{i = 1..L} (tok[p - i] + 1) * B^(i - 1) mod (2^61 - 1), a polynomial
hash kept in a 64-bit word, B = HASH_BASE. Collision probability: for two DIFFERENT token strings
of the same length L the difference of their hashes is a nonzero polynomial in B of degree at most
L - 1 whose coefficients have absolute value below 2^16 < 2^61 - 1, so it has at most L - 1 roots
in the field Z / (2^61 - 1); over a uniformly random base P[H_L(x) = H_L(y)] <= (L - 1) / (2^61 - 1)
<= 63 / (2^61 - 1), about 2.7e-17 per compared pair and length [derived]. Union bound over every
pair of positions of one document and the 60 lengths: 6.6e10 pairs on the validation split, so
P[any collision] < 1.9e-6, and 3.1e13 pairs on the full training split, below 9e-4 [derived from the
document lengths of data/wikitext103/bpe8000]. The bases are fixed constants (determinism), so these
are the usual random-base statements.
- Batch (`doc_features`): one hash. A collision CANNOT change any output: the pointer found through
  the hashes is verified token by token, and a failed verification sends that one position to an
  exact scan of its window (`DocFeatures.n_fallback` counts such positions). Proof that verifying
  the final (L, q) suffices: equal strings always have equal hashes, so the largest L with a hash
  match in the window is >= the true L(p), and the hashed "most recent q" at that L is at least as
  recent as the most recent true match; if that q verifies, both are the true values.
- Streaming (`doc_features_stream`): a trie node is keyed by TWO independent hashes and its depth,
  P[both collide] <= ((L - 1) / (2^61 - 1))^2 <= 7.5e-34 per pair, union bound 5e-23 (validation)
  and 2.4e-20 (training) [derived]. Every answer is still verified token by token and every list is
  checked on eviction; a failed check marks the document, which is recomputed with the exact batch
  function (`n_fallback` then counts documents). Only a double collision that no check notices could
  make the streaming output differ from the batch output.

Implementations (they return identical `DocFeatures`, which the tests check):
- `doc_features`: batch, sort based and vectorized with numpy (np.lexsort per order and per
  pointer length, O(n log n)); positions that cannot extend a match are dropped from the next
  pointer length, so the pointer loop shrinks quickly. Memory O(n) int64 arrays: meant for the fit
  documents and the evaluation splits, not for the 135M-token training stream.
- `doc_features_stream`: one pass per document in numba (comment block of the streaming section).
  Counts: one exact open-addressing table per document (linear probing, backward-shift deletion),
  one lookup-and-increment per order per token plus one decrement per order for the position that
  leaves the window; n_m comes from the identity n_m(p) = g_(m-1)(p - 1). Pointer: a sparse trie of
  the reversed left contexts whose nodes keep FIFO lists of window positions; a position registers
  at depths ptr_min .. min(ptr_max, max(ptr_min, L(p) + 1)) only, the query is a binary search
  over the depths ptr_min .. min(ptr_max, max(ptr_min, L(p - 1) + 1)) plus one token-by-token
  check, and the evicted position is popped from the heads of its lists. A ring buffer holds the
  last N + max(ptr_max, 4) + 2 tokens. No step depends on N; the live table entries are bounded by
  the window (at most max_order + 1 count entries and ptr_max - ptr_min + 1 trie list entries per
  window position). On text most positions register at depth ptr_min only: the pointer fires at
  5.2 percent of the validation positions [measured, smoke:
  results/20260924T054807Z_cvtm_doc_clauses_mixture_smoke/].
- `doc_candidates`: the exact successor counts per position and order as CSR rows (full
  distributions, top-1, sampling). Cost O(sum of n_m(p)) over the requested positions.
- `ns_per_token`: median wall time per token of `doc_features_stream` (H8.7c).

CuPy is not used (it was optional in the design): everything here is single-threaded CPU code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit

EDGE = 0xFFFF                                  # symbol read before the document start
MAX_ORDER = 4                                  # 4 x 16 bits = one exact uint64 key
M61 = np.uint64((1 << 61) - 1)                 # Mersenne prime of the pointer hash
HASH_BASE = 0x1C0FFEE5EED5A1B                  # fixed base, < 2^61 - 1
_U32 = np.uint64(0xFFFFFFFF)
_U29 = np.uint64(0x1FFFFFFF)


# ------------------------------------------------------------------ result type
@dataclass(frozen=True)
class DocFeatures:
    """Per-position document-clause features (arrays indexed by stream position).

    n[i, p], g[i, p]: n_m(p) and g_m(p) for m = orders[i] (int32).
    ptr_len[p]: L(p), 0 when the pointer does not fire (int32).
    ptr_cand[p]: tok[q] of the pointer position, -1 when none (int32).
    ptr_pos[p]: the pointer position q, -1 when none (int64).
    n_fallback: batch, positions whose hashed pointer failed exact verification and were decided
    by an exact scan; streaming, documents whose checks failed and were recomputed by the batch
    function. 0 unless a hash collision happened (module docstring)."""

    orders: tuple[int, ...]
    window: int | None
    ptr_min: int
    ptr_max: int
    n: np.ndarray
    g: np.ndarray
    ptr_len: np.ndarray
    ptr_cand: np.ndarray
    ptr_pos: np.ndarray
    n_fallback: int

    def count(self, m: int) -> np.ndarray:
        return self.n[self.orders.index(m)]

    def gold_count(self, m: int) -> np.ndarray:
        return self.g[self.orders.index(m)]

    def same_values(self, other: "DocFeatures") -> bool:
        """True when every per-position output is identical (n_fallback is not compared)."""
        return (self.orders == other.orders and self.window == other.window
                and self.ptr_min == other.ptr_min and self.ptr_max == other.ptr_max
                and np.array_equal(self.n, other.n) and np.array_equal(self.g, other.g)
                and np.array_equal(self.ptr_len, other.ptr_len)
                and np.array_equal(self.ptr_cand, other.ptr_cand)
                and np.array_equal(self.ptr_pos, other.ptr_pos))


@dataclass(frozen=True)
class CandidateCSR:
    """Earlier successors of every requested position for one order: row r covers
    tok[indptr[r]:indptr[r + 1]] (ascending token ids) with their counts cnt (int64)."""

    indptr: np.ndarray
    tok: np.ndarray
    cnt: np.ndarray

    def row(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.indptr[r]), int(self.indptr[r + 1])
        return self.tok[a:b], self.cnt[a:b]


# ------------------------------------------------------------------ argument checks
def _check_stream(tok, doc_start) -> tuple[np.ndarray, np.ndarray]:
    tok = np.ascontiguousarray(np.asarray(tok).astype(np.int64))
    ds = np.ascontiguousarray(np.asarray(doc_start).astype(np.int64))
    if tok.ndim != 1 or ds.ndim != 1 or ds.shape[0] < 1:
        raise ValueError("tok and doc_start must be 1-D, doc_start non-empty")
    if ds[0] != 0 or ds[-1] != tok.shape[0] or np.any(np.diff(ds) < 0):
        raise ValueError("doc_start must start at 0, end at len(tok) and be non-decreasing")
    if tok.shape[0] and (tok.min() < 0 or tok.max() >= EDGE):
        raise ValueError(f"token ids must lie in 0..{EDGE - 1} (exact 16-bit keys)")
    return tok, ds


def _check_orders(orders) -> tuple[int, ...]:
    out = tuple(int(m) for m in orders)
    if len(out) == 0 or list(out) != sorted(set(out)) or out[0] < 0 or out[-1] > MAX_ORDER:
        raise ValueError(f"orders must be strictly increasing values in 0..{MAX_ORDER}")
    return out


def _check_window(window) -> int | None:
    if window is None:
        return None
    w = int(window)
    if w < 0:
        raise ValueError("window must be None or >= 0")
    return w


def _check_ptr(ptr_min: int, ptr_max: int) -> tuple[int, int]:
    ptr_min, ptr_max = int(ptr_min), int(ptr_max)
    if not (1 <= ptr_min <= ptr_max):
        raise ValueError("need 1 <= ptr_min <= ptr_max")
    return ptr_min, ptr_max


def _doc_index(ds: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.diff(ds)
    doc_of = np.repeat(np.arange(lengths.shape[0], dtype=np.int64), lengths)
    return doc_of, ds[doc_of] if n else np.zeros(0, np.int64)


# ------------------------------------------------------------------ arithmetic mod 2^61 - 1
def _mulmod61_np(a: np.ndarray, b) -> np.ndarray:
    """a * b mod (2^61 - 1) for uint64 arrays or scalars below 2^61 - 1."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    a_hi, a_lo = a >> np.uint64(32), a & _U32
    b_hi, b_lo = b >> np.uint64(32), b & _U32
    lo = a_lo * b_lo
    mid = a_hi * b_lo + a_lo * b_hi
    hi = a_hi * b_hi
    r = ((hi << np.uint64(3)) + (mid >> np.uint64(29)) + ((mid & _U29) << np.uint64(32))
         + (lo >> np.uint64(61)) + (lo & M61))
    r = (r & M61) + (r >> np.uint64(61))
    return (r - np.where(r >= M61, M61, np.uint64(0))).astype(np.uint64)


def _addmod61_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.asarray(a, dtype=np.uint64) + np.asarray(b, dtype=np.uint64)
    return (r - np.where(r >= M61, M61, np.uint64(0))).astype(np.uint64)


@njit(cache=False, inline="always")
def _mulmod61(a, b):
    a_hi = a >> np.uint64(32)
    a_lo = a & np.uint64(0xFFFFFFFF)
    b_hi = b >> np.uint64(32)
    b_lo = b & np.uint64(0xFFFFFFFF)
    lo = a_lo * b_lo
    mid = a_hi * b_lo + a_lo * b_hi
    hi = a_hi * b_hi
    r = ((hi << np.uint64(3)) + (mid >> np.uint64(29)) + ((mid & np.uint64(0x1FFFFFFF)) << np.uint64(32))
         + (lo >> np.uint64(61)) + (lo & M61))
    r = (r & M61) + (r >> np.uint64(61))
    if r >= M61:
        r = r - M61
    return r


@njit(cache=False, inline="always")
def _addmod61(a, b):
    r = a + b
    if r >= M61:
        r = r - M61
    return r


# ------------------------------------------------------------------ batch (sort based)
def _shifted_sym(tok: np.ndarray, doc_lo: np.ndarray, i: int) -> np.ndarray:
    """tok[p - i] when p - i >= doc_lo(p), else EDGE (int64)."""
    n = tok.shape[0]
    out = np.full(n, EDGE, dtype=np.int64)
    src = np.arange(n, dtype=np.int64) - i
    ok = src >= doc_lo
    out[ok] = tok[src[ok]]
    return out


def _window_counts(keys: tuple[np.ndarray, ...], doc_of: np.ndarray, lo_w: np.ndarray) -> np.ndarray:
    """For every position p: #{q in [lo_w(p), p) : same document and keys[k][q] == keys[k][p] for all k}.
    keys are ordered from most to least significant."""
    n = doc_of.shape[0]
    pos = np.arange(n, dtype=np.int64)
    order = np.lexsort((pos,) + tuple(reversed(keys)))       # primary keys[0], ..., then position
    s = pos[order]
    new = np.ones(n, dtype=bool)
    if n > 1:
        diff = doc_of[s[1:]] != doc_of[s[:-1]]
        for k in keys:
            ks = k[s]
            diff |= ks[1:] != ks[:-1]
        new[1:] = diff
    gid = np.cumsum(new) - 1
    comb = gid * np.int64(n + 1) + s                          # ascending: groups, then positions
    left = np.searchsorted(comb, gid * np.int64(n + 1) + lo_w[s], side="left")
    out = np.empty(n, dtype=np.int64)
    out[s] = np.arange(n, dtype=np.int64) - left
    return out


@njit(cache=False)
def _pointer_scan_exact(tok, p, a, lo_w, ptr_min, ptr_max):
    """Exact pointer of position p by scanning its window (most recent q first)."""
    best_l = 0
    best_q = -1
    for q in range(p - 1, lo_w - 1, -1):
        length = 0
        while length < ptr_max and q - 1 - length >= a and tok[p - 1 - length] == tok[q - 1 - length]:
            length += 1
        if length > best_l:
            best_l = length
            best_q = q
    if best_l < ptr_min:
        return 0, -1
    return best_l, best_q


def _pointer_batch(tok, doc_of, doc_lo, lo_w, ptr_min, ptr_max, base):
    n = tok.shape[0]
    best_l = np.zeros(n, dtype=np.int64)
    best_q = np.full(n, -1, dtype=np.int64)
    h = np.zeros(n, dtype=np.uint64)
    tp1 = tok.astype(np.uint64) + np.uint64(1)
    act = np.arange(n, dtype=np.int64)
    pw = np.uint64(1)
    b = np.uint64(base % int(M61))
    for length in range(1, ptr_max + 1):
        act = act[act - length >= doc_lo[act]]
        if act.shape[0] == 0:
            break
        h[act] = _addmod61_np(h[act], _mulmod61_np(tp1[act - length], pw))
        pw = np.uint64(_mulmod61_np(pw, b))
        if length < ptr_min:
            continue
        order = np.lexsort((act, h[act]))                     # hash, then position
        s = act[order]
        same = (h[s[1:]] == h[s[:-1]]) & (doc_of[s[1:]] == doc_of[s[:-1]])
        prev = np.full(s.shape[0], -1, dtype=np.int64)
        prev[1:] = np.where(same, s[:-1], -1)
        hit = (prev >= 0) & (prev >= lo_w[s])
        best_l[s[hit]] = length
        best_q[s[hit]] = prev[hit]
        member = np.zeros(s.shape[0], dtype=bool)
        member[1:] |= same
        member[:-1] |= same
        act = s[member]
    # exact verification of every hashed pointer, exact scan where it fails
    rows = np.flatnonzero(best_l >= ptr_min)
    offs = np.arange(1, ptr_max + 1, dtype=np.int64)
    n_fallback = 0
    for c0 in range(0, rows.shape[0], 1 << 15):
        r = rows[c0:c0 + (1 << 15)]
        lr = best_l[r]
        use = offs[None, :] <= lr[:, None]
        ia = np.where(use, r[:, None] - offs[None, :], 0)
        ib = np.where(use, best_q[r][:, None] - offs[None, :], 0)
        bad = np.any(use & (tok[ia] != tok[ib]), axis=1)
        for p in r[bad]:
            n_fallback += 1
            bl, bq = _pointer_scan_exact(tok, int(p), int(doc_lo[p]), int(lo_w[p]), ptr_min, ptr_max)
            best_l[p] = bl
            best_q[p] = bq
    return best_l, best_q, n_fallback


def doc_features(tok, doc_start, orders=(0, 1, 2, 3, 4), window=None, ptr_min: int = 5,
                 ptr_max: int = 64, *, _hash_base: int = HASH_BASE) -> DocFeatures:
    """Batch document-clause features of every position (docs/11 section 11.3); see the module
    docstring for the definitions. `_hash_base` exists for tests that force hash collisions."""
    tok, ds = _check_stream(tok, doc_start)
    orders = _check_orders(orders)
    window = _check_window(window)
    ptr_min, ptr_max = _check_ptr(ptr_min, ptr_max)
    n = tok.shape[0]
    doc_of, doc_lo = _doc_index(ds, n)
    pos = np.arange(n, dtype=np.int64)
    lo_w = doc_lo.copy() if window is None else np.maximum(doc_lo, pos - window)
    cnt_n = np.zeros((len(orders), n), dtype=np.int32)
    cnt_g = np.zeros((len(orders), n), dtype=np.int32)
    ctx = np.zeros(n, dtype=np.uint64)
    m_done = 0
    for i, m in enumerate(orders):
        while m_done < m:
            m_done += 1
            ctx |= _shifted_sym(tok, doc_lo, m_done).astype(np.uint64) << np.uint64(16 * (m_done - 1))
        cnt_n[i] = _window_counts((ctx,), doc_of, lo_w)
        cnt_g[i] = _window_counts((ctx, tok), doc_of, lo_w)
    if n:
        best_l, best_q, n_fb = _pointer_batch(tok, doc_of, doc_lo, lo_w, ptr_min, ptr_max, int(_hash_base))
    else:
        best_l, best_q, n_fb = np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    cand = np.where(best_q >= 0, tok[np.maximum(best_q, 0)] if n else best_q, -1)
    return DocFeatures(orders=orders, window=window, ptr_min=ptr_min, ptr_max=ptr_max, n=cnt_n, g=cnt_g,
                       ptr_len=best_l.astype(np.int32), ptr_cand=cand.astype(np.int32),
                       ptr_pos=best_q.astype(np.int64), n_fallback=int(n_fb))


# ------------------------------------------------------------------ candidates (full distributions)
@njit(cache=False)
def _cand_kernel(tok, s, rank, left, req, fill, indptr, out_tok, out_cnt, scratch, touched):
    """Pass 1 (fill = False): number of distinct successors per requested row into indptr[1:].
    Pass 2 (fill = True): the rows themselves, ascending token ids."""
    for r in range(req.shape[0]):
        p = req[r]
        i = rank[p]
        nt = 0
        for j in range(left[p], i):
            w = tok[s[j]]
            if scratch[w] == 0:
                touched[nt] = w
                nt += 1
            scratch[w] += 1
        if fill:
            ws = np.sort(touched[:nt])
            base = indptr[r]
            for k in range(nt):
                out_tok[base + k] = ws[k]
                out_cnt[base + k] = scratch[ws[k]]
        else:
            indptr[r + 1] = nt
        for k in range(nt):
            scratch[touched[k]] = 0


def doc_candidates(tok, doc_start, positions, orders=(0, 1, 2, 3, 4), window=None) -> dict[int, CandidateCSR]:
    """Exact document-local successor counts of the requested positions, per order:
    {m: CandidateCSR} where row r lists the distinct tok[q] over q in W(positions[r]) with
    c_m(q) = c_m(positions[r]), and their counts (the counts of a row sum to n_m). The work and
    the output grow with sum of n_m over the requested positions: call it on chunks."""
    tok, ds = _check_stream(tok, doc_start)
    orders = _check_orders(orders)
    window = _check_window(window)
    req = np.ascontiguousarray(np.asarray(positions).astype(np.int64))
    n = tok.shape[0]
    if req.size and (req.min() < 0 or req.max() >= n):
        raise ValueError("positions out of range")
    doc_of, doc_lo = _doc_index(ds, n)
    pos = np.arange(n, dtype=np.int64)
    lo_w = doc_lo.copy() if window is None else np.maximum(doc_lo, pos - window)
    out: dict[int, CandidateCSR] = {}
    ctx = np.zeros(n, dtype=np.uint64)
    m_done = 0
    scratch = np.zeros(EDGE + 1, dtype=np.int64)
    for m in orders:
        while m_done < m:
            m_done += 1
            ctx |= _shifted_sym(tok, doc_lo, m_done).astype(np.uint64) << np.uint64(16 * (m_done - 1))
        order = np.lexsort((pos, ctx))
        s = pos[order]
        new = np.ones(n, dtype=bool)
        if n > 1:
            new[1:] = (ctx[s[1:]] != ctx[s[:-1]]) | (doc_of[s[1:]] != doc_of[s[:-1]])
        gid = np.cumsum(new) - 1
        comb = gid * np.int64(n + 1) + s
        left_sorted = np.searchsorted(comb, gid * np.int64(n + 1) + lo_w[s], side="left")
        rank = np.empty(n, dtype=np.int64)
        rank[s] = np.arange(n, dtype=np.int64)
        left = np.empty(n, dtype=np.int64)
        left[s] = left_sorted
        s = np.ascontiguousarray(s)
        indptr = np.zeros(req.shape[0] + 1, dtype=np.int64)
        touched = np.zeros(EDGE + 1, dtype=np.int64)
        dummy_t = np.zeros(0, dtype=np.int64)
        _cand_kernel(tok, s, rank, left, req, False, indptr, dummy_t, dummy_t, scratch, touched)
        indptr = np.cumsum(indptr)
        out_tok = np.zeros(int(indptr[-1]), dtype=np.int64)
        out_cnt = np.zeros(int(indptr[-1]), dtype=np.int64)
        _cand_kernel(tok, s, rank, left, req, True, indptr, out_tok, out_cnt, scratch, touched)
        out[m] = CandidateCSR(indptr=indptr, tok=out_tok, cnt=out_cnt)
    return out




# ------------------------------------------------------------------ streaming (numba)
# Count table: exact keys (packed m-token context, (m << 17) | token) of the (m + 1)-gram ENDING at a
# position, one fresh table per document. Only g_m needs a table: n_0(p) = p - lo_w(p) and, for
# m >= 1, n_m(p) = g_(m-1)(p - 1) (the positions q in W(p) with c_m(q) = c_m(p) are exactly q' + 1 for
# the q' in W(p - 1) whose m-gram ending at q' equals the one ending at p - 1; q' = doc_lo - 1 would
# read EDGE and never matches). The batch path does not use this identity, so the tests check it.
#
# Pointer: a trie over the reversed left contexts, stored sparsely. Node (L, context of length L) is
# keyed by two independent hashes (bases HASH_BASE and HASH_BASE2) and L, and holds a FIFO list
# (head = oldest, tail = most recent) of positions in the window, linked through
# nxt[(q mod R) * n_len + L - ptr_min]. Registration rule: position q enters the lists of its nodes
# at depths ptr_min .. R(q) = min(ptr_max, q - doc_lo, max(ptr_min, L*(q) + 1)) with L*(q) its own
# pointer length (below ptr_min: only depth ptr_min), and a node created at depth L <= L*(q) also
# receives q's pointer position first (lazily). Invariant (proved by induction in the notes of
# `doc_features_stream`): a node's list is either absent or exactly its positions in the current
# window, and a node is absent only if it holds at most one window position that did not register
# there. Query: the deepest depth D <= min(ptr_max, L*(p - 1) + 1) with a node (binary search, the
# existence of a node is monotone in D); the only possible deeper match is the head of that node
# (the unique oldest position), checked token by token; otherwise the answer is (D, tail). So a
# position costs O(1 + R(q) - ptr_min) table operations instead of ptr_max - ptr_min + 1.
HASH_BASE2 = 0x0B5AD4ECEDA1CE2A              # second base, < 2^61 - 1, independent of HASH_BASE


@njit(cache=False, inline="always")
def _mix3(k1, k2, k3):
    z = k1 * np.uint64(0x9E3779B97F4A7C15) + k2
    z = z * np.uint64(0xD6E8FEB86659FD93) + k3
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=False)
def _c_find(k1, k2, ck1, ck2, cused, mask):
    i = np.int64(_mix3(k1, k2, np.uint64(0)) & np.uint64(mask))
    while cused[i]:
        if ck1[i] == k1 and ck2[i] == k2:
            return i
        i = (i + 1) & mask
    return -i - 1


@njit(cache=False)
def _c_delete(i, ck1, ck2, cval, cused, mask):
    """Backward-shift deletion (linear probing stays valid without tombstones)."""
    j = i
    while True:
        j = (j + 1) & mask
        if not cused[j]:
            break
        h = np.int64(_mix3(ck1[j], ck2[j], np.uint64(0)) & np.uint64(mask))
        if i <= j:
            stay = (i < h) and (h <= j)
        else:
            stay = (i < h) or (h <= j)
        if not stay:
            ck1[i] = ck1[j]
            ck2[i] = ck2[j]
            cval[i] = cval[j]
            i = j
    cused[i] = 0


@njit(cache=False)
def _c_inc(k1, k2, ck1, ck2, cval, cused, mask):
    idx = _c_find(k1, k2, ck1, ck2, cused, mask)
    if idx >= 0:
        old = cval[idx]
        cval[idx] = old + 1
        return old
    i = -idx - 1
    ck1[i] = k1
    ck2[i] = k2
    cval[i] = 1
    cused[i] = 1
    return 0


@njit(cache=False)
def _c_dec(k1, k2, ck1, ck2, cval, cused, mask):
    idx = _c_find(k1, k2, ck1, ck2, cused, mask)
    if idx < 0:
        return False
    v = cval[idx] - 1
    if v == 0:
        _c_delete(idx, ck1, ck2, cval, cused, mask)
    else:
        cval[idx] = v
    return True


@njit(cache=False)
def _n_find(k1, k2, k3, nk1, nk2, nk3, nused, mask):
    i = np.int64(_mix3(k1, k2, k3) & np.uint64(mask))
    while nused[i]:
        if nk1[i] == k1 and nk2[i] == k2 and nk3[i] == k3:
            return i
        i = (i + 1) & mask
    return -i - 1


@njit(cache=False)
def _n_delete(i, nk1, nk2, nk3, nhead, ntail, nused, mask):
    j = i
    while True:
        j = (j + 1) & mask
        if not nused[j]:
            break
        h = np.int64(_mix3(nk1[j], nk2[j], nk3[j]) & np.uint64(mask))
        if i <= j:
            stay = (i < h) and (h <= j)
        else:
            stay = (i < h) or (h <= j)
        if not stay:
            nk1[i] = nk1[j]
            nk2[i] = nk2[j]
            nk3[i] = nk3[j]
            nhead[i] = nhead[j]
            ntail[i] = ntail[j]
            i = j
    nused[i] = 0


@njit(cache=False)
def _n_grow(nk1, nk2, nk3, nhead, ntail, nused):
    cap = nk1.shape[0] * 2
    mask = cap - 1
    a1 = np.zeros(cap, np.uint64)
    a2 = np.zeros(cap, np.uint64)
    a3 = np.zeros(cap, np.uint64)
    ah = np.zeros(cap, np.int64)
    at = np.zeros(cap, np.int64)
    au = np.zeros(cap, np.uint8)
    for i in range(nk1.shape[0]):
        if nused[i]:
            j = np.int64(_mix3(nk1[i], nk2[i], nk3[i]) & np.uint64(mask))
            while au[j]:
                j = (j + 1) & mask
            a1[j] = nk1[i]
            a2[j] = nk2[i]
            a3[j] = nk3[i]
            ah[j] = nhead[i]
            at[j] = ntail[i]
            au[j] = 1
    return a1, a2, a3, ah, at, au


@njit(cache=False)
def _pow2(x):
    c = 16
    while c < x:
        c <<= 1
    return c


@njit(cache=False)
def _stream_kernel(tok, doc_start, orders, window, ptr_min, ptr_max, base1, base2, rsize,
                   out_n, out_g, out_l, out_q, bad_doc):
    n_ord = orders.shape[0]
    max_order = orders[n_ord - 1]
    n_len = ptr_max - ptr_min + 1
    ring = np.zeros(rsize, np.int64)
    nxt = np.empty(rsize * n_len, np.int64)
    pw1 = np.empty(ptr_max + 1, np.uint64)
    pw2 = np.empty(ptr_max + 1, np.uint64)
    pw1[0] = np.uint64(1)
    pw2[0] = np.uint64(1)
    for i in range(1, ptr_max + 1):
        pw1[i] = _mulmod61(pw1[i - 1], base1)
        pw2[i] = _mulmod61(pw2[i - 1], base2)
    h1 = np.zeros(ptr_max + 1, np.uint64)
    h2 = np.zeros(ptr_max + 1, np.uint64)
    gcur = np.zeros(max_order + 1, np.int64)
    gprev = np.zeros(max_order + 1, np.int64)
    edge = np.uint64(EDGE)
    for d in range(doc_start.shape[0] - 1):
        a = doc_start[d]
        b = doc_start[d + 1]
        if b <= a:
            continue
        live = b - a if window < 0 else min(window, b - a)
        cap_c = _pow2(2 * (max_order + 1) * (live + 1))
        mask_c = cap_c - 1
        ck1 = np.zeros(cap_c, np.uint64)
        ck2 = np.zeros(cap_c, np.uint64)
        cval = np.zeros(cap_c, np.int64)
        cused = np.zeros(cap_c, np.uint8)
        cap_n = _pow2(4 * (live + 1))
        nk1 = np.zeros(cap_n, np.uint64)
        nk2 = np.zeros(cap_n, np.uint64)
        nk3 = np.zeros(cap_n, np.uint64)
        nhead = np.zeros(cap_n, np.int64)
        ntail = np.zeros(cap_n, np.int64)
        nused = np.zeros(cap_n, np.uint8)
        n_ent = 0
        prev_l = 0
        for p in range(a, b):
            t = tok[p]
            ring[p % rsize] = t
            # ---------------- the position that leaves the window
            if window >= 0:
                e = p - 1 - window
                if e >= a:
                    te = np.uint64(ring[e % rsize])
                    cw = np.uint64(0)
                    for m in range(max_order + 1):
                        if m > 0:
                            sym = np.uint64(ring[(e - m) % rsize]) if e - m >= a else edge
                            cw = cw | (sym << np.uint64(16 * (m - 1)))
                        if not _c_dec(cw, np.uint64(m << 17) | te, ck1, ck2, cval, cused, mask_c):
                            bad_doc[d] = 1
                    le = min(ptr_max, e - a)
                    x1 = np.uint64(0)
                    x2 = np.uint64(0)
                    for length in range(1, le + 1):
                        s = np.uint64(ring[(e - length) % rsize] + 1)
                        x1 = _addmod61(x1, _mulmod61(s, pw1[length - 1]))
                        x2 = _addmod61(x2, _mulmod61(s, pw2[length - 1]))
                        if length >= ptr_min:
                            idx = _n_find(x1, x2, np.uint64(length), nk1, nk2, nk3, nused, cap_n - 1)
                            if idx < 0:
                                break
                            if nhead[idx] != e:
                                bad_doc[d] = 1
                                break
                            if ntail[idx] == e:
                                _n_delete(idx, nk1, nk2, nk3, nhead, ntail, nused, cap_n - 1)
                                n_ent -= 1
                            else:
                                nhead[idx] = nxt[(e % rsize) * n_len + length - ptr_min]
            # ---------------- document n-gram counts
            cw = np.uint64(0)
            tu = np.uint64(t)
            for m in range(max_order + 1):
                if m > 0:
                    sym = np.uint64(ring[(p - m) % rsize]) if p - m >= a else edge
                    cw = cw | (sym << np.uint64(16 * (m - 1)))
                gcur[m] = _c_inc(cw, np.uint64(m << 17) | tu, ck1, ck2, cval, cused, mask_c)
            lo_w = a if window < 0 else max(a, p - window)
            for oi in range(n_ord):
                m = orders[oi]
                if m == 0:
                    out_n[oi, p] = p - lo_w
                elif p > a:
                    out_n[oi, p] = gprev[m - 1]
                else:
                    out_n[oi, p] = 0
                out_g[oi, p] = gcur[m]
            for m in range(max_order + 1):
                gprev[m] = gcur[m]
            # ---------------- pointer query
            lav = min(ptr_max, p - a)
            lstar = 0
            qstar = -1
            top = 0
            if lav >= ptr_min:
                prev_m = prev_l if prev_l >= ptr_min else ptr_min - 1
                hi = min(lav, prev_m + 1)
                top = min(lav, hi + 1)
                x1 = np.uint64(0)
                x2 = np.uint64(0)
                for length in range(1, top + 1):
                    s = np.uint64(ring[(p - length) % rsize] + 1)
                    x1 = _addmod61(x1, _mulmod61(s, pw1[length - 1]))
                    x2 = _addmod61(x2, _mulmod61(s, pw2[length - 1]))
                    h1[length] = x1
                    h2[length] = x2
                i0 = _n_find(h1[ptr_min], h2[ptr_min], np.uint64(ptr_min), nk1, nk2, nk3, nused, cap_n - 1)
                if i0 >= 0:
                    dd = ptr_min
                    idd = i0
                    if hi > ptr_min:
                        ih = _n_find(h1[hi], h2[hi], np.uint64(hi), nk1, nk2, nk3, nused, cap_n - 1)
                        if ih >= 0:
                            dd = hi
                            idd = ih
                        else:
                            lo = ptr_min
                            hi2 = hi
                            while hi2 - lo > 1:
                                mid = (lo + hi2) // 2
                                im = _n_find(h1[mid], h2[mid], np.uint64(mid), nk1, nk2, nk3, nused, cap_n - 1)
                                if im >= 0:
                                    lo = mid
                                    idd = im
                                else:
                                    hi2 = mid
                            dd = lo
                    head = nhead[idd]
                    tail = ntail[idd]
                    lstar = dd
                    qstar = tail
                    ok = tail >= lo_w and tail < p and tail - dd >= a
                    if ok:
                        for i in range(1, dd + 1):
                            if ring[(p - i) % rsize] != ring[(tail - i) % rsize]:
                                ok = False
                                break
                    if dd < lav and head >= lo_w and head < p:
                        length = 0
                        while (length < ptr_max and head - 1 - length >= a
                               and ring[(p - 1 - length) % rsize] == ring[(head - 1 - length) % rsize]):
                            length += 1
                        if length >= dd + 1:
                            lstar = length
                            qstar = head
                            ok = True
                    if not ok:
                        bad_doc[d] = 1
            out_l[p] = lstar
            out_q[p] = qstar
            # ---------------- registration of p
            if lav >= ptr_min:
                rr = ptr_min if lstar < ptr_min else min(ptr_max, lstar + 1)
                rr = min(rr, top)
                for length in range(ptr_min, rr + 1):
                    idx = _n_find(h1[length], h2[length], np.uint64(length), nk1, nk2, nk3, nused, cap_n - 1)
                    off = length - ptr_min
                    nxt[(p % rsize) * n_len + off] = -1
                    if idx >= 0:
                        nxt[(ntail[idx] % rsize) * n_len + off] = p
                        ntail[idx] = p
                    else:
                        i = -idx - 1
                        nk1[i] = h1[length]
                        nk2[i] = h2[length]
                        nk3[i] = np.uint64(length)
                        nused[i] = 1
                        if lstar >= length:
                            nhead[i] = qstar
                            nxt[(qstar % rsize) * n_len + off] = p
                        else:
                            nhead[i] = p
                        ntail[i] = p
                        n_ent += 1
                        if 2 * n_ent > cap_n:
                            nk1, nk2, nk3, nhead, ntail, nused = _n_grow(nk1, nk2, nk3, nhead, ntail, nused)
                            cap_n = nk1.shape[0]
            prev_l = lstar


def doc_features_stream(tok, doc_start, orders=(0, 1, 2, 3, 4), window=None, ptr_min: int = 5,
                        ptr_max: int = 64, *, _hash_base: int = HASH_BASE,
                        _hash_base2: int = HASH_BASE2) -> DocFeatures:
    """Streaming document-clause features: one pass per document in numba with O(1) hash-table
    work per order and O(1 + R(p) - ptr_min) pointer-table work per token (comment block above),
    none of it growing with the window N. Returns exactly what `doc_features` returns.

    Exactness: n_m and g_m use exact keys. The pointer trie keys a node by two independent
    61-bit polynomial hashes and its depth; for two different contexts of the same length L the
    probability (over random bases) that both hashes collide is at most ((L - 1) / (2^61 - 1))^2,
    below 7.5e-34 [derived], so a merged node needs a collision of that size. Every pointer answer is
    still checked token by token, and the lists are checked on eviction (the evicted position must
    be the head); a failed check marks the document, which is then recomputed with the exact batch
    function (`n_fallback` counts such documents; the `_hash_base*` arguments exist to test this).

    Proof sketch of the trie invariant (window W(t), R(q) as above): (i) among the window
    positions of a node at depth L, all but the oldest registered there, because the oldest lies
    in their own windows and shares L tokens with them, so their L* >= L and R >= L + 1 > L;
    (ii) a node is created by the first position z that registers at L while the node has no
    list; by (i) at most one older window position y shares the node, and y exists iff L*(z) >= L,
    in which case y is z's pointer position (the only candidate sharing >= L tokens), which the
    lazy add puts in front; (iii) positions leave the window in time order, so the evicted one is
    the head of every list it is in, and those lists are exactly its nodes from ptr_min up to the
    first missing one (existence is monotone in depth: a list at L has a member registered at L,
    hence at L - 1); (iv) at the query, if no node exists at D + 1 but a window position y shares
    D + 1 tokens with p, then by (i) y did not register at D + 1, so no older window position
    shares D tokens with y: y is the head of the node at D; any other position of that node shares
    exactly D tokens with p, and the most recent is the tail. Together with the bound
    L*(p) <= L*(p - 1) + 1 (a match of length l at q gives one of length l - 1 at q - 1, inside
    W(p - 1)) this yields the exact (L(p), q)."""
    tok, ds = _check_stream(tok, doc_start)
    orders = _check_orders(orders)
    window = _check_window(window)
    ptr_min, ptr_max = _check_ptr(ptr_min, ptr_max)
    n = tok.shape[0]
    n_docs = ds.shape[0] - 1
    max_doc = int(np.diff(ds).max()) if n_docs > 0 else 0
    rsize = max_doc + 1 if window is None else min(window + max(ptr_max, MAX_ORDER) + 2, max_doc + 1)
    rsize = max(rsize, 1)
    out_n = np.zeros((len(orders), n), dtype=np.int32)
    out_g = np.zeros((len(orders), n), dtype=np.int32)
    out_l = np.zeros(n, dtype=np.int32)
    out_q = np.full(n, -1, dtype=np.int64)
    bad = np.zeros(max(n_docs, 1), dtype=np.int8)
    if n:
        _stream_kernel(tok, ds, np.asarray(orders, dtype=np.int64), -1 if window is None else window,
                       ptr_min, ptr_max, np.uint64(int(_hash_base) % int(M61)),
                       np.uint64(int(_hash_base2) % int(M61)), rsize, out_n, out_g, out_l, out_q, bad)
    n_fb = 0
    for d in np.flatnonzero(bad[:n_docs]):
        a, b = int(ds[d]), int(ds[d + 1])
        fb = doc_features(tok[a:b], np.array([0, b - a]), orders, window, ptr_min, ptr_max)
        out_n[:, a:b] = fb.n
        out_g[:, a:b] = fb.g
        out_l[a:b] = fb.ptr_len
        out_q[a:b] = np.where(fb.ptr_pos >= 0, fb.ptr_pos + a, -1)
        n_fb += 1
    cand = np.where(out_q >= 0, tok[np.maximum(out_q, 0)] if n else out_q, -1).astype(np.int32)
    return DocFeatures(orders=orders, window=window, ptr_min=ptr_min, ptr_max=ptr_max, n=out_n, g=out_g,
                       ptr_len=out_l, ptr_cand=cand, ptr_pos=out_q, n_fallback=int(n_fb))


def ns_per_token(tok, doc_start, window, repeats: int = 5, orders=(0, 1, 2, 3, 4), ptr_min: int = 5,
                 ptr_max: int = 64, return_all: bool = False):
    """Median wall time per token (ns) of `doc_features_stream` over `repeats` timed passes, after
    one untimed warm-up pass (JIT compilation). Single-threaded; the host is shared, so the value
    is a contended measurement and must be labelled as such. With return_all=True returns
    (median, list of per-pass ns per token)."""
    tok, ds = _check_stream(tok, doc_start)
    n = max(1, tok.shape[0])
    doc_features_stream(tok, ds, orders, window, ptr_min, ptr_max)
    per = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter_ns()
        doc_features_stream(tok, ds, orders, window, ptr_min, ptr_max)
        per.append((time.perf_counter_ns() - t0) / n)
    med = float(np.median(per))
    return (med, per) if return_all else med
