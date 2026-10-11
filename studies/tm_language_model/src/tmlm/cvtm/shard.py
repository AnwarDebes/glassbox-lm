"""Sharded Kneser-Ney counting (docs/11 section 11.6 "Sharded counting"; docs/13 A3 H8.8 (b)).

`fit_kn_sharded(tok, doc_start, shards=S, ...)` returns a `tmlm.eval.baselines.ngram.KNModel` that
must be IDENTICAL (keys, counts, continuation counts, per-context statistics, discounts, and hence
the suffix clauses of `tmlm.cvtm.suffix`) to the unsharded `tmlm.cvtm.kn_gpu.fit_kn` of the same
data. The partition is by a hash of the n-gram CONTEXT, so every statistic that is a function of
one context (its total, N1, N2, N3+, the followers) is computed inside one shard; everything else
is an integer sum across shards. This module does not import or change `kn_gpu.fit_kn`; it reuses
only its backend helpers, the input checks and the discount formula.

Context hash (a content hash of the context's token tuple, independent of the corpus it came from):

    th_0            = mix64(SHARD_SALT)                                   the empty context
    th_k(type g)    = mix64(th_(k-1)(context of g) XOR ((token of g + 1) * TOKEN_MUL))   (mod 2^64)
    shard(context)  = th(context) mod S

with `mix64` the SplitMix64 step of docs/11 section 3 (`tmlm.engine.rng.mix64`), a level-k type
g = (context c, token w) encoded as the key (id of c << 16) | w exactly as in `kn_gpu`, and the
level-1 types the tokens 0..V (V = BOS) with the empty context. Two occurrences of the same token
tuple get the same hash in any corpus; a hash collision between two different contexts only puts
them into the same shard (it cannot change a count), so correctness never depends on the hash.

Counting, level k = 2..K (the level-synchronous exchange of a multi-device count):

1. Routing: the key of the k-gram ending at stream position i is (ctx(i) << 16) | stream[i] with
   ctx(i) the id of the (k-1)-gram ending at i - 1 (ctx(0) = 0: the reference's wrap-around
   key[0] = stream[0]); position i goes to shard th_(k-1)(ctx(i)) mod S.
2. Each shard, independently (its own device arrays, its own stable sort): the sorted unique keys of
   its positions, the first occurrence of every type (its minimum stream position), the local
   inverse ids, the raw counts (bincount over its counting positions), the suffix id of every type
   (the id of the (k-1)-gram ending at its first occurrence) and the type hash th_k of its types.
3. Merge: the shards' unique keys are disjoint (a type's context fixes its shard; this is checked),
   so concatenating them and re-sorting gives the unsharded key list; local ids map to global ranks,
   and every per-type array is permuted with them (docs/11 11.6: "counted independently,
   concatenated and re-sorted").

Level 1 has one context (the empty one), so its unigram counts live in the shard of th_0.
Continuation counts, top down: cont_k[g] = number of level-(k+1) types with a positive count whose
suffix is g; every shard contributes the bincount over its own level-(k+1) types and the partial
vectors are summed (an all-reduce of integers). Level statistics: every shard computes, for the
contexts it owns, the exact int64 total, N1, N2, N3+ and N_any of their followers, and its counts
of counts (types with count 1, 2, 3, 4); the per-context arrays are scattered into the global arrays
(contexts without followers stay 0) and the counts of counts are summed before
`kn_gpu.discounts_from_counts_of_counts` (the same float operations as the reference), so the
discounts are bit-identical whenever the integer sums are.

Pruning (`min_counts`) is not part of the sharded fit: H8.8 (b) compares the unpruned counting.

`fit_kn_sharded(...).fit_info` holds the per-phase wall times, the number of positions and types per
shard and level (load balance), the peak CuPy pool bytes, and the hash definition.
`compare_suffix_clauses(a, b)` lists every difference between two `SuffixClauses` (integer and
float64 arrays, bit for bit); `kn_gpu.compare_models` does the same for two KNModels.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..engine.rng import mix64 as _mix64_scalar
from ..eval.baselines.ngram import KNModel, _Level
from .kn_gpu import _Backend, _check_inputs, _Phases, discounts_from_counts_of_counts

SHARD_SALT = 0x5EED_C0DE_5A4D_0011           # th_0 = mix64(SHARD_SALT), the empty context
TOKEN_MUL = 0xD6E8FEB86659FD93               # the stream multiplier of docs/11 section 3
_GOLD = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
_SHIFT = 16
_MASK16 = 0xFFFF


# ------------------------------------------------------------------ hashing
def mix64_array(xp: Any, z: Any) -> Any:
    """Vectorised `tmlm.engine.rng.mix64` over a uint64 array of numpy or CuPy (wraps mod 2^64)."""
    u = xp.uint64
    with np.errstate(over="ignore"):
        z = z + u(_GOLD)
        z = (z ^ (z >> u(30))) * u(_C1)
        z = (z ^ (z >> u(27))) * u(_C2)
        return z ^ (z >> u(31))


def empty_context_hash() -> int:
    """th_0 (python int)."""
    return int(_mix64_scalar(SHARD_SALT))


def type_hash(xp: Any, th_ctx: Any, tokens: Any) -> Any:
    """th of the types (context hash th_ctx, token) : mix64(th_ctx XOR ((token + 1) * TOKEN_MUL))."""
    u = xp.uint64
    t = xp.asarray(tokens).astype(u)
    with np.errstate(over="ignore"):
        return mix64_array(xp, xp.asarray(th_ctx, dtype=u) ^ ((t + u(1)) * u(TOKEN_MUL)))


def unigram_type_hash(xp: Any, vocab: int) -> Any:
    """th_1 [V + 1]: the hash of every level-1 type (token 0..V with the empty context)."""
    return type_hash(xp, xp.full(vocab + 1, empty_context_hash(), dtype=xp.uint64), xp.arange(vocab + 1))


def tuple_hash(tokens) -> int:
    """th of one token tuple (t_1, ..., t_k) (python ints; the context of a level-(k+1) type)."""
    h = np.uint64(empty_context_hash())
    for t in tokens:
        h = type_hash(np, np.asarray([h], dtype=np.uint64), np.asarray([int(t)]))[0]
    return int(h)


def shard_of_hash(xp: Any, th: Any, shards: int) -> Any:
    """int8 shard ids th mod S."""
    return (th % xp.uint64(int(shards))).astype(xp.int8)


# ------------------------------------------------------------------ counting pass
def _stream(be: _Backend, tok: np.ndarray, ds: np.ndarray, n: int, vocab: int, pad: bool) -> tuple[Any, Any, int]:
    """The padded stream of `KNModel._padded_stream` on the backend (the stream phase of kn_gpu)."""
    xp = be.xp
    n_tok = int(tok.shape[0])
    n_doc = int(ds.shape[0]) - 1
    length = (n - 1) + n_tok + ((n - 1) * n_doc if pad else 0)
    if length >= 2**31 - 1:
        raise ValueError("stream too long for int32 type ids (2**31 positions)")
    stream = xp.full(length, vocab, dtype=xp.int32)
    real = xp.zeros(length, dtype=bool)
    if n_tok:
        idx = xp.arange(n_tok, dtype=xp.int64)
        ds_d = be.dev(ds)
        doc_of = xp.searchsorted(ds_d, idx, side="right") - 1
        if pad:
            pos = idx + (n - 1) * (doc_of + 2)
            stream[pos] = be.dev(tok).astype(xp.int32)
            real[pos] = True
        else:
            pos = idx + (n - 1)
            stream[pos] = be.dev(tok).astype(xp.int32)
            ok = (idx - ds_d[doc_of]) >= (n - 1)
            real[pos[ok]] = True
            del ok
        del idx, ds_d, doc_of, pos
    return stream, real, length


def _count_shard(be: _Backend, pos_s: Any, stream: Any, real: Any, ids_prev: Any, th_prev: Any) -> dict:
    """One shard of one level, on its own device arrays (released before the next shard; the results
    are staged on the host until the merge): the keys of its positions, their sorted unique keys,
    the local inverse ids, the raw counts, the suffix ids (global ids of the previous level at the
    first occurrence) and the type hashes."""
    xp = be.xp
    m = int(pos_s.shape[0])
    ctx = xp.where(pos_s == 0, 0, ids_prev[xp.maximum(pos_s - 1, 0)]).astype(xp.uint64)
    key_s = (ctx << xp.uint64(_SHIFT)) | stream[pos_s].astype(xp.uint64)
    del ctx
    real_s = real[pos_s]
    perm = be.argsort_stable(key_s)
    aux = key_s[perm]
    del key_s
    mask = xp.empty(m, dtype=bool)
    if m:
        mask[:1] = True
        mask[1:] = aux[1:] != aux[:-1]
    ukeys = aux[mask]
    del aux
    first_pos = pos_s[perm[mask]]
    grp = xp.cumsum(mask, dtype=xp.int32)
    grp -= 1
    inv_local = xp.empty(m, dtype=xp.int32)
    inv_local[perm] = grp
    del grp, perm, mask
    n_u = int(ukeys.shape[0])
    raw = be.bincount(inv_local[real_s], n_u)
    del real_s
    sfx = ids_prev[first_pos].astype(xp.int32)
    del first_pos
    th = type_hash(xp, th_prev[(ukeys >> xp.uint64(_SHIFT)).astype(xp.int64)], ukeys & xp.uint64(_MASK16))
    out = {"pos": be.host(pos_s), "ukeys": be.host(ukeys), "inv_local": be.host(inv_local), "raw": be.host(raw),
           "sfx": be.host(sfx), "th": be.host(th)}
    del ukeys, inv_local, raw, sfx, th
    return out


def _sharded_count_pass(be: _Backend, tok: np.ndarray, ds: np.ndarray, n: int, vocab: int, pad: bool,
                        shards: int, ph: _Phases) -> dict:
    xp = be.xp
    S = int(shards)
    with ph("stream"):
        stream, real, length = _stream(be, tok, ds, n, vocab, pad)
        n_train_tokens = int(real.sum())
    keys: list = [None] * (n + 1)
    raw: list = [None] * (n + 1)
    sfx: list = [None] * (n + 1)
    tshard: list = [None] * (n + 1)
    balance: dict = {}
    s0 = int(empty_context_hash() % S)
    with ph("level1"):
        keys[1] = np.arange(vocab + 1, dtype=np.uint64)
        raw[1] = be.host(be.bincount(stream[real], vocab + 1))       # counted in the shard of the empty context
        tshard[1] = np.full(vocab + 1, s0, dtype=np.int8)
        th_prev = unigram_type_hash(xp, vocab)
    balance["level1"] = {"shard_of_empty_context": s0, "positions": int(length), "types": int(vocab + 1)}
    ids_prev = stream
    for k in range(2, n + 1):
        with ph(f"level{k}.route"):
            # the shard of every previous-level type (a context of level k), then of every position
            ctx_shard = shard_of_hash(xp, th_prev, S)
            ctx = xp.empty(length, dtype=xp.int32)
            if length:
                ctx[:1] = 0
                ctx[1:] = ids_prev[:-1]
            sh = ctx_shard[ctx]
            del ctx, ctx_shard
        parts: list = []
        t_shard: list[float] = []
        n_pos: list[int] = []
        for s in range(S):
            be.sync()
            t0 = time.perf_counter()
            with ph(f"level{k}.shard_count"):
                pos_s = xp.flatnonzero(sh == s)
                n_pos.append(int(pos_s.shape[0]))
                parts.append(_count_shard(be, pos_s, stream, real, ids_prev, th_prev))
                del pos_s
                be.free()
            t_shard.append(time.perf_counter() - t0)
        del sh
        with ph(f"level{k}.merge"):
            n_s = [int(p["ukeys"].shape[0]) for p in parts]
            offs = np.concatenate([[0], np.cumsum(n_s)]).astype(np.int64)
            n_k = int(offs[-1])
            allk = be.dev(np.concatenate([p["ukeys"] for p in parts]))
            order = be.argsort_stable(allk)
            keys_k = allk[order]
            del allk
            if n_k > 1 and not bool((keys_k[1:] > keys_k[:-1]).all()):
                raise RuntimeError(f"level {k}: a type was counted in two shards (the routing is not a function "
                                   "of the context)")
            keys[k] = be.host(keys_k)
            del keys_k
            rank = xp.empty(n_k, dtype=xp.int32)
            rank[order] = xp.arange(n_k, dtype=xp.int32)

            def permuted(name: str) -> Any:
                a = be.dev(np.concatenate([p[name] for p in parts]))
                r = a[order]
                del a
                return r

            raw[k] = be.host(permuted("raw"))
            sfx[k] = be.host(permuted("sfx")).astype(np.int32, copy=False)
            th_k = permuted("th")
            tshard[k] = be.host(xp.concatenate([xp.full(n_s[s], s, dtype=xp.int8) for s in range(S)])[order])
            del order
            inv = xp.empty(length, dtype=xp.int32)
            for s in range(S):
                inv[be.dev(parts[s]["pos"])] = rank[int(offs[s]) + be.dev(parts[s]["inv_local"]).astype(xp.int64)]
            del rank, parts
        balance[f"level{k}"] = {"positions": n_pos,
                                "types": [int(x) for x in np.bincount(tshard[k].astype(np.int64), minlength=S)],
                                "shard_count_s": [round(x, 4) for x in t_shard]}
        ids_prev = inv
        th_prev = th_k
        del inv, th_k
        be.free()
    del ids_prev, th_prev, stream, real
    be.free()
    return {"keys": keys, "raw": raw, "sfx": sfx, "tshard": tshard, "n_train_tokens": n_train_tokens,
            "stream_length": length, "balance": balance}


def _sharded_continuation(be: _Backend, raw: list, sfx: list, tshard: list, n_types: list[int], n: int,
                          shards: int) -> list:
    """cont[k] (k = n-1..1) as the sum over shards of the bincount of the suffixes of the shard's own
    level-(k+1) types with a positive count (raw at the top level, continuation below)."""
    xp = be.xp
    cont: list = [None] * (n + 1)
    if n < 2:
        return cont
    upper = raw[n]
    for k in range(n - 1, 0, -1):
        up = be.dev(upper)
        sf = be.dev(sfx[k + 1]).astype(xp.int64)
        ts = be.dev(tshard[k + 1])
        acc = xp.zeros(n_types[k], dtype=xp.int64)
        for s in range(int(shards)):
            sel = (ts == s) & (up > 0)
            acc += be.bincount(sf[sel], n_types[k])
            del sel
        del up, sf, ts
        cont[k] = be.host(acc)
        upper = cont[k]
        del acc
        be.free()
    return cont


def _sharded_level_stats(be: _Backend, k: int, keys_k: np.ndarray, n_ctx: int, cnt_host: np.ndarray,
                         tshard_k: np.ndarray, shards: int, discount_mode: str, discount: float) -> _Level:
    """The statistics of `KNModel._level_stats`, computed per shard over the contexts it owns."""
    xp = be.xp
    if k == 1:
        n_ctx = 1
    cnt = be.dev(cnt_host).astype(xp.int64, copy=False)
    ctx_all = (xp.zeros(cnt.shape[0], dtype=xp.int64) if k == 1
               else (be.dev(keys_k) >> xp.uint64(_SHIFT)).astype(xp.int64))
    ts = be.dev(tshard_k)
    total_i = xp.zeros(n_ctx, dtype=xp.int64)
    n1 = xp.zeros(n_ctx, dtype=xp.int32)
    n2 = xp.zeros(n_ctx, dtype=xp.int32)
    n3 = xp.zeros(n_ctx, dtype=xp.int32)
    coc = [0, 0, 0, 0]
    cmax = 0
    for s in range(int(shards)):
        idx = xp.flatnonzero(ts == s)
        m = int(idx.shape[0])
        if m == 0:
            continue
        c = cnt[idx]
        x = ctx_all[idx]                   # ascending: keys are sorted and idx is ascending
        del idx
        start = xp.empty(m, dtype=bool)
        start[:1] = True
        start[1:] = x[1:] != x[:-1]
        seg_ctx = x[start]
        n_seg = int(seg_ctx.shape[0])
        seg_id = xp.cumsum(start, dtype=xp.int64) - 1
        csum = xp.zeros(m + 1, dtype=xp.int64)
        xp.cumsum(c, out=csum[1:])
        b = xp.concatenate([xp.flatnonzero(start), xp.asarray([m], dtype=xp.int64)])
        total_i[seg_ctx] = csum[b[1:]] - csum[b[:-1]]
        n1[seg_ctx] = be.bincount(seg_id[c == 1], n_seg).astype(xp.int32)
        n2[seg_ctx] = be.bincount(seg_id[c == 2], n_seg).astype(xp.int32)
        n3[seg_ctx] = be.bincount(seg_id[c >= 3], n_seg).astype(xp.int32)
        for j in range(4):
            coc[j] += int((c == j + 1).sum())
        cmax = max(cmax, int(c.max()))
        del c, x, start, seg_ctx, seg_id, csum, b
    n_any = (n1 + n2 + n3).astype(xp.int32)
    if discount_mode == "modified":
        d = discounts_from_counts_of_counts(*coc)
    else:
        d = (float(discount),) * 3
    if cmax >= 2**31 - 1:
        raise OverflowError("counts do not fit int32")
    lv = _Level(cnt=be.host(cnt.astype(xp.int32)), total=be.host(total_i.astype(xp.float64)), n1=be.host(n1),
                n2=be.host(n2), n3=be.host(n3), n_any=be.host(n_any), d=d)
    del cnt, ctx_all, ts, total_i, n1, n2, n3, n_any
    return lv


# ------------------------------------------------------------------ public API
def fit_kn_sharded(tok: np.ndarray, doc_start: np.ndarray, order: int = 5, vocab: int | None = None,
                   shards: int = 4, backend: str = "cupy", min_counts=None, *, discount_mode: str = "modified",
                   discount: float = 0.75, unk_interp: bool = True, pad: bool = True,
                   max_device_bytes: int | None = None) -> KNModel:
    """`KNModel(order, vocab, discount_mode, discount, unk_interp, pad)` fitted on (tok, doc_start) by
    S = `shards` context-hash shards (module docstring). backend "cupy" (GPU) or "numpy" (host)."""
    if vocab is None:
        raise ValueError("vocab is required")
    if min_counts is not None:
        raise NotImplementedError("the sharded fit covers the unpruned counting only (docs/13 A3 H8.8 b)")
    S = int(shards)
    if not 1 <= S <= 127:
        raise ValueError("shards must lie in 1..127 (int8 shard ids)")
    kn = KNModel(order, vocab, discount_mode=discount_mode, discount=discount, unk_interp=unk_interp, pad=pad)
    n = kn.order
    tok, ds = _check_inputs(tok, doc_start, vocab)
    t_all = time.perf_counter()
    be = _Backend(backend, max_device_bytes)
    ph = _Phases(be)
    try:
        cp = _sharded_count_pass(be, tok, ds, n, vocab, pad, S, ph)
        keys, raw, sfx, tshard = cp["keys"], cp["raw"], cp["sfx"], cp["tshard"]
        n_types = [0] + [int(keys[k].shape[0]) for k in range(1, n + 1)]
        with ph("cont_counts"):
            cont = _sharded_continuation(be, raw, sfx, tshard, n_types, n, S)
        for k in range(1, n + 1):
            sfx[k] = None
        with ph("level_stats"):
            for k in range(1, n + 1):
                n_ctx = int(keys[k - 1].shape[0]) if k >= 2 else 1
                kn.keys[k] = keys[k]
                kn.n_types[k] = int(keys[k].shape[0])
                kn.raw[k] = _sharded_level_stats(be, k, keys[k], n_ctx, raw[k], tshard[k], S, kn.discount_mode,
                                                 kn.discount)
                raw[k] = None
                be.free()
                kn.cont[k] = (_sharded_level_stats(be, k, keys[k], n_ctx, cont[k], tshard[k], S, kn.discount_mode,
                                                   kn.discount) if k < n else None)
                cont[k] = None
                be.free()
        kn.n_train_tokens = int(cp["n_train_tokens"])
        balance = cp["balance"]
        del cp, keys, raw, cont, sfx, tshard
    finally:
        be.close()
    kn.fit_info = {
        "backend": backend,
        "shards": S,
        "phases_s": dict(ph.seconds),
        "total_s": time.perf_counter() - t_all,
        "peak_pool_bytes": int(be.peak_pool_bytes),
        "pool_note": "bytes held by CuPy's default memory pool, sampled before every release; excludes the CUDA context",
        "balance": balance,
        "partition": ("level-k types and positions go to shard th_(k-1)(context) mod S; th_0 = mix64(SHARD_SALT), "
                      "th_k(c, w) = mix64(th_(k-1)(c) XOR ((w + 1) * TOKEN_MUL)); level 1 in the shard of th_0"),
        "shard_salt": hex(SHARD_SALT),
        "pruning": None,
    }
    return kn


def type_shards(kn: KNModel, shards: int) -> list:
    """The shard of every type of every level of a fitted KNModel (int8 arrays, [K+1] list), recomputed
    from the keys alone with the context hash; equal to the routing of `fit_kn_sharded` (tested)."""
    S = int(shards)
    out: list = [None] * (kn.order + 1)
    V = int(kn.vocab)
    out[1] = np.full(V + 1, empty_context_hash() % S, dtype=np.int8)
    th_prev = unigram_type_hash(np, V)
    for k in range(2, kn.order + 1):
        ctx = (kn.keys[k] >> np.uint64(_SHIFT)).astype(np.int64)
        out[k] = shard_of_hash(np, th_prev[ctx], S)
        th_prev = type_hash(np, th_prev[ctx], kn.keys[k] & np.uint64(_MASK16))
    return out


# ------------------------------------------------------------------ comparison of suffix clauses
_SC_LIST_FIELDS = ("keys", "fires", "d", "v", "dq", "vq")
_SC_ARRAY_FIELDS = ("b", "bq", "p1")


def compare_suffix_clauses(a, b) -> list[str]:
    """Every difference between two `SuffixClauses`: order, vocab, and bit for bit every per-level
    array (keys, fires, d, v, dq, vq) and the bias arrays (b, bq, p1); empty list: identical votes."""
    diffs: list[str] = []
    if a.order != b.order or a.vocab != b.vocab:
        return [f"order/vocab differ: ({a.order}, {a.vocab}) != ({b.order}, {b.vocab})"]

    def cmp(name: str, x, y) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape:
            diffs.append(f"{name}: {x.dtype}{x.shape} != {y.dtype}{y.shape}")
        elif x.dtype.kind == "f":
            # bit for bit (the same bytes), which also compares -inf and signed zeros exactly
            if not np.array_equal(x.view(np.uint64 if x.itemsize == 8 else np.uint32),
                                  y.view(np.uint64 if y.itemsize == 8 else np.uint32)):
                diffs.append(f"{name}: float bits differ at {int(np.count_nonzero(x.view(np.uint64) != y.view(np.uint64)))} entries")
        elif not np.array_equal(x, y):
            diffs.append(f"{name}: values differ at {int(np.count_nonzero(x != y))} entries")

    for f in _SC_LIST_FIELDS:
        la, lb = getattr(a, f), getattr(b, f)
        for j in range(2 if f != "keys" else 1, a.order + 1):
            if la[j] is None and lb[j] is None:
                continue
            if la[j] is None or lb[j] is None:
                diffs.append(f"{f}[{j}]: present in one only")
                continue
            cmp(f"{f}[{j}]", la[j], lb[j])
    for f in _SC_ARRAY_FIELDS:
        cmp(f, getattr(a, f), getattr(b, f))
    if a.state_hash() != b.state_hash():
        diffs.append("state_hash differs")
    return diffs
