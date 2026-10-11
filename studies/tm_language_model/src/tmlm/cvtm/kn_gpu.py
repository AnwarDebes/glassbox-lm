"""Kneser-Ney count fit on the GPU (docs/11 section 11.2 and 11.6; docs/13 A3 H8.8 and H8.9).

`fit_kn` returns a `tmlm.eval.baselines.ngram.KNModel` whose arrays are IDENTICAL (same dtypes,
same values, same discounts to the last bit) to `KNModel(order, vocab, ...).fit(tok, doc_start)`
when `min_counts` is None. It follows `KNModel.fit` step by step, on an array backend `xp`
(`backend="cupy"`: CuPy on the GPU; `backend="numpy"`: the same code on the host):

1. The padded stream of `KNModel._padded_stream`: n-1 global BOS tokens, and with `pad` n-1 BOS
   tokens before every document; `real` marks the positions that carry counts.
2. Per level k = 2..n: the key of the k-gram ending at every position,
   `(id of the (k-1)-gram ending one position earlier) << 16 | token` (`key[0] = stream[0]`, the
   same wrap-around as the reference), a STABLE argsort of the keys (CuPy's 1-D argsort is
   thrust::stable_sort_by_key, a radix sort; numpy `kind="stable"`), the run starts of the sorted
   keys (unique keys, the first occurrence of every type, the inverse ids), and the raw counts
   `bincount(ids[real])`. The suffix of every level-k type is `ids[k-1][first occurrence]`, as in
   the reference's continuation step.
3. Continuation counts, top down: cont_k[g] = number of level-(k+1) types with a positive count
   (raw at the top level, continuation below) whose suffix is g.
4. Level statistics of `KNModel._level_stats`: totals as exact integer segment sums over the
   sorted keys (converted to float64; the reference's float64 bincount of integer counts is exact
   below 2**53, so the values are the same), n1, n2, n3+ and n_any per context, and the modified
   discounts from the counts of counts with the float operations of
   `ngram.modified_discounts` in the same order (`discounts_from_counts_of_counts`).

Differences to the reference that are not visible in the result: inverse ids are int32 (the
reference holds int64; the values are the same, and `fit_kn` refuses streams of 2**31 positions or
more), totals are summed as integers, and only the ids of the previous level are kept (the
reference keeps the ids of every level on the host). Integer counting on the GPU uses atomic adds
of integers, so every array is deterministic and independent of thread order.

Pruning (`min_counts`, docs/13 A3 H8.9). `min_counts` maps an order k (3 <= k <= order) to the
minimum RAW count m_k of a type at that order; unlisted orders have m_k = 0 (no pruning), and
m_3 <= m_4 <= ... <= m_order must hold (non-decreasing; orders 1 and 2 are never pruned). The
semantics, applied to the unpruned fit:

- Kept types, top down: a level-k type is kept iff raw >= m_k, OR it is the context (prefix) of a
  kept level-(k+1) type. The second clause is needed because keys encode their context by index:
  it keeps the all-BOS contexts (raw count 0: BOS positions carry no count) that start every
  document. Because the thresholds are non-decreasing and the raw count of a suffix is at least
  the raw count of the type, the suffix of every type kept by its count is kept too, so the pruned
  type set stays prefix-closed and suffix-closed (checked: a violation raises RuntimeError).
- Keys are re-encoded with the new (order-preserving) context indices, so they stay sorted.
- Counts of the kept types: raw counts are data and do not change. The continuation counts of
  the lower orders depend on the type inventory of the order above. With the default
  `prune_cont="full"` every kept type keeps the continuation count of the unpruned inventory
  (KenLM-style: continuation counts are counted before pruning). `prune_cont="recount"`
  RECOMPUTES them from the kept types: cont_k[g] = number of KEPT level-(k+1) types with a
  positive (recomputed) count whose suffix is g, cascading down to the unigram level; this
  collapses the lower orders (it removes most of the continuation mass once the singletons of the
  top order are gone) and is kept only for comparison.
- Mass of the pruned followers (`prune_mass`). Default `"backoff"` (the SRILM/KenLM rule, needs
  `prune_cont="full"`): the pruned model is the unpruned model with the pruned followers removed
  and their probability mass moved to the interpolation weight. Per context u of a pruned level:
  the total stays the UNPRUNED total T(u), the discounts stay those of the unpruned fit,
  alpha(u, w) = (c - D(c)) / T(u) for every kept follower, and
  gamma(u) = (D-mass of the kept followers + count mass of the pruned followers) / T(u). The
  pruned count mass is stored per context as the float64 array `extra` of a `_PrunedLevel`
  (a `_Level` whose `gamma_mass` adds it), so `KNModel.prob`, `KNModel.dist` and
  `tmlm.cvtm.suffix.SuffixClauses.from_kn` evaluate this estimator unchanged; in the vote form it
  is folded into the one clause constant ln gamma(u) per context. `"renormalize"`: totals, n1, n2,
  n3+, n_any and the discounts are recomputed from the kept types and their counts, level by level,
  with the functions of the unpruned fit (the pruned mass is spread over the kept followers). With
  the singletons of an order removed, its n1 is 0 and the recomputed modified discounts collapse
  to the clamp values (1, 2, 3); other orders can get D1 = 0 and hence contexts with gamma = 0
  (zero probability for every non-follower, which `SuffixClauses.from_kn` refuses). KN-5 fitted on
  the training split without its last 200 documents and scored on those 200 held-out TRAINING
  documents, with {3: 2, 4: 2, 5: 2} (22 percent of the unpruned clause bytes), the loss over the
  unpruned model is about 0.194 bits per byte (recount), 0.094 (full, renormalize) and 0.054 (full,
  backoff) [unverified: scratch smoke of the stage-1 review of 2026-09-24, not under run_logged.py;
  no validation or test data were read].
- Every context's distribution sums to one (`KNModel.dist` rows sum to 1) in every mode, and the
  context of every kept type is present (its prefix index is valid).
- A pruned type is absent from `keys`: a query that needs it sees count 0, exactly as for an
  unseen n-gram. A kept type may end up with count 0 at its level (all its extensions pruned
  under "recount"); it stays in `keys` (it is only removed by its raw count).

`fit_kn(...).fit_info` holds the per-phase wall times (seconds, device synchronised at every
phase boundary), the backend, the peak bytes held by CuPy's memory pool (sampled before every
release; the CUDA context itself is not included), and the pruning report.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from ..common import hashing
from ..eval.baselines.ngram import KNModel, _Level

PRUNE_CONT_MODES = ("full", "recount")
PRUNE_MASS_MODES = ("backoff", "renormalize")
_SHIFT = np.uint64(16)
_MASK16 = np.uint64(0xFFFF)
_LEVEL_FIELDS = ("cnt", "total", "n1", "n2", "n3", "n_any")


@dataclass
class _PrunedLevel(_Level):
    """A pruned level under `prune_mass="backoff"` (module docstring): `total` is the unpruned total
    of every context, `d` the unpruned discounts, n1/n2/n3/n_any count the KEPT followers, and
    `extra` (float64 [n_ctx]) is the count mass of the pruned followers, added to the gamma mass."""

    extra: np.ndarray | None = None

    def gamma_mass(self, ctx: np.ndarray, single: float | None) -> np.ndarray:
        c = np.where(ctx >= 0, ctx, 0)
        return np.where(ctx >= 0, super().gamma_mass(ctx, single) + self.extra[c], 0.0)


# ------------------------------------------------------------------ discounts
def discounts_from_counts_of_counts(n1: int, n2: int, n3: int, n4: int) -> tuple[float, float, float]:
    """(D1, D2, D3+) of Chen and Goodman 1999 from the numbers of types with count 1, 2, 3 and 4:
    the float operations of `ngram.modified_discounts`, in the same order (bit-identical)."""
    n1 = float(n1)
    n2 = float(n2)
    n3 = float(n3)
    n4 = float(n4)
    y = n1 / (n1 + 2.0 * n2) if (n1 + 2.0 * n2) > 0 else 0.0

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    d1 = 1.0 - 2.0 * y * ratio(n2, n1)
    d2 = 2.0 - 3.0 * y * ratio(n3, n2)
    d3 = 3.0 - 4.0 * y * ratio(n4, n3)
    return (min(max(d1, 0.0), 1.0), min(max(d2, 0.0), 2.0), min(max(d3, 0.0), 3.0))


# ------------------------------------------------------------------ backend
class _Backend:
    """numpy or CuPy, with host/device transfers, synchronisation and pool accounting."""

    def __init__(self, name: str, max_device_bytes: int | None = None) -> None:
        if name not in ("numpy", "cupy"):
            raise ValueError(f"backend must be 'numpy' or 'cupy', got {name!r}")
        self.name = name
        self.gpu = name == "cupy"
        self.peak_pool_bytes = 0
        self._old_limit = None
        if self.gpu:
            import cupy

            self.xp = cupy
            self.pool = cupy.get_default_memory_pool()
            # host-to-device copies of pageable arrays go through CuPy's pinned staging pool, which
            # keeps its blocks cached (a smoke run, not logged, saw GiBs of host RSS held this way
            # after a full-split fit); free() drains it
            self.pinned = cupy.get_default_pinned_memory_pool()
            self.pool.free_all_blocks()
            if max_device_bytes is not None:
                self._old_limit = self.pool.get_limit()
                self.pool.set_limit(size=int(max_device_bytes))
        else:
            self.xp = np
            self.pool = None

    def close(self) -> None:
        self.free()
        if self.gpu and self._old_limit is not None:
            self.pool.set_limit(size=self._old_limit)

    def sync(self) -> None:
        if self.gpu:
            self.xp.cuda.Device().synchronize()

    def free(self) -> None:
        """Record the pool size (the peak since the last release), then release cached device blocks
        and cached pinned host blocks."""
        if self.gpu:
            self.sync()
            self.peak_pool_bytes = max(self.peak_pool_bytes, int(self.pool.total_bytes()))
            self.pool.free_all_blocks()
            self.pinned.free_all_blocks()

    def dev(self, a: Any) -> Any:
        return self.xp.asarray(a)

    def host(self, a: Any) -> np.ndarray:
        if self.gpu and not isinstance(a, np.ndarray):
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def argsort_stable(self, a: Any) -> Any:
        if self.gpu:
            return self.xp.argsort(a)          # 1-D: thrust::stable_sort_by_key (radix, stable)
        return np.argsort(a, kind="stable")

    def bincount(self, x: Any, n: int) -> Any:
        """int64 [n] counts of the non-negative integers `x` (empty input allowed)."""
        xp = self.xp
        if x.shape[0] == 0:
            return xp.zeros(n, dtype=xp.int64)
        return xp.bincount(x, minlength=n).astype(xp.int64, copy=False)


class _Phases:
    def __init__(self, be: _Backend) -> None:
        self.be = be
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        self.be.sync()
        t0 = time.perf_counter()
        yield
        self.be.sync()
        self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - t0)


# ------------------------------------------------------------------ counting pass
def _check_inputs(tok: np.ndarray, doc_start: np.ndarray, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    tok = np.asarray(tok)
    ds = np.asarray(doc_start).astype(np.int64)
    if tok.ndim != 1 or ds.ndim != 1 or ds.shape[0] < 1:
        raise ValueError("tok and doc_start must be 1-D, doc_start with at least one entry")
    if int(ds[0]) != 0 or int(ds[-1]) != tok.shape[0] or bool(np.any(np.diff(ds) < 0)):
        raise ValueError("doc_start must start at 0, end at len(tok) and be non-decreasing")
    if tok.shape[0] and (int(tok.min()) < 0 or int(tok.max()) >= vocab):
        raise ValueError("token ids must lie in [0, vocab)")
    return tok, ds


def _count_pass(be: _Backend, tok: np.ndarray, ds: np.ndarray, n: int, vocab: int, pad: bool,
                ph: _Phases) -> dict:
    """Keys, raw counts and first-occurrence suffixes of every level, continuation counts."""
    xp = be.xp
    n_tok = int(tok.shape[0])
    n_doc = int(ds.shape[0]) - 1
    length = (n - 1) + n_tok + ((n - 1) * n_doc if pad else 0)
    if length >= 2**31 - 1:
        raise ValueError("stream too long for int32 type ids (2**31 positions)")
    with ph("stream"):
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
        n_train_tokens = int(real.sum())
    keys: list = [None] * (n + 1)
    raw: list = [None] * (n + 1)
    suffix: list = [None] * (n + 1)
    with ph("level1"):
        keys[1] = np.arange(vocab + 1, dtype=np.uint64)
        raw[1] = be.host(be.bincount(stream[real], vocab + 1))
    ids_prev = stream
    for k in range(2, n + 1):
        with ph(f"level{k}.sort"):
            key = xp.empty(length, dtype=xp.uint64)
            key[:1] = stream[:1].astype(xp.uint64)
            if length > 1:
                tmp = ids_prev[:-1].astype(xp.uint64)
                xp.left_shift(tmp, _SHIFT, out=tmp)
                xp.bitwise_or(tmp, stream[1:].astype(xp.uint64), out=tmp)
                key[1:] = tmp
                del tmp
            perm = be.argsort_stable(key)
            aux = key[perm]
            del key
        with ph(f"level{k}.unique"):
            mask = xp.empty(length, dtype=bool)
            mask[:1] = True
            if length > 1:
                mask[1:] = aux[1:] != aux[:-1]
            keys_k = aux[mask]
            del aux
            first = perm[mask]
            grp = xp.cumsum(mask, dtype=xp.int32)
            grp -= 1
            del mask
            inv = xp.empty(length, dtype=xp.int32)
            inv[perm] = grp
            del grp, perm
            sfx = ids_prev[first]
            del first
        with ph(f"level{k}.counts"):
            rk = be.bincount(inv[real], int(keys_k.shape[0]))
        with ph("to_host"):
            keys[k] = be.host(keys_k)
            raw[k] = be.host(rk)
            suffix[k] = be.host(sfx).astype(np.int32, copy=False)
        del keys_k, rk, sfx
        ids_prev = inv
        be.free()
    del ids_prev, stream, real
    be.free()
    with ph("cont_counts"):
        cont = _continuation(be, raw, suffix, [int(keys[k].shape[0]) if keys[k] is not None else 0
                                               for k in range(n + 1)], n)
    be.free()
    return {"keys": keys, "raw": raw, "suffix": suffix, "cont": cont, "n_train_tokens": n_train_tokens,
            "stream_length": length}


def _continuation(be: _Backend, top: list, suffix: list, n_types: list[int], n: int,
                  check_suffix: bool = False) -> list:
    """cont[k] (k = n-1..1): number of level-(k+1) types with a positive count whose suffix is g.
    `top[n]` is the top-level raw count; below, the continuation count just computed is used."""
    xp = be.xp
    cont: list = [None] * (n + 1)
    if n < 2:
        return cont
    upper = be.dev(top[n])
    for k in range(n - 1, 0, -1):
        live = upper > 0
        sfx = be.dev(suffix[k + 1])[live]
        if check_suffix and sfx.shape[0] and int(sfx.min()) < 0:
            raise RuntimeError(f"a kept level-{k + 1} type with a positive count lost its suffix: "
                               "the pruned type set is not suffix-closed")
        ck = be.bincount(sfx.astype(xp.int64, copy=False), n_types[k])
        del sfx, live
        cont[k] = be.host(ck)
        upper = ck
    del upper
    return cont


# ------------------------------------------------------------------ level statistics
def _level_stats(be: _Backend, k: int, keys_k: np.ndarray, n_ctx: int, cnt_host: np.ndarray,
                 discount_mode: str, discount: float) -> _Level:
    """`KNModel._level_stats` on the backend."""
    xp = be.xp
    cnt = be.dev(cnt_host).astype(xp.int64, copy=False)
    if k == 1:
        n_ctx = 1
        ctx = xp.zeros(cnt.shape[0], dtype=xp.int64)
    else:
        ctx = (be.dev(keys_k) >> _SHIFT).astype(xp.int64)
    # totals: exact integer sums over the runs of equal context (keys are sorted by context)
    csum = xp.zeros(cnt.shape[0] + 1, dtype=xp.int64)
    xp.cumsum(cnt, out=csum[1:])
    bounds = xp.searchsorted(ctx, xp.arange(n_ctx + 1, dtype=xp.int64), side="left")
    total = (csum[bounds[1:]] - csum[bounds[:-1]]).astype(xp.float64)
    del csum, bounds
    n1 = be.bincount(ctx[cnt == 1], n_ctx).astype(xp.int32)
    n2 = be.bincount(ctx[cnt == 2], n_ctx).astype(xp.int32)
    n3 = be.bincount(ctx[cnt >= 3], n_ctx).astype(xp.int32)
    n_any = (n1 + n2 + n3).astype(xp.int32)
    if discount_mode == "modified":
        coc = [int((cnt == c).sum()) for c in (1, 2, 3, 4)]
        d = discounts_from_counts_of_counts(*coc)
    else:
        d = (float(discount),) * 3
    if int(cnt.max()) >= 2**31 - 1:
        raise OverflowError("counts do not fit int32")
    lv = _Level(cnt=be.host(cnt.astype(xp.int32)), total=be.host(total), n1=be.host(n1), n2=be.host(n2),
                n3=be.host(n3), n_any=be.host(n_any), d=d)
    del cnt, ctx, total, n1, n2, n3, n_any
    return lv


# ------------------------------------------------------------------ pruning
def _thresholds(min_counts: Mapping[int, int] | None, n: int) -> dict[int, int] | None:
    if min_counts is None:
        return None
    m = {k: 0 for k in range(1, n + 1)}
    for k, v in dict(min_counts).items():
        k = int(k)
        if k < 3 or k > n:
            raise ValueError(f"min_counts orders must lie in 3..{n}, got {k}")
        if int(v) != v or int(v) < 0:
            raise ValueError(f"min_counts[{k}] must be a non-negative integer, got {v!r}")
        m[k] = int(v)
    for k in range(4, n + 1):
        if m[k] < m[k - 1]:
            raise ValueError(f"min_counts must be non-decreasing in the order (m_{k - 1} = {m[k - 1]} > m_{k} = {m[k]}; "
                             "unlisted orders count as 0)")
    return m


def _prune(be: _Backend, cnt_pass: dict, n: int, m: dict[int, int], prune_cont: str,
           ph: _Phases) -> tuple[dict, dict, list]:
    xp = be.xp
    keys, raw, suffix, cont = cnt_pass["keys"], cnt_pass["raw"], cnt_pass["suffix"], cnt_pass["cont"]
    keep: list = [None] * (n + 1)          # None: every type of the level is kept
    by_count = [0] * (n + 1)
    by_closure = [0] * (n + 1)
    with ph("prune.select"):
        for k in range(n, 0, -1):
            if m[k] <= 0:
                continue                   # thresholds are non-decreasing: no level below is pruned either
            kp = be.dev(raw[k]) >= m[k]
            by_count[k] = int(kp.sum())
            if k < n and keep[k + 1] is not None:
                upper_keys = be.dev(keys[k + 1])[be.dev(keep[k + 1])]
                pref = (upper_keys >> _SHIFT).astype(xp.int64)
                kp[pref] = True
                del upper_keys, pref
            by_closure[k] = int(kp.sum()) - by_count[k]
            keep[k] = be.host(kp)
            del kp
        be.free()
    newidx: list = [None] * (n + 1)
    out_keys: list = [None] * (n + 1)
    out_raw: list = [None] * (n + 1)
    out_sfx: list = [None] * (n + 1)
    with ph("prune.remap"):
        for k in range(1, n + 1):
            if keep[k] is not None:                # type ids fit int32 (the count pass refuses 2**31 positions)
                kp = be.dev(keep[k])
                newidx[k] = be.host(xp.cumsum(kp, dtype=xp.int32) - 1)
                newidx[k][~keep[k]] = -1
                del kp
            if keep[k] is None and (k == 1 or keep[k - 1] is None):
                out_keys[k], out_raw[k], out_sfx[k] = keys[k], raw[k], suffix[k]
                continue
            sel = be.dev(keep[k]) if keep[k] is not None else None
            kk = be.dev(keys[k]) if sel is None else be.dev(keys[k])[sel]
            if k >= 2 and keep[k - 1] is not None:
                ni = be.dev(newidx[k - 1])
                pre = ni[(kk >> _SHIFT).astype(xp.int64)]
                if pre.shape[0] and int(pre.min()) < 0:
                    raise RuntimeError(f"a kept level-{k} type lost its context: the pruned type set is not prefix-closed")
                kk = (pre.astype(xp.uint64) << _SHIFT) | (kk & _MASK16)
                del pre
            out_keys[k] = be.host(kk)
            out_raw[k] = raw[k] if sel is None else be.host(be.dev(raw[k])[sel])
            if k >= 2:
                s = be.dev(suffix[k]) if sel is None else be.dev(suffix[k])[sel]
                if keep[k - 1] is not None:
                    s = be.dev(newidx[k - 1])[s.astype(xp.int64)]
                out_sfx[k] = be.host(s.astype(xp.int32))
                del s
            del kk, sel
            be.free()
    n_types = [0] + [int(out_keys[k].shape[0]) for k in range(1, n + 1)]
    with ph("prune.counts"):
        if prune_cont == "recount":
            out_cont = _continuation(be, out_raw, out_sfx, n_types, n, check_suffix=True)
        else:
            out_cont = [None] * (n + 1)
            for k in range(1, n):
                out_cont[k] = cont[k] if keep[k] is None else cont[k][keep[k]]
        be.free()
    report = {
        "min_counts": {str(k): m[k] for k in range(3, n + 1)},
        "prune_cont": prune_cont,
        "n_types_unpruned": [0] + [int(keys[k].shape[0]) for k in range(1, n + 1)],
        "n_types_pruned": n_types,
        "kept_by_count": by_count,
        "kept_by_context_closure": by_closure,
    }
    return {"keys": out_keys, "raw": out_raw, "suffix": out_sfx, "cont": out_cont,
            "n_train_tokens": cnt_pass["n_train_tokens"]}, report, keep


def _unpruned_context_stats(cnt_pass: dict, keep: list, n: int) -> dict:
    """For every PRUNED level k (keep[k] is not None) and both count variants: the unpruned total
    of every KEPT context (float64, in the pruned context indexing) and the unpruned discounts."""
    keys, raw, cont = cnt_pass["keys"], cnt_pass["raw"], cnt_pass["cont"]
    out: dict = {}
    for k in range(2, n + 1):
        if keep[k] is None:
            continue
        ctx = (keys[k] >> _SHIFT).astype(np.int64)
        n_ctx = int(keys[k - 1].shape[0])
        kept_ctx = np.flatnonzero(keep[k - 1]) if keep[k - 1] is not None else None
        for variant, cnt in (("raw", raw[k]), ("cont", cont[k] if k < n else None)):
            if cnt is None:
                continue
            tot = np.bincount(ctx, weights=cnt.astype(np.float64), minlength=n_ctx)
            coc = [int(np.count_nonzero(cnt == c)) for c in (1, 2, 3, 4)]
            out[(variant, k)] = (tot if kept_ctx is None else tot[kept_ctx],
                                 discounts_from_counts_of_counts(*coc))
        del ctx
    return out


def _as_backoff(lv: _Level, total_unpruned: np.ndarray, d_unpruned: tuple[float, float, float],
                discount_mode: str, discount: float) -> _PrunedLevel:
    """The level of the kept types with the pruned followers' count mass moved to the gamma mass."""
    extra = total_unpruned - lv.total                     # exact: integer sums below 2**53 in float64
    if extra.shape != lv.total.shape or bool(np.any(extra < 0)):
        raise RuntimeError("pruned totals exceed the unpruned totals (internal error)")
    d = d_unpruned if discount_mode == "modified" else (float(discount),) * 3
    return _PrunedLevel(cnt=lv.cnt, total=np.ascontiguousarray(total_unpruned), n1=lv.n1, n2=lv.n2, n3=lv.n3,
                        n_any=lv.n_any, d=d, extra=extra)


# ------------------------------------------------------------------ public API
def fit_kn(tok: np.ndarray, doc_start: np.ndarray, order: int = 5, vocab: int | None = None,
           backend: str = "cupy", min_counts: Mapping[int, int] | None = None, *,
           prune_cont: str = "full", prune_mass: str = "backoff", discount_mode: str = "modified",
           discount: float = 0.75, unk_interp: bool = True, pad: bool = True,
           max_device_bytes: int | None = None) -> KNModel:
    """Fit `KNModel(order, vocab, discount_mode, discount, unk_interp, pad)` on (tok, doc_start).

    backend: "cupy" (GPU) or "numpy" (host). min_counts: None (the exact reference fit) or the
    pruning thresholds of the module docstring; prune_cont: "full" (default) or "recount";
    prune_mass: "backoff" (default, requires prune_cont="full") or "renormalize".
    max_device_bytes: a limit for CuPy's memory pool during the fit (restored afterwards).
    The returned model carries `fit_info` (dict): backend, phase wall times, peak pool bytes,
    pruning report."""
    if vocab is None:
        raise ValueError("vocab is required")
    if prune_cont not in PRUNE_CONT_MODES:
        raise ValueError(f"prune_cont must be one of {PRUNE_CONT_MODES}")
    if prune_mass not in PRUNE_MASS_MODES:
        raise ValueError(f"prune_mass must be one of {PRUNE_MASS_MODES}")
    if prune_mass == "backoff" and prune_cont != "full":
        raise ValueError('prune_mass="backoff" needs prune_cont="full" (the unpruned totals are those of the '
                         'unpruned continuation counts)')
    kn = KNModel(order, vocab, discount_mode=discount_mode, discount=discount, unk_interp=unk_interp, pad=pad)
    n = kn.order
    m = _thresholds(min_counts, n)
    tok, ds = _check_inputs(tok, doc_start, vocab)
    t_all = time.perf_counter()
    be = _Backend(backend, max_device_bytes)
    ph = _Phases(be)
    try:
        cnt_pass = _count_pass(be, tok, ds, n, vocab, pad, ph)
        report = None
        unpruned: dict = {}
        if m is not None:
            full_pass = cnt_pass
            cnt_pass, report, keep = _prune(be, full_pass, n, m, prune_cont, ph)
            report["prune_mass"] = prune_mass
            if prune_mass == "backoff":
                with ph("prune.unpruned_totals"):
                    unpruned = _unpruned_context_stats(full_pass, keep, n)
            del full_pass, keep
        keys, raw, cont = cnt_pass["keys"], cnt_pass["raw"], cnt_pass["cont"]
        cnt_pass["suffix"] = None                  # not needed below; release host memory early
        with ph("level_stats"):
            for k in range(1, n + 1):
                n_ctx = int(keys[k - 1].shape[0]) if k >= 2 else 1
                kn.keys[k] = keys[k]
                kn.n_types[k] = int(keys[k].shape[0])
                kn.raw[k] = _level_stats(be, k, keys[k], n_ctx, raw[k], kn.discount_mode, kn.discount)
                if ("raw", k) in unpruned:
                    kn.raw[k] = _as_backoff(kn.raw[k], *unpruned.pop(("raw", k)), kn.discount_mode, kn.discount)
                raw[k] = None                      # the int64 counts of this level are no longer needed
                be.free()
                kn.cont[k] = (_level_stats(be, k, keys[k], n_ctx, cont[k], kn.discount_mode, kn.discount)
                              if k < n else None)
                if ("cont", k) in unpruned:
                    kn.cont[k] = _as_backoff(kn.cont[k], *unpruned.pop(("cont", k)), kn.discount_mode, kn.discount)
                cont[k] = None
                be.free()
        kn.n_train_tokens = int(cnt_pass["n_train_tokens"])
        del cnt_pass, keys, raw, cont
    finally:
        be.close()
    kn.fit_info = {
        "backend": backend,
        "phases_s": dict(ph.seconds),
        "total_s": time.perf_counter() - t_all,
        "peak_pool_bytes": int(be.peak_pool_bytes),
        "pool_note": "bytes held by CuPy's default memory pool, sampled before every release; excludes the CUDA context",
        "pruning": report,
    }
    return kn


# ------------------------------------------------------------------ comparison and accounting
def model_arrays(kn: KNModel) -> dict[str, np.ndarray]:
    """Every array of a fitted KNModel under a stable name (keys.k, raw.k.field, cont.k.field)."""
    out: dict[str, np.ndarray] = {}
    for k in range(1, kn.order + 1):
        out[f"keys.{k}"] = kn.keys[k]
        for variant, levels in (("raw", kn.raw), ("cont", kn.cont)):
            lv = levels[k]
            if lv is None:
                continue
            for f in _LEVEL_FIELDS + (("extra",) if isinstance(lv, _PrunedLevel) else ()):
                out[f"{variant}.{k}.{f}"] = getattr(lv, f)
    return out


def model_scalars(kn: KNModel) -> dict[str, Any]:
    """The non-array state: order, vocab, n_types, n_train_tokens and the discounts as float.hex."""
    d = {"order": kn.order, "vocab": kn.vocab, "n_types": list(kn.n_types), "n_train_tokens": int(kn.n_train_tokens),
         "discount_mode": kn.discount_mode, "discount": kn.discount.hex(), "unk_interp": kn.unk_interp, "pad": kn.pad}
    for k in range(1, kn.order + 1):
        for variant, levels in (("raw", kn.raw), ("cont", kn.cont)):
            if levels[k] is not None:
                d[f"{variant}.{k}.d"] = [float(x).hex() for x in levels[k].d]
    return d


def model_hashes(kn: KNModel) -> dict[str, Any]:
    """sha256 (tmlm.common.hashing.sha256_array: dtype, shape, bytes) of every array, plus the
    scalars of `model_scalars`: equal dicts mean identical models (for cross-process checks)."""
    out: dict[str, Any] = {name: hashing.sha256_array(np.asarray(a)) for name, a in model_arrays(kn).items()}
    out["scalars"] = model_scalars(kn)
    return out


def compare_models(a: KNModel, b: KNModel) -> list[str]:
    """Every difference between two fitted models (empty list: identical arrays, dtypes and discounts)."""
    diffs: list[str] = []
    sa, sb = model_scalars(a), model_scalars(b)
    for key in sorted(set(sa) | set(sb)):
        if sa.get(key) != sb.get(key):
            diffs.append(f"{key}: {sa.get(key)!r} != {sb.get(key)!r}")
    aa, ab = model_arrays(a), model_arrays(b)
    for name in sorted(set(aa) | set(ab)):
        x, y = aa.get(name), ab.get(name)
        if x is None or y is None:
            diffs.append(f"{name}: present in one model only")
        elif x.dtype != y.dtype or x.shape != y.shape:
            diffs.append(f"{name}: {x.dtype}{x.shape} != {y.dtype}{y.shape}")
        elif not np.array_equal(x, y):
            diffs.append(f"{name}: values differ at {int(np.count_nonzero(x != y))} entries")
    return diffs


def model_nbytes(kn: KNModel) -> dict[str, int]:
    """Bytes of a fitted KNModel: keys, per-type counts, per-context statistics, total."""
    keys = sum(int(kn.keys[k].nbytes) for k in range(1, kn.order + 1))
    per_type = per_ctx = 0
    for k in range(1, kn.order + 1):
        for levels in (kn.raw, kn.cont):
            lv = levels[k]
            if lv is None:
                continue
            per_type += int(lv.cnt.nbytes)
            per_ctx += sum(int(getattr(lv, f).nbytes) for f in ("total", "n1", "n2", "n3", "n_any")
                           + (("extra",) if isinstance(lv, _PrunedLevel) else ()))
    return {"keys": keys, "counts": per_type, "context_stats": per_ctx, "total": keys + per_type + per_ctx}
