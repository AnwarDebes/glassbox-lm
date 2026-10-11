"""CKM post-hoc count read-out probe (EXPLORATORY: not a pre-registered result, no threshold of
docs/13 is tested here). The question,
measured BEFORE anything is built into the engine: if every clause of the saved masked tiny-LM
bank kept EXACT counts of the targets it fires on, and the fired clauses were mixed into a proper
distribution, how good would the prediction be, and do the TM-learned clauses add anything over
plain n-gram contexts under the SAME estimator?

Nothing here trains or changes the engine. The layer-1 state (results/20260923T125642Z_H2_row_d,
config experiments/configs/tiny_lm.json, 4 epochs) is loaded into the C++ engine and only the
engine's own forward pass is called (`tm_fired_clauses`, `predict_scores`).

Semantics of fired_clauses(epoch, example, hidden), read from src/tmlm/engine/core/tm_core.h
(tm_hidden, tm_ctx, tm_make_example, tm_positive_true, tm_fires) and engine.cpp (fired, forward):
  * the target position p is ALWAYS hidden (tm_ctx returns SYM_MASK for q == p); a position outside
    the example's document reads SYM_EDGE;
  * hidden=None in masked mode: any other position q is hidden iff
    rng32(seed, S_MASK, epoch, q, 0) < mask_rate_q32 (0.15), the random training mask of that epoch;
    `epoch` selects that mask and nothing else at prediction time (ex.a only feeds the diffusion rate);
  * hidden=<bitmap>: the bitmap REPLACES the random mask and `epoch` has no effect at all; an
    all-zero bitmap hides only the target;
  * bag literals respect the same hiding and skip stop tokens;
  * fired = the candidates of the anchor index (16 exact segments, all four symbol kinds) that pass
    tm_fires (alive, has an anchor, every included literal true), ascending clause ids.

Choices (all logged): training counts use an all-zero bitmap (only the target hidden), as
specified; CKM inside the engine would count under the random 15 percent training mask instead,
so clauses with a positive MASK literal get no counts here. Validation uses the frozen hidden mask
of the eval set. About 5 percent of the training DOCUMENTS are held out from counting and used for
every fitted parameter; validation is never used for fitting or selection.

    OMP_NUM_THREADS=1 ... TMLM_CPU_SLOTS=1 nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name ckm_probe_masked_exploratory --config <scratch json> --no-gpu-sampler -- \\
        .venv-tm/bin/python -u experiments/ckm_posthoc_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from numba import njit

from engine_common import ROOT, EngineConfig, TinyLmTask, load_json, make_engine

EPOCH = 4
H2_DIR = ROOT / "results" / "20260923T125642Z_H2_row_d"
NGRAM_TYPES = (("L1", (-1,)), ("L2", (-2, -1)), ("L3", (-3, -2, -1)), ("R1", (1,)), ("R2", (1, 2)),
               ("R3", (1, 2, 3)), ("L1R1", (-1, 1)), ("L2R1", (-2, -1, 1)), ("L1R2", (-1, 1, 2)),
               ("L2R2", (-2, -1, 1, 2)))
LAM_CAP = 0.999        # recursive JM: keeps the unigram floor weight > 0 so NLL stays finite
NINC_CAP = 12
LOGN_CAP = 23
T0 = time.time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}


def flush_metrics() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


# ------------------------------------------------------------------ numba kernels
@njit(cache=False)
def k_accumulate(ptr, ids, tgt, counts):
    for r in range(ptr.shape[0] - 1):
        t = tgt[r]
        for e in range(ptr[r], ptr[r + 1]):
            counts[ids[e], t] += 1


@njit(cache=False)
def k_dense_to_csr(counts):
    C, V = counts.shape
    indptr = np.zeros(C + 1, np.int64)
    for j in range(C):
        n = 0
        for c in range(V):
            if counts[j, c] != 0:
                n += 1
        indptr[j + 1] = indptr[j] + n
    idx = np.empty(indptr[C], np.int32)
    val = np.empty(indptr[C], np.int32)
    for j in range(C):
        o = indptr[j]
        for c in range(V):
            if counts[j, c] != 0:
                idx[o] = c
                val[o] = counts[j, c]
                o += 1
    return indptr, idx, val


@njit(cache=False)
def k_vote_scores(ptr, ids, rows, vote_cls, vote_w, V):
    n = rows.shape[0]
    out = np.zeros((n, V), np.int32)
    S = vote_cls.shape[1]
    for i in range(n):
        r = rows[i]
        for e in range(ptr[r], ptr[r + 1]):
            j = ids[e]
            for s in range(S):
                c = vote_cls[j, s]
                if c != 0xFFFF:
                    out[i, c] += vote_w[j, s]
    return out


@njit(cache=False)
def k_truncate(indptr, idx, val, K):
    n = indptr.shape[0] - 1
    new_ptr = np.zeros(n + 1, np.int64)
    for k in range(n):
        d = indptr[k + 1] - indptr[k]
        new_ptr[k + 1] = new_ptr[k] + (d if d < K else K)
    nidx = np.empty(new_ptr[n], np.int32)
    nval = np.empty(new_ptr[n], np.int32)
    ssum = np.zeros(n, np.int64)
    for k in range(n):
        lo, hi = indptr[k], indptr[k + 1]
        o = new_ptr[k]
        if hi - lo <= K:
            for e in range(lo, hi):
                nidx[o] = idx[e]
                nval[o] = val[e]
                ssum[k] += val[e]
                o += 1
        else:
            order = np.argsort(-val[lo:hi].astype(np.int64), kind="mergesort")  # stable: ties -> lower class id
            keep = np.sort(order[:K])
            for t in range(K):
                e = lo + keep[t]
                nidx[o] = idx[e]
                nval[o] = val[e]
                ssum[k] += val[e]
                o += 1
    return new_ptr, nidx, nval, ssum


@njit(cache=False)
def k_row_argmax(indptr, idx, val):
    n = indptr.shape[0] - 1
    out = np.full(n, -1, np.int64)
    for k in range(n):
        best = -1
        bc = -1
        for e in range(indptr[k], indptr[k + 1]):
            if val[e] > best:  # indices ascending, strict > keeps the lowest class id on ties
                best = val[e]
                bc = idx[e]
        out[k] = bc
    return out


@njit(cache=False)
def k_pair_pk(rowptr, comp, gold, indptr, idx, val, inv_norm):
    pk = np.zeros(comp.shape[0])
    for r in range(rowptr.shape[0] - 1):
        g = gold[r]
        for i in range(rowptr[r], rowptr[r + 1]):
            k = comp[i]
            lo, hi = indptr[k], indptr[k + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if idx[mid] < g:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < indptr[k + 1] and idx[lo] == g:
                pk[i] = val[lo] * inv_norm[k]
    return pk


@njit(cache=False)
def k_oracle(rowptr, comp, gold, indptr, idx, val, amax):
    n = rowptr.shape[0] - 1
    any_argmax = np.zeros(n, np.uint8)
    in_support = np.zeros(n, np.uint8)
    for r in range(n):
        g = gold[r]
        for i in range(rowptr[r], rowptr[r + 1]):
            k = comp[i]
            if amax[k] == g:
                any_argmax[r] = 1
            lo, hi = indptr[k], indptr[k + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if idx[mid] < g:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < indptr[k + 1] and idx[lo] == g:
                in_support[r] = 1
    return any_argmax, in_support


@njit(cache=False)
def k_recursive_weights(rowptr, lam):
    n = rowptr.shape[0] - 1
    w = np.empty(lam.shape[0])
    wuni = np.empty(n)
    for r in range(n):
        rem = 1.0
        for i in range(rowptr[r], rowptr[r + 1]):
            w[i] = rem * lam[i]
            rem *= 1.0 - lam[i]
        wuni[r] = rem
    return w, wuni


@njit(cache=False)
def k_em_jm(rowptr, pk, bkt, pu_g, lam_b, iters, cap):
    n = rowptr.shape[0] - 1
    nb = lam_b.shape[0]
    maxlen = 1
    for r in range(n):
        if rowptr[r + 1] - rowptr[r] > maxlen:
            maxlen = rowptr[r + 1] - rowptr[r]
    tmp = np.empty(maxlen)
    trace = np.empty(iters + 1)
    for it in range(iters + 1):
        stop = np.zeros(nb)
        pas = np.zeros(nb)
        ll = 0.0
        for r in range(n):
            lo, hi = rowptr[r], rowptr[r + 1]
            rem = 1.0
            p = 0.0
            for i in range(lo, hi):
                lm = lam_b[bkt[i]]
                wi = rem * lm
                tmp[i - lo] = wi * pk[i]
                p += tmp[i - lo]
                rem *= 1.0 - lm
            pu_part = rem * pu_g[r]
            p += pu_part
            ll += np.log(p)
            tail = pu_part / p
            for i in range(hi - 1, lo - 1, -1):
                ri = tmp[i - lo] / p
                stop[bkt[i]] += ri
                pas[bkt[i]] += tail
                tail += ri
        trace[it] = ll
        if it == iters:
            break
        for b in range(nb):
            if stop[b] + pas[b] > 0.0:
                v = stop[b] / (stop[b] + pas[b])
                lam_b[b] = v if v < cap else cap
    return lam_b, trace


@njit(cache=False)
def k_lin_obj(rowptr, pk, bkt, pu_g, alpha, a0):
    n = rowptr.shape[0] - 1
    nb = alpha.shape[0]
    num = np.zeros(nb)
    den = np.zeros(nb)
    ll = 0.0
    num0 = 0.0
    den0 = 0.0
    for r in range(n):
        A = a0 * pu_g[r]
        Z = a0
        for i in range(rowptr[r], rowptr[r + 1]):
            A += alpha[bkt[i]] * pk[i]
            Z += alpha[bkt[i]]
        ll += np.log(A) - np.log(Z)
        for i in range(rowptr[r], rowptr[r + 1]):
            num[bkt[i]] += pk[i] / A
            den[bkt[i]] += 1.0 / Z
        num0 += pu_g[r] / A
        den0 += 1.0 / Z
    return ll, num, den, num0, den0


@njit(cache=False)
def k_induction_stats(ptr, ids, ng, pos, doc_lo, tok, n_comp):
    last = np.full(n_comp, -1, np.int64)
    tries = np.zeros(n_comp, np.int64)
    hits = np.zeros(n_comp, np.int64)
    for r in range(pos.shape[0]):
        p = pos[r]
        dl = doc_lo[r]
        y = tok[p]
        for e in range(ptr[r], ptr[r + 1]):
            k = ids[e]
            q = last[k]
            if q >= dl:
                tries[k] += 1
                if tok[q] == y:
                    hits[k] += 1
            last[k] = p
        for t in range(ng.shape[1]):
            k = ng[r, t]
            if k >= 0:
                q = last[k]
                if q >= dl:
                    tries[k] += 1
                    if tok[q] == y:
                        hits[k] += 1
                last[k] = p
    return tries, hits


@njit(cache=False)
def k_induction_candidates(pres_ptr, pres_ids, pres_ng, pres_pos, pres_doc_lo, vis, row_of,
                           rowptr, comp, tok, n_comp):
    """Walk the presence positions in ascending order; at a row (target) position read, for every
    pair of that row, the most recent earlier VISIBLE position of the same document where the same
    component was present; then, if the position is visible, record it."""
    last = np.full(n_comp, -1, np.int64)
    tstar = np.full(comp.shape[0], -1, np.int64)
    for a in range(pres_pos.shape[0]):
        p = pres_pos[a]
        dl = pres_doc_lo[a]
        r = row_of[a]
        if r >= 0:
            for i in range(rowptr[r], rowptr[r + 1]):
                q = last[comp[i]]
                if q >= dl:
                    tstar[i] = tok[q]
        if vis[p] != 0:
            for e in range(pres_ptr[a], pres_ptr[a + 1]):
                last[pres_ids[e]] = p
            for t in range(pres_ng.shape[1]):
                k = pres_ng[a, t]
                if k >= 0:
                    last[k] = p
    return tstar


@njit(cache=False)
def k_gold_eval(rowptr, w, pkfull, wuni, pu_g, gold, pos, doc_lo, tok, vis, V):
    """Row gold probabilities of the mixture, plus the document-cache gold probability (visible
    earlier tokens of the same document) and whether the cache is non-empty. Rows sorted by position."""
    n = rowptr.shape[0] - 1
    pm = np.empty(n)
    pc = np.zeros(n)
    has = np.zeros(n, np.uint8)
    wsum_err = 0.0
    hist = np.zeros(V, np.int64)
    cur = -1
    cursor = 0
    nvis = 0
    for r in range(n):
        s = wuni[r] * pu_g[r]
        ws = wuni[r]
        for i in range(rowptr[r], rowptr[r + 1]):
            s += w[i] * pkfull[i]
            ws += w[i]
        e = abs(ws - 1.0)
        if e > wsum_err:
            wsum_err = e
        pm[r] = s
        p = pos[r]
        if doc_lo[r] != cur:
            for c in range(V):
                hist[c] = 0
            nvis = 0
            cur = doc_lo[r]
            cursor = cur
        while cursor < p:
            if vis[cursor] != 0:
                hist[tok[cursor]] += 1
                nvis += 1
            cursor += 1
        if nvis > 0:
            has[r] = 1
            pc[r] = hist[gold[r]] / nvis
    return pm, pc, has, wsum_err


@njit(cache=False)
def k_dense_eval(rowptr, comp, w, hh, tstar, indptr, idx, val, inv_norm, wuni, pu, gold, pos, doc_lo,
                 tok, vis, mu, use_cache):
    n = rowptr.shape[0] - 1
    V = pu.shape[0]
    acc = np.empty(V)
    hist = np.zeros(V, np.int64)
    pg = np.empty(n)
    rank = np.empty(n, np.int64)
    maxerr = 0.0
    cur = -1
    cursor = 0
    nvis = 0
    for r in range(n):
        for c in range(V):
            acc[c] = wuni[r] * pu[c]
        for i in range(rowptr[r], rowptr[r + 1]):
            wi = w[i]
            if wi == 0.0:
                continue
            k = comp[i]
            hk = hh[i] if tstar[i] >= 0 else 0.0
            s = wi * (1.0 - hk) * inv_norm[k]
            for e in range(indptr[k], indptr[k + 1]):
                acc[idx[e]] += s * val[e]
            if tstar[i] >= 0:
                acc[tstar[i]] += wi * hk
        if use_cache:
            p = pos[r]
            if doc_lo[r] != cur:
                for c in range(V):
                    hist[c] = 0
                nvis = 0
                cur = doc_lo[r]
                cursor = cur
            while cursor < p:
                if vis[cursor] != 0:
                    hist[tok[cursor]] += 1
                    nvis += 1
                cursor += 1
            if nvis > 0:
                for c in range(V):
                    acc[c] = (1.0 - mu) * acc[c] + mu * hist[c] / nvis
        tot = 0.0
        for c in range(V):
            tot += acc[c]
        e = abs(tot - 1.0)
        if e > maxerr:
            maxerr = e
        g = gold[r]
        pgr = acc[g]
        pg[r] = pgr
        rk = 0
        for c in range(V):
            if acc[c] > pgr or (acc[c] == pgr and c < g):
                rk += 1
        rank[r] = rk
    return pg, rank, maxerr


@njit(cache=False)
def k_gval(indptr, idx, val, lam, inv_n, pu):
    g = np.zeros(idx.shape[0])
    for k in range(indptr.shape[0] - 1):
        if inv_n[k] == 0.0:
            continue
        l = lam[k]
        for e in range(indptr[k], indptr[k + 1]):
            g[e] = np.log(1.0 + l * val[e] * inv_n[k] / ((1.0 - l) * pu[idx[e]]))
    return g


@njit(cache=False)
def k_pair_count(comp, target, indptr, idx, val):
    out = np.zeros(comp.shape[0])
    for i in range(comp.shape[0]):
        k = comp[i]
        g = target[i]
        lo, hi = indptr[k], indptr[k + 1]
        while lo < hi:
            mid = (lo + hi) // 2
            if idx[mid] < g:
                lo = mid + 1
            else:
                hi = mid
        if lo < indptr[k + 1] and idx[lo] == g:
            out[i] = val[lo]
    return out


@njit(cache=False)
def k_loglin(rowptr, comp, bkt, alpha, beta0, indptr, idx, gval, logpu, gold, tstar, bonus, nb, want_grad):
    n = rowptr.shape[0] - 1
    V = logpu.shape[0]
    s = np.empty(V)
    P = np.empty(V)
    grad = np.zeros(nb)
    gbeta = 0.0
    ll = 0.0
    pg = np.empty(n)
    rank = np.empty(n, np.int64)
    maxerr = 0.0
    for r in range(n):
        for c in range(V):
            s[c] = beta0 * logpu[c]
        for i in range(rowptr[r], rowptr[r + 1]):
            k = comp[i]
            a = alpha[bkt[i]]
            for e in range(indptr[k], indptr[k + 1]):
                s[idx[e]] += a * gval[e]
            if tstar[i] >= 0:
                s[tstar[i]] += a * bonus[i]
        mx = s[0]
        for c in range(1, V):
            if s[c] > mx:
                mx = s[c]
        tot = 0.0
        for c in range(V):
            P[c] = np.exp(s[c] - mx)
            tot += P[c]
        chk = 0.0
        for c in range(V):
            P[c] /= tot
            chk += P[c]
        e1 = abs(chk - 1.0)
        if e1 > maxerr:
            maxerr = e1
        g = gold[r]
        pgr = P[g]
        pg[r] = pgr
        ll += np.log(pgr)
        rk = 0
        for c in range(V):
            if P[c] > pgr or (P[c] == pgr and c < g):
                rk += 1
        rank[r] = rk
        if want_grad:
            el = 0.0
            for c in range(V):
                el += P[c] * logpu[c]
            gbeta += logpu[g] - el
            for i in range(rowptr[r], rowptr[r + 1]):
                k = comp[i]
                gy = 0.0
                eg = 0.0
                for e in range(indptr[k], indptr[k + 1]):
                    eg += P[idx[e]] * gval[e]
                    if idx[e] == g:
                        gy = gval[e]
                if tstar[i] >= 0:
                    eg += P[tstar[i]] * bonus[i]
                    if tstar[i] == g:
                        gy += bonus[i]
                grad[bkt[i]] += gy - eg
    return ll, grad, gbeta, pg, rank, maxerr


# ------------------------------------------------------------------ engine passes
def fired_pass(eng, positions: np.ndarray, bitmap: np.ndarray, label: str):
    """The engine's own fired lists (tm_fired_clauses, the same call as CpuEngine.fired_clauses
    but with one reused output buffer) for every position; CSR (ptr int64, ids uint16)."""
    lib, h = eng.lib, eng.handle
    C = int(eng.cfg.n_clauses)
    assert C <= 65536
    buf = np.zeros(C, dtype=np.uint32)
    bufp = buf.ctypes.data
    bm = np.ascontiguousarray(bitmap, dtype=np.uint8)
    assert bm.shape == (int(eng.corpus.tok.shape[0]),)
    bmp = bm.ctypes.data
    n = int(positions.shape[0])
    ptr = np.zeros(n + 1, dtype=np.int64)
    cap = max(4096, n * 112)
    ids = np.empty(cap, dtype=np.uint16)
    fn = lib.tm_fired_clauses
    o = 0
    t0 = time.time()
    step = max(1, n // 10)
    for i, p in enumerate(positions.tolist()):
        m = fn(h, EPOCH, p, bmp, bufp, C, 0)
        if o + m > cap:
            cap = int(cap * 1.5) + m
            ids2 = np.empty(cap, dtype=np.uint16)
            ids2[:o] = ids[:o]
            ids = ids2
        ids[o:o + m] = buf[:m]
        o += m
        ptr[i + 1] = o
        if (i + 1) % step == 0:
            el = time.time() - t0
            log(f"  {label}: {i + 1}/{n} positions, {(i + 1) / el:.0f} pos/s, fired/pos {o / (i + 1):.1f}")
    wall = time.time() - t0
    return ptr, ids[:o].copy(), wall


def check_fired_against_api(eng, positions, bitmap, ptr, ids, n_check: int, n_brute: int) -> None:
    rng = np.random.default_rng(12345)
    pick = rng.choice(positions.shape[0], size=min(n_check, positions.shape[0]), replace=False)
    for i in pick:
        a = eng.fired_clauses(EPOCH, int(positions[i]), bitmap)
        assert np.array_equal(a, ids[ptr[i]:ptr[i + 1]].astype(np.uint32)), "fired list differs from the API"
    for i in pick[:n_brute]:
        b = eng.fired_clauses_bruteforce(EPOCH, int(positions[i]), bitmap)
        assert np.array_equal(b, ids[ptr[i]:ptr[i + 1]].astype(np.uint32)), "fired list differs from the brute force"


# ------------------------------------------------------------------ n-gram contexts
def ngram_keys(tok, doc_lo, doc_hi, pos, offsets, vis, V):
    n = tok.shape[0]
    key = np.zeros(pos.shape[0], dtype=np.int64)
    valid = np.ones(pos.shape[0], dtype=bool)
    for o in offsets:
        q = pos + o
        inside = (q >= doc_lo) & (q < doc_hi)
        qc = np.clip(q, 0, n - 1)
        t = np.where(inside, tok[qc].astype(np.int64), V)  # V = EDGE symbol
        if vis is not None:
            valid &= (~inside) | (vis[qc] != 0)
        key = key * (V + 1) + t
    return key, valid


def doc_bounds(doc_start: np.ndarray, pos: np.ndarray):
    ds = doc_start.astype(np.int64)
    d = np.searchsorted(ds, pos, side="right") - 1
    return ds[d], ds[d + 1]


def pairs_from_csr(ptr, ids, keep_mask_per_comp):
    n = ptr.shape[0] - 1
    rowid = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    comp = ids.astype(np.int64)
    keep = keep_mask_per_comp[comp]
    return rowid[keep], comp[keep]


def pairs_from_matrix(mat):
    n, t = mat.shape
    rowid = np.repeat(np.arange(n, dtype=np.int64), t)
    comp = mat.reshape(-1).astype(np.int64)
    keep = comp >= 0
    return rowid[keep], comp[keep]


def merge_pairs(a, b):
    rowid = np.concatenate([a[0], b[0]])
    comp = np.concatenate([a[1], b[1]])
    o = np.argsort(rowid, kind="stable")
    return rowid[o], comp[o]


def rowptr_of(rowid, n):
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(rowid, minlength=n))
    return ptr


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(H2_DIR / "layer1_seed1.npz"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--max-count-pos", type=int, default=0, help="0 = every counting position")
    ap.add_argument("--max-heldout-pos", type=int, default=0, help="0 = every held-out position")
    ap.add_argument("--val-max-pos", type=int, default=0, help="smoke only: validation positions below this")
    ap.add_argument("--kn", type=int, default=1, help="1: KN-5 comparators with bits per token on the same targets")
    ap.add_argument("--loglin", type=int, default=1, help="1: the extra log-linear pooling rows")
    ap.add_argument("--loglin-fit-rows", type=int, default=3000)
    ap.add_argument("--loglin-iters", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    M("mode_smoke", bool(args.smoke))
    M("state_file", str(Path(args.state).relative_to(ROOT)) if Path(args.state).is_absolute() else args.state)

    # ---------------------------------------------------------------- data and engine
    task = TinyLmTask(vocab=2000, n_tokens=5_000_000, mode="masked")
    train, val = task.train, task.val
    V = int(task.vocab)
    hidden = task.hidden.astype(np.uint8)
    targets_all = task.targets.astype(np.int64)
    gold_all = task.gold.astype(np.int64)
    assert np.all(np.diff(targets_all) > 0) and np.all(hidden[targets_all] == 1)
    for k, v in task.metrics().items():
        METRICS[f"task.{k}"] = {**v, "exploratory": True}
    log(f"task: train {train.tok.shape[0]} tokens, {train.doc_start.shape[0] - 1} docs; val {val.tok.shape[0]} tokens, "
        f"{targets_all.shape[0]} targets; bigram {task.left_bigram_top1:.4f} count probe {task.count_probe_top1:.4f}")
    l1 = load_json(ROOT / "experiments" / "configs" / "tiny_lm.json")
    cfg = EngineConfig(**{**l1["engine"], "seed": args.seed})
    C = int(cfg.n_clauses)
    st = dict(np.load(args.state))
    eng_tr = make_engine(cfg, train)
    eng_tr.load_state(st)
    eng_va = make_engine(cfg, val)
    eng_va.load_state(st)
    M("engine.state_hash", eng_tr.state_hash())
    M("engine.omp_threads", int(eng_tr.lib.tm_omp_max_threads()))
    alive = st["alive"].astype(bool)
    inc = (st["lit_state"] >= 128) & (st["lit_id"] != 0xFFFFFFFF) & alive[:, None]
    n_inc_eng = inc.sum(axis=1).astype(np.int64)
    from tmlm.engine.config import sym_mask
    lit = st["lit_id"].astype(np.int64)
    pos_mask_lit = inc & ((lit & 1) == 0) & (((lit >> 1) & 0xFFFF) == sym_mask(V)) & ((lit >> 17) < 16)
    M("engine.alive_clauses", int(alive.sum()))
    M("engine.n_inc_histogram", np.bincount(n_inc_eng, minlength=13).tolist(), note="included literals per clause (positive and negated, exact and bag)")
    M("engine.clauses_with_positive_mask_literal", int(pos_mask_lit.any(axis=1).sum()))

    # ---------------------------------------------------------------- validation fired lists (frozen mask)
    n_val_pos = int(val.tok.shape[0]) if args.val_max_pos <= 0 else min(args.val_max_pos, int(val.tok.shape[0]))
    val_pos = np.arange(n_val_pos, dtype=np.int64)
    sel = targets_all < n_val_pos
    targets, gold = targets_all[sel], gold_all[sel]
    nT = int(targets.shape[0])
    M("val.positions_scanned", n_val_pos)
    M("val.targets_evaluated", nT)
    log(f"validation fired pass over {n_val_pos} positions (frozen mask)")
    vptr, vids, vwall = fired_pass(eng_va, val_pos, hidden, "val")
    M("throughput.val_fired_pos_per_s", n_val_pos / vwall, note="single thread, nice 19, contended host")
    check_fired_against_api(eng_va, val_pos, hidden, vptr, vids, 200, 5)

    # ---------------------------------------------------------------- M1 pipeline check
    t0 = time.time()
    hits = 0
    mism = 0
    for i in range(0, targets_all.shape[0], 4096):
        tp = targets_all[i:i + 4096]
        s = eng_va.predict_scores(EPOCH, tp.astype(np.uint32), hidden)
        hits += int((s.argmax(axis=1) == gold_all[i:i + 4096]).sum())
        inside = tp < n_val_pos
        if inside.any():
            rec = k_vote_scores(vptr, vids, tp[inside], st["vote_cls"], st["vote_w"].astype(np.int32), V)
            mism += int((rec != s[inside]).any(axis=1).sum())
    top1_engine = hits / targets_all.shape[0]
    h2 = json.loads((H2_DIR / "metrics.json").read_text())
    expect = h2[f"row_d.seed{args.seed}.layer1_top1"]["value"]
    M("M1.engine_signed_vote_top1", top1_engine, note="predict_scores argmax (ties to the lowest id), frozen mask, all targets")
    M("M1.expected_top1_from_H2", expect, source=str(H2_DIR.relative_to(ROOT) / "metrics.json"))
    M("M1.passed", bool(top1_engine == expect))
    M("M1.score_rows_mismatch_vs_fired_lists", mism, note="vote sums rebuilt from our fired lists vs predict_scores, targets scanned")
    log(f"M1: engine top-1 {top1_engine:.6f}, expected {expect:.6f}, rebuilt-score mismatches {mism} ({time.time() - t0:.0f} s)")
    flush_metrics()
    if top1_engine != expect or mism != 0:
        log("M1 FAILED: stop")
        return 3

    # ---------------------------------------------------------------- training fired lists (only the target hidden)
    tok_tr = train.tok.astype(np.int64)
    ds_tr = train.doc_start.astype(np.int64)
    n_doc = ds_tr.shape[0] - 1
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, args.seed]))
    n_ho = max(1, int(round(args.heldout_frac * n_doc)))
    ho_docs = np.sort(rng.choice(n_doc, size=n_ho, replace=False))
    doc_of = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_ho = np.isin(doc_of, ho_docs)
    cnt_pos = np.flatnonzero(~is_ho).astype(np.int64)
    ho_pos = np.flatnonzero(is_ho).astype(np.int64)
    if args.max_count_pos > 0:
        cnt_pos = cnt_pos[:args.max_count_pos]
    if args.max_heldout_pos > 0:
        ho_pos = ho_pos[:args.max_heldout_pos]
    M("split.heldout_docs", int(n_ho), note="SeedSequence([20260923, 715, seed]), about 5 percent of the slice's documents")
    M("split.counting_positions", int(cnt_pos.shape[0]))
    M("split.heldout_positions", int(ho_pos.shape[0]))
    zero = np.zeros(train.tok.shape[0], dtype=np.uint8)
    log(f"training fired pass: {cnt_pos.shape[0]} counting + {ho_pos.shape[0]} held-out positions (only the target hidden)")
    cptr, cids, cwall = fired_pass(eng_tr, cnt_pos, zero, "count")
    check_fired_against_api(eng_tr, cnt_pos, zero, cptr, cids, 200, 5)
    hptr, hids, hwall = fired_pass(eng_tr, ho_pos, zero, "heldout")
    M("throughput.train_fired_pos_per_s", (cnt_pos.shape[0] + ho_pos.shape[0]) / (cwall + hwall), note="single thread, nice 19, contended host")
    M("throughput.train_fired_wall_s", cwall + hwall)
    M("train.fired_per_position", float(cptr[-1] / max(1, cnt_pos.shape[0])))
    flush_metrics()

    # ---------------------------------------------------------------- engine count tables (M2)
    t0 = time.time()
    counts = np.zeros((C, V), dtype=np.int32)
    k_accumulate(cptr, cids, tok_tr[cnt_pos], counts)
    e_ptr, e_idx, e_val = k_dense_to_csr(counts)
    del counts
    D_e = np.diff(e_ptr)
    N_e = np.bincount(np.repeat(np.arange(C), D_e), weights=e_val, minlength=C).astype(np.int64)
    M("M2.engine_pairs_counted", int(cptr[-1]))
    M("M2.engine_clauses_with_counts", int((N_e > 0).sum()))
    M("M2.engine_N_median_of_counted", float(np.median(N_e[N_e > 0])) if (N_e > 0).any() else 0.0)
    M("M2.engine_D_mean_of_counted", float(D_e[N_e > 0].mean()) if (N_e > 0).any() else 0.0)
    M("M2.engine_D_gt64_fraction", float((D_e[N_e > 0] > 64).mean()) if (N_e > 0).any() else 0.0)
    log(f"engine counts: {int((N_e > 0).sum())} clauses with counts, median N {np.median(N_e[N_e > 0]):.0f}, "
        f"mean D {D_e[N_e > 0].mean():.1f} ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- n-gram count tables (M4)
    t0 = time.time()
    lo_c, hi_c = doc_bounds(ds_tr, cnt_pos)
    ptr_parts, idx_parts, val_parts = [e_ptr], [e_idx], [e_val]
    N_parts, D_parts, ninc_parts, fam_parts = [N_e], [D_e], [n_inc_eng], [np.zeros(C, np.int64)]
    base = C
    types = []
    ng_cnt = np.full((cnt_pos.shape[0], len(NGRAM_TYPES)), -1, dtype=np.int64)
    for t, (name, offs) in enumerate(NGRAM_TYPES):
        key, _ = ngram_keys(tok_tr, lo_c, hi_c, cnt_pos, offs, None, V)
        u, cnt = np.unique(key * V + tok_tr[cnt_pos], return_counts=True)
        ctx = u // V
        tg = (u % V).astype(np.int32)
        ckeys, start, inv = np.unique(ctx, return_index=True, return_inverse=True)
        n_ctx = ckeys.shape[0]
        indptr = np.append(start, u.shape[0]).astype(np.int64)
        Nt = np.add.reduceat(cnt, start).astype(np.int64)
        ptr_parts.append(indptr[1:] + ptr_parts[-1][-1])
        idx_parts.append(tg)
        val_parts.append(cnt.astype(np.int32))
        N_parts.append(Nt)
        D_parts.append(np.diff(indptr))
        ninc_parts.append(np.full(n_ctx, len(offs), np.int64))
        fam_parts.append(np.ones(n_ctx, np.int64))
        ng_cnt[:, t] = base + np.searchsorted(ckeys, key)
        types.append({"name": name, "offs": offs, "base": base, "keys": ckeys})
        M(f"M4.ngram_contexts.{name}", int(n_ctx))
        base += n_ctx
    n_comp = base
    indptr = np.concatenate(ptr_parts)
    idx = np.concatenate(idx_parts)
    valc = np.concatenate(val_parts)
    N = np.concatenate(N_parts)
    D = np.concatenate(D_parts)
    ninc = np.concatenate(ninc_parts)
    fam = np.concatenate(fam_parts)
    del ptr_parts, idx_parts, val_parts
    assert indptr.shape[0] == n_comp + 1 and indptr[-1] == idx.shape[0]
    inv_N = np.where(N > 0, 1.0 / np.maximum(N, 1), 0.0)
    lam_exact = np.where(N > 0, N / np.maximum(N + D, 1), 0.0)
    t_ptr, t_idx, t_val, S64 = k_truncate(indptr, idx, valc, 64)
    inv_S64 = np.where(S64 > 0, 1.0 / np.maximum(S64, 1), 0.0)
    lam_64 = np.where(N > 0, S64 / np.maximum(N + D, 1), 0.0)
    amax = k_row_argmax(indptr, idx, valc)
    uni = np.bincount(tok_tr[cnt_pos], minlength=V).astype(np.float64)
    p_uni = (uni + 1.0) / (uni.sum() + V)
    assert abs(p_uni.sum() - 1.0) < 1e-12
    logn = np.minimum(np.floor(np.log2(np.maximum(N, 1))).astype(np.int64), LOGN_CAP)
    bucket = (fam * (NINC_CAP + 1) + np.minimum(ninc, NINC_CAP)) * (LOGN_CAP + 1) + logn
    n_buckets = 2 * (NINC_CAP + 1) * (LOGN_CAP + 1)
    M("M4.components_total", int(n_comp))
    M("M4.count_nnz_total", int(idx.shape[0]))
    log(f"n-gram tables: {n_comp - C} contexts, {idx.shape[0]} (component, target) entries ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- row sets
    def ng_lookup(tok, ds, pos, vis):
        lo, hi = doc_bounds(ds, pos)
        out = np.full((pos.shape[0], len(types)), -1, dtype=np.int64)
        for t, ty in enumerate(types):
            key, valid = ngram_keys(tok, lo, hi, pos, ty["offs"], vis, V)
            j = np.searchsorted(ty["keys"], key)
            jc = np.minimum(j, ty["keys"].shape[0] - 1)
            found = valid & (j < ty["keys"].shape[0]) & (ty["keys"][jc] == key)
            out[found, t] = ty["base"] + j[found]
        return out, lo

    has_counts = N > 0
    val_vis = (1 - hidden).astype(np.uint8)
    ng_val_all, val_lo_all = ng_lookup(val.tok.astype(np.int64), val.doc_start, val_pos, val_vis)
    ng_ho, ho_lo = ng_lookup(tok_tr, ds_tr, ho_pos, None)
    sets = {}
    # validation rows = targets (positions == indices into the validation pass)
    t_ptr_rows = np.zeros(nT + 1, np.int64)
    t_ptr_rows[1:] = np.cumsum(vptr[targets + 1] - vptr[targets])
    t_ids = np.concatenate([vids[vptr[p]:vptr[p + 1]] for p in targets.tolist()]) if nT else np.zeros(0, np.uint16)
    row_of_val = np.full(n_val_pos, -1, np.int64)
    row_of_val[targets] = np.arange(nT)
    sets["val"] = dict(n=nT, pos=targets, gold=gold, doc_lo=val_lo_all[targets], tok=val.tok.astype(np.int64), vis=val_vis,
                       eptr=t_ptr_rows, eids=t_ids, ng=ng_val_all[targets],
                       pres=(vptr, vids, ng_val_all, val_pos, val_lo_all, row_of_val))
    ho_vis = np.ones(train.tok.shape[0], np.uint8)
    sets["ho"] = dict(n=int(ho_pos.shape[0]), pos=ho_pos, gold=tok_tr[ho_pos], doc_lo=ho_lo, tok=tok_tr, vis=ho_vis,
                      eptr=hptr, eids=hids, ng=ng_ho,
                      pres=(hptr, hids, ng_ho, ho_pos, ho_lo, np.arange(ho_pos.shape[0], dtype=np.int64)))
    for sname, s in sets.items():
        fired_n = np.diff(s["eptr"])
        rowid, comp = pairs_from_csr(s["eptr"], s["eids"], has_counts)
        with_counts = np.bincount(rowid, minlength=s["n"]) > 0
        M(f"{sname}.engine_fired_per_target", float(fired_n.mean()))
        M(f"{sname}.frac_targets_with_fired_clause_N_ge_1", float(with_counts.mean()))
        M(f"{sname}.frac_targets_with_no_fired_clause", float((fired_n == 0).mean()))
        eng_pairs = (rowid, comp)
        ng_pairs = pairs_from_matrix(s["ng"])
        s["arms"] = {"engine": eng_pairs, "ngram": ng_pairs, "union": merge_pairs(eng_pairs, ng_pairs)}
        M(f"{sname}.ngram_contexts_available_per_target", float(ng_pairs[0].shape[0] / max(1, s["n"])))
    log("row sets built")

    # ---------------------------------------------------------------- induction statistics on the counting positions
    t0 = time.time()
    tries, ihits = k_induction_stats(cptr, cids, ng_cnt, cnt_pos, lo_c, tok_tr, n_comp)
    h_comp = np.where(tries > 0, ihits / np.maximum(tries, 1), 0.0)
    eng_tr_mask = (tries[:C] > 0)
    M("M5.engine_clauses_with_induction_tries", int(eng_tr_mask.sum()))
    M("M5.engine_h_weighted_mean", float(ihits[:C].sum() / max(1, tries[:C].sum())), note="hits over tries, counting positions, document order")
    M("M5.ngram_h_weighted_mean", float(ihits[C:].sum() / max(1, tries[C:].sum())))
    for sname, s in sets.items():
        pp, pi, png, ppos, plo, prow = s["pres"]
        s["tstar"] = {}
        for arm, (rowid, comp) in s["arms"].items():
            rp = rowptr_of(rowid, s["n"])
            s["tstar"][arm] = k_induction_candidates(pp, pi, png, ppos, plo, s["vis"], prow, rp, comp,
                                                     s["tok"], n_comp)
        te = s["tstar"]["engine"]
        M(f"{sname}.engine_pairs_with_induction_candidate", float((te >= 0).mean()) if te.shape[0] else 0.0)
    log(f"induction statistics ({time.time() - t0:.0f} s)")
    del ng_cnt

    # ---------------------------------------------------------------- estimators
    ROWS: dict[str, dict] = {}
    per_target: dict[str, np.ndarray] = {}
    base_cache: dict = {}

    def order_perm(rowid, comp, how):
        if how == "spec":   # n_inc desc, N desc, id asc
            return np.lexsort((comp, -N[comp], -ninc[comp], rowid))
        if how == "nasc":   # N asc (rarest context first), then n_inc desc, id asc
            return np.lexsort((comp, -ninc[comp], N[comp], rowid))
        return np.arange(rowid.shape[0])  # pairs are already grouped by row

    def base_of(sname, arm, how, trunc, induction):
        key = (sname, arm, how, trunc, induction)
        if key in base_cache:
            base_cache[key] = base_cache.pop(key)  # most recently used goes last
            return base_cache[key]
        while len(base_cache) >= 6:
            base_cache.pop(next(iter(base_cache)))
        s = sets[sname]
        rowid0, comp0 = s["arms"][arm]
        perm = order_perm(rowid0, comp0, how)
        rowid, comp = rowid0[perm], comp0[perm]
        rp = rowptr_of(rowid, s["n"])
        if induction:
            ts = s["tstar"][arm][perm]
            hh = np.where(ts >= 0, h_comp[comp], 0.0)
        else:
            ts = np.full(comp.shape[0], -1, np.int64)
            hh = np.zeros(comp.shape[0])
        csr = (t_ptr, t_idx, t_val, inv_S64) if trunc else (indptr, idx, valc, inv_N)
        pk = k_pair_pk(rp, comp, s["gold"], csr[0], csr[1], csr[2], csr[3])
        pkfull = np.where(ts >= 0, hh * (ts == s["gold"][rowid]) + (1.0 - hh) * pk, pk)
        del pk
        b = dict(rp=rp, rowid=rowid, comp=comp, hh=hh, ts=ts, csr=csr, pkfull=pkfull, m=np.diff(rp))
        base_cache[key] = b
        return b

    def build(sname, arm, est, params, induction):
        """Per-pair weights (ordered pairs), per-pair h and t*, CSR choice, per-row unigram weight."""
        s = sets[sname]
        how = "spec" if est in ("wb", "wb64", "jm") else ("nasc" if est == "wb_nasc" else "none")
        b = dict(base_of(sname, arm, how, est in ("wb64", "lin64"), induction))
        rowid, comp, rp, m = b["rowid"], b["comp"], b["rp"], b["m"]
        if est in ("wb", "wb_nasc"):
            w, wuni = k_recursive_weights(rp, lam_exact[comp])
        elif est == "wb64":
            w, wuni = k_recursive_weights(rp, lam_64[comp])
        elif est == "jm":
            w, wuni = k_recursive_weights(rp, params["lam_b"][bucket[comp]])
        elif est == "snm":
            beta = params["beta"]
            w = np.where(m[rowid] > 0, (1.0 - beta) / np.maximum(m[rowid], 1), 0.0)
            wuni = np.where(m > 0, beta, 1.0)
        elif est in ("lin", "lin64"):
            a = params["alpha"][bucket[comp]]
            Z = np.bincount(rowid, weights=a, minlength=s["n"]) + params["a0"]
            w = a / Z[rowid]
            wuni = params["a0"] / Z
        else:
            raise ValueError(est)
        b["w"], b["wuni"] = w, wuni
        return b

    def fit(arm, est, induction):
        """Parameters fitted on the held-out chunk only."""
        if est in ("wb", "wb_nasc", "wb64"):
            return {}, None
        s = sets["ho"]
        b = base_of("ho", arm, "spec" if est == "jm" else "none", est == "lin64", induction)
        pu_g = p_uni[s["gold"]]
        if est == "snm":
            m = b["m"]
            pf = np.bincount(b["rowid"], weights=b["pkfull"], minlength=s["n"]) / np.maximum(m, 1)
            ok = m > 0
            beta = 0.5
            for _ in range(200):
                r = beta * pu_g[ok] / ((1 - beta) * pf[ok] + beta * pu_g[ok])
                beta = float(r.mean())
            return {"beta": beta}, {"beta": beta}
        if est == "jm":
            # same order as WB; initial lambda = the mean WB lambda of the bucket's components
            bk = bucket[b["comp"]]
            init = np.bincount(bucket[has_counts], weights=lam_exact[has_counts], minlength=n_buckets)
            cntb = np.bincount(bucket[has_counts], minlength=n_buckets)
            lam_b = np.minimum(np.where(cntb > 0, init / np.maximum(cntb, 1), 0.5), LAM_CAP)
            lam_b, trace = k_em_jm(b["rp"], b["pkfull"], bk, pu_g, lam_b.copy(), 40, LAM_CAP)
            assert np.all(np.diff(trace) >= -1e-6 * np.abs(trace[:-1])), "EM log-likelihood decreased"
            return {"lam_b": lam_b}, {"heldout_ll_first": float(trace[0]), "heldout_ll_last": float(trace[-1]), "em_iters": 40}
        if est in ("lin", "lin64"):
            from scipy.optimize import minimize
            bk = bucket[b["comp"]]
            used = np.unique(bk)

            def obj(theta):
                alpha = np.zeros(n_buckets)
                alpha[used] = np.exp(theta)
                ll, num, den, _, _ = k_lin_obj(b["rp"], b["pkfull"], bk, pu_g, alpha, 1.0)
                g = alpha[used] * (num[used] - den[used])
                return -ll, -g
            res = minimize(obj, np.zeros(used.shape[0]), jac=True, method="L-BFGS-B", options={"maxiter": 100})
            alpha = np.zeros(n_buckets)
            alpha[used] = np.exp(res.x)
            return {"alpha": alpha, "a0": 1.0}, {"heldout_ll": float(-res.fun), "converged": bool(res.success), "iters": int(res.nit),
                                                 "optimizer": "L-BFGS-B on log weights (normalized linear mixture)"}
        raise ValueError(est)

    def heldout_bits(arm, est, params, induction, mu=None):
        s = sets["ho"]
        b = build("ho", arm, est, params, induction)
        pm, pc, has, werr = k_gold_eval(b["rp"], b["w"], b["pkfull"], b["wuni"], p_uni[s["gold"]], s["gold"], s["pos"],
                                        s["doc_lo"], s["tok"], s["vis"], V)
        assert werr < 1e-9, f"weights do not sum to one: {werr}"
        p = pm if mu is None else np.where(has == 1, (1 - mu) * pm + mu * pc, pm)
        assert np.all(p > 0)
        return float(-np.log(p).mean() / np.log(2.0)), pm, pc, has

    def record(name, row, pg):
        ROWS[name] = row
        per_target[name] = pg.astype(np.float32)
        for k in ("top1", "top5", "bits_per_token", "heldout_bits_per_token"):
            M(f"rows.{name}.{k}", row[k])
        M(f"rows.{name}.spec", {k: v for k, v in row.items() if k not in ("top1", "top5", "bits_per_token", "heldout_bits_per_token")})
        flush_metrics()

    def evaluate(name, arm, est, params, induction, mu=None, fit_info=None):
        s = sets["val"]
        b = build("val", arm, est, params, induction)
        t0 = time.time()
        pg, rank, maxerr = k_dense_eval(b["rp"], b["comp"], b["w"], b["hh"], b["ts"], b["csr"][0], b["csr"][1], b["csr"][2],
                                        b["csr"][3], b["wuni"], p_uni, s["gold"], s["pos"], s["doc_lo"], s["tok"], s["vis"],
                                        0.0 if mu is None else float(mu), mu is not None)
        assert maxerr < 1e-9, f"{name}: a probability row does not sum to one (max error {maxerr})"
        assert np.all(pg > 0), f"{name}: zero probability on a gold token"
        ho_bits = heldout_bits(arm, est, params, induction, mu)[0]
        row = {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
               "bits_per_token": float(-np.log(pg).mean() / np.log(2.0)), "heldout_bits_per_token": ho_bits,
               "arm": arm, "estimator": est, "induction": bool(induction), "doc_cache_mu": mu, "max_row_sum_error": float(maxerr)}
        if fit_info is not None:
            row["fit"] = fit_info
        if est in ("wb", "wb64", "wb_nasc", "jm"):
            # where the recursion puts its mass: the first three fired components and the unigram floor
            cum3 = np.zeros(s["n"])
            for k in range(3):
                sel = b["m"] > k
                cum3[sel] += b["w"][b["rp"][:-1][sel] + k]
            row["mean_weight_first3"] = float(cum3.mean())
            row["median_unigram_weight"] = float(np.median(b["wuni"]))
        record(name, row, pg)
        log(f"{name:34s} top1 {row['top1']:.4f} top5 {row['top5']:.4f} bits {row['bits_per_token']:.4f} "
            f"(held-out {ho_bits:.4f}) [{time.time() - t0:.1f} s]")
        return row

    ests = ("wb", "wb64", "wb_nasc", "snm", "jm", "lin", "lin64")
    best = {}
    for arm in ("engine", "ngram", "union"):
        for est in ests:
            params, info = fit(arm, est, False)
            evaluate(f"{arm}.{est}", arm, est, params, False, fit_info=info)
        # (e) oracle bounds
        s = sets["val"]
        rowid, comp = s["arms"][arm]
        rp = rowptr_of(rowid, s["n"])
        am, sup = k_oracle(rp, comp, s["gold"], indptr, idx, valc, amax)
        M(f"rows.{arm}.oracle.some_fired_argmax_is_gold", float(am.mean()))
        M(f"rows.{arm}.oracle.gold_in_union_of_supports", float(sup.mean()))
        log(f"{arm}: oracle argmax {am.mean():.4f}, support {sup.mean():.4f}")
        # best estimator of the arm by HELD-OUT bits per token (validation is never used to choose)
        best[arm] = min(ests, key=lambda e: ROWS[f"{arm}.{e}"]["heldout_bits_per_token"])
        M(f"best_estimator_by_heldout.{arm}", best[arm])

    # ---------------------------------------------------------------- M5 in-context recall on the best estimator of each arm
    for arm in ("engine", "ngram", "union"):
        est = best[arm]
        p_ind, info_ind = fit(arm, est, True)
        evaluate(f"{arm}.{est}+induction", arm, est, p_ind, True, fit_info=info_ind)
        p0, _ = fit(arm, est, False)
        for ind, params in ((False, p0), (True, p_ind)):
            _, pm, pc, has = heldout_bits(arm, est, params, ind)
            ok = has == 1
            mu = 0.1
            for _ in range(200):
                r = mu * pc[ok] / ((1 - mu) * pm[ok] + mu * pc[ok])
                mu = float(r.mean())
            name = f"{arm}.{est}+{'induction+' if ind else ''}doc_cache"
            evaluate(name, arm, est, params, ind, mu=mu)

    # ---------------------------------------------------------------- extra (not in the specification): log-linear pooling
    # P(c) proportional to P_uni(c)^beta0 * prod_k (Pt_k(c) / P_uni(c))^alpha_b(k), with Pt_k the WB-smoothed clause
    # distribution lam_k P_k + (1 - lam_k) P_uni. It is the form of the engine's own read-out (a SUM of per-clause
    # votes followed by a softmax), with votes that are count-derived log ratios instead of learned integers.
    if args.loglin:
        gval = k_gval(indptr, idx, valc, lam_exact, inv_N, p_uni)
        logpu = np.log(p_uni)
        rng_ll = np.random.default_rng(np.random.SeedSequence([20260923, 716]))
        ho_perm = rng_ll.permutation(sets["ho"]["n"])
        fit_rows = np.sort(ho_perm[:args.loglin_fit_rows])
        chk_rows = np.sort(ho_perm[args.loglin_fit_rows:2 * args.loglin_fit_rows])

        def sub(b, rows, sname):
            keep = np.zeros(sets[sname]["n"], bool)
            keep[rows] = True
            newid = np.cumsum(keep) - 1
            sel = keep[b["rowid"]]
            rid = newid[b["rowid"][sel]]
            return rowptr_of(rid, rows.shape[0]), b["comp"][sel], b["ts"][sel], b["hh"][sel]

        def bonus_of(comp, ts, hh):
            hc = np.minimum(hh, 0.999)
            has = ts >= 0
            cnt_t = k_pair_count(comp, np.where(has, ts, 0), indptr, idx, valc)
            lam = lam_exact[comp]
            pt = lam * cnt_t * inv_N[comp] + (1.0 - lam) * p_uni[np.where(has, ts, 0)]
            return np.where(has & (hc > 0), np.log(hc / (1.0 - hc) + pt) - np.log(pt), 0.0)

        for arm in ("engine", "ngram", "union"):
            for ind in (False, True):
                t0 = time.time()
                bh = base_of("ho", arm, "none", False, ind)
                rpf, cf, tf, hf = sub(bh, fit_rows, "ho")
                bf = bonus_of(cf, tf, hf)
                bkf = bucket[cf]
                used = np.unique(bkf)
                goldf = sets["ho"]["gold"][fit_rows]
                a_init = 1.0 / max(1.0, float(np.diff(rpf).mean()))
                from scipy.optimize import minimize

                def obj(theta):
                    alpha = np.zeros(n_buckets)
                    alpha[used] = theta[:-1]
                    ll, grad, gb, _, _, _ = k_loglin(rpf, cf, bkf, alpha, theta[-1], indptr, idx, gval, logpu, goldf, tf, bf, n_buckets, True)
                    R = goldf.shape[0]
                    return -ll / R, -np.append(grad[used], gb) / R
                res = minimize(obj, np.append(np.full(used.shape[0], a_init), 1.0), jac=True, method="L-BFGS-B",
                               options={"maxiter": args.loglin_iters})
                alpha = np.zeros(n_buckets)
                alpha[used] = res.x[:-1]
                beta0 = float(res.x[-1])
                rpc, cc, tc, hc_ = sub(bh, chk_rows, "ho")
                llc, _, _, _, _, _ = k_loglin(rpc, cc, bucket[cc], alpha, beta0, indptr, idx, gval, logpu,
                                              sets["ho"]["gold"][chk_rows], tc, bonus_of(cc, tc, hc_), n_buckets, False)
                bv = base_of("val", arm, "none", False, ind)
                _, _, _, pg, rank, maxerr = k_loglin(bv["rp"], bv["comp"], bucket[bv["comp"]], alpha, beta0, indptr, idx, gval, logpu,
                                                     sets["val"]["gold"], bv["ts"], bonus_of(bv["comp"], bv["ts"], bv["hh"]), n_buckets, False)
                assert maxerr < 1e-9 and np.all(pg > 0)
                name = f"{arm}.loglin" + ("+induction" if ind else "")
                row = {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
                       "bits_per_token": float(-np.log(pg).mean() / np.log(2.0)),
                       "heldout_bits_per_token": float(-llc / chk_rows.shape[0] / np.log(2.0)),
                       "arm": arm, "estimator": "loglin", "induction": ind, "max_row_sum_error": float(maxerr),
                       "fit": {"fit_rows": int(fit_rows.shape[0]), "heldout_check_rows": int(chk_rows.shape[0]), "iters": int(res.nit),
                               "converged": bool(res.success), "beta0": beta0, "n_params": int(used.shape[0] + 1),
                               "optimizer": "L-BFGS-B on the held-out fit rows"}}
                record(name, row, pg)
                log(f"{name:34s} top1 {row['top1']:.4f} top5 {row['top5']:.4f} bits {row['bits_per_token']:.4f} "
                    f"(held-out {row['heldout_bits_per_token']:.4f}) [{time.time() - t0:.1f} s, {res.nit} iters]")

    # ---------------------------------------------------------------- KN-5 comparators with probabilities
    if args.kn:
        from tmlm.eval.baselines.ngram import BidirectionalKN, reverse_docs, reversed_positions
        t0 = time.time()
        # the SAME counting documents (held-out docs removed) as every row above
        keep_docs = np.setdiff1d(np.arange(n_doc), ho_docs)
        seg = [tok_tr[ds_tr[d]:ds_tr[d + 1]] for d in keep_docs]
        tok_k = np.concatenate(seg).astype(np.uint16) if args.max_count_pos <= 0 else tok_tr[cnt_pos].astype(np.uint16)
        if args.max_count_pos <= 0:
            ds_k = np.zeros(len(seg) + 1, np.int64)
            ds_k[1:] = np.cumsum([x.shape[0] for x in seg])
        else:
            ds_k = np.array([0, tok_k.shape[0]], np.int64)
        bi = BidirectionalKN.fit(tok_k, ds_k, 5, V)
        vt = val.tok
        vds = val.doc_start.astype(np.int64)
        rtok = reverse_docs(vt, vds)
        rhid = reverse_docs(hidden, vds)
        rpos = reversed_positions(targets, vds)
        res = {k: [] for k in ("left", "both")}
        for i in range(0, nT, 2048):
            ctx, valid = bi.fwd.contexts(vt, vds, targets[i:i + 2048], hidden, 5)
            pf = bi.fwd.dist(ctx, valid, 5)
            ctx, valid = bi.bwd.contexts(rtok, vds, rpos[i:i + 2048], rhid, 5)
            pb = bi.bwd.dist(ctx, valid, 5)
            prod = pf * pb
            prod /= prod.sum(axis=1, keepdims=True)
            for k, P in (("left", pf), ("both", prod)):
                assert np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)
                g = gold[i:i + 2048]
                pgk = P[np.arange(g.shape[0]), g]
                rk = (P > pgk[:, None]).sum(axis=1) + ((P == pgk[:, None]) & (np.arange(V)[None, :] < g[:, None])).sum(axis=1)
                res[k].append((pgk, rk))
        for k in res:
            pgk = np.concatenate([a for a, _ in res[k]])
            rk = np.concatenate([b for _, b in res[k]])
            nm = {"left": "kn5_left_visible", "both": "kn5_both_visible_renormalized_product"}[k]
            ROWS[f"comparator.{nm}"] = {"top1": float((rk == 0).mean()), "top5": float((rk < 5).mean()),
                                        "bits_per_token": float(-np.log(pgk).mean() / np.log(2.0))}
            per_target[f"comparator.{nm}"] = pgk.astype(np.float32)
            for kk, vv in ROWS[f"comparator.{nm}"].items():
                M(f"rows.comparator.{nm}.{kk}", vv, note="modified KN-5 fitted on the counting documents of this probe; "
                  "the product row is renormalized over V, so it is a proper distribution")
            log(f"comparator {nm}: {ROWS[f'comparator.{nm}']}")
        M("comparator.kn5_fit_and_eval_wall_s", time.time() - t0)

    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        np.savez_compressed(Path(run_dir) / "per_target_gold_prob.npz", targets=targets, gold=gold, **per_target)
        Path(run_dir, "rows.json").write_text(json.dumps(ROWS, indent=1, default=float) + "\n")
    M("wall_s_total", time.time() - T0)
    flush_metrics()
    if not run_dir:
        print(json.dumps(ROWS, indent=1, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
