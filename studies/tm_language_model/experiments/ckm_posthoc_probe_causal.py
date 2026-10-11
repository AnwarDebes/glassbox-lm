"""CKM post-hoc count read-out probe, CAUSAL, on the corpus-scale bank (EXPLORATORY: not a
pre-registered result, no threshold of docs/13 is tested here; validation split only, the test split
is never loaded). The causal twin of experiments/ckm_posthoc_probe.py (masked tiny LM), in the
setting of the pre-registered headline (protocol CP: causal, V 8000, bits per byte vs KN-5).

Question (measured BEFORE anything is
built into the engine): if every clause of the saved causal bank kept EXACT counts of the next
tokens it fires on over an N-token training prefix, and the fired clauses were mixed into a proper
distribution, how many bits per byte would that give, against (a) the engine's own signed votes
(2.1749 bits per byte after 4 epochs over all 135M tokens), (b) KN-2 and KN-5 refitted on the SAME
prefix, and (c) plain left n-gram contexts (left-1 to left-5) under the SAME estimators? And how
much does in-context recall (clause-keyed induction, document cache) add?

Nothing here trains or changes the engine. The epoch-4 state of results/20260923T042350Z_CP_anchor
(checkpoint_epoch4.npz; causal, V 8000, C 1,048,576, 13 bags) is loaded into the CUDA engine and
only its own forward pass is called: CudaEngine._forward (the batched forward that predict_scores
and fired_clauses use; read-only, no engine code is modified), checked against the public
fired_clauses, fired_clauses_bruteforce and predict_scores. In causal mode the hidden() function is
0 everywhere (tm_core.h tm_hidden), so `epoch` has no effect and every earlier in-document position
is visible; the target itself reads SYM_MASK and positions outside the document read SYM_EDGE.

Pieces that run on the GPU (shared lock): the forward pass, the exact (clause, next token) counts
(radix sort of 32-bit keys per clause range), gold-count lookups, and the dense mixture rows used
for top-1, top-5 and the log-linear rows (one block per row, V accumulators in shared memory,
integer fixed-point atomics so that every result is deterministic). Bits of the linear mixtures come
from the float64 CPU gold-only path; the GPU gold probabilities are cross-checked against it.

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMLM_CPU_SLOTS=1
    source env/env.sh
    scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- \\
        scripts/run_logged.py --name ckm_probe_causal_exploratory --config <scratch json> -- \\
        .venv-tm/bin/python -u experiments/ckm_posthoc_probe_causal.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from engine_common import ROOT
from ckm_posthoc_probe import (k_gval, k_induction_candidates, k_lin_obj, k_loglin,  # noqa: E402
                               k_recursive_weights, k_row_argmax, k_truncate, k_vote_scores, merge_pairs,
                               pairs_from_matrix, rowptr_of)

sys.path.insert(0, str(ROOT / "src"))
CP_DIR = ROOT / "results" / "20260923T042350Z_CP_anchor"
EPOCH = 4                      # irrelevant in causal mode (tm_hidden is 0), kept for the record
CORPUS = "wikitext103"
NG_ORDERS = 5                  # left-1 .. left-5 contexts
LAM_CAP = 0.999
NINC_CAP = 12
LOGN_CAP = 23
RBITS = 18                     # clause ranges of 2^18 ids: (j & (2^18 - 1)) * 8000 + y < 2^32
LN2 = math.log(2.0)
T0 = time.time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}


def flush_metrics() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


# ------------------------------------------------------------------ CPU kernels (new here)
@njit(cache=False)
def k_ind_stats_skip(ptr, ids, ng, doc_lo, tok, is_cnt, n_comp):
    """Copy statistics over the COUNTING positions of the prefix, in position order: for every
    component present at p (fired clause or found n-gram context), the most recent earlier position
    q of the same document where it was also present; try if such q exists, hit if tok[q] == tok[p].
    Held-out documents are skipped entirely (they never touch `last`)."""
    last = np.full(n_comp, -1, np.int64)
    tries = np.zeros(n_comp, np.int64)
    hits = np.zeros(n_comp, np.int64)
    for p in range(ptr.shape[0] - 1):
        if is_cnt[p] == 0:
            continue
        dl = doc_lo[p]
        y = tok[p]
        for e in range(ptr[p], ptr[p + 1]):
            k = ids[e]
            q = last[k]
            if q >= dl:
                tries[k] += 1
                if tok[q] == y:
                    hits[k] += 1
            last[k] = p
        for t in range(ng.shape[1]):
            k = ng[p, t]
            if k >= 0:
                q = last[k]
                if q >= dl:
                    tries[k] += 1
                    if tok[q] == y:
                        hits[k] += 1
                last[k] = p
    return tries, hits


@njit(cache=False)
def k_sort_rows(rp, key):
    """Permutation that sorts the pairs of every row by `key` (keys unique inside a row)."""
    perm = np.empty(key.shape[0], np.int64)
    for r in range(rp.shape[0] - 1):
        lo, hi = rp[r], rp[r + 1]
        if hi - lo == 0:
            continue
        o = np.argsort(key[lo:hi])
        for t in range(hi - lo):
            perm[lo + t] = lo + o[t]
    return perm


@njit(cache=False)
def k_recursive_logweights(rp, lam):
    """Recursive interpolation weights in LOG space: w_i = lam_i prod_{j<i} (1 - lam_j), floor = prod (1 - lam_j).
    With about 270 fired clauses the linear-space products underflow float64; the log form does not."""
    n = rp.shape[0] - 1
    lw = np.empty(lam.shape[0])
    lwu = np.empty(n)
    for r in range(n):
        rem = 0.0
        for i in range(rp[r], rp[r + 1]):
            l = lam[i]
            lw[i] = (rem + np.log(l)) if l > 0.0 else -np.inf
            rem += np.log1p(-l) if l < 1.0 else -np.inf
        lwu[r] = rem
    return lw, lwu


@njit(cache=False)
def k_gold_eval_log(rowptr, lw, pkfull, lwuni, pu_g, gold, pos, doc_lo, tok, V):
    """Log gold probability of a linear mixture given log weights (log-sum-exp, no underflow), the weight-sum
    error, and the unigram document cache gold probability (causal: every earlier in-document token)."""
    n = rowptr.shape[0] - 1
    lp = np.empty(n)
    pc = np.zeros(n)
    has = np.zeros(n, np.uint8)
    werr = 0.0
    hist = np.zeros(V, np.int64)
    cur = -1
    cursor = 0
    for r in range(n):
        base = lwuni[r] + np.log(pu_g[r])
        m = base
        for i in range(rowptr[r], rowptr[r + 1]):
            if pkfull[i] > 0.0 and lw[i] > -np.inf:
                t = lw[i] + np.log(pkfull[i])
                if t > m:
                    m = t
        sacc = np.exp(base - m) if base > -np.inf else 0.0
        ws = np.exp(lwuni[r])
        for i in range(rowptr[r], rowptr[r + 1]):
            ws += np.exp(lw[i])
            if pkfull[i] > 0.0 and lw[i] > -np.inf:
                sacc += np.exp(lw[i] + np.log(pkfull[i]) - m)
        lp[r] = m + np.log(sacc)
        e = abs(ws - 1.0)
        if e > werr:
            werr = e
        if doc_lo[r] != cur:
            for c in range(V):
                hist[c] = 0
            cur = doc_lo[r]
            cursor = cur
        while cursor < pos[r]:
            hist[tok[cursor]] += 1
            cursor += 1
        nv = pos[r] - cur
        if nv > 0:
            has[r] = 1
            pc[r] = hist[gold[r]] / nv
    return lp, pc, has, werr


@njit(cache=False)
def k_em_jm_log(rowptr, pk, bkt, pu_g, lam_b, iters, cap):
    """EM for bucketed recursive (Jelinek-Mercer style deleted) interpolation, in log space."""
    n = rowptr.shape[0] - 1
    nb = lam_b.shape[0]
    maxlen = 1
    for r in range(n):
        if rowptr[r + 1] - rowptr[r] > maxlen:
            maxlen = rowptr[r + 1] - rowptr[r]
    lt = np.empty(maxlen)
    trace = np.empty(iters + 1)
    for it in range(iters + 1):
        stop = np.zeros(nb)
        pas = np.zeros(nb)
        ll = 0.0
        for r in range(n):
            lo, hi = rowptr[r], rowptr[r + 1]
            rem = 0.0
            m = -np.inf
            for i in range(lo, hi):
                lm = lam_b[bkt[i]]
                lt[i - lo] = (rem + np.log(lm) + np.log(pk[i])) if (pk[i] > 0.0 and lm > 0.0) else -np.inf
                if lt[i - lo] > m:
                    m = lt[i - lo]
                rem += np.log1p(-lm)
            lpu = rem + np.log(pu_g[r])
            if lpu > m:
                m = lpu
            s = np.exp(lpu - m)
            for i in range(lo, hi):
                if lt[i - lo] > -np.inf:
                    s += np.exp(lt[i - lo] - m)
            lpr = m + np.log(s)
            ll += lpr
            tail = np.exp(lpu - lpr)
            for i in range(hi - 1, lo - 1, -1):
                ri = np.exp(lt[i - lo] - lpr) if lt[i - lo] > -np.inf else 0.0
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
def k_cache_gold(pos, doc_lo, tok, gold, V):
    """Unigram document cache at every row (rows sorted by position): gold probability and whether
    the cache is non-empty (causal: every earlier in-document token is visible)."""
    n = pos.shape[0]
    pc = np.zeros(n)
    has = np.zeros(n, np.uint8)
    hist = np.zeros(V, np.int64)
    cur = -1
    cursor = 0
    for r in range(n):
        if doc_lo[r] != cur:
            for c in range(V):
                hist[c] = 0
            cur = doc_lo[r]
            cursor = cur
        while cursor < pos[r]:
            hist[tok[cursor]] += 1
            cursor += 1
        nv = pos[r] - cur
        if nv > 0:
            has[r] = 1
            pc[r] = hist[gold[r]] / nv
    return pc, has


@njit(cache=False)
def k_cache_rows(pos, doc_lo, tok, V):
    """Dense unigram cache rows (count / n, zero rows where the cache is empty); rows sorted by position."""
    n = pos.shape[0]
    out = np.zeros((n, V))
    has = np.zeros(n, np.uint8)
    hist = np.zeros(V, np.int64)
    cur = -1
    cursor = 0
    for r in range(n):
        if doc_lo[r] != cur:
            for c in range(V):
                hist[c] = 0
            cur = doc_lo[r]
            cursor = cur
        while cursor < pos[r]:
            hist[tok[cursor]] += 1
            cursor += 1
        nv = pos[r] - cur
        if nv > 0:
            has[r] = 1
            for c in range(V):
                out[r, c] = hist[c] / nv
    return out, has


# ------------------------------------------------------------------ GPU kernels (new here)
GPU_SRC = r"""
#define BS 512
extern "C" __global__ void pair_count(const long long n, const int* __restrict__ comp, const int* __restrict__ tgt,
        const long long* __restrict__ indptr, const unsigned short* __restrict__ idx, const int* __restrict__ val,
        int* __restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int t = tgt[i];
        if (t < 0) { out[i] = 0; continue; }
        const int k = comp[i];
        long long lo = indptr[k], hi = indptr[k + 1];
        const long long end = hi;
        while (lo < hi) { const long long mid = (lo + hi) >> 1; if ((int)idx[mid] < t) lo = mid + 1; else hi = mid; }
        out[i] = (lo < end && (int)idx[lo] == t) ? val[lo] : 0;
    }
}

extern "C" __global__ void __launch_bounds__(BS) mix_rows(
        const long long* __restrict__ rowptr, const int* __restrict__ comp, const double* __restrict__ coef,
        const int* __restrict__ tstar, const double* __restrict__ tcoef, const double* __restrict__ tbon,
        const long long* __restrict__ indptr, const unsigned short* __restrict__ idx,
        const int* __restrict__ evi, const double* __restrict__ evd, const int use_evd,
        const double* __restrict__ base_row, const double* __restrict__ base_vec,
        const int mode, const double scale,
        const double mu, const long long* __restrict__ row_pos, const long long* __restrict__ row_dl,
        const unsigned short* __restrict__ tok, int* __restrict__ hist_g,
        const int* __restrict__ gold, const int n_rows, const int V,
        double* __restrict__ out_pg, int* __restrict__ out_rank, int* __restrict__ out_top1, double* __restrict__ out_sum,
        double* __restrict__ out_lpg,
        const int want_grad, double* __restrict__ out_gpair, double* __restrict__ out_gbeta) {
    extern __shared__ long long acc[];
    double* P = (double*)acc;
    __shared__ double rd[BS];
    __shared__ double rb[BS];
    __shared__ int ri[BS];
    __shared__ int rk[BS];
    const int tid = threadIdx.x;
    int* hist = hist_g + (long long)blockIdx.x * V;
    for (int r = blockIdx.x; r < n_rows; r += gridDim.x) {
        const double b0 = base_row[r];
        for (int c = tid; c < V; c += BS) acc[c] = llrint(b0 * base_vec[c] * scale);
        const long long nctx = (mu > 0.0) ? (row_pos[r] - row_dl[r]) : 0;
        if (nctx > 0) for (int c = tid; c < V; c += BS) hist[c] = 0;
        __syncthreads();
        const long long p0 = rowptr[r], p1 = rowptr[r + 1];
        for (long long i = p0; i < p1; ++i) {
            const double cf = coef[i];
            if (cf != 0.0) {
                const int k = comp[i];
                const long long e1 = indptr[k + 1];
                if (use_evd) { for (long long e = indptr[k] + tid; e < e1; e += BS) atomicAdd((unsigned long long*)&acc[idx[e]], (unsigned long long)llrint(cf * evd[e] * scale)); }
                else { for (long long e = indptr[k] + tid; e < e1; e += BS) atomicAdd((unsigned long long*)&acc[idx[e]], (unsigned long long)llrint(cf * (double)evi[e] * scale)); }
            }
            if (tid == 0) {
                const int t = tstar[i];
                if (t >= 0) { const double tc = tcoef[i]; if (tc != 0.0) atomicAdd((unsigned long long*)&acc[t], (unsigned long long)llrint(tc * scale)); }
            }
        }
        if (nctx > 0) for (long long q = row_dl[r] + tid; q < row_pos[r]; q += BS) atomicAdd(&hist[tok[q]], 1);
        __syncthreads();
        for (int c = tid; c < V; c += BS) P[c] = (double)acc[c] / scale;
        __syncthreads();
        const int g = gold[r];
        double lpg = 0.0;
        if (mode == 1) {
            const double sg = P[g];
            double m = -1e300;
            for (int c = tid; c < V; c += BS) m = fmax(m, P[c]);
            rd[tid] = m; __syncthreads();
            for (int st = BS / 2; st > 0; st >>= 1) { if (tid < st) rd[tid] = fmax(rd[tid], rd[tid + st]); __syncthreads(); }
            m = rd[0]; __syncthreads();
            double z = 0.0;
            for (int c = tid; c < V; c += BS) { const double v = exp(P[c] - m); P[c] = v; z += v; }
            rd[tid] = z; __syncthreads();
            for (int st = BS / 2; st > 0; st >>= 1) { if (tid < st) rd[tid] += rd[tid + st]; __syncthreads(); }
            z = rd[0]; __syncthreads();
            lpg = sg - m - log(z);
            for (int c = tid; c < V; c += BS) P[c] = P[c] / z;
            __syncthreads();
        }
        if (nctx > 0) {
            const double inv = mu / (double)nctx;
            for (int c = tid; c < V; c += BS) P[c] = (1.0 - mu) * P[c] + inv * (double)hist[c];
            __syncthreads();
        }
        const double pg = P[g];
        if (mode == 0) lpg = log(pg);
        double s = 0.0, best = -1.0; int cnt = 0, bi = V;
        for (int c = tid; c < V; c += BS) {
            const double v = P[c];
            s += v;
            if (v > pg || (v == pg && c < g)) ++cnt;
            if (v > best) { best = v; bi = c; }
        }
        rd[tid] = s; rb[tid] = best; ri[tid] = cnt; rk[tid] = bi;
        __syncthreads();
        for (int st = BS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                rd[tid] += rd[tid + st]; ri[tid] += ri[tid + st];
                const double ob = rb[tid + st]; const int ok = rk[tid + st];
                if (ob > rb[tid] || (ob == rb[tid] && ok < rk[tid])) { rb[tid] = ob; rk[tid] = ok; }
            }
            __syncthreads();
        }
        if (tid == 0) { out_pg[r] = pg; out_rank[r] = ri[0]; out_top1[r] = rk[0]; out_sum[r] = rd[0]; out_lpg[r] = lpg; }
        __syncthreads();
        if (want_grad) {
            double el = 0.0;
            for (int c = tid; c < V; c += BS) el += P[c] * base_vec[c];
            rd[tid] = el; __syncthreads();
            for (int st = BS / 2; st > 0; st >>= 1) { if (tid < st) rd[tid] += rd[tid + st]; __syncthreads(); }
            if (tid == 0) out_gbeta[r] = base_vec[g] - rd[0];
            const int lane = tid & 31, w = tid >> 5;
            for (long long i = p0 + w; i < p1; i += BS / 32) {
                const int k = comp[i];
                double eg = 0.0, gy = 0.0;
                const long long e1 = indptr[k + 1];
                for (long long e = indptr[k] + lane; e < e1; e += 32) { const double gv = evd[e]; const int c = idx[e]; eg += P[c] * gv; if (c == g) gy += gv; }
                for (int o = 16; o > 0; o >>= 1) { eg += __shfl_down_sync(0xffffffffu, eg, o); gy += __shfl_down_sync(0xffffffffu, gy, o); }
                if (lane == 0) { const int t = tstar[i]; if (t >= 0) { eg += P[t] * tbon[i]; if (t == g) gy += tbon[i]; } out_gpair[i] = gy - eg; }
            }
            __syncthreads();
        }
    }
}
"""
SCALE_LIN = float(2 ** 60)     # fixed point for probabilities (each term <= 1, row sum ~ 1 < 2^3)
SCALE_LOG = float(2 ** 40)     # fixed point for log-linear scores


class Gpu:
    """The two probe kernels and the CSR copies on the device."""

    def __init__(self, V: int, grid: int = 1024) -> None:
        import cupy as cp
        self.cp = cp
        self.V = V
        self.grid = grid
        mod = cp.RawModule(code=GPU_SRC, options=("-std=c++14",))
        self.k_pair = mod.get_function("pair_count")
        self.k_mix = mod.get_function("mix_rows")
        self.smem = 8 * V
        self.k_mix.max_dynamic_shared_size_bytes = self.smem
        self.hist = cp.zeros(grid * V, dtype=cp.int32)
        self.dummy_d = cp.zeros(1, dtype=cp.float64)
        self.dummy_i = cp.zeros(1, dtype=cp.int32)
        self.dummy_l = cp.zeros(1, dtype=cp.int64)
        self.csr: dict = {}

    def put_csr(self, name, indptr, idx, val=None, evd=None):
        cp = self.cp
        self.csr[name] = dict(indptr=cp.asarray(indptr.astype(np.int64)), idx=cp.asarray(idx.astype(np.uint16)),
                              val=cp.asarray(val.astype(np.int32)) if val is not None else None,
                              evd=cp.asarray(evd.astype(np.float64)) if evd is not None else None)

    def drop_csr(self, name):
        self.csr.pop(name, None)
        self.cp.get_default_memory_pool().free_all_blocks()

    def pair_count(self, name, comp, tgt, chunk=1 << 25):
        cp = self.cp
        c = self.csr[name]
        out = np.zeros(comp.shape[0], np.int64)
        for lo in range(0, comp.shape[0], chunk):
            dc = cp.asarray(comp[lo:lo + chunk].astype(np.int32))
            dt = cp.asarray(tgt[lo:lo + chunk].astype(np.int32))
            o = cp.zeros(dc.shape[0], dtype=cp.int32)
            n = int(dc.shape[0])
            self.k_pair((min(4096, (n + 255) // 256),), (256,), (np.int64(n), dc, dt, c["indptr"], c["idx"], c["val"], o))
            out[lo:lo + n] = o.get()
        return out

    def mix(self, name, rp, comp, coef, tstar, tcoef, base_row, base_vec, mode, gold, *, mu=0.0, pos=None, dl=None,
            tok_dev=None, use_evd=False, want_grad=False, tbon=None, rows_per_chunk=24576):
        cp = self.cp
        c = self.csr[name]
        n = rp.shape[0] - 1
        out = dict(pg=np.zeros(n), rank=np.zeros(n, np.int64), top1=np.zeros(n, np.int64), sum=np.zeros(n), lpg=np.zeros(n))
        if want_grad:
            out["gpair"] = np.zeros(comp.shape[0])
            out["gbeta"] = np.zeros(n)
        scale = SCALE_LOG if mode == 1 else SCALE_LIN
        for r0 in range(0, n, rows_per_chunk):
            r1 = min(n, r0 + rows_per_chunk)
            a, b = int(rp[r0]), int(rp[r1])
            nr = r1 - r0
            d_rp = cp.asarray((rp[r0:r1 + 1] - a).astype(np.int64))
            d_comp = cp.asarray(comp[a:b].astype(np.int32))
            d_coef = cp.asarray(coef[a:b].astype(np.float64))
            d_ts = cp.asarray(tstar[a:b].astype(np.int32))
            d_tc = cp.asarray(tcoef[a:b].astype(np.float64))
            d_tb = cp.asarray(tbon[a:b].astype(np.float64)) if tbon is not None else self.dummy_d
            d_base = cp.asarray(np.broadcast_to(np.asarray(base_row, np.float64), (n,))[r0:r1].copy())
            d_vec = cp.asarray(np.asarray(base_vec, np.float64))
            d_gold = cp.asarray(gold[r0:r1].astype(np.int32))
            d_pos = cp.asarray(pos[r0:r1].astype(np.int64)) if mu > 0 else self.dummy_l
            d_dl = cp.asarray(dl[r0:r1].astype(np.int64)) if mu > 0 else self.dummy_l
            o_pg = cp.zeros(nr, dtype=cp.float64)
            o_rank = cp.zeros(nr, dtype=cp.int32)
            o_top = cp.zeros(nr, dtype=cp.int32)
            o_sum = cp.zeros(nr, dtype=cp.float64)
            o_lpg = cp.zeros(nr, dtype=cp.float64)
            o_gp = cp.zeros(max(1, b - a), dtype=cp.float64) if want_grad else self.dummy_d
            o_gb = cp.zeros(nr, dtype=cp.float64) if want_grad else self.dummy_d
            grid = min(self.grid, nr)
            self.k_mix((grid,), (512,), (d_rp, d_comp, d_coef, d_ts, d_tc, d_tb, c["indptr"], c["idx"],
                                         c["val"] if c["val"] is not None else self.dummy_i,
                                         c["evd"] if c["evd"] is not None else self.dummy_d, np.int32(1 if use_evd else 0),
                                         d_base, d_vec, np.int32(mode), np.float64(scale), np.float64(mu), d_pos, d_dl,
                                         tok_dev if tok_dev is not None else cp.zeros(1, dtype=cp.uint16), self.hist,
                                         d_gold, np.int32(nr), np.int32(self.V), o_pg, o_rank, o_top, o_sum, o_lpg,
                                         np.int32(1 if want_grad else 0), o_gp, o_gb), shared_mem=self.smem)
            out["pg"][r0:r1] = o_pg.get()
            out["rank"][r0:r1] = o_rank.get()
            out["top1"][r0:r1] = o_top.get()
            out["sum"][r0:r1] = o_sum.get()
            out["lpg"][r0:r1] = o_lpg.get()
            if want_grad:
                out["gpair"][a:b] = o_gp.get()[: b - a]
                out["gbeta"][r0:r1] = o_gb.get()
        return out


# ------------------------------------------------------------------ engine passes
def forward_pass(eng, n: int, chunk: int, label: str, *, tok_dev=None, is_ho_dev=None, V=8000, scores_fn=None):
    """Fired lists of positions 0..n-1 of the engine's corpus through CudaEngine._forward (the batched forward
    of predict_scores; read-only). Returns the CSR (ptr int64, ids uint32 ascending per position), the per-range
    uint32 count keys of the counting positions (if tok_dev is given), candidate total and wall."""
    cp = eng.cp
    ptr = np.zeros(n + 1, dtype=np.int64)
    ids_parts = []
    n_ranges = (int(eng.cfg.n_clauses) + (1 << RBITS) - 1) >> RBITS
    key_parts = [[] for _ in range(n_ranges)]
    n_cand = 0
    t0 = time.time()
    step = max(1, (n // chunk) // 10)
    for ci, lo in enumerate(range(0, n, chunk)):
        hi = min(n, lo + chunk)
        ids = cp.arange(lo, hi, dtype=cp.uint32)
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, ids, None, 0)
        eng._marks = []
        n_cand += int(fw.n_candidates)
        fe, fj = fw.fired_ex, fw.fired_j
        if fe.size:
            assert bool(cp.all(fe[1:] >= fe[:-1])), "fired_ex is not example-major"
            order = cp.argsort((fe.astype(cp.uint64) << cp.uint64(32)) | fj.astype(cp.uint64))
            fe, fj = fe[order], fj[order]
        cnts = cp.bincount(fe.astype(cp.int64), minlength=hi - lo) if fe.size else cp.zeros(hi - lo, dtype=cp.int64)
        ptr[lo + 1:hi + 1] = ptr[lo] + np.cumsum(cnts.get())
        ids_parts.append(fj.get().astype(np.uint32))
        if tok_dev is not None and fe.size:
            p = fe.astype(cp.int64) + lo
            keep = is_ho_dev[p] == 0
            y = tok_dev[p].astype(cp.uint32)
            key = (fj & cp.uint32((1 << RBITS) - 1)) * cp.uint32(V) + y
            rng = fj >> cp.uint32(RBITS)
            for r in range(n_ranges):
                sel = keep & (rng == r)
                key_parts[r].append(key[sel].get())
        if scores_fn is not None:
            scores_fn(lo, hi, fw.scores)
        if (ci + 1) % step == 0:
            el = time.time() - t0
            log(f"  {label}: {hi}/{n} positions, {hi / el:.0f} pos/s, fired/pos {ptr[hi] / hi:.1f}, cand/pos {n_cand / hi:.0f}")
    eng.synchronize()
    wall = time.time() - t0
    ids_all = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.uint32)
    return ptr, ids_all, key_parts, n_cand, wall


def gpu_counts(key_parts, V: int, C: int):
    """Exact (clause, target) counts from the per-range 32-bit keys: radix sort + run lengths on the GPU."""
    import cupy as cp
    js, ys, vs = [], [], []
    total = 0
    for r, parts in enumerate(key_parts):
        if not parts:
            continue
        k = np.concatenate(parts)
        total += int(k.shape[0])
        if k.shape[0] == 0:
            continue
        d = cp.asarray(k)
        d.sort()
        brk = cp.flatnonzero(d[1:] != d[:-1]) + 1
        starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
        uk = d[starts]
        cnt = cp.diff(cp.concatenate([starts, cp.asarray([int(d.shape[0])], dtype=starts.dtype)]))
        ukh = uk.get().astype(np.int64)
        js.append((r << RBITS) + ukh // V)
        ys.append((ukh % V).astype(np.uint16))
        vs.append(cnt.get().astype(np.int32))
        del d, brk, starts, uk, cnt
        cp.get_default_memory_pool().free_all_blocks()
    j = np.concatenate(js)
    idx = np.concatenate(ys)
    val = np.concatenate(vs)
    assert np.all(np.diff(j) >= 0)
    indptr = np.zeros(C + 1, np.int64)
    indptr[1:] = np.cumsum(np.bincount(j, minlength=C))
    return indptr, idx, val, total


# ------------------------------------------------------------------ n-gram contexts
def ngram_level_ids(tok_a, lo_a, tok_b, lo_b, K: int, V: int):
    """Joint ids of the left-k contexts (k = 1..K) of every position of two streams (prefix, validation);
    a token before the document start is the edge symbol V. Level k id identifies (t[p-k], ..., t[p-1])."""
    na = tok_a.shape[0]
    pa = np.arange(na, dtype=np.int64)
    pb = np.arange(tok_b.shape[0], dtype=np.int64)
    out = []
    prev = None
    for k in range(1, K + 1):
        qa, qb = pa - k, pb - k
        ta = np.where(qa >= lo_a, tok_a[np.maximum(qa, 0)], V).astype(np.int64)
        tb = np.where(qb >= lo_b, tok_b[np.maximum(qb, 0)], V).astype(np.int64)
        if k == 1:
            ia, ib = ta, tb
        else:
            key = np.concatenate([prev[0] * (V + 1) + ta, prev[1] * (V + 1) + tb])
            _, inv = np.unique(key, return_inverse=True)
            inv = inv.reshape(-1).astype(np.int64)
            ia, ib = inv[:na], inv[na:]
        out.append((ia, ib))
        prev = (ia, ib)
    return out


def doc_lo_of(doc_start: np.ndarray, n: int) -> np.ndarray:
    ds = doc_start.astype(np.int64)
    return np.repeat(ds[:-1], np.diff(ds))[:n]


def csr_rows(ptr, ids, positions):
    """Sub-CSR of the given positions (ascending)."""
    starts, ends = ptr[positions], ptr[positions + 1]
    lens = ends - starts
    rp = np.zeros(positions.shape[0] + 1, np.int64)
    rp[1:] = np.cumsum(lens)
    take = np.repeat(starts - rp[:-1], lens) + np.arange(rp[-1], dtype=np.int64)
    return rp, ids[take]


def doc_boot(bits_a, bits_b, doc_of_row, n_docs, B=2000, seed=0):
    """Document-block bootstrap of the mean per-token bits difference a - b (ratio of sums)."""
    d = np.bincount(doc_of_row, weights=bits_a - bits_b, minlength=n_docs)
    t = np.bincount(doc_of_row, minlength=n_docs).astype(np.float64)
    ok = t > 0
    d, t = d[ok], t[ok]
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 717, seed]))
    idx = rng.integers(0, d.shape[0], size=(B, d.shape[0]))
    stats = d[idx].sum(axis=1) / t[idx].sum(axis=1)
    return float(d.sum() / t.sum()), [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))], int(d.shape[0])


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CP_DIR / "checkpoint_epoch4.npz"))
    ap.add_argument("--n-tokens", type=int, default=5_000_000, help="training prefix, cut at a document boundary")
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--val-max-pos", type=int, default=0, help="smoke only: validation positions below this")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--fit-every", type=int, default=3, help="linear fits use every k-th held-out row")
    ap.add_argument("--loglin-fit-rows", type=int, default=12000)
    ap.add_argument("--loglin-iters", type=int, default=60)
    ap.add_argument("--lin-iters", type=int, default=300)
    ap.add_argument("--kn", type=int, default=1)
    ap.add_argument("--loglin", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    M("mode_smoke", bool(args.smoke))
    M("args", vars(args))

    import cupy as cp
    from tmlm.data.corpus_store import load_split
    from tmlm.decode.bank import class_bias_from_counts
    from tmlm.engine.checkpoint import load_checkpoint
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import contiguous, split_meta

    # ---------------------------------------------------------------- state, data
    cfg, st, meta = load_checkpoint(args.checkpoint)
    cp_metrics = json.loads((CP_DIR / "metrics.json").read_text())
    logged_hash = cp_metrics["epoch4.checkpoint"]["value"]["state_hash"]
    assert meta["state_hash"] == logged_hash, "checkpoint state hash differs from the logged epoch-4 hash"
    assert cfg.mode == "causal"
    V, C = int(cfg.vocab), int(cfg.n_clauses)
    M("state.checkpoint", str(Path(args.checkpoint).relative_to(ROOT)))
    M("state.state_hash", meta["state_hash"], note="equals epoch4.checkpoint.state_hash of results/20260923T042350Z_CP_anchor")
    M("state.engine_config", cfg.to_dict())
    full = load_split(CORPUS, V, "train")
    train_counts_full = np.bincount(np.asarray(full.tok).astype(np.int64), minlength=V)
    tr = contiguous(slice_train(full, args.n_tokens))
    del full
    val = contiguous(load_split(CORPUS, V, "validation"))
    val_meta, _ = split_meta(V, "validation")
    n_tr, n_va_tok = int(tr.tok.shape[0]), int(val.tok.shape[0])
    n_bytes = int(val_meta["n_bytes_utf8"])
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    tok_va = np.asarray(val.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    ds_va = np.asarray(val.doc_start).astype(np.int64)
    lo_tr = doc_lo_of(ds_tr, n_tr)
    lo_va = doc_lo_of(ds_va, n_va_tok)
    n_doc = ds_tr.shape[0] - 1
    M("data.prefix_tokens", n_tr, note="first tokens of WikiText-103 train at V 8000, cut at a document boundary (m3_tiny_lm.slice_train)")
    M("data.prefix_docs", n_doc)
    M("data.validation_tokens", n_va_tok)
    M("data.validation_bytes_utf8", n_bytes)
    M("data.validation_docs", int(ds_va.shape[0] - 1))
    log(f"prefix {n_tr} tokens {n_doc} docs; validation {n_va_tok} tokens {n_bytes} bytes")

    # held-out documents of the prefix (every fitted parameter and every estimator choice comes from them)
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, 1]))
    n_ho = max(1, int(round(args.heldout_frac * n_doc)))
    ho_docs = np.sort(rng.choice(n_doc, size=n_ho, replace=False))
    doc_of_tr = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_ho = np.isin(doc_of_tr, ho_docs)
    cnt_pos = np.flatnonzero(~is_ho)
    ho_pos = np.flatnonzero(is_ho)
    M("split.heldout_docs", int(n_ho), note="SeedSequence([20260923, 715, 1]), about 5 percent of the prefix documents")
    M("split.counting_positions", int(cnt_pos.shape[0]))
    M("split.heldout_positions", int(ho_pos.shape[0]))

    # ---------------------------------------------------------------- C1 forward passes
    pool = cp.get_default_memory_pool()
    t0 = time.time()
    eng = CudaEngine(cfg, tr)
    eng.load_state(st)
    eng.synchronize()
    M("engine.prefix_build_s", time.time() - t0)
    tok_dev = cp.asarray(np.asarray(tr.tok).astype(np.uint16))
    ho_dev = cp.asarray(is_ho.astype(np.uint8))
    log("prefix forward pass (causal, every position a target)")
    ptr_tr, ids_tr, key_parts, cand_tr, wall_tr = forward_pass(eng, n_tr, args.chunk, "prefix", tok_dev=tok_dev, is_ho_dev=ho_dev, V=V)
    M("throughput.prefix_forward_pos_per_s", n_tr / wall_tr, note="CudaEngine._forward + fired-list and count-key download; shared GPU (a Track A E1 cell runs next to it)")
    M("prefix.fired_per_position", float(ptr_tr[-1] / n_tr))
    M("prefix.candidates_per_position", float(cand_tr / n_tr))
    # API checks on the prefix engine
    rs = np.random.default_rng(12345)
    chk = np.sort(rs.choice(n_tr, size=100, replace=False))
    for i, p in enumerate(chk.tolist()):
        a = eng.fired_clauses(EPOCH, p)
        assert np.array_equal(a, ids_tr[ptr_tr[p]:ptr_tr[p + 1]]), f"prefix fired list differs from the API at {p}"
        if i < 3:
            b = eng.fired_clauses_bruteforce(EPOCH, p)
            assert np.array_equal(np.sort(b), ids_tr[ptr_tr[p]:ptr_tr[p + 1]]), f"prefix fired list differs from brute force at {p}"
    M("check.prefix_fired_vs_api", "100 random positions equal to fired_clauses, 3 equal to fired_clauses_bruteforce")
    del eng, tok_dev, ho_dev
    pool.free_all_blocks()
    t0 = time.time()
    indptr_e, idx_e, val_e, n_events = gpu_counts(key_parts, V, C)
    del key_parts
    pool.free_all_blocks()
    cs = np.concatenate([[0], np.cumsum(val_e.astype(np.int64))])
    N_e = (cs[indptr_e[1:]] - cs[indptr_e[:-1]]).astype(np.int64)
    del cs
    D_e = np.diff(indptr_e).astype(np.int64)
    assert int(N_e.sum()) == n_events
    M("counts.events", int(n_events), note="(clause, next token) pairs over the counting positions")
    M("counts.distinct_pairs", int(idx_e.shape[0]))
    M("counts.clauses_with_counts", int((N_e > 0).sum()))
    M("counts.N_median_of_counted", float(np.median(N_e[N_e > 0])))
    M("counts.D_mean_of_counted", float(D_e[N_e > 0].mean()))
    M("counts.D_gt64_fraction", float((D_e[N_e > 0] > 64).mean()))
    M("counts.gpu_sort_s", time.time() - t0)
    log(f"engine counts: {n_events} events, {idx_e.shape[0]} distinct pairs, {(N_e > 0).sum()} clauses ({time.time() - t0:.0f} s)")
    flush_metrics()

    # validation pass + pipeline check (C1)
    n_va = n_va_tok if args.val_max_pos <= 0 else min(args.val_max_pos, n_va_tok)
    cal = cp_metrics["epoch4.calibration"]["value"]
    a_q16, b_q16 = int(cal["a_q16"]), int(cal["b_q16"])
    bias = class_bias_from_counts(train_counts_full, V)
    d_bias = cp.asarray(bias.astype(np.int64))
    d_vtok = cp.asarray(tok_va.astype(np.int64))
    acc = {"raw_hits": 0, "cal_hits": 0, "cal_top5": 0, "cal_nats": 0.0, "worst_norm": 0.0}
    eng_lp = np.zeros(n_va)

    def on_scores(lo, hi, scores):
        g = d_vtok[lo:hi]
        rows = cp.arange(hi - lo)
        acc["raw_hits"] += int((cp.argmax(scores, axis=1) == g).sum())
        num = scores.astype(cp.int64) * (a_q16 * 256) + (b_q16 * d_bias)[None, :]
        acc["cal_hits"] += int((cp.argmax(num, axis=1) == g).sum())
        ng = num[rows, g][:, None]
        rank = (num > ng).sum(axis=1) + ((num == ng) & (cp.arange(V)[None, :] < g[:, None])).sum(axis=1)
        acc["cal_top5"] += int((rank < 5).sum())
        lg = num.astype(cp.float64) / float(1 << 24)
        m = lg.max(axis=1, keepdims=True)
        z = lg - m
        lse = cp.log(cp.exp(z).sum(axis=1, keepdims=True))
        lp = z - lse
        acc["worst_norm"] = max(acc["worst_norm"], float(cp.abs(cp.log(cp.exp(lp).sum(axis=1))).max()))
        lpg = lp[rows, g]
        acc["cal_nats"] += float(-lpg.sum())
        eng_lp[lo:hi] = lpg.get()

    t0 = time.time()
    eng_v = CudaEngine(cfg, val)
    eng_v.load_state(st)
    log("validation forward pass")
    ptr_va, ids_va, _, cand_va, wall_va = forward_pass(eng_v, n_va, 8192, "validation", scores_fn=on_scores)
    assert acc["worst_norm"] < 1e-6
    top1_raw = acc["raw_hits"] / n_va
    logged_raw = cp_metrics["epoch4.validation.causal_top1_raw_scores"]["value"]
    logged_bpb = cp_metrics["epoch4.validation.bits_per_byte"]["value"]
    M("throughput.validation_forward_pos_per_s", n_va / wall_va)
    M("validation.fired_per_position", float(ptr_va[-1] / n_va))
    M("validation.candidates_per_position", float(cand_va / n_va))
    M("C1.engine_raw_top1", top1_raw, note="argmax of the class sums (ties to the lowest id), our forward pass")
    M("C1.logged_raw_top1", logged_raw, source="results/20260923T042350Z_CP_anchor/metrics.json epoch4.validation.causal_top1_raw_scores")
    M("C1.engine_calibrated_top1", acc["cal_hits"] / n_va, note="logged epoch-4 calibration (a_q16, b_q16)")
    M("C1.engine_calibrated_top5", acc["cal_top5"] / n_va)
    M("C1.engine_bits_per_token_logged_calibration", acc["cal_nats"] / LN2 / n_va)
    if n_va == n_va_tok:
        bpb = acc["cal_nats"] / LN2 / n_bytes
        M("C1.engine_bits_per_byte_logged_calibration", bpb)
        M("C1.logged_bits_per_byte", logged_bpb)
        M("C1.passed", bool(top1_raw == logged_raw and abs(bpb - logged_bpb) < 1e-9))
    else:
        M("C1.passed", None, label="unverified", note="validation subset (smoke): the logged values cover the full split")
    # vote sums rebuilt from our fired lists vs the public predict_scores; fired lists vs the public API
    rows = np.sort(rs.choice(n_va, size=min(2000, n_va), replace=False))
    s_api = eng_v.predict_scores(EPOCH, rows.astype(np.uint32))
    rec = k_vote_scores(ptr_va, ids_va, rows, st["vote_cls"], st["vote_w"].astype(np.int32), V)
    mism = int((rec != s_api).any(axis=1).sum())
    M("C1.score_rows_mismatch_vs_predict_scores", mism, note="2000 random validation positions: vote sums rebuilt from our fired lists vs predict_scores")
    for i, p in enumerate(rows[:100].tolist()):
        assert np.array_equal(eng_v.fired_clauses(EPOCH, p), ids_va[ptr_va[p]:ptr_va[p + 1]])
        if i < 3:
            assert np.array_equal(np.sort(eng_v.fired_clauses_bruteforce(EPOCH, p)), ids_va[ptr_va[p]:ptr_va[p + 1]])
    M("check.validation_fired_vs_api", "100 positions equal to fired_clauses, 3 equal to fired_clauses_bruteforce")
    log(f"C1: raw top-1 {top1_raw:.6f} (logged {logged_raw:.6f}); calibrated bits/token "
        f"{acc['cal_nats'] / LN2 / n_va:.4f}; rebuilt-score mismatches {mism} ({time.time() - t0:.0f} s)")
    del eng_v, d_bias, d_vtok
    pool.free_all_blocks()
    flush_metrics()
    if mism != 0 or (n_va == n_va_tok and not METRICS["C1.passed"]["value"]):
        log("C1 FAILED: stop")
        return 3

    # ---------------------------------------------------------------- n-gram components (left-1 .. left-5)
    t0 = time.time()
    va_pos = np.arange(n_va, dtype=np.int64)
    levels = ngram_level_ids(tok_tr, lo_tr, tok_va[:n_va], lo_va[:n_va], NG_ORDERS, V)
    y_cnt = tok_tr[cnt_pos]
    ptr_parts, idx_parts, val_parts = [indptr_e], [idx_e], [val_e]
    alive = st["alive"].astype(bool)
    inc = (st["lit_state"] >= 128) & (st["lit_id"] != 0xFFFFFFFF) & alive[:, None]
    ninc_e = inc.sum(axis=1).astype(np.int64)
    M("engine.alive_clauses", int(alive.sum()))
    M("engine.n_inc_histogram", np.bincount(ninc_e, minlength=13).tolist(), note="included literals per clause (positive and negated, exact and bag)")
    N_parts, D_parts, ninc_parts, fam_parts = [N_e], [D_e], [ninc_e], [np.zeros(C, np.int64)]
    base = C
    ng_tr = np.full((n_tr, NG_ORDERS), -1, np.int64)
    ng_va = np.full((n_va, NG_ORDERS), -1, np.int64)
    for k, (ia, ib) in enumerate(levels):
        u, cnt = np.unique(ia[cnt_pos] * V + y_cnt, return_counts=True)
        ctx = u // V
        tg = (u % V).astype(np.uint16)
        ckeys, start = np.unique(ctx, return_index=True)
        indptr = np.append(start, u.shape[0]).astype(np.int64)
        ptr_parts.append(indptr[1:] + ptr_parts[-1][-1])
        idx_parts.append(tg)
        val_parts.append(cnt.astype(np.int32))
        N_parts.append(np.add.reduceat(cnt, start).astype(np.int64))
        D_parts.append(np.diff(indptr))
        ninc_parts.append(np.full(ckeys.shape[0], k + 1, np.int64))
        fam_parts.append(np.ones(ckeys.shape[0], np.int64))
        for ids_, out in ((ia, ng_tr), (ib, ng_va)):
            j = np.searchsorted(ckeys, ids_)
            jc = np.minimum(j, ckeys.shape[0] - 1)
            found = ckeys[jc] == ids_
            out[found, k] = base + jc[found]
        M(f"ngram.contexts_left{k + 1}", int(ckeys.shape[0]))
        base += ckeys.shape[0]
    del levels
    n_comp = base
    indptr = np.concatenate(ptr_parts)
    idx = np.concatenate(idx_parts)
    valc = np.concatenate(val_parts)
    N = np.concatenate(N_parts)
    D = np.concatenate(D_parts)
    ninc = np.concatenate(ninc_parts)
    fam = np.concatenate(fam_parts)
    del ptr_parts, idx_parts, val_parts, idx_e, val_e
    assert indptr.shape[0] == n_comp + 1 and indptr[-1] == idx.shape[0]
    inv_N = np.where(N > 0, 1.0 / np.maximum(N, 1), 0.0)
    lam_exact = np.where(N > 0, N / np.maximum(N + D, 1), 0.0)
    t_ptr, t_idx, t_val, S64 = k_truncate(indptr, idx, valc, 64)
    inv_S64 = np.where(S64 > 0, 1.0 / np.maximum(S64, 1), 0.0)
    lam_64 = np.where(N > 0, S64 / np.maximum(N + D, 1), 0.0)
    amax = k_row_argmax(indptr, idx, valc)
    uni = np.bincount(y_cnt, minlength=V).astype(np.float64)
    p_uni = (uni + 1.0) / (uni.sum() + V)
    assert abs(p_uni.sum() - 1.0) < 1e-12
    logn = np.minimum(np.floor(np.log2(np.maximum(N, 1))).astype(np.int64), LOGN_CAP)
    bucket = (fam * (NINC_CAP + 1) + np.minimum(ninc, NINC_CAP)) * (LOGN_CAP + 1) + logn
    n_buckets = 2 * (NINC_CAP + 1) * (LOGN_CAP + 1)
    has_counts = N > 0
    # spec order key: n_inc descending, N descending, id ascending
    spec_rank = np.empty(n_comp, np.int64)
    spec_rank[np.lexsort((np.arange(n_comp), -N, -ninc))] = np.arange(n_comp)
    M("components.total", int(n_comp))
    M("components.count_entries_total", int(idx.shape[0]))
    M("components.truncated64_entries_total", int(t_idx.shape[0]))
    log(f"n-gram tables and truncation: {n_comp - C} contexts, {idx.shape[0]} entries ({time.time() - t0:.0f} s)")
    flush_metrics()

    # ---------------------------------------------------------------- induction statistics on the counting positions
    t0 = time.time()
    tries, ihits = k_ind_stats_skip(ptr_tr, ids_tr, ng_tr, lo_tr, tok_tr, (~is_ho).astype(np.uint8), n_comp)
    h_comp = np.where(tries > 0, ihits / np.maximum(tries, 1), 0.0)
    M("induction.engine_hits_over_tries", float(ihits[:C].sum() / max(1, tries[:C].sum())), note="counting positions, document order, exact most recent earlier firing")
    M("induction.ngram_hits_over_tries", float(ihits[C:].sum() / max(1, tries[C:].sum())))
    M("induction.stats_s", time.time() - t0)
    log(f"induction statistics ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- row sets
    t0 = time.time()
    sets = {}
    ho_rp, ho_ids = csr_rows(ptr_tr, ids_tr, ho_pos)
    del ids_tr
    for sname, rp_e, ids_e, pos, tok, lo, ng in (("val", ptr_va, ids_va, va_pos, tok_va, lo_va[:n_va], ng_va),
                                                  ("ho", ho_rp, ho_ids, ho_pos, tok_tr, lo_tr[ho_pos], ng_tr[ho_pos])):
        n = pos.shape[0]
        gold = tok[pos]
        rowid_all = np.repeat(np.arange(n, dtype=np.int64), np.diff(rp_e))
        comp_all = ids_e.astype(np.int64)
        keep = has_counts[comp_all]
        eng_pairs = (rowid_all[keep], comp_all[keep])
        ng_pairs = pairs_from_matrix(ng)
        arms = {"engine": eng_pairs, "ngram": ng_pairs, "union": merge_pairs(eng_pairs, ng_pairs)}
        s = dict(n=n, pos=pos, gold=gold, tok=tok, lo=lo, arms={})
        vis = np.ones(tok.shape[0], np.uint8)
        for arm, (rowid, comp) in arms.items():
            rp = rowptr_of(rowid, n)
            ts = k_induction_candidates(rp_e, ids_e, ng, pos, lo, vis, np.arange(n, dtype=np.int64), rp, comp, tok, n_comp)
            s["arms"][arm] = dict(rowid=rowid.astype(np.int64), comp=comp.astype(np.int64), rp=rp, m=np.diff(rp), ts=ts.astype(np.int64))
        M(f"{sname}.engine_fired_per_row", float(rp_e[-1] / max(1, n)))
        M(f"{sname}.engine_fired_with_counts_per_row", float(eng_pairs[0].shape[0] / max(1, n)))
        M(f"{sname}.ngram_contexts_found_per_row", float(ng_pairs[0].shape[0] / max(1, n)))
        M(f"{sname}.frac_rows_without_counted_engine_clause", float((np.diff(s["arms"]["engine"]["rp"]) == 0).mean()))
        sets[sname] = s
    del ho_ids, ids_va, ng_tr
    log(f"row sets and induction candidates ({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------- GPU mixer, gold counts per pair
    t0 = time.time()
    gpu = Gpu(V)
    gpu.put_csr("full", indptr, idx, valc)
    gpu.put_csr("trunc", t_ptr, t_idx, t_val)
    tok_va_dev = cp.asarray(tok_va.astype(np.uint16))
    for sname, s in sets.items():
        for arm, A in s["arms"].items():
            g = s["gold"][A["rowid"]]
            A["cnt_full"] = gpu.pair_count("full", A["comp"], g)
            A["cnt_trunc"] = gpu.pair_count("trunc", A["comp"], g)
            A["cnt_ts"] = gpu.pair_count("full", A["comp"], np.where(A["ts"] >= 0, A["ts"], -1))
            A["spec"] = k_sort_rows(A["rp"], spec_rank[A["comp"]])
            # oracle bounds
            if sname == "val":
                n = s["n"]
                in_sup = np.bincount(A["rowid"], weights=(A["cnt_full"] > 0), minlength=n) > 0
                am = np.bincount(A["rowid"], weights=(amax[A["comp"]] == g), minlength=n) > 0
                M(f"oracle.{arm}.gold_in_union_of_supports", float(in_sup.mean()))
                M(f"oracle.{arm}.some_component_argmax_is_gold", float(am.mean()))
    # cross-check of the GPU gold counts against a CPU binary search on a sample of pairs
    A = sets["val"]["arms"]["union"]
    samp = np.random.default_rng(7).choice(A["comp"].shape[0], size=min(20000, A["comp"].shape[0]), replace=False)
    g = sets["val"]["gold"][A["rowid"][samp]]
    for i, k in enumerate(A["comp"][samp].tolist()):
        seg = idx[indptr[k]:indptr[k + 1]]
        j = np.searchsorted(seg, g[i])
        ref = int(valc[indptr[k] + j]) if j < seg.shape[0] and seg[j] == g[i] else 0
        assert ref == A["cnt_full"][samp[i]], "GPU gold count differs from the CPU lookup"
    M("check.gpu_gold_counts_vs_cpu", "20000 random validation union pairs equal")
    log(f"gold counts, spec orders, oracles ({time.time() - t0:.0f} s)")
    flush_metrics()

    # ---------------------------------------------------------------- estimators
    ROWS: dict[str, dict] = {}
    per_tok: dict[str, np.ndarray] = {}
    sv = sets["val"]
    doc_of_val = np.searchsorted(ds_va, np.arange(n_va), side="right") - 1
    n_val_docs = int(ds_va.shape[0] - 1)

    def arrays(sname, arm, order, trunc, induction):
        s = sets[sname]
        A = s["arms"][arm]
        perm = A["spec"] if order == "spec" else None
        pick = (lambda x: x[perm]) if perm is not None else (lambda x: x)
        comp = pick(A["comp"])
        cnt = pick(A["cnt_trunc"] if trunc else A["cnt_full"])
        inv = (inv_S64 if trunc else inv_N)[comp]
        pk = cnt * inv
        if induction:
            ts = pick(A["ts"])
            hh = np.where(ts >= 0, h_comp[comp], 0.0)
            pkfull = np.where(ts >= 0, hh * (ts == s["gold"][A["rowid"]]) + (1.0 - hh) * pk, pk)
        else:
            ts = np.full(comp.shape[0], -1, np.int64)
            hh = np.zeros(comp.shape[0])
            pkfull = pk
        return dict(rowid=A["rowid"], comp=comp, rp=A["rp"], m=A["m"], ts=ts, hh=hh, pkfull=pkfull, inv=inv)

    def weights(b, est, params, n):
        rowid, comp, rp, m = b["rowid"], b["comp"], b["rp"], b["m"]
        if est in ("wb", "wb64"):
            w, wuni = k_recursive_weights(rp, (lam_64 if est == "wb64" else lam_exact)[comp])
        elif est == "jm":
            w, wuni = k_recursive_weights(rp, params["lam_b"][bucket[comp]])
        elif est == "snm":
            beta = params["beta"]
            w = np.where(m[rowid] > 0, (1.0 - beta) / np.maximum(m[rowid], 1), 0.0)
            wuni = np.where(m > 0, beta, 1.0)
        elif est in ("lin", "lin64"):
            a = params["alpha"][bucket[comp]]
            Z = np.bincount(rowid, weights=a, minlength=n) + params["a0"]
            w = a / Z[rowid]
            wuni = params["a0"] / Z
        else:
            raise ValueError(est)
        return w, wuni

    def order_of(est):
        return "spec" if est in ("wb", "wb64", "jm") else "none"

    def sub_rows(b, keep_rows):
        """Restrict an arrays() bundle to the rows where keep_rows is True."""
        newid = np.cumsum(keep_rows) - 1
        sel = keep_rows[b["rowid"]]
        rid = newid[b["rowid"][sel]]
        n2 = int(keep_rows.sum())
        rp = rowptr_of(rid, n2)
        return dict(rowid=rid, comp=b["comp"][sel], rp=rp, m=np.diff(rp), ts=b["ts"][sel], hh=b["hh"][sel],
                    pkfull=b["pkfull"][sel], inv=b["inv"][sel]), n2

    ho = sets["ho"]
    fit_rows = (np.arange(ho["n"]) % args.fit_every) == 0

    def fit(arm, est, induction):
        if est in ("wb", "wb64"):
            return {}, None
        b0 = arrays("ho", arm, order_of(est), est == "lin64", induction)
        b, n2 = sub_rows(b0, fit_rows)
        pu_g = p_uni[ho["gold"][fit_rows]]
        if est == "snm":
            m = b["m"]
            pf = np.bincount(b["rowid"], weights=b["pkfull"], minlength=n2) / np.maximum(m, 1)
            ok = m > 0
            beta = 0.5
            for _ in range(200):
                beta = float((beta * pu_g[ok] / ((1 - beta) * pf[ok] + beta * pu_g[ok])).mean())
            return {"beta": beta}, {"beta": beta, "fit_rows": n2}
        if est == "jm":
            bk = bucket[b["comp"]]
            init = np.bincount(bucket[has_counts], weights=lam_exact[has_counts], minlength=n_buckets)
            cntb = np.bincount(bucket[has_counts], minlength=n_buckets)
            lam_b = np.minimum(np.where(cntb > 0, init / np.maximum(cntb, 1), 0.5), LAM_CAP)
            lam_b, trace = k_em_jm_log(b["rp"], b["pkfull"], bk, pu_g, lam_b.copy(), 40, LAM_CAP)
            assert np.all(np.diff(trace) >= -1e-6 * np.abs(trace[:-1])), "EM log-likelihood decreased"
            return {"lam_b": lam_b}, {"heldout_ll_first": float(trace[0]), "heldout_ll_last": float(trace[-1]), "em_iters": 40, "fit_rows": n2}
        if est in ("lin", "lin64"):
            from scipy.optimize import minimize
            bk = bucket[b["comp"]]
            used = np.unique(bk)

            def obj(theta):
                alpha = np.zeros(n_buckets)
                alpha[used] = np.exp(theta)
                ll, num, den, _, _ = k_lin_obj(b["rp"], b["pkfull"], bk, pu_g, alpha, 1.0)
                return -ll, -(alpha[used] * (num[used] - den[used]))
            res = minimize(obj, np.zeros(used.shape[0]), jac=True, method="L-BFGS-B", options={"maxiter": args.lin_iters})
            alpha = np.zeros(n_buckets)
            alpha[used] = np.exp(res.x)
            return {"alpha": alpha, "a0": 1.0}, {"heldout_ll": float(-res.fun), "converged": bool(res.success), "iters": int(res.nit),
                                                 "n_params": int(used.shape[0]), "fit_rows": n2}
        raise ValueError(est)

    def gold_bits(sname, arm, est, params, induction, mu=None):
        """Natural-log gold probability of the mixture (log space: the recursive estimators underflow float64)."""
        s = sets[sname]
        b = arrays(sname, arm, order_of(est), est in ("wb64", "lin64"), induction)
        w, wuni = weights(b, est, params, s["n"])
        if est in ("wb", "wb64", "jm"):
            lam = (lam_64 if est == "wb64" else lam_exact)[b["comp"]] if est != "jm" else params["lam_b"][bucket[b["comp"]]]
            lw, lwu = k_recursive_logweights(b["rp"], lam)
        else:
            with np.errstate(divide="ignore"):
                lw, lwu = np.log(w), np.log(wuni)
        lpm, pc, has, werr = k_gold_eval_log(b["rp"], lw, b["pkfull"], lwu, p_uni[s["gold"]], s["gold"], s["pos"], s["lo"], s["tok"], V)
        assert werr < 1e-9, f"weights do not sum to one: {werr}"
        if mu is None:
            lp = lpm
        else:
            with np.errstate(divide="ignore"):
                lp = np.where(has == 1, np.logaddexp(np.log1p(-mu) + lpm, np.log(mu) + np.log(pc)), lpm)
        assert np.all(np.isfinite(lp)), "a gold log probability is not finite"
        return lp, lpm, pc, has, b, w, wuni

    def record(name, row, bits_tok):
        ROWS[name] = row
        per_tok[name] = bits_tok.astype(np.float32)
        for k in ("bits_per_byte", "bits_per_token", "top1", "top5", "heldout_bits_per_token"):
            if k in row:
                M(f"rows.{name}.{k}", row[k])
        M(f"rows.{name}.spec", {k: v for k, v in row.items() if k not in ("bits_per_byte", "bits_per_token", "top1", "top5", "heldout_bits_per_token")})
        flush_metrics()
        f = lambda k: (f"{row[k]:.4f}" if isinstance(row.get(k), float) else "n/a")  # noqa: E731
        log(f"{name:40s} bpb {f('bits_per_byte')} bits/tok {f('bits_per_token')} top1 {f('top1')} top5 {f('top5')} held-out {f('heldout_bits_per_token')}")

    def bpb_of(bits_tok):
        return float(bits_tok.sum() / n_bytes) if n_va == n_va_tok else None

    def evaluate(name, arm, est, params, induction, mu=None, fit_info=None):
        t1 = time.time()
        lp, _, _, _, b, w, wuni = gold_bits("val", arm, est, params, induction, mu)
        p = np.exp(lp)
        hh = b["hh"]
        coef = w * (1.0 - hh) * b["inv"]
        tcoef = w * hh
        trunc = est in ("wb64", "lin64")
        out = gpu.mix("trunc" if trunc else "full", b["rp"], b["comp"], coef, b["ts"], tcoef, wuni, p_uni, 0, sv["gold"],
                      mu=(mu or 0.0), pos=sv["pos"], dl=sv["lo"], tok_dev=tok_va_dev)
        maxerr = float(np.abs(out["sum"] - 1.0).max())
        assert maxerr < 1e-9, f"{name}: a probability row does not sum to one ({maxerr})"
        rel = float(np.abs(out["pg"] - p).max())
        assert rel < 1e-9, f"{name}: GPU gold probability differs from the CPU path (max abs {rel})"
        bits_tok = -lp / LN2
        ho_lp = gold_bits("ho", arm, est, params, induction, mu)[0]
        row = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()), "top1": float((out["rank"] == 0).mean()),
               "top5": float((out["rank"] < 5).mean()), "heldout_bits_per_token": float((-ho_lp / LN2).mean()),
               "arm": arm, "estimator": est, "induction": bool(induction), "doc_cache_mu": mu, "max_row_sum_error": maxerr,
               "gpu_vs_cpu_gold_prob_max_abs_diff": rel, "eval_s": time.time() - t1}
        if fit_info is not None:
            row["fit"] = fit_info
        if est in ("wb", "wb64", "jm"):
            cum3 = np.zeros(sv["n"])
            for k in range(3):
                sel = b["m"] > k
                cum3[sel] += w[b["rp"][:-1][sel] + k]
            row["mean_weight_first3"] = float(cum3.mean())
            row["median_unigram_weight"] = float(np.median(wuni))
        record(name, row, bits_tok)
        return row

    def fit_mu(arm, est, params, induction):
        _, lpm, pc, has, _, _, _ = gold_bits("ho", arm, est, params, induction)
        pm = np.exp(lpm)
        ok = (has == 1) & ((pm + pc) > 0)
        mu = 0.1
        for _ in range(200):
            mu = float((mu * pc[ok] / ((1 - mu) * pm[ok] + mu * pc[ok])).mean())
        return mu

    # KN-5 on the counting documents (the comparator; its gold probabilities also anchor the kn5+ rows below)
    from tmlm.eval.baselines.ngram import KNModel
    t0 = time.time()
    keep_docs = np.setdiff1d(np.arange(n_doc), ho_docs)

    def docs_stream(docs):
        seg = [tok_tr[ds_tr[d]:ds_tr[d + 1]] for d in docs]
        ds = np.zeros(len(seg) + 1, np.int64)
        ds[1:] = np.cumsum([x.shape[0] for x in seg])
        return np.concatenate(seg).astype(np.int64), ds
    tk, dk = docs_stream(keep_docs)
    th, dh = docs_stream(ho_docs)
    assert np.array_equal(th, tok_tr[ho_pos]), "held-out stream order differs from the held-out rows"
    kn = KNModel(5, V, discount_mode="modified").fit(tk, dk)
    M("kn.fit_s", time.time() - t0)
    vt, vds = tok_va[:n_va], np.append(ds_va[ds_va < n_va], n_va)
    kn_lp2 = {"val": kn.logprob(vt, vds, order=5), "ho": kn.logprob(th, dh, order=5)}

    ests = ("wb", "wb64", "snm", "jm", "lin", "lin64")
    best = {}
    fitted = {}
    for arm in ("engine", "ngram", "union"):
        for est in ests:
            params, info = fit(arm, est, False)
            fitted[(arm, est, False)] = params
            evaluate(f"{arm}.{est}", arm, est, params, False, fit_info=info)
        best[arm] = min(ests, key=lambda e: ROWS[f"{arm}.{e}"]["heldout_bits_per_token"])
        M(f"best_estimator_by_heldout.{arm}", best[arm])
    # specified rows (R1 is engine.wb64 above): R2 analogue = wb64 + induction, as specified (no fitted scalar)
    evaluate("engine.wb64+induction", "engine", "wb64", {}, True)
    evaluate("engine.wb+induction", "engine", "wb", {}, True)
    # C5 in-context recall on the best linear estimator of every arm
    for arm in ("engine", "ngram", "union"):
        est = best[arm]
        p_ind, info_ind = fit(arm, est, True)
        fitted[(arm, est, True)] = p_ind
        evaluate(f"{arm}.{est}+induction", arm, est, p_ind, True, fit_info=info_ind)
        for ind, params in ((False, fitted[(arm, est, False)]), (True, p_ind)):
            mu = fit_mu(arm, est, params, ind)
            evaluate(f"{arm}.{est}+{'induction+' if ind else ''}doc_cache", arm, est, params, ind, mu=mu)

    # ---------------------------------------------------------------- KN-5-anchored linear mixtures (extra; bits only)
    # KN-5 (fitted on the counting documents) is one more expert with its own weight; the question is whether the
    # engine clauses add anything on top of the best classical model at equal data.
    def kn_bundle(sname, arm, induction, keep=None):
        s = sets[sname]
        b = arrays(sname, arm, "none", False, induction)
        n = s["n"]
        pkn = np.exp2(kn_lp2[sname])
        gold, pos, lo = s["gold"], s["pos"], s["lo"]
        if keep is not None:
            b, n = sub_rows(b, keep)
            pkn, gold, pos, lo = pkn[keep], gold[keep], pos[keep], lo[keep]
        rowid = np.concatenate([b["rowid"], np.arange(n)])
        pk = np.concatenate([b["pkfull"], pkn])
        bk = np.concatenate([bucket[b["comp"]], np.full(n, n_buckets)])
        o = np.argsort(rowid, kind="stable")
        return dict(rowid=rowid[o], pk=pk[o], bk=bk[o], rp=rowptr_of(rowid[o], n), n=n, gold=gold, pos=pos, lo=lo, tok=s["tok"])

    def kn_eval(bd, alpha):
        a = alpha[bd["bk"]]
        Z = np.bincount(bd["rowid"], weights=a, minlength=bd["n"]) + 1.0
        lw = np.log(a / Z[bd["rowid"]])
        lwu = np.log(1.0 / Z)
        lpm, pc, has, werr = k_gold_eval_log(bd["rp"], lw, bd["pk"], lwu, p_uni[bd["gold"]], bd["gold"], bd["pos"], bd["lo"], bd["tok"], V)
        assert werr < 1e-9
        return lpm, pc, has

    def with_cache(lpm, pc, has, mu):
        with np.errstate(divide="ignore"):
            return np.where(has == 1, np.logaddexp(np.log1p(-mu) + lpm, np.log(mu) + np.log(pc)), lpm)

    t0 = time.time()
    from scipy.optimize import minimize
    for arm in ("engine", "ngram", "union"):
        for ind in (False, True):
            t1 = time.time()
            bf = kn_bundle("ho", arm, ind, fit_rows)
            used = np.unique(bf["bk"])
            pu_g = p_uni[bf["gold"]]

            def obj(theta):
                alpha = np.zeros(n_buckets + 1)
                alpha[used] = np.exp(theta)
                ll, num, den, _, _ = k_lin_obj(bf["rp"], bf["pk"], bf["bk"], pu_g, alpha, 1.0)
                return -ll, -(alpha[used] * (num[used] - den[used]))
            res = minimize(obj, np.zeros(used.shape[0]), jac=True, method="L-BFGS-B", options={"maxiter": args.lin_iters})
            alpha = np.zeros(n_buckets + 1)
            alpha[used] = np.exp(res.x)
            bh = kn_bundle("ho", arm, ind)
            lpm_h, pc_h, has_h = kn_eval(bh, alpha)
            bv = kn_bundle("val", arm, ind)
            lpm_v, pc_v, has_v = kn_eval(bv, alpha)
            assert np.all(np.isfinite(lpm_v))
            name = f"kn5+{arm}.lin" + ("+induction" if ind else "")
            info = {"heldout_ll": float(-res.fun), "converged": bool(res.success), "iters": int(res.nit), "n_params": int(used.shape[0]),
                    "fit_rows": int(bf["n"]), "kn_expert_weight_over_a0": float(alpha[n_buckets])}
            bits_tok = -lpm_v / LN2
            record(name, {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                          "heldout_bits_per_token": float((-lpm_h / LN2).mean()), "arm": arm, "estimator": "lin with a KN-5 expert",
                          "induction": ind, "top1": None, "top5": None, "fit": info, "eval_s": time.time() - t1,
                          "note": "bits only (the dense KN-5 rows are not combined with the clause rows on the GPU)"}, bits_tok)
            ok = (has_h == 1)
            pm = np.exp(lpm_h)
            mu = 0.1
            for _ in range(200):
                mu = float((mu * pc_h[ok] / ((1 - mu) * pm[ok] + mu * pc_h[ok])).mean())
            bits_tok = -with_cache(lpm_v, pc_v, has_v, mu) / LN2
            record(name + "+doc_cache", {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                         "heldout_bits_per_token": float((-with_cache(lpm_h, pc_h, has_h, mu) / LN2).mean()),
                                         "arm": arm, "estimator": "lin with a KN-5 expert", "induction": ind, "doc_cache_mu": mu,
                                         "top1": None, "top5": None}, bits_tok)
    M("kn_anchor.total_s", time.time() - t0)

    # ---------------------------------------------------------------- log-linear pooling (extra, the engine's read-out form)
    if args.loglin:
        t0 = time.time()
        gval = k_gval(indptr, idx, valc, lam_exact, inv_N, p_uni)
        gpu.drop_csr("trunc")
        gpu.csr["gval"] = dict(indptr=gpu.csr["full"]["indptr"], idx=gpu.csr["full"]["idx"], val=None, evd=cp.asarray(gval))
        logpu = np.log(p_uni)
        rng_ll = np.random.default_rng(np.random.SeedSequence([20260924, 716]))
        ho_perm = rng_ll.permutation(ho["n"])
        nf = min(args.loglin_fit_rows, ho["n"] // 2)
        fit_mask = np.zeros(ho["n"], bool)
        fit_mask[ho_perm[:nf]] = True
        chk_mask = np.zeros(ho["n"], bool)
        chk_mask[ho_perm[nf:2 * nf]] = True
        from scipy.optimize import minimize

        def ll_bundle(sname, arm, induction, keep=None):
            s = sets[sname]
            A = s["arms"][arm]
            b = dict(rowid=A["rowid"], comp=A["comp"], ts=A["ts"] if induction else np.full(A["comp"].shape[0], -1, np.int64),
                     cnt_ts=A["cnt_ts"], rp=A["rp"])
            if induction:
                hc = np.minimum(h_comp[b["comp"]], 0.999)
                has = b["ts"] >= 0
                tsafe = np.where(has, b["ts"], 0)
                lam = lam_exact[b["comp"]]
                pt = lam * b["cnt_ts"] * inv_N[b["comp"]] + (1.0 - lam) * p_uni[tsafe]
                bon = np.where(has & (hc > 0), np.log(hc / (1.0 - hc) + pt) - np.log(pt), 0.0)
            else:
                bon = np.zeros(b["comp"].shape[0])
            b["bon"] = bon
            gold = s["gold"]
            if keep is not None:
                newid = np.cumsum(keep) - 1
                sel = keep[b["rowid"]]
                rid = newid[b["rowid"][sel]]
                n2 = int(keep.sum())
                b = dict(rowid=rid, comp=b["comp"][sel], ts=b["ts"][sel], bon=bon[sel], rp=rowptr_of(rid, n2))
                gold = gold[keep]
                pos, lo = s["pos"][keep], s["lo"][keep]
            else:
                pos, lo = s["pos"], s["lo"]
            b["gold"], b["pos"], b["lo"] = gold, pos, lo
            return b

        ll_sol: dict = {}
        for arm in ("engine", "ngram", "union"):
            for ind in (False, True):
                t1 = time.time()
                bf = ll_bundle("ho", arm, ind, fit_mask)
                bk = bucket[bf["comp"]]
                used = np.unique(bk)
                R = bf["gold"].shape[0]
                nev = [0]

                def obj(theta):
                    alpha = np.zeros(n_buckets)
                    alpha[used] = theta[:-1]
                    beta0 = float(theta[-1])
                    a_p = alpha[bk]
                    o = gpu.mix("gval", bf["rp"], bf["comp"], a_p, bf["ts"], a_p * bf["bon"], beta0, logpu, 1, bf["gold"],
                                use_evd=True, want_grad=True, tbon=bf["bon"])
                    nev[0] += 1
                    ll = float(o["lpg"].sum())
                    grad = np.bincount(bk, weights=o["gpair"], minlength=n_buckets)[used]
                    return -ll / R, -np.append(grad, o["gbeta"].sum()) / R
                a_init = 1.0 / max(1.0, float(np.diff(bf["rp"]).mean()))
                # warm start: +induction from the same arm without induction; union from the n-gram solution with the
                # engine buckets at 0 (so the union fit starts exactly at ngram.loglin and can only improve on the fit rows)
                if ind:
                    x0a, b0 = ll_sol[(arm, False)]
                    init = "warm: same arm without induction"
                elif arm == "union":
                    x0a, b0 = ll_sol[("ngram", False)]
                    init = "warm: ngram.loglin, engine buckets 0"
                else:
                    x0a, b0 = np.full(n_buckets, a_init), 1.0
                    init = f"uniform {a_init:.4g}"
                x0 = np.append(np.clip(x0a[used], -20.0, 20.0), b0)
                res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                               bounds=[(-20.0, 20.0)] * used.shape[0] + [(-5.0, 5.0)], options={"maxiter": args.loglin_iters})
                alpha = np.zeros(n_buckets)
                alpha[used] = res.x[:-1]
                beta0 = float(res.x[-1])
                ll_sol[(arm, ind)] = (alpha.copy(), beta0)
                bc = ll_bundle("ho", arm, ind, chk_mask)
                a_c = alpha[bucket[bc["comp"]]]
                oc = gpu.mix("gval", bc["rp"], bc["comp"], a_c, bc["ts"], a_c * bc["bon"], beta0, logpu, 1, bc["gold"], use_evd=True)
                bv = ll_bundle("val", arm, ind)
                a_v = alpha[bucket[bv["comp"]]]
                ov = gpu.mix("gval", bv["rp"], bv["comp"], a_v, bv["ts"], a_v * bv["bon"], beta0, logpu, 1, bv["gold"], use_evd=True)
                maxerr = float(np.abs(ov["sum"] - 1.0).max())
                assert maxerr < 1e-9 and np.all(np.isfinite(ov["lpg"]))
                # CPU cross-check (masked probe's k_loglin) on 200 validation rows
                keep = np.zeros(sv["n"], bool)
                keep[np.random.default_rng(3).choice(sv["n"], size=min(200, sv["n"]), replace=False)] = True
                bx = ll_bundle("val", arm, ind, keep)
                _, _, _, pgx, rkx, _ = k_loglin(bx["rp"], bx["comp"], bucket[bx["comp"]], alpha, beta0, indptr, idx, gval, logpu,
                                                bx["gold"], bx["ts"], bx["bon"], n_buckets, False)
                relx = float(np.abs(np.log(pgx) - ov["lpg"][keep]).max())
                assert relx < 1e-6, f"loglin GPU vs CPU gold probability differ ({relx})"
                name = f"{arm}.loglin" + ("+induction" if ind else "")
                bits_tok = -ov["lpg"] / LN2
                row = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()), "top1": float((ov["rank"] == 0).mean()),
                       "top5": float((ov["rank"] < 5).mean()), "heldout_bits_per_token": float((-oc["lpg"] / LN2).mean()),
                       "arm": arm, "estimator": "loglin", "induction": ind, "max_row_sum_error": maxerr, "gpu_vs_cpu_abs_logprob_diff_200_rows": relx,
                       "fit": {"fit_rows": int(R), "heldout_check_rows": int(bc["gold"].shape[0]), "iters": int(res.nit), "fun_evals": nev[0],
                               "converged": bool(res.success), "beta0": beta0, "n_params": int(used.shape[0] + 1), "init": init,
                               "fit_objective_bits_per_token": float(res.fun / LN2),
                               "optimizer": "L-BFGS-B on the held-out fit rows (GPU objective and gradient)"},
                       "eval_s": time.time() - t1}
                record(name, row, bits_tok)
                # + unigram document cache on top (mu by EM on the check rows)
                pc, has = k_cache_gold(bc["pos"], bc["lo"], tok_tr, bc["gold"], V)
                ok = has == 1
                mu = 0.1
                for _ in range(200):
                    mu = float((mu * pc[ok] / ((1 - mu) * oc["pg"][ok] + mu * pc[ok])).mean()) if ok.any() else 0.0
                ovc = gpu.mix("gval", bv["rp"], bv["comp"], a_v, bv["ts"], a_v * bv["bon"], beta0, logpu, 1, bv["gold"], use_evd=True,
                              mu=mu, pos=bv["pos"], dl=bv["lo"], tok_dev=tok_va_dev)
                maxerr = float(np.abs(ovc["sum"] - 1.0).max())
                assert maxerr < 1e-9
                pcv, hasv = k_cache_gold(bv["pos"], bv["lo"], tok_va, bv["gold"], V)
                ref = np.where(hasv == 1, (1 - mu) * ov["pg"] + mu * pcv, ov["pg"])
                assert float(np.abs(ref - ovc["pg"]).max()) < 1e-9
                with np.errstate(divide="ignore"):
                    lpv = np.where(hasv == 1, np.logaddexp(np.log1p(-mu) + ov["lpg"], np.log(mu) + np.log(pcv)), ov["lpg"])
                    lph = np.where(has == 1, np.logaddexp(np.log1p(-mu) + oc["lpg"], np.log(mu) + np.log(pc)), oc["lpg"])
                assert np.all(np.isfinite(lpv))
                bits_tok = -lpv / LN2
                row2 = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()), "top1": float((ovc["rank"] == 0).mean()),
                        "top5": float((ovc["rank"] < 5).mean()), "heldout_bits_per_token": float((-lph / LN2).mean()),
                        "arm": arm, "estimator": "loglin", "induction": ind, "doc_cache_mu": mu, "max_row_sum_error": maxerr}
                record(name + "+doc_cache", row2, bits_tok)
        M("loglin.total_s", time.time() - t0)

    # ---------------------------------------------------------------- KN comparators on the SAME prefix
    if args.kn:
        t0 = time.time()
        for order in (1, 2, 3, 4, 5):
            lp2 = kn.logprob(vt, vds, order=order)
            bits_tok = -lp2
            row = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                   "heldout_bits_per_token": float(-kn.logprob(th, dh, order=order).mean()),
                   "fitted_on": "the counting documents of this probe (held-out documents removed)", "discounts": "modified KN (Chen and Goodman)"}
            ROWS[f"comparator.kn{order}"] = row
            per_tok[f"comparator.kn{order}"] = bits_tok.astype(np.float32)
        # top-1 / top-5 for KN-2 and KN-5, and KN-5 + unigram document cache (mu by EM on the held-out documents)
        lp_h = kn.logprob(th, dh, order=5)
        pos_h = np.arange(th.shape[0], dtype=np.int64)
        lo_h = doc_lo_of(dh, th.shape[0])
        pc_h, has_h = k_cache_gold(pos_h, lo_h, th, th, V)
        pm_h = 2.0 ** lp_h
        ok = has_h == 1
        mu_kn = 0.1
        for _ in range(200):
            mu_kn = float((mu_kn * pc_h[ok] / ((1 - mu_kn) * pm_h[ok] + mu_kn * pc_h[ok])).mean())
        lo_v = lo_va[:n_va]
        for order in (2, 5):
            ranks = np.zeros(n_va, np.int64)
            ranks_c = np.zeros(n_va, np.int64)
            pgc = np.zeros(n_va)
            for i in range(0, n_va, 2048):
                pos = np.arange(i, min(n_va, i + 2048))
                ctx, valid = kn.contexts(vt, vds, pos, None, order)
                P = kn.dist(ctx, valid, order)
                assert np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)
                g = vt[pos]
                ar = np.arange(V)[None, :]
                pg = P[np.arange(pos.shape[0]), g][:, None]
                ranks[pos] = (P > pg).sum(axis=1) + ((P == pg) & (ar < g[:, None])).sum(axis=1)
                if order == 5:
                    H, hs = k_cache_rows(pos, lo_v[pos], vt, V)
                    Pc = np.where(hs[:, None] == 1, (1 - mu_kn) * P + mu_kn * H, P)
                    assert np.all(np.abs(Pc.sum(axis=1) - 1.0) < 1e-9)
                    pgc_ = Pc[np.arange(pos.shape[0]), g][:, None]
                    ranks_c[pos] = (Pc > pgc_).sum(axis=1) + ((Pc == pgc_) & (ar < g[:, None])).sum(axis=1)
                    pgc[pos] = pgc_[:, 0]
            ROWS[f"comparator.kn{order}"]["top1"] = float((ranks == 0).mean())
            ROWS[f"comparator.kn{order}"]["top5"] = float((ranks < 5).mean())
            if order == 5:
                bits_tok = -np.log(pgc) / LN2
                pch = np.where(has_h == 1, (1 - mu_kn) * pm_h + mu_kn * pc_h, pm_h)
                ROWS["comparator.kn5+doc_cache"] = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                                    "top1": float((ranks_c == 0).mean()), "top5": float((ranks_c < 5).mean()),
                                                    "heldout_bits_per_token": float((-np.log2(pch)).mean()), "doc_cache_mu": mu_kn}
                per_tok["comparator.kn5+doc_cache"] = bits_tok.astype(np.float32)
        # equal total data: KN-5 on the whole prefix (counting + held-out documents), bits only
        kn_all = KNModel(5, V, discount_mode="modified").fit(tok_tr, ds_tr)
        for order in (2, 5):
            bits_tok = -kn_all.logprob(vt, vds, order=order)
            ROWS[f"comparator.kn{order}_whole_prefix"] = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean()),
                                                          "fitted_on": "the whole prefix (counting + held-out documents)"}
            per_tok[f"comparator.kn{order}_whole_prefix"] = bits_tok.astype(np.float32)
        # add-one unigram of the counting positions (the floor every mixture uses)
        bits_tok = -np.log2(p_uni[vt])
        ROWS["comparator.unigram_addone"] = {"bits_per_byte": bpb_of(bits_tok), "bits_per_token": float(bits_tok.mean())}
        per_tok["comparator.unigram_addone"] = bits_tok.astype(np.float32)
        for nm in [k for k in ROWS if k.startswith("comparator.")]:
            for kk, vv in ROWS[nm].items():
                M(f"rows.{nm}.{kk}", vv, note="in-house KN (src/tmlm/eval/baselines/ngram.py), same validation tokens and bytes")
            log(f"{nm}: {ROWS[nm]}")
        M("kn.total_s", time.time() - t0)
    # the engine's own row (logged calibration, reproduced in C1)
    per_tok["engine.signed_votes_logged_calibration"] = (-eng_lp / LN2).astype(np.float32)
    ROWS["engine.signed_votes_logged_calibration"] = {"bits_per_byte": bpb_of(-eng_lp / LN2), "bits_per_token": float((-eng_lp / LN2).mean()),
                                                      "top1_raw": top1_raw, "top1": METRICS["C1.engine_calibrated_top1"]["value"],
                                                      "top5": METRICS["C1.engine_calibrated_top5"]["value"],
                                                      "note": "trained on all 135M train tokens for 4 epochs; calibration fitted on validation (logged)"}

    # ---------------------------------------------------------------- paired document-block bootstrap on key differences
    def cmp(a, b, label):
        if a in per_tok and b in per_tok:
            mean, ci, nd = doc_boot(per_tok[a].astype(np.float64), per_tok[b].astype(np.float64), doc_of_val, n_val_docs)
            ratio = n_va / n_bytes if n_va == n_va_tok else None
            M(f"paired.{label}", {"a": a, "b": b, "bits_per_token_a_minus_b": mean, "ci95": ci, "docs": nd,
                                  "bits_per_byte_a_minus_b": (mean * ratio) if ratio else None}, label="derived",
              note="document-block bootstrap, 2000 resamples; bits per byte = bits per token x tokens / bytes of the full split")
    # does the TM add information? same estimator, union (engine clauses + n-grams) minus n-grams alone
    for suf in ("lin", "lin64", "jm", "snm", f"{best['union']}+induction", f"{best['union']}+induction+doc_cache", "loglin", "loglin+induction",
                "loglin+induction+doc_cache"):
        cmp(f"union.{suf}", f"ngram.{suf}", f"union_minus_ngram.{suf}")
    for suf in ("lin", "lin+doc_cache", "lin+induction", "lin+induction+doc_cache"):
        cmp(f"kn5+union.{suf}", f"kn5+ngram.{suf}", f"kn5_union_minus_kn5_ngram.{suf}")
        cmp(f"kn5+engine.{suf}", "comparator.kn5" if "doc_cache" not in suf else "comparator.kn5+doc_cache", f"kn5_engine_minus_kn5.{suf}")
    # count read-outs over the engine clauses vs the engine's own votes and vs KN-2
    for r in ("engine.wb", "engine.wb64", f"engine.{best['engine']}", "engine.loglin"):
        cmp(r, "engine.signed_votes_logged_calibration", f"minus_signed_votes.{r}")
        cmp(r, "comparator.kn2", f"minus_kn2.{r}")
    # in-context recall
    for arm in ("engine", "ngram", "union"):
        cmp(f"{arm}.{best[arm]}+induction", f"{arm}.{best[arm]}", f"induction_gain.{arm}.{best[arm]}")
        cmp(f"{arm}.loglin+induction", f"{arm}.loglin", f"induction_gain.{arm}.loglin")
    # best non-comparator row by HELD-OUT bits (validation is never used to choose) vs KN-5 and KN-5 + cache
    cands = [k for k in ROWS if not k.startswith("comparator.") and k != "engine.signed_votes_logged_calibration"
             and ROWS[k].get("heldout_bits_per_token") is not None]
    best_row = min(cands, key=lambda k: ROWS[k]["heldout_bits_per_token"])
    M("best_row_by_heldout_bits", best_row)
    cmp(best_row, "comparator.kn5", "best_minus_kn5")
    cmp(best_row, "comparator.kn5+doc_cache", "best_minus_kn5_doc_cache")
    cmp(best_row, "comparator.kn5_whole_prefix", "best_minus_kn5_whole_prefix")

    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        np.savez_compressed(Path(run_dir) / "per_token_bits.npz", positions=np.arange(n_va), doc=doc_of_val, **per_tok)
        Path(run_dir, "rows.json").write_text(json.dumps(ROWS, indent=1, default=float) + "\n")
    M("wall_s_total", time.time() - T0)
    flush_metrics()
    if not run_dir:
        print(json.dumps(ROWS, indent=1, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
