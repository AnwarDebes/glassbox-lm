"""Exact speedups of the R1 Q_pred stage (compute-reduction items O4 and O10; docs/13 A8 item 7).

A speedup enters a registered row only if it reproduces the registered estimator on logged cells under a rule written
before the check (docs/13 A8 item 7). Everything here is meant to be BIT-IDENTICAL to the registered computation of
experiments/ssl_r1_controls.py (stage_qpred, count_table, fit_rows_fired) and experiments/cvtm_t1_readout.py (T1Gpu,
T1_SRC), which stay unchanged because running chains import them. The verification rule and its runs are in
experiments/cvtm/configs/ssl_r1_speedup_verify.json and experiments/ssl_r1_speedup_verify.py.

O4, T1GpuG. The registered GPU objective (cvtm_t1_readout.T1_SRC, kernel t1_loglin) recomputes the per-pair log term

    g = log(1 + lam[k] * val[e] * inv_n[k] / ((1 - lam[k]) * pu[c]))

for every visited (row, fired clause, class) entry, once in the forward pass and once in the gradient pass, at every
L-BFGS evaluation, although g depends on the count tables only. T1GpuG evaluates g ONCE per count table, with the
identical expression, in a separate kernel (t1_gval) compiled with the same options, stores it per stored pair
(float64), and the objective kernel t1_loglin_g loads it. Every other line of t1_loglin_g is the registered kernel's
line (tests/test_ssl_r1_qpred_fast.py derives T1G_OBJECTIVE_SRC from T1_SRC by exactly the replacements listed in
REPLACEMENTS and compares). Why the bits cannot change: the expression contains no multiply feeding an add (a double
division feeds the add), so FMA contraction cannot differ between the two kernels; CUDA double division is correctly
rounded and log() is the same libdevice routine, so the same inputs give the same g; the uses of g (`cf * g * scale`
into an integer atomic, `eg += P[c] * gv`, `gy += gv`) are the registered expressions in the registered order. The
integer fixed-point class sums and the double reductions are untouched. Memory per stored pair on the device: idx
(2 bytes) + g (8 bytes), against idx + val (6 bytes) for T1Gpu; val, lam, inv_n and pu are freed after t1_gval.

O10, prefix_pass. The features stage's prefix fire counts (ssl_r1_pretext.fire_counts: every prefix position, the
masked banks' all-zero bitmap, epoch 0) and the Q_pred stage's counting pass (ssl_r1_controls.count_table over the
counting positions, the all-zero bitmap, epoch cvtm_t1_readout.EPOCH = 4) and fit-row pass (fit_rows_fired, same
bitmap and epoch) evaluate the SAME forward per position with the same tables:
  * with a bitmap the engine's hidden set is the bitmap (tm_core.h tm_hidden returns bitmap[q] and never reads the
    epoch); epoch and example id enter tm_make_example only as ex->e / ex->a, which only the masked no-bitmap rule, the
    diffusion no-bitmap rate and the training feedback read; for non-masked modes both passes pass no bitmap and
    tm_hidden returns 0 without reading the epoch;
  * each example's candidates are verified independently (k3_verify over (cand_ex, cand_j) pairs), so the fired set
    of a position does not depend on which other positions share its chunk;
  * both aggregates are integer counts (uint64 bincount; int32 increments), so the order of accumulation cannot change
    a bit.
prefix_pass therefore runs R1's fire-count loop unchanged (same chunks, epoch 0, same bitmap, same bincount) and, from
the same fired lists, also accumulates O[j, target] over the counting positions (one integer-atomic kernel per chunk,
no host synchronisation). The fired lists of the fit positions and of the sampled cross-engine check rows come from
the registered fit_rows_fired, bound by the caller. The loaded table must equal count_table recomputed bit for bit on
a logged run before the stage may use it (A8 item 7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCALE_LOG = float(2 ** 40)      # cvtm_t1_readout.SCALE_LOG (fixed point of the GPU class sums)

# The registered kernel with the log term loaded instead of recomputed. Derived from cvtm_t1_readout.T1_SRC by the
# replacements in REPLACEMENTS (checked by the tests), nothing else.
T1G_OBJECTIVE_SRC = r"""
#define BS 512
extern "C" __global__ void __launch_bounds__(BS) t1_loglin_g(
        const long long* __restrict__ rowptr, const int* __restrict__ comp, const int* __restrict__ bkt,
        const double* __restrict__ alpha, const long long* __restrict__ indptr,
        const unsigned short* __restrict__ idx, const double* __restrict__ gval,
        const double* __restrict__ logpu, const double beta0, const double scale,
        const int* __restrict__ gold, const int n_rows, const int V,
        double* __restrict__ out_lpg, double* __restrict__ out_pg, int* __restrict__ out_rank,
        int* __restrict__ out_top, double* __restrict__ out_sum,
        const int want_grad, double* __restrict__ out_gpair, double* __restrict__ out_gbeta) {
    extern __shared__ long long acc[];
    double* P = (double*)acc;
    __shared__ double rd[BS];
    __shared__ double rb[BS];
    __shared__ int ri[BS];
    __shared__ int rk[BS];
    const int tid = threadIdx.x;
    for (int r = blockIdx.x; r < n_rows; r += gridDim.x) {
        for (int c = tid; c < V; c += BS) acc[c] = llrint(beta0 * logpu[c] * scale);
        __syncthreads();
        const long long p0 = rowptr[r], p1 = rowptr[r + 1];
        for (long long i = p0; i < p1; ++i) {
            const double cf = alpha[bkt[i]];
            if (cf == 0.0) continue;
            const int k = comp[i];
            const long long e1 = indptr[k + 1];
            for (long long e = indptr[k] + tid; e < e1; e += BS) {
                const int c = idx[e];
                const double g = gval[e];
                atomicAdd((unsigned long long*)&acc[c], (unsigned long long)llrint(cf * g * scale));
            }
        }
        __syncthreads();
        for (int c = tid; c < V; c += BS) P[c] = (double)acc[c] / scale;
        __syncthreads();
        const int g = gold[r];
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
        const double lpg = sg - m - log(z);
        for (int c = tid; c < V; c += BS) P[c] = P[c] / z;
        __syncthreads();
        const double pg = P[g];
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
        if (tid == 0) { out_lpg[r] = lpg; out_pg[r] = pg; out_rank[r] = ri[0]; out_top[r] = rk[0]; out_sum[r] = rd[0]; }
        __syncthreads();
        if (want_grad) {
            double el = 0.0;
            for (int c = tid; c < V; c += BS) el += P[c] * logpu[c];
            rd[tid] = el; __syncthreads();
            for (int st = BS / 2; st > 0; st >>= 1) { if (tid < st) rd[tid] += rd[tid + st]; __syncthreads(); }
            if (tid == 0) out_gbeta[r] = logpu[g] - rd[0];
            const int lane = tid & 31, w = tid >> 5;
            for (long long i = p0 + w; i < p1; i += BS / 32) {
                const int k = comp[i];
                double eg = 0.0, gy = 0.0;
                const long long e1 = indptr[k + 1];
                for (long long e = indptr[k] + lane; e < e1; e += 32) {
                    const int c = idx[e];
                    const double gv = gval[e];
                    eg += P[c] * gv;
                    if (c == g) gy += gv;
                }
                for (int o = 16; o > 0; o >>= 1) { eg += __shfl_down_sync(0xffffffffu, eg, o); gy += __shfl_down_sync(0xffffffffu, gy, o); }
                if (lane == 0) out_gpair[i] = gy - eg;
            }
            __syncthreads();
        }
    }
}
"""

# One-off per count table: g per stored pair, the registered expression verbatim (same operands, same order).
T1G_GVAL_SRC = r"""
extern "C" __global__ void t1_gval(
        const long long* __restrict__ indptr, const unsigned short* __restrict__ idx, const int* __restrict__ val,
        const double* __restrict__ lam, const double* __restrict__ inv_n, const double* __restrict__ pu,
        const int n_clauses, double* __restrict__ gval) {
    for (int k = blockIdx.x; k < n_clauses; k += gridDim.x) {
        const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;
        const long long e1 = indptr[k + 1];
        for (long long e = indptr[k] + threadIdx.x; e < e1; e += blockDim.x) {
            const int c = idx[e];
            gval[e] = log(1.0 + l * (double)val[e] * inn / (oml * pu[c]));
        }
    }
}
"""

T1G_SRC = T1G_OBJECTIVE_SRC + T1G_GVAL_SRC
T1G_OPTIONS = ("-std=c++14",)   # the registered T1Gpu's RawModule options

# (old, new, expected count): T1G_OBJECTIVE_SRC == T1_SRC with these replacements applied in order (the 16-space
# line before the 12-space line, which is its substring).
REPLACEMENTS = (
    ("void __launch_bounds__(BS) t1_loglin(", "void __launch_bounds__(BS) t1_loglin_g(", 1),
    ("        const unsigned short* __restrict__ idx, const int* __restrict__ val,\n"
     "        const double* __restrict__ lam, const double* __restrict__ inv_n, const double* __restrict__ pu,\n"
     "        const double* __restrict__ logpu,",
     "        const unsigned short* __restrict__ idx, const double* __restrict__ gval,\n"
     "        const double* __restrict__ logpu,", 1),
    ("                const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;\n", "", 1),
    ("            const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;\n", "", 1),
    ("const double g = log(1.0 + l * (double)val[e] * inn / (oml * pu[c]));", "const double g = gval[e];", 1),
    ("const double gv = log(1.0 + l * (double)val[e] * inn / (oml * pu[c]));", "const double gv = gval[e];", 1),
)


def derive_objective_source(t1_src: str) -> str:
    """T1_SRC with REPLACEMENTS applied; each must occur exactly its expected number of times."""
    s = t1_src
    for old, new, n in REPLACEMENTS:
        if s.count(old) != n:
            raise ValueError(f"replacement source occurs {s.count(old)} times, expected {n}: {old[:60]!r}")
        s = s.replace(old, new)
    return s


def to_device_chunked(cp, host: np.ndarray, dtype, chunk_bytes: int = 1 << 28):
    """cvtm_t1_readout.to_device_chunked (a copy: only the upload, no host staging copy of the whole array)."""
    h = np.asarray(host)
    out = cp.empty(h.shape[0], dtype=dtype)
    step = max(1, chunk_bytes // max(1, np.dtype(dtype).itemsize))
    for a in range(0, h.shape[0], step):
        out[a:a + step].set(np.ascontiguousarray(h[a:a + step], dtype=dtype))
    return out


class T1GpuG:
    """Drop-in for cvtm_t1_readout.T1Gpu (same constructor arguments, prepare, run and the `cp` attribute, so
    cvtm_t1_readout.loglin_fit_gpu uses it unchanged) with the per-pair log term precomputed once (O4)."""

    def __init__(self, V: int, indptr, idx, val, lam, inv_n, p_uni, grid: int = 1024, gval_grid: int = 4096) -> None:
        import cupy as cp
        self.cp = cp
        self.V = int(V)
        self.grid = int(grid)
        mod = cp.RawModule(code=T1G_SRC, options=T1G_OPTIONS)
        self.k = mod.get_function("t1_loglin_g")
        kg = mod.get_function("t1_gval")
        self.smem = 8 * self.V
        self.k.max_dynamic_shared_size_bytes = self.smem
        indptr = np.asarray(indptr)
        self.d_indptr = to_device_chunked(cp, indptr, np.int64)
        self.d_idx = to_device_chunked(cp, idx, np.uint16)
        d_val = to_device_chunked(cp, val, np.int32)
        d_lam = cp.asarray(np.ascontiguousarray(lam, dtype=np.float64))
        d_inv = cp.asarray(np.ascontiguousarray(inv_n, dtype=np.float64))
        d_pu = cp.asarray(np.ascontiguousarray(p_uni, dtype=np.float64))
        self.d_logpu = cp.asarray(np.log(np.ascontiguousarray(p_uni, dtype=np.float64)))
        n_cl = int(indptr.shape[0] - 1)
        nnz = int(indptr[-1]) if n_cl >= 0 and indptr.shape[0] else 0
        self.n_pairs = nnz
        self.d_g = cp.empty(max(1, nnz), dtype=cp.float64)
        if n_cl > 0 and nnz > 0:
            kg((min(int(gval_grid), n_cl),), (256,), (self.d_indptr, self.d_idx, d_val, d_lam, d_inv, d_pu, np.int32(n_cl), self.d_g))
        cp.cuda.Device().synchronize()
        del d_val, d_lam, d_inv, d_pu

    def gval(self) -> np.ndarray:
        """The stored log terms (tests and verification)."""
        return self.d_g[:self.n_pairs].get()

    def prepare(self, rp: np.ndarray, comp: np.ndarray, bkt: np.ndarray, gold: np.ndarray, rows_per_chunk: int = 16384) -> list:
        """cvtm_t1_readout.T1Gpu.prepare, unchanged."""
        cp = self.cp
        n = rp.shape[0] - 1
        chunks = []
        for r0 in range(0, n, rows_per_chunk):
            r1 = min(n, r0 + rows_per_chunk)
            a, b = int(rp[r0]), int(rp[r1])
            chunks.append(dict(r0=r0, r1=r1, a=a, b=b,
                               rp=cp.asarray((rp[r0:r1 + 1] - a).astype(np.int64)),
                               comp=cp.asarray(np.ascontiguousarray(comp[a:b], dtype=np.int32)),
                               bkt=cp.asarray(np.ascontiguousarray(bkt[a:b], dtype=np.int32)),
                               gold=cp.asarray(np.ascontiguousarray(gold[r0:r1], dtype=np.int32))))
        return chunks

    def run(self, chunks: list, n_rows: int, n_pairs: int, alpha: np.ndarray, beta0: float, want_grad: bool) -> dict:
        """cvtm_t1_readout.T1Gpu.run with the loaded log terms: same launch shape, same outputs."""
        cp = self.cp
        d_alpha = cp.asarray(np.ascontiguousarray(alpha, dtype=np.float64))
        out = dict(lpg=np.empty(n_rows), pg=np.empty(n_rows), rank=np.empty(n_rows, np.int64),
                   top=np.empty(n_rows, np.int64), sum=np.empty(n_rows))
        if want_grad:
            out["gpair"] = np.empty(n_pairs)
            out["gbeta"] = np.empty(n_rows)
        dummy = cp.zeros(1, dtype=cp.float64)
        for ch in chunks:
            nr = ch["r1"] - ch["r0"]
            o_lpg = cp.empty(nr, cp.float64)
            o_pg = cp.empty(nr, cp.float64)
            o_rank = cp.empty(nr, cp.int32)
            o_top = cp.empty(nr, cp.int32)
            o_sum = cp.empty(nr, cp.float64)
            o_gp = cp.empty(max(1, ch["b"] - ch["a"]), cp.float64) if want_grad else dummy
            o_gb = cp.empty(nr, cp.float64) if want_grad else dummy
            self.k((min(self.grid, nr),), (512,),
                   (ch["rp"], ch["comp"], ch["bkt"], d_alpha, self.d_indptr, self.d_idx, self.d_g, self.d_logpu,
                    np.float64(beta0), np.float64(SCALE_LOG), ch["gold"], np.int32(nr), np.int32(self.V),
                    o_lpg, o_pg, o_rank, o_top, o_sum, np.int32(1 if want_grad else 0), o_gp, o_gb),
                   shared_mem=self.smem)
            r0, r1 = ch["r0"], ch["r1"]
            out["lpg"][r0:r1] = o_lpg.get()
            out["pg"][r0:r1] = o_pg.get()
            out["rank"][r0:r1] = o_rank.get()
            out["top"][r0:r1] = o_top.get()
            out["sum"][r0:r1] = o_sum.get()
            if want_grad:
                out["gpair"][ch["a"]:ch["b"]] = o_gp.get()[:ch["b"] - ch["a"]]
                out["gbeta"][r0:r1] = o_gb.get()
        return out


def outputs_bit_identical(a: dict, b: dict) -> dict:
    """Per output key: True iff the two arrays have the same dtype, shape and bytes (floats compared as bit
    patterns, so -0.0 != 0.0 and NaN payloads count)."""
    res = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            res[k] = False
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        res[k] = bool(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes())
    return res


# ------------------------------------------------------------------ O10: one prefix pass for fire counts and the count table
# Per chunk the pass adds ONE kernel launch to R1's fire-count loop and no host synchronisation: on a GPU shared by
# several processes every synchronisation waits for a time slice, and a first version with four per chunk (boolean
# compactions, per-chunk fit-row extraction) cost almost a second counting pass (logged in
# results/20260925T153833Z_ssl_r1_speedup_timing_A3_seed100/log.txt). Integer atomics: the order of the increments
# cannot change a count.
COUNT_SRC = r"""
extern "C" __global__ void count_events(const unsigned int* __restrict__ fired_ex, const unsigned int* __restrict__ fired_j,
        const long long n, const long long lo, const unsigned char* __restrict__ is_cnt, const long long* __restrict__ tok,
        const int V, int* __restrict__ counts, unsigned long long* __restrict__ n_events) {
    __shared__ unsigned long long blk;
    if (threadIdx.x == 0) blk = 0;
    __syncthreads();
    unsigned long long mine = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long q = lo + (long long)fired_ex[i];
        if (is_cnt[q]) {
            atomicAdd(&counts[(long long)fired_j[i] * V + tok[q]], 1);
            ++mine;
        }
    }
    if (mine) atomicAdd(&blk, mine);
    __syncthreads();
    if (threadIdx.x == 0 && blk) atomicAdd(n_events, blk);
}
"""


@dataclass
class PrefixPass:
    """The results of one pass over the prefix: R1's fire counts, the Q_pred count table O (CSR of the dense int32
    table of count_table: rows = clauses, ascending class ids), the counted events, the fit rows' fired lists (CSR
    over fit_pos, ascending ids per row: fit_rows_fired) and the check rows' fired lists (check_rows index cnt_pos)."""
    fire_counts: np.ndarray
    n_clauses: int
    vocab: int
    indptr: np.ndarray
    idx: np.ndarray
    val: np.ndarray
    n_events: int
    fit_ptr: np.ndarray
    fit_ids: np.ndarray
    check_rows: np.ndarray
    check_ptr: np.ndarray
    check_ids: np.ndarray

    def counts_dense(self) -> np.ndarray:
        """The dense int32 [C, V] table (what count_table returns)."""
        out = np.zeros((self.n_clauses, self.vocab), np.int32)
        rows = np.repeat(np.arange(self.n_clauses, dtype=np.int64), np.diff(self.indptr))
        out.reshape(-1)[rows * self.vocab + self.idx.astype(np.int64)] = self.val
        return out

    def got(self) -> dict[int, np.ndarray]:
        """{row index into cnt_pos: ascending fired ids} of the check rows (count_table's third return value)."""
        return {int(r): self.check_ids[self.check_ptr[i]:self.check_ptr[i + 1]].astype(np.uint32)
                for i, r in enumerate(self.check_rows.tolist())}


def _csr_of_dense_host(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = counts.shape[0]
    rows, cols = np.nonzero(counts)
    indptr = np.zeros(C + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=C), out=indptr[1:])
    return indptr, cols.astype(np.uint16), counts[rows, cols].astype(np.int32)


def _csr_of_dense_device(cp, dcounts, C: int, V: int, rows_per_block: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d2 = dcounts.reshape(C, V)
    nnz_row = cp.count_nonzero(d2, axis=1).get().astype(np.int64)
    indptr = np.zeros(C + 1, np.int64)
    np.cumsum(nnz_row, out=indptr[1:])
    idx = np.empty(int(indptr[-1]), np.uint16)
    val = np.empty(int(indptr[-1]), np.int32)
    for r0 in range(0, C, rows_per_block):
        r1 = min(C, r0 + rows_per_block)
        blk = d2[r0:r1]
        rr, cc = cp.nonzero(blk)
        key = rr.astype(cp.int64) * V + cc
        if key.size > 1 and not bool(cp.all(key[1:] > key[:-1])):
            raise RuntimeError("cupy.nonzero did not return row-major order")
        a, b = int(indptr[r0]), int(indptr[r1])
        if b - a != int(rr.size):
            raise RuntimeError("row counts and nonzero entries disagree")
        idx[a:b] = cc.astype(cp.uint16).get()
        val[a:b] = blk[rr, cc].get()
        del rr, cc, key
    return indptr, idx, val


def prefix_pass(eng, n_clauses: int, vocab: int, n_tok: int, chunk: int, hidden: np.ndarray | None, tok: np.ndarray,
                cnt_pos: np.ndarray, fit_pos: np.ndarray, check_rows: np.ndarray, fired_rows) -> PrefixPass:
    """One pass over the n_tok prefix positions: ssl_r1_pretext.fire_counts (its loop, unchanged) plus, from the same
    fired lists, the count table of count_table over cnt_pos (targets tok[pos]). The fit rows' and the check rows'
    fired lists (1.2 percent and 500 positions of the prefix) come from `fired_rows(positions) -> (ptr, ids)`, the
    registered ssl_r1_controls.fit_rows_fired with the all-zero bitmap (the caller binds it): the fit rows are then
    Q_pred's own computation, and the check rows' lists are the per-position fired sets that count_table extracts
    (a position's fired set does not depend on its chunk). `hidden` is ssl_r1_pretext.hidden_for(cfg, n_tok)."""
    C, V = int(n_clauses), int(vocab)
    if V > 1 << 16:
        raise ValueError("class ids are stored as uint16 (as T1Gpu uploads them): vocab must be at most 65536")
    tok = np.asarray(tok).astype(np.int64)
    cnt_pos = np.asarray(cnt_pos, np.int64)
    check_rows = np.asarray(check_rows, np.int64)
    is_cnt = np.zeros(n_tok, np.uint8)
    is_cnt[cnt_pos] = 1
    if not hasattr(eng, "_forward"):       # CPU / oracle engines: ssl_r1_pretext.fire_counts' public-API loop
        out = np.zeros(C, np.uint64)
        counts = np.zeros((C, V), np.int32)
        n_events = 0
        for q in range(n_tok):
            a = eng.fired_clauses(0, q, hidden)
            out[a.astype(np.int64)] += np.uint64(1)
            if is_cnt[q]:
                counts[np.asarray(a, np.int64), tok[q]] += 1
                n_events += int(a.shape[0])
        indptr, idx, val = _csr_of_dense_host(counts)
        del counts
    else:                                   # CUDA engine: ssl_r1_pretext.fire_counts' loop plus one kernel per chunk
        import cupy as cp
        kcount = cp.RawModule(code=COUNT_SRC).get_function("count_events")
        bitmap = eng._bitmap(hidden)
        acc = cp.zeros(C, dtype=cp.uint64)
        dcounts = cp.zeros(C * V, dtype=cp.int32)
        d_tok = cp.asarray(tok)
        d_cnt = cp.asarray(is_cnt)
        d_nev = cp.zeros(1, dtype=cp.uint64)
        for lo in range(0, n_tok, chunk):
            hi = min(n_tok, lo + chunk)
            eng._sync_state_struct()
            fw = eng._forward(0, cp.arange(lo, hi, dtype=cp.uint32), bitmap, 0)
            eng._marks = []
            if fw.fired_j.size:
                acc += cp.bincount(fw.fired_j.astype(cp.int64), minlength=C).astype(cp.uint64)
                nf = int(fw.fired_j.size)
                kcount((min(1024, (nf + 255) // 256),), (256,),
                       (cp.ascontiguousarray(fw.fired_ex, dtype=cp.uint32), cp.ascontiguousarray(fw.fired_j, dtype=cp.uint32),
                        np.int64(nf), np.int64(lo), d_cnt, d_tok, np.int32(V), dcounts, d_nev))
            del fw
        eng.synchronize()
        out = acc.get()
        n_events = int(d_nev.get()[0])
        indptr, idx, val = _csr_of_dense_device(cp, dcounts, C, V)
        del acc, dcounts, d_tok, d_cnt, d_nev
        cp.get_default_memory_pool().free_all_blocks()
    fit_ptr, fit_ids = fired_rows(np.asarray(fit_pos, np.int64))
    ck_ptr, ck_ids = fired_rows(cnt_pos[check_rows])
    return PrefixPass(fire_counts=out, n_clauses=C, vocab=V, indptr=indptr, idx=idx, val=val, n_events=int(n_events),
                      fit_ptr=np.asarray(fit_ptr, np.int64), fit_ids=np.asarray(fit_ids), check_rows=check_rows,
                      check_ptr=np.asarray(ck_ptr, np.int64), check_ids=np.asarray(ck_ids))


def array_sha256(a: np.ndarray) -> str:
    from tmlm.common import hashing
    return hashing.sha256_array(np.ascontiguousarray(a))


def save_prefix_counts(path: Path, pp: PrefixPass, meta: dict) -> str:
    """Uncompressed npz (fast to write and load); returns the file's sha256."""
    from tmlm.common import hashing
    np.savez(path, indptr=pp.indptr, idx=pp.idx, val=pp.val, n_events=np.int64(pp.n_events),
             n_clauses=np.int64(pp.n_clauses), vocab=np.int64(pp.vocab), fit_ptr=pp.fit_ptr, fit_ids=pp.fit_ids,
             check_rows=pp.check_rows, check_ptr=pp.check_ptr, check_ids=pp.check_ids,
             meta=np.array(json.dumps(meta, sort_keys=True)))
    return hashing.sha256_file(path)


def load_prefix_counts(path: Path) -> tuple[PrefixPass, dict]:
    with np.load(path, allow_pickle=False) as z:
        pp = PrefixPass(fire_counts=np.zeros(0, np.uint64), n_clauses=int(z["n_clauses"]), vocab=int(z["vocab"]),
                        indptr=z["indptr"], idx=z["idx"], val=z["val"], n_events=int(z["n_events"]), fit_ptr=z["fit_ptr"],
                        fit_ids=z["fit_ids"], check_rows=z["check_rows"], check_ptr=z["check_ptr"], check_ids=z["check_ids"])
        meta = json.loads(str(z["meta"]))
    return pp, meta


def pass_meta(state_hash: str, prefix, corpus_tables, fit_pos: np.ndarray, cnt_pos: np.ndarray, check_rows: np.ndarray,
              cfg, engine_kind: str) -> dict:
    """What a table was computed from; Q_pred loads it only when every entry equals its own (else it recomputes)."""
    return {"state_hash": str(state_hash),
            "prefix_tok_sha256": array_sha256(np.asarray(prefix.tok)),
            "prefix_doc_start_sha256": array_sha256(np.asarray(prefix.doc_start)),
            "kind_map_sha256": array_sha256(np.asarray(corpus_tables.kind_map)),
            "sym_freq_sha256": array_sha256(np.asarray(corpus_tables.sym_freq)),
            "fit_pos_sha256": array_sha256(np.asarray(fit_pos, np.int64)),
            "cnt_pos_sha256": array_sha256(np.asarray(cnt_pos, np.int64)),
            "check_rows_sha256": array_sha256(np.asarray(check_rows, np.int64)),
            "n_clauses": int(cfg.n_clauses), "vocab": int(cfg.vocab), "mode": str(cfg.mode),
            "engine": str(engine_kind), "epoch": 0,
            "hidden": "ssl_r1_pretext.hidden_for (all-zero bitmap for masked and diffusion banks, none otherwise)"}
