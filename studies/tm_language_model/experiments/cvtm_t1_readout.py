"""H8.1 of docs/13 amendment A3 (pre-registered): the count-vote stand-alone read-out "T1" of
docs/11 section 11.5 on a Tsetlin-automata-learned clause bank, against the SAME bank's own signed
votes. Validation split only; the test split is never loaded.

The read-out (docs/11 11.5, `loglin` of experiments/ckm_posthoc_probe.py on the engine arm, no
induction, no cache). The bank is frozen. One counting pass gives, for every clause j and class c,
O_jc = the number of counting positions where j fires and the target is c (exact integers),
N_j = sum over c of O_jc and D_j = #{c : O_jc > 0}. At a position the class score is

    s(c) = beta0 ln P_uni(c) + sum over fired j with N_j > 0 of alpha_g(j) q_jc,
    q_jc = ln(1 + lam_j O_jc / (N_j (1 - lam_j) P_uni(c))),  lam_j = N_j / (N_j + D_j) (Witten-Bell),

with g(j) = (min(n_inc_j, 12), min(floor(log2 N_j), 23)), P_uni the add-one unigram of the counting
positions, and P(c) = softmax over the V classes. alpha (one per group) and beta0 are fitted by
L-BFGS-B on rows of the fit documents H of docs/11 11.6 (training documents i with
splitmix64(i XOR 0xC0FFEE) mod 100 == 0, never counted), with the budget of the config (fixed before
the first logged run). The secondary row `lin` is the probe's normalised linear mixture on the same
counts and fit rows. Kernels of the two probe scripts are imported unchanged; `k_loglin` of the
masked probe is the float64 CPU reference.

Modes (experiments/cvtm/configs/H8_1_{masked,causal}.json):
  masked  tiny-LM bank (V 2000, C 65536) of results/20260923T125642Z_H2_row_d, seeds 2 to 5 (seed 1
          was used in exploration: --smoke only). Counts over the 5M-token slice with only the target
          hidden; validation = the 56,210 frozen targets under the frozen hidden mask. Validation
          fired lists and the signed-vote check come from the CPU engine (the H2 path); the counting
          pass uses the CUDA engine (bit-exact with the CPU engine, cross-checked here on samples);
          the fit runs on the GPU; every reported loglin and lin number comes from the CPU float64
          reference kernels of the probe.
  causal  CP anchor epoch-4 bank (V 8000, C 1,048,576), counts over the first 20,000,000 training
          tokens (cut at a document boundary), validation = every position of the validation
          split, bits per byte = total bits / UTF-8 bytes (the byte accounting of
          experiments/cp_causal_perplexity.py). Counting streams sorted (clause, class) keys through
          a bounded GPU buffer and merges them exactly on the host; scoring runs on the GPU
          (deterministic fixed-point accumulation) with a CPU float64 cross-check.

Pipeline checks (the run stops when one fails): the signed-vote numbers of the bank are reproduced
exactly from the logged runs; fired lists equal the engine's public fired_clauses (and the brute
force) on samples; vote sums rebuilt from our fired lists equal predict_scores; the counted events
equal the fired events of the counting positions; every probability row sums to one; the GPU
objective agrees with the CPU reference.

Launch pattern (the one of results/20260924T022628Z_ckm_probe_causal_controls_exploratory):
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMLM_CPU_SLOTS=1
    source env/env.sh
    scripts/gpu_run.sh shared -- nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --limit-gib 25 -- \\
        scripts/run_logged.py --name cvtm_H8_1_masked_seed2 --config experiments/cvtm/configs/H8_1_masked.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_t1_readout.py --config experiments/cvtm/configs/H8_1_masked.json --bank-seed 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from engine_common import ROOT, EngineConfig, load_json, make_engine  # noqa: E402
from ckm_posthoc_probe import (k_dense_eval, k_dense_to_csr, k_gval, k_lin_obj, k_loglin,  # noqa: E402
                               k_pair_pk, k_vote_scores, rowptr_of)
from tmlm.engine.rng import mix64  # noqa: E402

EPOCH = 4                   # as in the probes: selects the random training mask only when no bitmap is given
NINC_CAP = 12
LOGN_CAP = 23
N_BUCKETS = (NINC_CAP + 1) * (LOGN_CAP + 1)
FIT_SALT = 0xC0FFEE
FIT_MODULUS = 100
LN2 = math.log(2.0)
SCALE_LOG = float(2 ** 40)  # fixed point of the GPU class sums (integer atomics: order independent)
GIB = float(1 << 30)


# ------------------------------------------------------------------ bookkeeping
class Recorder:
    """Labelled metrics (measured / derived / unverified) flushed to the run dir of run_logged.py.
    In smoke mode every entry carries "smoke": true."""

    def __init__(self, smoke: bool) -> None:
        self.smoke = bool(smoke)
        self.metrics: dict = {}
        self.t0 = time.time()
        self.gpu_pool_peak = 0

    def log(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:8.1f} s] {msg}", flush=True)

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        value = _plain(value)
        e = {"value": value, "label": label, **{k: _plain(v) for k, v in extra.items()}}
        if self.smoke:
            e["smoke"] = True
        self.metrics[name] = e

    def flush(self) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_metrics(self.metrics, on_conflict="replace")
        elif getattr(self, "scratch", None):
            Path(self.scratch).mkdir(parents=True, exist_ok=True)
            (Path(self.scratch) / "metrics_unlogged.json").write_text(json.dumps(self.metrics, indent=1, default=str) + "\n")

    def dataset(self, name: str, digest: str) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_dataset_hashes({name: digest}, on_conflict="replace")

    def memory(self, where: str, cp=None) -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        msg = f"memory at {where}: peak rss {rss / GIB:.2f} GiB"
        if cp is not None:
            pool = cp.get_default_memory_pool()
            self.gpu_pool_peak = max(self.gpu_pool_peak, int(pool.total_bytes()))
            msg += f", gpu pool now {pool.total_bytes() / GIB:.2f} GiB (peak seen {self.gpu_pool_peak / GIB:.2f})"
        self.log(msg)


def _plain(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def out_dir(args) -> Path:
    d = os.environ.get("TMLM_RUN_DIR")
    if d:
        return Path(d)
    if not args.out_dir:
        raise SystemExit("not under run_logged.py: give --out-dir (a scratch directory)")
    p = Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ public helpers (tested in tests/test_cvtm_t1_readout.py)
def fit_document_mask(n_docs: int, salt: int = FIT_SALT, modulus: int = FIT_MODULUS) -> np.ndarray:
    """docs/11 11.6: training document i (its index in the training split) is a fit document iff
    splitmix64(i XOR salt) mod modulus == 0, with splitmix64 = mix64 of docs/11 section 3."""
    return np.fromiter((mix64(i ^ salt) % modulus == 0 for i in range(int(n_docs))), dtype=bool, count=int(n_docs))


def included_literals(state: dict) -> np.ndarray:
    """n_inc per clause: included literals (lit_state >= 128, a real literal id) of alive clauses."""
    alive = state["alive"].astype(bool)
    inc = (state["lit_state"] >= 128) & (state["lit_id"] != 0xFFFFFFFF) & alive[:, None]
    return inc.sum(axis=1).astype(np.int64)


def clause_groups(n_inc: np.ndarray, N: np.ndarray) -> np.ndarray:
    """g(j) of docs/11 11.5 as one index: min(n_inc, 12) * 24 + min(floor(log2 max(N, 1)), 23); equal to
    the probe's `bucket` for its engine family (fam = 0)."""
    logn = np.minimum(np.floor(np.log2(np.maximum(np.asarray(N, np.int64), 1))).astype(np.int64), LOGN_CAP)
    return np.minimum(np.asarray(n_inc, np.int64), NINC_CAP) * (LOGN_CAP + 1) + logn


def witten_bell(N: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lam, inv_N) per clause; 0 where N == 0 (such a clause never votes)."""
    N = np.asarray(N, np.int64)
    D = np.asarray(D, np.int64)
    lam = np.where(N > 0, N / np.maximum(N + D, 1), 0.0)
    inv_n = np.where(N > 0, 1.0 / np.maximum(N, 1), 0.0)
    return lam, inv_n


def add_one_unigram(targets: np.ndarray, vocab: int) -> np.ndarray:
    uni = np.bincount(np.asarray(targets, np.int64), minlength=vocab).astype(np.float64)
    return (uni + 1.0) / (uni.sum() + vocab)


def projected_gradient_inf_norm(x: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """||P(x - g) - x||_inf, the stationarity measure that L-BFGS-B compares with gtol."""
    return float(np.max(np.abs(np.clip(x - g, lo, hi) - x))) if x.shape[0] else 0.0


def doc_block_bootstrap(num_by_doc: np.ndarray, den_by_doc: np.ndarray, resamples: int, seed: int) -> dict:
    """Ratio-of-sums statistic sum(num) / sum(den) with a document-block bootstrap (docs/13 A3)."""
    num = np.asarray(num_by_doc, np.float64)
    den = np.asarray(den_by_doc, np.float64)
    ok = den > 0
    num, den = num[ok], den[ok]
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 81, int(seed)]))
    pick = rng.integers(0, num.shape[0], size=(int(resamples), num.shape[0]))
    stats = num[pick].sum(axis=1) / den[pick].sum(axis=1)
    return {"point": float(num.sum() / den.sum()), "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "docs": int(num.shape[0]), "resamples": int(resamples)}


# ------------------------------------------------------------------ exact streamed counting (causal)
@njit(cache=False)
def k_merge_sorted_counts(ak, ac, bk, bc):
    """Union of two sorted unique key arrays with summed counts (exact integers)."""
    na = ak.shape[0]
    nb = bk.shape[0]
    i = 0
    j = 0
    n = 0
    while i < na and j < nb:
        if ak[i] < bk[j]:
            i += 1
        elif ak[i] > bk[j]:
            j += 1
        else:
            i += 1
            j += 1
        n += 1
    n += (na - i) + (nb - j)
    ok = np.empty(n, ak.dtype)
    oc = np.empty(n, ac.dtype)
    i = 0
    j = 0
    o = 0
    while i < na and j < nb:
        if ak[i] < bk[j]:
            ok[o] = ak[i]
            oc[o] = ac[i]
            i += 1
        elif ak[i] > bk[j]:
            ok[o] = bk[j]
            oc[o] = bc[j]
            j += 1
        else:
            ok[o] = ak[i]
            oc[o] = ac[i] + bc[j]
            i += 1
            j += 1
        o += 1
    while i < na:
        ok[o] = ak[i]
        oc[o] = ac[i]
        i += 1
        o += 1
    while j < nb:
        ok[o] = bk[j]
        oc[o] = bc[j]
        j += 1
        o += 1
    return ok, oc


@njit(cache=False)
def k_row_sums(indptr, val):
    n = indptr.shape[0] - 1
    out = np.zeros(n, np.int64)
    for k in range(n):
        s = 0
        for e in range(indptr[k], indptr[k + 1]):
            s += val[e]
        out[k] = s
    return out


@njit(cache=False)
def k_accumulate_events(fe, fj, tgt_of_example, counts):
    for i in range(fe.shape[0]):
        counts[fj[i], tgt_of_example[fe[i]]] += 1


class SortedCountAccumulator:
    """Exact (clause, class) counts from batches of sorted unique 64-bit keys j * V + y with counts.
    Keys are held per clause range of 2^range_bits clauses as sorted uint32 local keys with int32 counts
    (8 bytes per distinct pair). The result does not depend on how the events were split into batches
    (merging is exact integer addition), which is the property docs/11 11.6 asks of sharded counting."""

    def __init__(self, n_clauses: int, vocab: int, range_bits: int = 16, max_events_per_pair: int = (1 << 31) - 1) -> None:
        self.C, self.V, self.rb = int(n_clauses), int(vocab), int(range_bits)
        self.span_clauses = 1 << self.rb
        self.span = self.span_clauses * self.V
        if self.span >= (1 << 32):
            raise ValueError("range_bits too large for uint32 local keys")
        if max_events_per_pair >= (1 << 31):
            raise ValueError("int32 counts need fewer than 2^31 events per pair")
        self.n_ranges = (self.C + self.span_clauses - 1) // self.span_clauses
        self.keys = [np.zeros(0, np.uint32) for _ in range(self.n_ranges)]
        self.cnts = [np.zeros(0, np.int32) for _ in range(self.n_ranges)]
        self.events = 0
        self.batches = 0

    @property
    def n_pairs(self) -> int:
        return int(sum(k.shape[0] for k in self.keys if k is not None))

    def add(self, keys: np.ndarray, counts: np.ndarray) -> None:
        keys = np.asarray(keys, np.uint64)
        counts = np.asarray(counts, np.int32)
        if keys.shape[0] == 0:
            return
        if keys.shape[0] > 1 and not bool(np.all(keys[1:] > keys[:-1])):
            raise ValueError("keys must be sorted and unique")
        if int(keys[-1]) >= self.C * self.V:
            raise ValueError("key out of range")
        bounds = np.searchsorted(keys, np.arange(self.n_ranges + 1, dtype=np.uint64) * np.uint64(self.span))
        for r in range(self.n_ranges):
            a, b = int(bounds[r]), int(bounds[r + 1])
            if a == b:
                continue
            self.add_range(r, (keys[a:b] - np.uint64(r * self.span)).astype(np.uint32), counts[a:b])
        self.batches += 1

    def add_range(self, r: int, local_keys: np.ndarray, counts: np.ndarray) -> None:
        """Merge sorted unique uint32 local keys (j - r * 2^range_bits) * V + y of clause range r."""
        local_keys = np.asarray(local_keys, np.uint32)
        counts = np.asarray(counts, np.int32)
        if local_keys.shape[0] == 0:
            return
        if local_keys.shape[0] > 1 and not bool(np.all(local_keys[1:] > local_keys[:-1])):
            raise ValueError("local keys must be sorted and unique")
        if int(local_keys[-1]) >= self.span or not 0 <= int(r) < self.n_ranges:
            raise ValueError("local key or range out of bounds")
        self.keys[r], self.cnts[r] = k_merge_sorted_counts(self.keys[r], self.cnts[r], local_keys, counts)
        self.events += int(counts.astype(np.int64).sum())

    def add_unsorted(self, keys: np.ndarray) -> None:
        """Convenience for CPU callers and tests: raw event keys, any order, repeats allowed."""
        u, c = np.unique(np.asarray(keys, np.uint64), return_counts=True)
        self.add(u, c.astype(np.int32))

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR over clauses: indptr int64 [C + 1], idx uint16 (ascending per clause), val int32.
        The per-range buffers are released while the CSR is filled."""
        if self.V > 65536:
            raise ValueError("uint16 class ids need V <= 65536")
        per_clause = np.zeros(self.n_ranges * self.span_clauses, np.int64)
        for r in range(self.n_ranges):
            if self.keys[r].shape[0]:
                per_clause[r * self.span_clauses:(r + 1) * self.span_clauses] = np.bincount(
                    (self.keys[r] // np.uint32(self.V)).astype(np.int64), minlength=self.span_clauses)
        per_clause = per_clause[:self.C]
        indptr = np.zeros(self.C + 1, np.int64)
        indptr[1:] = np.cumsum(per_clause)
        total = int(indptr[-1])
        idx = np.empty(total, np.uint16)
        val = np.empty(total, np.int32)
        o = 0
        for r in range(self.n_ranges):
            k = self.keys[r]
            n = k.shape[0]
            idx[o:o + n] = (k % np.uint32(self.V)).astype(np.uint16)
            val[o:o + n] = self.cnts[r]
            o += n
            self.keys[r] = np.zeros(0, np.uint32)
            self.cnts[r] = np.zeros(0, np.int32)
        assert o == total
        return indptr, idx, val


@njit(cache=False)
def k_merge_compact_pass(ptrA, clsA, c8A, ovkA, ovcA, kB, cB, V, fill, cls_o, c8_o, ovk_o, ovc_o, per_clause):
    """One pass of the exact merge of a compact range table A (clause offsets ptrA, uint16 class ids,
    uint8 counts with 255 = "see the overflow table", overflow keys and full counts in key order) with
    sorted unique local keys kB and counts cB. fill = False only sizes the output."""
    nA = clsA.shape[0]
    nB = kB.shape[0]
    ea = 0
    ja = 0
    oa = 0
    ib = 0
    n = 0
    nov = 0
    while ea < nA or ib < nB:
        ka = np.int64(-1)
        if ea < nA:
            while ptrA[ja + 1] <= ea:
                ja += 1
            ka = np.int64(ja) * V + np.int64(clsA[ea])
        kb = np.int64(kB[ib]) if ib < nB else np.int64(-1)
        if kb < 0 or (ka >= 0 and ka <= kb):
            key = ka
            if c8A[ea] == 255:
                if ovkA[oa] != ka:
                    raise ValueError("overflow table out of step")
                v = np.int64(ovcA[oa])
                oa += 1
            else:
                v = np.int64(c8A[ea])
            ea += 1
            if ka == kb:
                v += np.int64(cB[ib])
                ib += 1
        else:
            key = kb
            v = np.int64(cB[ib])
            ib += 1
        if fill:
            cls_o[n] = key % V
            per_clause[key // V] += 1
            if v >= 255:
                c8_o[n] = 255
                ovk_o[nov] = key
                ovc_o[nov] = v
            else:
                c8_o[n] = v
        if v >= 255:
            nov += 1
        n += 1
    return n, nov


class CompactCountAccumulator:
    """Exact (clause, class) counts at about 3 bytes per distinct pair: per clause range of 2^range_bits
    clauses, clause offsets, uint16 class ids and uint8 counts, with every count >= 255 kept exactly in a
    per-range overflow table (uint32 local key, int64 count). Same interface and same result as
    SortedCountAccumulator (tested); used by the causal run so that the 20M-token table (about 1.06e9
    pairs) fits the host memory left by the other jobs of the container."""

    def __init__(self, n_clauses: int, vocab: int, range_bits: int = 16) -> None:
        self.C, self.V, self.rb = int(n_clauses), int(vocab), int(range_bits)
        if self.V > 65536:
            raise ValueError("uint16 class ids need V <= 65536")
        self.span_clauses = 1 << self.rb
        self.span = self.span_clauses * self.V
        if self.span >= (1 << 32):
            raise ValueError("range_bits too large for uint32 local keys")
        self.n_ranges = (self.C + self.span_clauses - 1) // self.span_clauses
        S = self.span_clauses
        self.tab = [(np.zeros(S + 1, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.uint8), np.zeros(0, np.uint32),
                     np.zeros(0, np.int64)) for _ in range(self.n_ranges)]
        self.events = 0
        self.batches = 0

    @property
    def n_pairs(self) -> int:
        return int(sum(t[1].shape[0] for t in self.tab))

    @property
    def n_overflow(self) -> int:
        return int(sum(t[3].shape[0] for t in self.tab))

    def nbytes(self) -> int:
        return int(sum(sum(a.nbytes for a in t) for t in self.tab))

    def add(self, keys: np.ndarray, counts: np.ndarray) -> None:
        keys = np.asarray(keys, np.uint64)
        counts = np.asarray(counts, np.int64)
        if keys.shape[0] == 0:
            return
        if keys.shape[0] > 1 and not bool(np.all(keys[1:] > keys[:-1])):
            raise ValueError("keys must be sorted and unique")
        if int(keys[-1]) >= self.C * self.V:
            raise ValueError("key out of range")
        bounds = np.searchsorted(keys, np.arange(self.n_ranges + 1, dtype=np.uint64) * np.uint64(self.span))
        for r in range(self.n_ranges):
            a, b = int(bounds[r]), int(bounds[r + 1])
            if a < b:
                self.add_range(r, (keys[a:b] - np.uint64(r * self.span)).astype(np.uint32), counts[a:b])
        self.batches += 1

    def add_unsorted(self, keys: np.ndarray) -> None:
        u, c = np.unique(np.asarray(keys, np.uint64), return_counts=True)
        self.add(u, c.astype(np.int64))

    def add_range(self, r: int, local_keys: np.ndarray, counts: np.ndarray) -> None:
        kB = np.asarray(local_keys, np.uint32)
        cB = np.asarray(counts, np.int64)
        if kB.shape[0] == 0:
            return
        if kB.shape[0] > 1 and not bool(np.all(kB[1:] > kB[:-1])):
            raise ValueError("local keys must be sorted and unique")
        if not 0 <= int(r) < self.n_ranges:
            raise ValueError("range out of bounds")
        n_real = min(self.span_clauses, self.C - int(r) * self.span_clauses)
        if int(kB[-1]) >= n_real * self.V or int(cB.min()) <= 0:
            raise ValueError("local key or count out of bounds")
        ptrA, clsA, c8A, ovkA, ovcA = self.tab[r]
        z16, z8, z32, z64 = np.zeros(1, np.uint16), np.zeros(1, np.uint8), np.zeros(1, np.uint32), np.zeros(1, np.int64)
        n, nov = k_merge_compact_pass(ptrA, clsA, c8A, ovkA, ovcA, kB, cB, self.V, False, z16, z8, z32, z64, z64)
        cls_o = np.empty(n, np.uint16)
        c8_o = np.empty(n, np.uint8)
        ovk_o = np.empty(nov, np.uint32)
        ovc_o = np.empty(nov, np.int64)
        per = np.zeros(self.span_clauses, np.int64)
        k_merge_compact_pass(ptrA, clsA, c8A, ovkA, ovcA, kB, cB, self.V, True, cls_o, c8_o, ovk_o, ovc_o, per)
        ptr = np.zeros(self.span_clauses + 1, np.int64)
        ptr[1:] = np.cumsum(per)
        self.tab[r] = (ptr, cls_o, c8_o, ovk_o, ovc_o)
        self.events += int(cB.sum())

    def _range_arrays(self, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(clause offsets restricted to real clauses, class ids, exact int32 counts) of range r."""
        ptr, cls, c8, ovk, ovc = self.tab[r]
        val = c8.astype(np.int32)
        big = np.flatnonzero(c8 == 255)
        if big.shape[0] != ovk.shape[0]:
            raise ValueError("overflow table size mismatch")
        if big.shape[0]:
            clause = np.searchsorted(ptr, big, side="right") - 1
            if not np.array_equal(clause.astype(np.int64) * self.V + cls[big].astype(np.int64), ovk.astype(np.int64)):
                raise ValueError("overflow keys do not match the entries")
            if int(ovc.max()) >= (1 << 31):
                raise ValueError("a count does not fit int32")
            val[big] = ovc.astype(np.int32)
        n_real = min(self.span_clauses, self.C - r * self.span_clauses)
        return ptr[:n_real + 1], cls, val

    def _indptr(self) -> np.ndarray:
        indptr = np.zeros(self.C + 1, np.int64)
        o = 0
        for r in range(self.n_ranges):
            ptr = self.tab[r][0]
            n_real = min(self.span_clauses, self.C - r * self.span_clauses)
            indptr[r * self.span_clauses + 1:r * self.span_clauses + n_real + 1] = o + ptr[1:n_real + 1]
            o += int(ptr[n_real])
        return indptr

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host CSR (indptr int64 [C + 1], idx uint16, val int32), as SortedCountAccumulator.finalize."""
        indptr = self._indptr()
        idx = np.empty(int(indptr[-1]), np.uint16)
        val = np.empty(int(indptr[-1]), np.int32)
        for r in range(self.n_ranges):
            ptr, cls, v = self._range_arrays(r)
            a = int(indptr[r * self.span_clauses])
            idx[a:a + cls.shape[0]] = cls
            val[a:a + cls.shape[0]] = v
            self.tab[r] = None
        return indptr, idx, val

    def finalize_to_device(self, cp) -> tuple[np.ndarray, object, object, np.ndarray]:
        """Device CSR filled range by range (the host never holds the whole table): (indptr host,
        idx device uint16, val device int32, N per clause host int64). The ranges are released."""
        indptr = self._indptr()
        total = int(indptr[-1])
        d_idx = cp.empty(total, dtype=cp.uint16)
        d_val = cp.empty(total, dtype=cp.int32)
        N = np.zeros(self.C, np.int64)
        for r in range(self.n_ranges):
            ptr, cls, v = self._range_arrays(r)
            a = int(indptr[r * self.span_clauses])
            if cls.shape[0]:
                d_idx[a:a + cls.shape[0]].set(cls)
                d_val[a:a + cls.shape[0]].set(v)
                n_real = ptr.shape[0] - 1
                clause = np.repeat(np.arange(n_real, dtype=np.int32), np.diff(ptr))
                N[r * self.span_clauses:r * self.span_clauses + n_real] = np.bincount(clause, weights=v, minlength=n_real).astype(np.int64)
            self.tab[r] = None
        return indptr, d_idx, d_val, N


# ------------------------------------------------------------------ GPU read-out kernel (q computed on the fly)
T1_SRC = r"""
#define BS 512
extern "C" __global__ void __launch_bounds__(BS) t1_loglin(
        const long long* __restrict__ rowptr, const int* __restrict__ comp, const int* __restrict__ bkt,
        const double* __restrict__ alpha, const long long* __restrict__ indptr,
        const unsigned short* __restrict__ idx, const int* __restrict__ val,
        const double* __restrict__ lam, const double* __restrict__ inv_n, const double* __restrict__ pu,
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
            const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;
            const long long e1 = indptr[k + 1];
            for (long long e = indptr[k] + tid; e < e1; e += BS) {
                const int c = idx[e];
                const double g = log(1.0 + l * (double)val[e] * inn / (oml * pu[c]));
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
                const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;
                double eg = 0.0, gy = 0.0;
                const long long e1 = indptr[k + 1];
                for (long long e = indptr[k] + lane; e < e1; e += 32) {
                    const int c = idx[e];
                    const double gv = log(1.0 + l * (double)val[e] * inn / (oml * pu[c]));
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


def to_device_chunked(cp, host: np.ndarray, dtype, chunk_bytes: int = 1 << 28):
    """Copy a large host array to the device in slices of at most chunk_bytes, so that no host staging
    copy of the whole array is made (host memory stays bounded under the cgroup-wide guard)."""
    h = np.asarray(host)
    out = cp.empty(h.shape[0], dtype=dtype)
    step = max(1, chunk_bytes // max(1, np.dtype(dtype).itemsize))
    for a in range(0, h.shape[0], step):
        out[a:a + step].set(np.ascontiguousarray(h[a:a + step], dtype=dtype))
    return out


class T1Gpu:
    """The count-vote read-out on the GPU: one block per row, the V class sums in shared memory as
    int64 fixed point (scale 2^40, integer atomics: the result does not depend on thread order), q
    recomputed from the exact counts (no per-entry vote array, so the 20M-token table fits)."""

    def __init__(self, V: int, indptr, idx, val, lam, inv_n, p_uni, grid: int = 1024, device_csr=None) -> None:
        import cupy as cp
        self.cp = cp
        self.V = int(V)
        self.grid = int(grid)
        mod = cp.RawModule(code=T1_SRC, options=("-std=c++14",))
        self.k = mod.get_function("t1_loglin")
        self.smem = 8 * self.V
        self.k.max_dynamic_shared_size_bytes = self.smem
        if device_csr is None:
            self.d_indptr = to_device_chunked(cp, indptr, np.int64)
            self.d_idx = to_device_chunked(cp, idx, np.uint16)
            self.d_val = to_device_chunked(cp, val, np.int32)
        else:
            self.d_indptr, self.d_idx, self.d_val = device_csr
        self.d_lam = cp.asarray(np.ascontiguousarray(lam, dtype=np.float64))
        self.d_inv = cp.asarray(np.ascontiguousarray(inv_n, dtype=np.float64))
        self.d_pu = cp.asarray(np.ascontiguousarray(p_uni, dtype=np.float64))
        self.d_logpu = cp.asarray(np.log(np.ascontiguousarray(p_uni, dtype=np.float64)))

    def prepare(self, rp: np.ndarray, comp: np.ndarray, bkt: np.ndarray, gold: np.ndarray, rows_per_chunk: int = 16384) -> list:
        """Upload the fixed row data once (the fit evaluates the same rows many times)."""
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
                   (ch["rp"], ch["comp"], ch["bkt"], d_alpha, self.d_indptr, self.d_idx, self.d_val, self.d_lam, self.d_inv,
                    self.d_pu, self.d_logpu, np.float64(beta0), np.float64(SCALE_LOG), ch["gold"], np.int32(nr),
                    np.int32(self.V), o_lpg, o_pg, o_rank, o_top, o_sum, np.int32(1 if want_grad else 0), o_gp, o_gb),
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


# ------------------------------------------------------------------ fitting
def lbfgsb(fun, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: dict) -> tuple:
    """scipy L-BFGS-B with the config budget; returns (res, trace of objective values per evaluation)."""
    from scipy.optimize import minimize
    trace: list[float] = []

    def f(theta):
        v, g = fun(theta)
        trace.append(float(v))
        return v, g
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo.tolist(), hi.tolist())),
                   options={"maxiter": int(budget["maxiter"]), "maxfun": int(budget["maxfun"]), "ftol": float(budget["ftol"]),
                            "gtol": float(budget["gtol"]), "maxcor": int(budget.get("maxcor", 10))})
    return res, trace


def fit_info(res, trace, x, g, lo, hi, n_rows, extra=None) -> dict:
    msg = res.message.decode() if isinstance(res.message, bytes) else str(res.message)
    d = {"converged": bool(res.success), "message": msg, "iterations": int(res.nit), "function_evaluations": int(res.nfev),
         "objective_nats_per_row": float(res.fun), "projected_gradient_inf_norm": projected_gradient_inf_norm(x, g, lo, hi),
         "fit_rows": int(n_rows), "n_params": int(x.shape[0]),
         "objective_change_last_10_evals": float(trace[-11] - trace[-1]) if len(trace) > 10 else None,
         "params_at_bounds": int(np.sum((x <= lo + 1e-12) | (x >= hi - 1e-12)))}
    if extra:
        d.update(extra)
    return d


def fit_lin(rp, pk, bk, pu_g, budget) -> tuple[np.ndarray, dict]:
    """The probe's normalised linear mixture: P(gold) = (a0 P_uni(gold) + sum_j w_g(j) O_j,gold / N_j) / (a0 + sum_j w_g(j)),
    a0 = 1, w = exp(theta) per group; mean negative log-likelihood of the fit rows by L-BFGS-B."""
    used = np.unique(bk)
    R = rp.shape[0] - 1

    def fun(theta):
        alpha = np.zeros(N_BUCKETS)
        alpha[used] = np.exp(theta)
        ll, num, den, _, _ = k_lin_obj(rp, pk, bk, pu_g, alpha, 1.0)
        return -ll / R, -(alpha[used] * (num[used] - den[used])) / R
    lo = np.full(used.shape[0], float(budget["log_weight_bounds"][0]))
    hi = np.full(used.shape[0], float(budget["log_weight_bounds"][1]))
    res, trace = lbfgsb(fun, np.zeros(used.shape[0]), lo, hi, budget)
    f, g = fun(res.x)
    w = np.zeros(N_BUCKETS)
    w[used] = np.exp(res.x)
    return w, fit_info(res, trace, res.x, g, lo, hi, R, {"objective_recomputed": float(f), "used_groups": used.tolist()})


def lin_weights(rowid, comp_bkt, n_rows, w_group):
    a = w_group[comp_bkt]
    Z = np.bincount(rowid, weights=a, minlength=n_rows) + 1.0
    return a / Z[rowid], 1.0 / Z


# ------------------------------------------------------------------ rows
def csr_select(ptr: np.ndarray, ids: np.ndarray, keep_clause: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rowptr, rowid, comp) of a fired-list CSR restricted to the clauses with counts."""
    n = ptr.shape[0] - 1
    rowid = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    comp = ids.astype(np.int64)
    k = keep_clause[comp]
    rowid, comp = rowid[k], comp[k]
    return rowptr_of(rowid, n), rowid, comp


def subset_rows(ptr: np.ndarray, ids: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = ptr[rows], ptr[rows + 1]
    lens = ends - starts
    rp = np.zeros(rows.shape[0] + 1, np.int64)
    rp[1:] = np.cumsum(lens)
    take = np.repeat(starts - rp[:-1], lens) + np.arange(rp[-1], dtype=np.int64)
    return rp, ids[take]


def cpu_fired_lists(eng, positions: np.ndarray, bitmap: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The CPU engine's own fired lists (tm_fired_clauses, ascending ids) for every position."""
    lib, h = eng.lib, eng.handle
    C = int(eng.cfg.n_clauses)
    buf = np.zeros(C, dtype=np.uint32)
    bm = None if bitmap is None else np.ascontiguousarray(bitmap, dtype=np.uint8)
    bmp = None if bm is None else bm.ctypes.data
    n = int(positions.shape[0])
    ptr = np.zeros(n + 1, np.int64)
    parts = []
    for i, p in enumerate(positions.tolist()):
        m = lib.tm_fired_clauses(h, EPOCH, int(p), bmp, buf.ctypes.data, C, 0)
        parts.append(buf[:m].copy())
        ptr[i + 1] = ptr[i] + m
    return ptr, (np.concatenate(parts) if parts else np.zeros(0, np.uint32))


def cuda_rows(eng, positions: np.ndarray, bitmap_dev, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """CUDA fired lists (ascending ids per row) for the given positions."""
    cp = eng.cp
    n = int(positions.shape[0])
    ptr = np.zeros(n + 1, np.int64)
    parts = []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, cp.asarray(positions[lo:hi].astype(np.uint32)), bitmap_dev, 0)
        eng._marks = []
        fe, fj = fw.fired_ex, fw.fired_j
        if fe.size:
            order = cp.argsort((fe.astype(cp.uint64) << cp.uint64(32)) | fj.astype(cp.uint64))
            fe, fj = fe[order], fj[order]
            cnt = cp.bincount(fe.astype(cp.int64), minlength=hi - lo).get()
        else:
            cnt = np.zeros(hi - lo, np.int64)
        ptr[lo + 1:hi + 1] = ptr[lo] + np.cumsum(cnt)
        parts.append(fj.get().astype(np.uint32))
        del fw
    return ptr, (np.concatenate(parts) if parts else np.zeros(0, np.uint32))


def sample_positions(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 812, int(seed)]))
    return np.sort(rng.choice(pool, size=min(int(k), int(pool.shape[0])), replace=False))


def budget_of(cfg: dict, args, key: str) -> dict:
    b = dict(cfg["fit"][key])
    if args.maxiter:
        if not args.smoke:
            raise SystemExit("--maxiter is a smoke-only override")
        b["maxiter"] = int(args.maxiter)
    return b


# ------------------------------------------------------------------ the loglin fit and CPU reference
def loglin_fit_gpu(t1: T1Gpu, rows: dict, budget: dict, rec: Recorder, tag: str) -> tuple[np.ndarray, float, dict]:
    """alpha per group and beta0 by L-BFGS-B on the fit rows, objective = mean NLL (nats) on the GPU."""
    bk = rows["bkt"]
    used = np.unique(bk)
    R = rows["n"]
    chunks = t1.prepare(rows["rp"], rows["comp"], bk, rows["gold"])
    n_pairs = int(rows["comp"].shape[0])
    nev = [0]
    t_eval = [0.0]

    def fun(theta):
        t = time.time()
        alpha = np.zeros(N_BUCKETS)
        alpha[used] = theta[:-1]
        o = t1.run(chunks, R, n_pairs, alpha, float(theta[-1]), True)
        f = -float(o["lpg"].sum()) / R
        g = -np.append(np.bincount(bk, weights=o["gpair"], minlength=N_BUCKETS)[used], o["gbeta"].sum()) / R
        nev[0] += 1
        t_eval[0] += time.time() - t
        if nev[0] % 25 == 0:
            rec.log(f"  {tag} fit: {nev[0]} evaluations, objective {f:.8f} nats/row, {t_eval[0] / nev[0]:.2f} s/eval")
        return f, g
    a_init = 1.0 / max(1.0, float(np.diff(rows["rp"]).mean()))
    lo = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][0])), float(budget["beta0_bounds"][0]))
    hi = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][1])), float(budget["beta0_bounds"][1]))
    x0 = np.append(np.full(used.shape[0], a_init), 1.0)
    t0 = time.time()
    scale = np.ones(x0.shape[0])
    pre = budget.get("precondition", "none")
    if pre == "empirical_fisher_diagonal_at_init":
        # A fixed diagonal reparametrisation u = x * s, s = sqrt of the per-parameter mean squared per-row gradient at x0.
        # The model and its optimum are unchanged; only the optimizer's path (conditioning) is.
        alpha0 = np.zeros(N_BUCKETS)
        alpha0[used] = x0[:-1]
        o = t1.run(chunks, R, n_pairs, alpha0, float(x0[-1]), True)
        per_row = np.bincount(rows["rowid"].astype(np.int64) * N_BUCKETS + bk, weights=o["gpair"], minlength=R * N_BUCKETS).reshape(R, N_BUCKETS)
        fisher = np.append((per_row[:, used] ** 2).mean(axis=0), (o["gbeta"] ** 2).mean())
        scale = np.sqrt(fisher + 1e-12)
        del o, per_row
    elif pre != "none":
        raise ValueError(f"unknown precondition {pre!r}")
    res, trace = lbfgsb(lambda u: (lambda fg: (fg[0], fg[1] / scale))(fun(u / scale)), x0 * scale, lo * scale, hi * scale, budget)
    wall = time.time() - t0
    x = res.x / scale
    f, g = fun(x)
    alpha = np.zeros(N_BUCKETS)
    alpha[used] = x[:-1]
    beta0 = float(x[-1])
    info = fit_info(res, trace, x, g, lo, hi, R, {"init": f"alpha {a_init:.6g} (1 / mean counted fired clauses per fit row), beta0 1",
                                                  "precondition": pre, "preconditioner_min_max": [float(scale.min()), float(scale.max())],
                                                  "projected_gradient_inf_norm_optimizer_space": projected_gradient_inf_norm(res.x, g / scale, lo * scale, hi * scale),
                                                  "wall_s": wall, "s_per_evaluation": t_eval[0] / max(1, nev[0]),
                                                  "objective_gpu_recomputed": float(f), "used_groups": used.tolist(),
                                                  "objective_trace_every_10th": trace[::10]})
    del chunks
    t1.cp.get_default_memory_pool().free_all_blocks()
    return alpha, beta0, info


def cpu_loglin(rows: dict, alpha: np.ndarray, beta0: float, indptr, idx, gval, logpu, want_grad: bool):
    n_pairs = rows["comp"].shape[0]
    tstar = np.full(n_pairs, -1, np.int64)
    bonus = np.zeros(n_pairs)
    return k_loglin(rows["rp"], rows["comp"], rows["bkt"].astype(np.int64), alpha, float(beta0), indptr, idx, gval, logpu,
                    rows["gold"], tstar, bonus, N_BUCKETS, want_grad)


def verdict_masked(signed_top1: float, loglin_top1: float, gain_min: float) -> str:
    return "MET" if loglin_top1 - signed_top1 >= gain_min else "NOT MET"


def verdict_causal(signed_bpb: float, loglin_bpb: float, drop_min: float) -> str:
    return "MET" if signed_bpb - loglin_bpb >= drop_min else "NOT MET"


# ------------------------------------------------------------------ masked
def run_masked(cfg: dict, args, rec: Recorder) -> int:
    from engine_common import TinyLmTask
    from tmlm.common import hashing, locks
    seed = int(args.bank_seed)
    if seed not in cfg["bank_seeds_reported"] and not args.smoke:
        raise SystemExit(f"bank seed {seed} is not a reported seed of the config; use --smoke")
    bank = ROOT / cfg["banks"][str(seed)]
    V = int(cfg["data"]["vocab"])
    rec.m("setup.bank_seed", seed)
    rec.m("setup.bank_file", str(bank.relative_to(ROOT)), sha256=hashing.sha256_file(bank))
    rec.dataset(f"bank_layer1_seed{seed}", hashing.sha256_file(bank))
    rec.m("setup.config", cfg)

    # ------------------------------------------------ data
    task = TinyLmTask(vocab=V, n_tokens=int(cfg["data"]["train_tokens"]), mode="masked")
    train, val = task.train, task.val
    hidden = task.hidden.astype(np.uint8)
    targets = task.targets.astype(np.int64)
    gold = task.gold.astype(np.int64)
    assert np.all(np.diff(targets) > 0) and np.all(hidden[targets] == 1)
    for k, v in task.metrics().items():
        rec.m(f"task.{k}", v["value"], v.get("label", "measured"), **{kk: vv for kk, vv in v.items() if kk not in ("value", "label")})
    from tmlm.data.corpus_store import split_dir
    for split in ("train", "validation"):
        meta = json.loads((split_dir("wikitext103", V, split) / "meta.json").read_text())
        rec.dataset(f"{split}_tok_V{V}", meta["sha256"]["tok.npy"])
    from tmlm.data.evalsets import load_evalset
    rec.dataset(f"evalset_validation_subword_V{V}", load_evalset("wikitext103", V, "validation", "subword")["meta"]["sha256_combined"])
    val_rows = targets.shape[0] if not args.val_max_rows else min(int(args.val_max_rows), targets.shape[0])
    if args.val_max_rows and not args.smoke:
        raise SystemExit("--val-max-rows is a smoke-only override")
    rec.log(f"task: train {train.tok.shape[0]} tokens, {train.doc_start.shape[0] - 1} docs; validation {val.tok.shape[0]} tokens, "
            f"{targets.shape[0]} targets")

    # ------------------------------------------------ bank and engines
    l1 = load_json(ROOT / cfg["engine_config"])
    ecfg = EngineConfig(**{**l1["engine"], "seed": seed})
    C = int(ecfg.n_clauses)
    st = dict(np.load(bank))
    eng_va = make_engine(ecfg, val)
    eng_va.load_state(st)
    rec.m("bank.state_hash", eng_va.state_hash())
    n_inc = included_literals(st)
    rec.m("bank.alive_clauses", int(st["alive"].astype(bool).sum()))

    # ------------------------------------------------ validation fired lists (CPU engine, frozen mask) + M1
    t0 = time.time()
    vptr, vids = cpu_fired_lists(eng_va, targets, hidden)
    rec.m("throughput.validation_fired_rows_per_s", targets.shape[0] / (time.time() - t0), note="CPU engine, one thread, nice 19, contended host")
    for p_i in sample_positions(np.arange(targets.shape[0]), 200, 1).tolist():
        a = eng_va.fired_clauses(EPOCH, int(targets[p_i]), hidden)
        assert np.array_equal(a, vids[vptr[p_i]:vptr[p_i + 1]]), "validation fired list differs from the API"
    for p_i in sample_positions(np.arange(targets.shape[0]), 5, 2).tolist():
        b = eng_va.fired_clauses_bruteforce(EPOCH, int(targets[p_i]), hidden)
        assert np.array_equal(b, vids[vptr[p_i]:vptr[p_i + 1]]), "validation fired list differs from the brute force"
    hits = 0
    top5 = 0
    mism = 0
    signed_correct = np.zeros(targets.shape[0], np.int8)
    for i in range(0, targets.shape[0], 4096):
        tp = targets[i:i + 4096]
        s = eng_va.predict_scores(EPOCH, tp.astype(np.uint32), hidden)
        g = gold[i:i + 4096]
        am = s.argmax(axis=1)
        signed_correct[i:i + 4096] = (am == g)
        hits += int((am == g).sum())
        sg = s[np.arange(g.shape[0]), g][:, None]
        rank = (s > sg).sum(axis=1) + ((s == sg) & (np.arange(V)[None, :] < g[:, None])).sum(axis=1)
        top5 += int((rank < 5).sum())
        rebuilt = k_vote_scores(vptr, vids, np.arange(i, i + tp.shape[0], dtype=np.int64), st["vote_cls"], st["vote_w"].astype(np.int32), V)
        mism += int((rebuilt != s).any(axis=1).sum())
    signed_top1 = hits / targets.shape[0]
    h2 = json.loads((ROOT / cfg["expected_signed_top1"]["source"]).read_text())
    expect = float(h2[cfg["expected_signed_top1"]["key"].format(seed=seed)]["value"])
    passed = bool(signed_top1 == expect and mism == 0)
    rec.m("check.signed_top1_reproduced", signed_top1, expected=expect, source=cfg["expected_signed_top1"]["source"], passed=passed,
          note="predict_scores argmax (ties to the lowest id) over all frozen validation targets, CPU engine")
    rec.m("check.rebuilt_vote_sums_mismatch_rows", mism, note="vote sums rebuilt from our fired lists vs predict_scores, every target")
    rec.m("rows.signed.top1", signed_top1)
    rec.m("rows.signed.top5", top5 / targets.shape[0], note="rank of the gold class in the raw class sums, ties to the lowest id")
    rec.log(f"pipeline check: signed top-1 {signed_top1:.6f} (expected {expect:.6f}), rebuilt mismatches {mism}")
    rec.flush()
    if not passed:
        rec.log("PIPELINE CHECK FAILED: stop")
        return 3

    # ------------------------------------------------ fit documents and counting positions
    tok_tr = train.tok.astype(np.int64)
    ds_tr = train.doc_start.astype(np.int64)
    n_doc = ds_tr.shape[0] - 1
    fit_doc = fit_document_mask(n_doc)
    doc_of = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_fit = fit_doc[doc_of]
    fit_pos = np.flatnonzero(is_fit).astype(np.int64)
    cnt_pos = np.flatnonzero(~is_fit).astype(np.int64)
    if args.max_count_pos:
        if not args.smoke:
            raise SystemExit("--max-count-pos is a smoke-only override")
        cnt_pos = cnt_pos[:int(args.max_count_pos)]
    rec.m("split.fit_documents", np.flatnonzero(fit_doc).tolist(), rule="splitmix64(i XOR 0xC0FFEE) mod 100 == 0 (docs/11 11.6)")
    rec.m("split.fit_positions", int(fit_pos.shape[0]))
    rec.m("split.counting_positions", int(cnt_pos.shape[0]))
    rec.log(f"fit documents {np.flatnonzero(fit_doc).tolist()} ({fit_pos.shape[0]} positions); counting positions {cnt_pos.shape[0]}")

    # ------------------------------------------------ counting pass (CUDA engine, only the target hidden)
    locks.yield_gpu_if_requested()
    import cupy as cp
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB))
    eng_g = make_engine(ecfg, train, engine="cuda")
    eng_g.load_state(st)
    assert eng_g.state_hash() == rec.metrics["bank.state_hash"]["value"]
    zero = np.zeros(train.tok.shape[0], dtype=np.uint8)
    zero_dev = cp.asarray(zero)
    counts = np.zeros((C, V), dtype=np.int32)
    chunk = int(cfg["compute"]["forward_chunk"])
    check_cnt = sample_positions(np.arange(cnt_pos.shape[0]), int(cfg["checks"]["cross_engine_count_rows"]), 3)
    got_cnt: dict[int, np.ndarray] = {}
    n_events = 0
    t0 = time.time()
    ci = 0
    for lo in range(0, cnt_pos.shape[0], chunk):
        hi = min(cnt_pos.shape[0], lo + chunk)
        eng_g._sync_state_struct()
        fw = eng_g._forward(EPOCH, cp.asarray(cnt_pos[lo:hi].astype(np.uint32)), zero_dev, 0)
        eng_g._marks = []
        fe = fw.fired_ex.get().astype(np.int64)
        fj = fw.fired_j.get().astype(np.int64)
        del fw
        assert fe.shape[0] == 0 or bool(np.all(fe[1:] >= fe[:-1])), "fired_ex is not example-major"
        k_accumulate_events(fe, fj, tok_tr[cnt_pos[lo:hi]], counts)
        n_events += int(fe.shape[0])
        a, b = np.searchsorted(check_cnt, [lo, hi])
        for r in check_cnt[a:b].tolist():
            s0, s1 = np.searchsorted(fe, [r - lo, r - lo + 1])
            got_cnt[r] = np.sort(fj[s0:s1]).astype(np.uint32)
        ci += 1
        if ci % 20 == 0:
            rec.log(f"  count pass: {hi}/{cnt_pos.shape[0]} positions, {hi / (time.time() - t0):.0f} pos/s, fired/pos {n_events / hi:.1f}")
    wall_count = time.time() - t0
    rec.m("throughput.count_pass_pos_per_s", cnt_pos.shape[0] / wall_count, note="CUDA engine forward, shared GPU")
    rec.m("counts.events", n_events)
    fptr, fids = cuda_rows(eng_g, fit_pos, zero_dev, chunk)
    rec.memory("after the counting pass", cp)
    # cross-engine check: CPU engine fired lists on samples of the counting and fit rows
    eng_c = make_engine(ecfg, train)
    eng_c.load_state(st)
    bad = 0
    for r in check_cnt.tolist():
        ref = cpu_fired_lists(eng_c, np.array([cnt_pos[r]]), zero)[1]
        bad += int(not np.array_equal(ref, got_cnt[r]))
    check_fit = sample_positions(np.arange(fit_pos.shape[0]), int(cfg["checks"]["cross_engine_fit_rows"]), 4)
    for r in check_fit.tolist():
        ref = cpu_fired_lists(eng_c, np.array([fit_pos[r]]), zero)[1]
        bad += int(not np.array_equal(ref, fids[fptr[r]:fptr[r + 1]]))
    for r in check_fit[:5].tolist():
        bad += int(not np.array_equal(eng_c.fired_clauses_bruteforce(EPOCH, int(fit_pos[r]), zero), fids[fptr[r]:fptr[r + 1]]))
    rec.m("check.cuda_vs_cpu_fired_list_mismatches", bad, rows=int(check_cnt.shape[0] + check_fit.shape[0]),
          note="CUDA counting-pass lists vs the CPU engine's tm_fired_clauses (and 5 brute-force rows)")
    rec.log(f"cross-engine fired lists: {bad} mismatches over {check_cnt.shape[0] + check_fit.shape[0]} rows")
    rec.flush()
    if bad:
        rec.log("PIPELINE CHECK FAILED: stop")
        return 3
    del eng_c, eng_g, zero_dev
    pool.free_all_blocks()

    # ------------------------------------------------ count tables and read-out constants
    indptr, idx, valc = k_dense_to_csr(counts)
    del counts
    N = k_row_sums(indptr, valc)
    D = np.diff(indptr)
    assert int(N.sum()) == n_events, "counted events differ from fired events"
    lam, inv_n = witten_bell(N, D)
    p_uni = add_one_unigram(tok_tr[cnt_pos], V)
    logpu = np.log(p_uni)
    grp = clause_groups(n_inc, N)
    gval = k_gval(indptr, idx, valc, lam, inv_n, p_uni)
    has = N > 0
    rec.m("counts.clauses_with_counts", int(has.sum()))
    rec.m("counts.distinct_pairs", int(idx.shape[0]))
    rec.m("counts.N_median_of_counted", float(np.median(N[has])))
    rec.m("counts.D_mean_of_counted", float(D[has].mean()))
    rec.m("check.counted_events_equal_fired_events", True)

    def make_rows(ptr, ids, gold_rows):
        rp, rowid, comp = csr_select(ptr, ids, has)
        return dict(n=int(ptr.shape[0] - 1), rp=rp, rowid=rowid, comp=comp, bkt=grp[comp].astype(np.int64), gold=np.asarray(gold_rows, np.int64))
    fit_rows = make_rows(fptr, fids, tok_tr[fit_pos])
    vr = np.arange(val_rows)
    vrp, vids_sub = subset_rows(vptr, vids, vr)
    val_rows_d = make_rows(vrp, vids_sub, gold[:val_rows])
    rec.m("rows.fit.counted_fired_per_row", float(fit_rows["comp"].shape[0] / max(1, fit_rows["n"])))
    rec.m("rows.validation.counted_fired_per_row", float(val_rows_d["comp"].shape[0] / max(1, val_rows_d["n"])))
    rec.m("rows.validation.frac_rows_without_counted_clause", float((np.diff(val_rows_d["rp"]) == 0).mean()))

    # ------------------------------------------------ loglin fit (GPU objective), CPU reference at the solution
    budget = budget_of(cfg, args, "loglin")
    locks.yield_gpu_if_requested()
    t1 = T1Gpu(V, indptr, idx, valc, lam, inv_n, p_uni)
    rec.log(f"loglin fit on {fit_rows['n']} rows ({fit_rows['comp'].shape[0]} pairs), budget {budget}")
    alpha, beta0, info = loglin_fit_gpu(t1, fit_rows, budget, rec, "loglin")
    ll, grad, gb, _, _, _ = cpu_loglin(fit_rows, alpha, beta0, indptr, idx, gval, logpu, True)
    used = np.array(info["used_groups"], np.int64)
    x = np.append(alpha[used], beta0)
    g_cpu = -np.append(grad[used], gb) / fit_rows["n"]
    lo = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][0])), float(budget["beta0_bounds"][0]))
    hi = np.append(np.full(used.shape[0], float(budget["alpha_bounds"][1])), float(budget["beta0_bounds"][1]))
    info["objective_cpu_float64"] = float(-ll / fit_rows["n"])
    info["projected_gradient_inf_norm_cpu_float64"] = projected_gradient_inf_norm(x, g_cpu, lo, hi)
    info["abs_diff_objective_gpu_vs_cpu"] = abs(info["objective_cpu_float64"] - info["objective_gpu_recomputed"])
    rec.m("fit.loglin", info)
    rec.log(f"loglin fit: converged {info['converged']} ({info['message']}), {info['iterations']} iterations, "
            f"objective {info['objective_cpu_float64']:.8f} nats/row, projected gradient {info['projected_gradient_inf_norm_cpu_float64']:.3g}")
    assert info["abs_diff_objective_gpu_vs_cpu"] < 1e-6, "GPU objective differs from the CPU reference"

    # ------------------------------------------------ validation: loglin (CPU reference; GPU cross-check)
    results = {}
    if args.skip_val:
        if not args.smoke:
            raise SystemExit("--skip-val is a smoke-only option")
        rec.log("--skip-val: no validation scoring (budget calibration smoke)")
    else:
        _, _, _, pg, rank, maxerr = cpu_loglin(val_rows_d, alpha, beta0, indptr, idx, gval, logpu, False)
        assert maxerr < 1e-9 and np.all(pg > 0), "a loglin probability row does not sum to one"
        vch = t1.prepare(val_rows_d["rp"], val_rows_d["comp"], val_rows_d["bkt"], val_rows_d["gold"])
        og = t1.run(vch, val_rows_d["n"], int(val_rows_d["comp"].shape[0]), alpha, beta0, False)
        gpu_diff = float(np.abs(og["lpg"] - np.log(pg)).max())
        rank_mism = int((og["rank"] != rank).sum())
        results["loglin"] = dict(top1=float((rank == 0).mean()), top5=float((rank < 5).mean()),
                                 bits_per_token=float(-np.log(pg).mean() / LN2), max_row_sum_error=float(maxerr),
                                 gpu_vs_cpu_max_abs_logprob_diff=gpu_diff, gpu_vs_cpu_rank_mismatch_rows=rank_mism)
        assert gpu_diff < 1e-6, f"GPU vs CPU validation log probabilities differ ({gpu_diff})"
        pg_loglin, rank_loglin = pg, rank
        # lin (secondary): the probe's normalised linear mixture, fitted on the same rows
        lb = budget_of(cfg, args, "lin")
        pk_fit = k_pair_pk(fit_rows["rp"], fit_rows["comp"], fit_rows["gold"], indptr, idx, valc, inv_n)
        w_group, linfo = fit_lin(fit_rows["rp"], pk_fit, fit_rows["bkt"], p_uni[fit_rows["gold"]], lb)
        rec.m("fit.lin", linfo)
        w, wuni = lin_weights(val_rows_d["rowid"], val_rows_d["bkt"], val_rows_d["n"], w_group)
        dummy_pos = np.arange(val_rows_d["n"], dtype=np.int64)
        pgl, rankl, maxerrl = k_dense_eval(val_rows_d["rp"], val_rows_d["comp"], w, np.zeros(w.shape[0]),
                                           np.full(w.shape[0], -1, np.int64), indptr, idx, valc, inv_n, wuni, p_uni,
                                           val_rows_d["gold"], dummy_pos, dummy_pos * 0, dummy_pos * 0,
                                           np.ones(1, np.uint8), 0.0, False)
        assert maxerrl < 1e-9 and np.all(pgl > 0), "a lin probability row does not sum to one"
        results["lin"] = dict(top1=float((rankl == 0).mean()), top5=float((rankl < 5).mean()),
                              bits_per_token=float(-np.log(pgl).mean() / LN2), max_row_sum_error=float(maxerrl))
        for name, row in results.items():
            for k, v in row.items():
                rec.m(f"rows.{name}.{k}", v)
            rec.log(f"{name:8s} top1 {row['top1']:.4f} top5 {row['top5']:.4f} bits/token {row['bits_per_token']:.4f}")
        # H8.1 verdict and the paired document bootstrap of the top-1 difference
        gain_min = float(cfg["threshold"]["masked_top1_gain_min"])
        n_eval = val_rows_d["n"]
        full = n_eval == targets.shape[0]
        delta = results["loglin"]["top1"] - signed_top1 if full else None
        rec.m("H8_1.top1_gain_loglin_minus_signed", delta, label="derived", threshold=gain_min)
        rec.m("H8_1.verdict", verdict_masked(signed_top1, results["loglin"]["top1"], gain_min) if full else "not judged (smoke subset)",
              label="derived", rule=f"loglin top-1 minus signed top-1 >= {gain_min} (docs/13 A3, H8.1, masked, per bank)")
        vds = val.doc_start.astype(np.int64)
        doc_t = np.searchsorted(vds, targets[:n_eval], side="right") - 1
        diff = (rank_loglin == 0).astype(np.float64) - signed_correct[:n_eval]
        bs = doc_block_bootstrap(np.bincount(doc_t, weights=diff, minlength=vds.shape[0] - 1),
                                 np.bincount(doc_t, minlength=vds.shape[0] - 1).astype(np.float64),
                                 int(cfg["bootstrap"]["resamples"]), seed)
        rec.m("H8_1.top1_gain_bootstrap", bs, label="derived", note="document-block bootstrap over validation documents (ratio of sums)")
        rec.log(f"H8.1 masked seed {seed}: signed {signed_top1:.4f}, loglin {results['loglin']['top1']:.4f}, gain {delta}, "
                f"verdict {rec.metrics['H8_1.verdict']['value']}, bootstrap {bs['ci95']}")
        od = out_dir(args)
        np.savez_compressed(od / "per_target.npz", targets=targets[:n_eval], gold=gold[:n_eval], signed_correct=signed_correct[:n_eval],
                            loglin_gold_prob=pg_loglin, loglin_rank=rank_loglin, lin_gold_prob=pgl, lin_rank=rankl)
        (od / "rows.json").write_text(json.dumps({"signed": {"top1": signed_top1, "top5": top5 / targets.shape[0]}, **results}, indent=1) + "\n")
    od = out_dir(args)
    (od / "readout_params.json").write_text(json.dumps({"alpha_by_group": {int(g): float(alpha[g]) for g in np.flatnonzero(alpha)},
                                                         "beta0": beta0, "group_rule": "min(n_inc,12)*24 + min(floor(log2 N),23)"}, indent=1) + "\n")
    rec.memory("end", cp)
    rec.m("memory.peak_rss_gib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB)
    rec.m("memory.gpu_pool_peak_seen_gib", rec.gpu_pool_peak / GIB, note="CuPy pool total at the logged checkpoints")
    rec.m("wall_s_total", time.time() - rec.t0)
    rec.flush()
    return 0


# ------------------------------------------------------------------ causal
def run_causal(cfg: dict, args, rec: Recorder) -> int:
    import cupy as cp
    from tmlm.common import locks
    from tmlm.data.corpus_store import load_split, raw_dir, split_dir
    from tmlm.decode.bank import class_bias_from_counts
    from tmlm.engine.checkpoint import load_checkpoint
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import contiguous, split_meta

    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(cfg["compute"]["gpu_pool_limit_gib"]) * GIB))
    rec.m("setup.config", cfg)
    ckpt = ROOT / cfg["bank"]["checkpoint"]
    ecfg, st, meta = load_checkpoint(ckpt)
    cpm = json.loads((ROOT / cfg["bank"]["metrics"]).read_text())
    assert meta["state_hash"] == cpm["epoch4.checkpoint"]["value"]["state_hash"], "checkpoint hash differs from the logged epoch-4 hash"
    assert ecfg.mode == "causal"
    V, C = int(ecfg.vocab), int(ecfg.n_clauses)
    rec.m("bank.state_hash", meta["state_hash"], source=cfg["bank"]["metrics"])
    n_inc = included_literals(st)
    rec.m("bank.alive_clauses", int(st["alive"].astype(bool).sum()))

    n_tokens = int(cfg["data"]["train_prefix_tokens"])
    if args.prefix_tokens:
        if not args.smoke:
            raise SystemExit("--prefix-tokens is a smoke-only override")
        n_tokens = int(args.prefix_tokens)
    full = load_split("wikitext103", V, "train")
    train_counts_full = np.zeros(V, np.int64)
    for a in range(0, int(full.tok.shape[0]), 1 << 24):  # chunked: no 1 GB int64 copy of the whole split
        train_counts_full += np.bincount(np.asarray(full.tok[a:a + (1 << 24)]).astype(np.int64), minlength=V)
    tr = contiguous(slice_train(full, n_tokens))
    del full
    val = contiguous(load_split("wikitext103", V, "validation"))
    vmeta, _ = split_meta(V, "validation")
    n_bytes = int(vmeta["n_bytes_utf8"])
    for split in ("train", "validation"):
        m_ = json.loads((split_dir("wikitext103", V, split) / "meta.json").read_text())
        rec.dataset(f"{split}_tok_V{V}", m_["sha256"]["tok.npy"])
    raw_meta = json.loads((raw_dir("wikitext103", "validation") / "meta.json").read_text())
    rec.dataset("validation_text_utf8", raw_meta["sha256"]["text.utf8"])
    doc_bytes_off = np.load(raw_dir("wikitext103", "validation") / "doc_bytes.npy").astype(np.int64)
    doc_bytes = np.diff(doc_bytes_off)
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    tok_va = np.asarray(val.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    ds_va = np.asarray(val.doc_start).astype(np.int64)
    assert doc_bytes.shape[0] == ds_va.shape[0] - 1 and int(doc_bytes.sum()) == n_bytes, "per-document bytes do not match the split"
    n_tr, n_va_all = int(tok_tr.shape[0]), int(tok_va.shape[0])
    n_va = n_va_all if not args.val_max_rows else min(int(args.val_max_rows), n_va_all)
    if args.val_max_rows and not args.smoke:
        raise SystemExit("--val-max-rows is a smoke-only override")
    n_doc = ds_tr.shape[0] - 1
    rec.m("data.prefix_tokens", n_tr, note="first tokens of WikiText-103 train at V 8000, cut at a document boundary (m3_tiny_lm.slice_train)")
    rec.m("data.prefix_docs", n_doc)
    rec.m("data.validation_tokens", n_va_all)
    rec.m("data.validation_bytes_utf8", n_bytes)
    rec.log(f"prefix {n_tr} tokens {n_doc} docs; validation {n_va_all} tokens {n_bytes} bytes")

    fit_doc = fit_document_mask(n_doc)
    doc_of = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_fit = fit_doc[doc_of]
    fit_pos = np.flatnonzero(is_fit).astype(np.int64)
    n_cnt = int(n_tr - fit_pos.shape[0])
    rec.m("split.fit_documents", np.flatnonzero(fit_doc).tolist(), rule="splitmix64(i XOR 0xC0FFEE) mod 100 == 0 (docs/11 11.6)")
    rec.m("split.fit_positions", int(fit_pos.shape[0]))
    rec.m("split.counting_positions", n_cnt)

    # ------------------------------------------------ validation forward pass: fired lists and the signed-vote check
    locks.yield_gpu_if_requested()
    cal = cpm["epoch4.calibration"]["value"]
    a_q16, b_q16 = int(cal["a_q16"]), int(cal["b_q16"])
    bias = class_bias_from_counts(train_counts_full, V)
    d_bias = cp.asarray(bias.astype(np.int64))
    d_vtok = cp.asarray(tok_va.astype(np.int64))
    acc = {"raw_hits": 0, "cal_hits": 0, "cal_top5": 0, "worst_norm": 0.0}
    signed_lp = np.zeros(n_va)
    t0 = time.time()
    eng_v = CudaEngine(ecfg, val)
    eng_v.load_state(st)
    vchunk = int(cfg["compute"]["validation_chunk"])
    ptr_va = np.zeros(n_va + 1, np.int64)
    ids_parts = []
    for lo in range(0, n_va, vchunk):
        hi = min(n_va, lo + vchunk)
        eng_v._sync_state_struct()
        fw = eng_v._forward(EPOCH, cp.arange(lo, hi, dtype=cp.uint32), None, 0)
        eng_v._marks = []
        fe, fj = fw.fired_ex, fw.fired_j
        if fe.size:
            order = cp.argsort((fe.astype(cp.uint64) << cp.uint64(32)) | fj.astype(cp.uint64))
            fe, fj = fe[order], fj[order]
            cnt = cp.bincount(fe.astype(cp.int64), minlength=hi - lo).get()
        else:
            cnt = np.zeros(hi - lo, np.int64)
        ptr_va[lo + 1:hi + 1] = ptr_va[lo] + np.cumsum(cnt)
        ids_parts.append(fj.get().astype(np.uint32))
        scores = fw.scores
        g = d_vtok[lo:hi]
        rows = cp.arange(hi - lo)
        acc["raw_hits"] += int((cp.argmax(scores, axis=1) == g).sum())
        num = scores.astype(cp.int64) * (a_q16 * 256) + (b_q16 * d_bias)[None, :]
        acc["cal_hits"] += int((cp.argmax(num, axis=1) == g).sum())
        ng = num[rows, g][:, None]
        rank = (num > ng).sum(axis=1) + ((num == ng) & (cp.arange(V)[None, :] < g[:, None])).sum(axis=1)
        acc["cal_top5"] += int((rank < 5).sum())
        lg = num.astype(cp.float64) / float(1 << 24)
        mx = lg.max(axis=1, keepdims=True)
        zz = lg - mx
        lp = zz - cp.log(cp.exp(zz).sum(axis=1, keepdims=True))
        acc["worst_norm"] = max(acc["worst_norm"], float(cp.abs(cp.log(cp.exp(lp).sum(axis=1))).max()))
        signed_lp[lo:hi] = lp[rows, g].get()
        del fw, scores, num, lg, zz, lp
    ids_va = np.concatenate(ids_parts)
    del ids_parts
    signed_bits = -signed_lp / LN2
    raw_top1 = acc["raw_hits"] / n_va
    logged_raw = float(cpm["epoch4.validation.causal_top1_raw_scores"]["value"])
    logged_bpb = float(cpm["epoch4.validation.bits_per_byte"]["value"])
    signed_bpb = float(signed_bits.sum() / n_bytes) if n_va == n_va_all else None
    rs = np.random.default_rng(12345)
    chk = np.sort(rs.choice(n_va, size=min(2000, n_va), replace=False))
    s_api = eng_v.predict_scores(EPOCH, chk.astype(np.uint32))
    rebuilt = k_vote_scores(ptr_va, ids_va, chk.astype(np.int64), st["vote_cls"], st["vote_w"].astype(np.int32), V)
    mism = int((rebuilt != s_api).any(axis=1).sum())
    for i, p in enumerate(chk[:100].tolist()):
        assert np.array_equal(eng_v.fired_clauses(EPOCH, p), ids_va[ptr_va[p]:ptr_va[p + 1]]), "validation fired list differs from the API"
        if i < 3:
            assert np.array_equal(np.sort(eng_v.fired_clauses_bruteforce(EPOCH, p)), ids_va[ptr_va[p]:ptr_va[p + 1]]), \
                "validation fired list differs from the brute force"
    del eng_v, d_bias, d_vtok
    pool.free_all_blocks()
    assert acc["worst_norm"] < 1e-6
    passed = bool(raw_top1 == logged_raw and mism == 0 and (signed_bpb is None or abs(signed_bpb - logged_bpb) < 1e-9))
    rec.m("check.signed_raw_top1_reproduced", raw_top1, expected=logged_raw, source=cfg["bank"]["metrics"])
    rec.m("check.signed_bpb_reproduced", signed_bpb, expected=logged_bpb, source=cfg["bank"]["metrics"],
          note="logged epoch-4 calibration (a_q16, b_q16), total bits / UTF-8 bytes of the validation split")
    rec.m("check.rebuilt_vote_sums_mismatch_rows", mism, note="2000 random validation positions vs predict_scores")
    rec.m("check.passed_signed_votes", passed if n_va == n_va_all else None)
    rec.m("rows.signed.bits_per_byte", signed_bpb)
    rec.m("rows.signed.bits_per_token", float(signed_bits.mean()))
    rec.m("rows.signed.top1_raw", raw_top1)
    rec.m("rows.signed.top1_calibrated", acc["cal_hits"] / n_va)
    rec.m("rows.signed.top5_calibrated", acc["cal_top5"] / n_va)
    rec.m("throughput.validation_forward_pos_per_s", n_va / (time.time() - t0))
    rec.log(f"pipeline check: raw top-1 {raw_top1:.6f} (logged {logged_raw:.6f}), bpb {signed_bpb} (logged {logged_bpb:.6f}), rebuilt mismatches {mism}")
    rec.flush()
    if (n_va == n_va_all and not passed) or mism != 0:
        rec.log("PIPELINE CHECK FAILED: stop")
        return 3

    # ------------------------------------------------ prefix pass: streamed exact counts + fit-row fired lists
    locks.yield_gpu_if_requested()
    t0 = time.time()
    eng = CudaEngine(ecfg, tr)
    eng.load_state(st)
    del st  # every later step uses n_inc and the engines only (host memory)
    tok_dev = cp.asarray(tok_tr.astype(np.uint64))
    isfit_dev = cp.asarray(is_fit.astype(np.uint8))
    cap = int(cfg["compute"]["count_buffer_events"])
    buf = cp.empty(cap, dtype=cp.uint64)
    nbuf = 0
    accum = CompactCountAccumulator(C, V, range_bits=int(cfg["compute"]["count_range_bits"]))
    fit_j_parts, fit_n_parts = [], []
    chk_tr = sample_positions(np.arange(n_tr), 100, 5)
    got_tr: dict[int, np.ndarray] = {}
    n_events = 0
    flush_s = [0.0]
    n_flush = [0]
    span = np.uint64(accum.span)

    def flush_buffer(n: int) -> None:
        """Sort and run-length encode the buffered keys on the GPU, then merge them per clause range on the
        host (one range downloaded at a time, as uint32 local keys: bounded host memory)."""
        t = time.time()
        k = buf[:n]
        k.sort()
        brk = cp.flatnonzero(k[1:] != k[:-1]) + 1
        starts = cp.concatenate([cp.zeros(1, dtype=brk.dtype), brk])
        uk = k[starts]
        cnt = cp.diff(cp.concatenate([starts, cp.asarray([n], dtype=starts.dtype)])).astype(cp.int32)
        del brk, starts
        bounds = cp.searchsorted(uk, cp.arange(accum.n_ranges + 1, dtype=cp.uint64) * cp.uint64(span)).get()
        for r in range(accum.n_ranges):
            a, b = int(bounds[r]), int(bounds[r + 1])
            if a < b:
                accum.add_range(r, (uk[a:b] - cp.uint64(r * int(span))).astype(cp.uint32).get(), cnt[a:b].get())
        accum.batches += 1
        n_flush[0] += 1
        del uk, cnt
        flush_s[0] += time.time() - t

    chunk = int(cfg["compute"]["forward_chunk"])
    ci = 0
    for lo in range(0, n_tr, chunk):
        hi = min(n_tr, lo + chunk)
        eng._sync_state_struct()
        fw = eng._forward(EPOCH, cp.arange(lo, hi, dtype=cp.uint32), None, 0)
        eng._marks = []
        fe, fj = fw.fired_ex, fw.fired_j
        del fw
        p = fe.astype(cp.int64) + lo
        isf = isfit_dev[p] != 0
        keep = ~isf
        key = fj[keep].astype(cp.uint64) * cp.uint64(V) + tok_dev[p[keep]]
        m = int(key.size)
        if nbuf + m > cap:
            flush_buffer(nbuf)
            nbuf = 0
        if m > cap:
            raise RuntimeError("count buffer smaller than one forward chunk")
        buf[nbuf:nbuf + m] = key
        nbuf += m
        n_events += m
        del key, keep
        if bool(isf.any()):
            pe, je = p[isf], fj[isf]
            order = cp.argsort((pe.astype(cp.uint64) << cp.uint64(32)) | je.astype(cp.uint64))
            fit_j_parts.append(je[order].get().astype(np.uint32))
            per_pos = cp.bincount(pe - lo, minlength=hi - lo).get()
            fit_n_parts.append(per_pos[is_fit[lo:hi]])
            del pe, je, order
        a, b = np.searchsorted(chk_tr, [lo, hi])
        if b > a:
            feh = fe.get()
            fjh = fj.get()
            for r in chk_tr[a:b].tolist():
                s0, s1 = np.searchsorted(feh, [r - lo, r - lo + 1])
                got_tr[r] = np.sort(fjh[s0:s1]).astype(np.uint32)
        del fe, fj, p, isf
        ci += 1
        if ci % 100 == 0:
            el = time.time() - t0
            rec.log(f"  prefix pass: {hi}/{n_tr} positions, {hi / el:.0f} pos/s, counted events {n_events}, "
                    f"distinct pairs so far {accum.n_pairs}, flush {flush_s[0]:.0f} s")
    if nbuf:
        flush_buffer(nbuf)
        nbuf = 0
    eng.synchronize()
    wall_prefix = time.time() - t0
    rec.m("throughput.prefix_pass_pos_per_s", n_tr / wall_prefix, note="CUDA forward + streamed counting, shared GPU")
    rec.m("throughput.count_flush_s", flush_s[0])
    rec.memory("after the prefix pass", cp)
    bad = 0
    for i, r in enumerate(chk_tr.tolist()):
        bad += int(not np.array_equal(eng.fired_clauses(EPOCH, r), got_tr[r]))
        if i < 3:
            bad += int(not np.array_equal(np.sort(eng.fired_clauses_bruteforce(EPOCH, r)), got_tr[r]))
    rec.m("check.prefix_fired_list_mismatches", bad, note="100 random prefix positions vs fired_clauses, 3 vs the brute force")
    del eng, tok_dev, isfit_dev, buf
    pool.free_all_blocks()
    if bad:
        rec.flush()
        rec.log("PIPELINE CHECK FAILED: stop")
        return 3
    assert accum.events == n_events
    t0 = time.time()
    n_pairs, n_ovf, host_table_gib = accum.n_pairs, accum.n_overflow, accum.nbytes() / GIB
    rec.memory("before the count table moves to the GPU", cp)
    # the count table goes to the GPU range by range; the host keeps only indptr (lookups and the CPU cross-check read the device copy)
    locks.yield_gpu_if_requested()
    indptr, d_idx, d_val, N = accum.finalize_to_device(cp)
    del accum
    D = np.diff(indptr)
    assert int(N.sum()) == n_events, "counted events differ from the streamed events"
    assert int(indptr[-1]) == n_pairs
    has = N > 0
    rec.m("counts.events", n_events, note="(clause, next token) pairs over the counting positions")
    rec.m("counts.distinct_pairs", n_pairs)
    rec.m("counts.pairs_with_count_ge_255", n_ovf, note="kept exactly in the overflow tables of the compact accumulator")
    rec.m("memory.host_count_table_gib", host_table_gib, note="compact accumulator at the end of the prefix pass")
    rec.m("counts.clauses_with_counts", int(has.sum()))
    rec.m("counts.N_median_of_counted", float(np.median(N[has])))
    rec.m("counts.D_mean_of_counted", float(D[has].mean()))
    rec.m("counts.flush_batches", n_flush[0])
    rec.m("check.counted_events_equal_fired_events", True)
    rec.m("throughput.finalize_s", time.time() - t0)
    rec.log(f"counts: {n_events} events, {n_pairs} distinct pairs, {int(has.sum())} clauses, table {host_table_gib:.2f} GiB on the host before the move")
    lam, inv_n = witten_bell(N, D)
    tgt_cnt = tok_tr[~is_fit]
    p_uni = add_one_unigram(tgt_cnt, V)
    logpu = np.log(p_uni)
    grp = clause_groups(n_inc, N)
    fit_j = np.concatenate(fit_j_parts) if fit_j_parts else np.zeros(0, np.uint32)
    fit_n = np.concatenate(fit_n_parts) if fit_n_parts else np.zeros(0, np.int64)
    del fit_j_parts, fit_n_parts
    assert fit_n.shape[0] == fit_pos.shape[0] and int(fit_n.sum()) == fit_j.shape[0]
    fptr = np.zeros(fit_pos.shape[0] + 1, np.int64)
    fptr[1:] = np.cumsum(fit_n)
    t1 = T1Gpu(V, indptr, None, None, lam, inv_n, p_uni, device_csr=(to_device_chunked(cp, indptr, np.int64), d_idx, d_val))
    del d_idx, d_val
    rec.memory("count table on the GPU", cp)

    # ------------------------------------------------ rows: fit / check subsets of the fit documents, validation
    fcfg = cfg["fit"]
    n_fit = min(int(args.fit_rows or fcfg["fit_rows"]), fit_pos.shape[0])
    if args.fit_rows and not args.smoke:
        raise SystemExit("--fit-rows is a smoke-only override")
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 813]))
    perm = rng.permutation(fit_pos.shape[0])
    fit_sel = np.sort(perm[:n_fit])
    chk_sel = np.sort(perm[n_fit:n_fit + int(fcfg["check_rows"])])

    def make_rows(ptr, ids, gold_rows):
        """Rows restricted to clauses with counts; int32 pair arrays (80M validation pairs)."""
        n = int(ptr.shape[0] - 1)
        keep = has[ids]
        rowid = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))[keep]
        comp = ids[keep].astype(np.int32)
        return dict(n=n, rp=rowptr_of(rowid, n), rowid=rowid, comp=comp, bkt=grp[comp].astype(np.int32),
                    gold=np.asarray(gold_rows, np.int64))
    rp_f, ids_f = subset_rows(fptr, fit_j, fit_sel)
    fit_rows = make_rows(rp_f, ids_f, tok_tr[fit_pos[fit_sel]])
    rp_c, ids_c = subset_rows(fptr, fit_j, chk_sel)
    chk_rows = make_rows(rp_c, ids_c, tok_tr[fit_pos[chk_sel]])
    del fit_j, rp_f, ids_f, rp_c, ids_c
    val_rows = make_rows(ptr_va, ids_va, tok_va[:n_va])
    del ids_va
    rec.m("rows.fit.n", fit_rows["n"])
    rec.m("rows.check.n", chk_rows["n"])
    rec.m("rows.fit.counted_fired_per_row", float(fit_rows["comp"].shape[0] / max(1, fit_rows["n"])))
    rec.m("rows.validation.counted_fired_per_row", float(val_rows["comp"].shape[0] / max(1, val_rows["n"])))
    rec.m("rows.validation.frac_rows_without_counted_clause", float((np.diff(val_rows["rp"]) == 0).mean()))

    # ------------------------------------------------ loglin fit and validation
    locks.yield_gpu_if_requested()
    budget = budget_of(cfg, args, "loglin")
    rec.log(f"loglin fit on {fit_rows['n']} rows ({fit_rows['comp'].shape[0]} pairs), budget {budget}")
    alpha, beta0, info = loglin_fit_gpu(t1, fit_rows, budget, rec, "loglin")
    used = np.array(info["used_groups"], np.int64)
    info["check_rows"] = int(chk_rows["n"])
    info["check_rows_bits_per_token"] = None
    if chk_rows["n"]:
        cch = t1.prepare(chk_rows["rp"], chk_rows["comp"], chk_rows["bkt"], chk_rows["gold"])
        oc = t1.run(cch, chk_rows["n"], int(chk_rows["comp"].shape[0]), alpha, beta0, False)
        info["check_rows_bits_per_token"] = float(-oc["lpg"].mean() / LN2)
        del cch
    rec.log(f"loglin fit: converged {info['converged']} ({info['message']}), {info['iterations']} iterations, "
            f"objective {info['objective_nats_per_row']:.8f} nats/row, projected gradient {info['projected_gradient_inf_norm']:.3g}, "
            f"check rows {info['check_rows_bits_per_token']} bits/token")
    # CPU float64 cross-check on sampled validation and fit rows (sub-CSR of the clauses they use)

    def cpu_crosscheck(rows: dict, gpu_lpg: np.ndarray, k: int, seed_: int, batch: int = 20) -> float:
        """float64 CPU reference (k_loglin) on k sampled rows, in batches of `batch` rows, each with a
        sub-table of the clauses it uses gathered from the device copy (bounded host memory)."""
        pick_all = sample_positions(np.arange(rows["n"]), k, seed_)
        worst = 0.0
        for b0 in range(0, pick_all.shape[0], batch):
            pick = pick_all[b0:b0 + batch]
            rp_s, comp_s = subset_rows(rows["rp"], rows["comp"], pick)
            _, bkt_s = subset_rows(rows["rp"], rows["bkt"], pick)
            uc, inv = np.unique(comp_s, return_inverse=True)
            sub_ptr = np.zeros(uc.shape[0] + 1, np.int64)
            sub_ptr[1:] = np.cumsum(indptr[uc + 1] - indptr[uc])
            take = np.concatenate([np.arange(indptr[c], indptr[c + 1]) for c in uc.tolist()]) if uc.shape[0] else np.zeros(0, np.int64)
            d_take = cp.asarray(take)
            sub_idx, sub_val = t1.d_idx[d_take].get().astype(np.int64), t1.d_val[d_take].get()
            del d_take, take
            sub_g = k_gval(sub_ptr, sub_idx, sub_val, lam[uc], inv_n[uc], p_uni)
            sub_rows = dict(rp=rp_s, comp=inv.astype(np.int64).reshape(-1), bkt=bkt_s, gold=rows["gold"][pick])
            _, _, _, pgx, _, errx = cpu_loglin(sub_rows, alpha, beta0, sub_ptr, sub_idx, sub_g, logpu, False)
            assert errx < 1e-9
            worst = max(worst, float(np.abs(np.log(pgx) - gpu_lpg[pick]).max()))
        return worst
    ofit = t1.run(t1.prepare(fit_rows["rp"], fit_rows["comp"], fit_rows["bkt"], fit_rows["gold"]), fit_rows["n"],
                  int(fit_rows["comp"].shape[0]), alpha, beta0, False)
    info["cpu_vs_gpu_max_abs_logprob_diff_fit_rows"] = cpu_crosscheck(fit_rows, ofit["lpg"], int(cfg["checks"]["cpu_crosscheck_rows"]), 6)
    del ofit
    vch = t1.prepare(val_rows["rp"], val_rows["comp"], val_rows["bkt"], val_rows["gold"])
    ov = t1.run(vch, val_rows["n"], int(val_rows["comp"].shape[0]), alpha, beta0, False)
    del vch
    pool.free_all_blocks()
    maxerr = float(np.abs(ov["sum"] - 1.0).max())
    assert maxerr < 1e-9 and np.all(np.isfinite(ov["lpg"])), "a loglin probability row does not sum to one"
    info["cpu_vs_gpu_max_abs_logprob_diff_validation_rows"] = cpu_crosscheck(val_rows, ov["lpg"], int(cfg["checks"]["cpu_crosscheck_rows"]), 7)
    assert info["cpu_vs_gpu_max_abs_logprob_diff_validation_rows"] < 1e-6 and info["cpu_vs_gpu_max_abs_logprob_diff_fit_rows"] < 1e-6
    rec.m("fit.loglin", info)
    loglin_bits = -ov["lpg"] / LN2
    results = {"loglin": dict(bits_per_byte=float(loglin_bits.sum() / n_bytes) if n_va == n_va_all else None,
                              bits_per_token=float(loglin_bits.mean()), top1=float((ov["rank"] == 0).mean()),
                              top5=float((ov["rank"] < 5).mean()), max_row_sum_error=maxerr)}

    # ------------------------------------------------ lin (secondary): fit on the same rows, dense rows through the probe's GPU mixer
    from ckm_posthoc_probe_causal import Gpu
    lb = budget_of(cfg, args, "lin")
    gpu = Gpu(V)
    gpu.csr["full"] = dict(indptr=t1.d_indptr, idx=t1.d_idx, val=t1.d_val, evd=None)

    def gold_pk(rows: dict) -> np.ndarray:
        """O_j,gold / N_j per pair (exact counts looked up on the device copy of the table)."""
        cnt = gpu.pair_count("full", rows["comp"], rows["gold"][rows["rowid"]])
        return cnt * inv_n[rows["comp"]]
    pk_fit = gold_pk(fit_rows)
    w_group, linfo = fit_lin(fit_rows["rp"], pk_fit, fit_rows["bkt"], p_uni[fit_rows["gold"]], lb)
    del pk_fit
    # validation rows in blocks (bounded host memory): gold probability in float64 on the CPU, dense rows (top-1, top-5, row sums) on the GPU
    n_v = val_rows["n"]
    p_lin = np.empty(n_v)
    olin = {"rank": np.empty(n_v, np.int64)}
    lin_err = 0.0
    lin_diff = 0.0
    for r0 in range(0, n_v, 32768):
        r1 = min(n_v, r0 + 32768)
        a, b = int(val_rows["rp"][r0]), int(val_rows["rp"][r1])
        rid = val_rows["rowid"][a:b].astype(np.int64) - r0
        comp_c, bkt_c, gold_c = val_rows["comp"][a:b], val_rows["bkt"][a:b], val_rows["gold"][r0:r1]
        w_c, wuni_c = lin_weights(rid, bkt_c, r1 - r0, w_group)
        pk_c = gpu.pair_count("full", comp_c, gold_c[rid]) * inv_n[comp_c]
        p_c = np.bincount(rid, weights=w_c * pk_c, minlength=r1 - r0) + wuni_c * p_uni[gold_c]
        o = gpu.mix("full", val_rows["rp"][r0:r1 + 1] - a, comp_c, w_c * inv_n[comp_c], np.broadcast_to(np.int64(-1), (b - a,)),
                    np.broadcast_to(0.0, (b - a,)), wuni_c, p_uni, 0, gold_c)
        p_lin[r0:r1] = p_c
        olin["rank"][r0:r1] = o["rank"]
        lin_err = max(lin_err, float(np.abs(o["sum"] - 1.0).max()))
        lin_diff = max(lin_diff, float(np.abs(o["pg"] - p_c).max()))
        del w_c, pk_c, o
    assert np.all(p_lin > 0)
    lp_lin = np.log(p_lin)
    assert lin_err < 1e-9 and lin_diff < 1e-9, f"lin rows: sum error {lin_err}, GPU vs CPU {lin_diff}"
    rec.m("fit.lin", linfo)
    lin_bits = -lp_lin / LN2
    results["lin"] = dict(bits_per_byte=float(lin_bits.sum() / n_bytes) if n_va == n_va_all else None, bits_per_token=float(lin_bits.mean()),
                          top1=float((olin["rank"] == 0).mean()), top5=float((olin["rank"] < 5).mean()), max_row_sum_error=lin_err,
                          gpu_vs_cpu_max_abs_gold_prob_diff=lin_diff)
    del gpu, t1
    pool.free_all_blocks()
    for name, row in results.items():
        for k, v in row.items():
            rec.m(f"rows.{name}.{k}", v)
        rec.log(f"{name:8s} bpb {row['bits_per_byte']} bits/token {row['bits_per_token']:.4f} top1 {row['top1']:.4f} top5 {row['top5']:.4f}")

    # ------------------------------------------------ H8.1 verdict and the paired document bootstrap (bits per byte)
    drop_min = float(cfg["threshold"]["causal_bpb_drop_min"])
    if n_va == n_va_all:
        delta = results["loglin"]["bits_per_byte"] - signed_bpb
        rec.m("H8_1.bpb_loglin_minus_signed", delta, label="derived", threshold=-drop_min)
        rec.m("H8_1.verdict", verdict_causal(signed_bpb, results["loglin"]["bits_per_byte"], drop_min), label="derived",
              rule=f"loglin validation bits per byte at least {drop_min} lower than the signed votes (docs/13 A3, H8.1, causal)")
        doc_v = np.searchsorted(ds_va, np.arange(n_va), side="right") - 1
        bs = doc_block_bootstrap(np.bincount(doc_v, weights=loglin_bits - signed_bits, minlength=ds_va.shape[0] - 1),
                                 doc_bytes.astype(np.float64), int(cfg["bootstrap"]["resamples"]), 1)
        rec.m("H8_1.bpb_difference_bootstrap", bs, label="derived",
              note="document-block bootstrap over validation documents: sum of bit differences / sum of UTF-8 bytes of the resampled documents")
        rec.log(f"H8.1 causal: signed {signed_bpb:.4f} bpb, loglin {results['loglin']['bits_per_byte']:.4f} bpb, difference {delta:.4f}, "
                f"verdict {rec.metrics['H8_1.verdict']['value']}, bootstrap {bs['ci95']}")
    else:
        rec.m("H8_1.verdict", "not judged (smoke subset)", label="derived")
    od = out_dir(args)
    np.savez_compressed(od / "per_token_bits.npz", signed=signed_bits.astype(np.float32), loglin=loglin_bits.astype(np.float32),
                        lin=lin_bits.astype(np.float32), loglin_rank=ov["rank"].astype(np.int32), lin_rank=olin["rank"].astype(np.int32))
    (od / "rows.json").write_text(json.dumps({"signed": {"bits_per_byte": signed_bpb, "top1_raw": raw_top1}, **results}, indent=1) + "\n")
    (od / "readout_params.json").write_text(json.dumps({"alpha_by_group": {int(g): float(alpha[g]) for g in np.flatnonzero(alpha)},
                                                         "beta0": beta0, "lin_log_weight_by_group": {int(g): float(np.log(w_group[g])) for g in np.flatnonzero(w_group)},
                                                         "group_rule": "min(n_inc,12)*24 + min(floor(log2 N),23)"}, indent=1) + "\n")
    rec.m("memory.peak_rss_gib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB)
    rec.m("memory.gpu_pool_peak_seen_gib", rec.gpu_pool_peak / GIB, note="CuPy pool total at the logged checkpoints")
    rec.m("wall_s_total", time.time() - rec.t0)
    rec.flush()
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--bank-seed", type=int, default=0, help="masked: which layer-1 bank (2 to 5 reported; 1 smoke only)")
    ap.add_argument("--smoke", action="store_true", help="label everything smoke; allows the overrides below")
    ap.add_argument("--max-count-pos", type=int, default=0, help="smoke, masked: counting positions")
    ap.add_argument("--prefix-tokens", type=int, default=0, help="smoke, causal: training prefix")
    ap.add_argument("--val-max-rows", type=int, default=0, help="smoke: first validation rows only")
    ap.add_argument("--fit-rows", type=int, default=0, help="smoke, causal: fit rows")
    ap.add_argument("--maxiter", type=int, default=0, help="smoke: L-BFGS-B iterations")
    ap.add_argument("--skip-val", action="store_true", help="smoke: fit only, no validation scoring")
    ap.add_argument("--out-dir", default="", help="output directory when not run under run_logged.py")
    args = ap.parse_args()
    cfg = load_json(args.config)
    rec = Recorder(args.smoke)
    rec.scratch = args.out_dir or None
    rec.m("mode_smoke", bool(args.smoke))
    rec.m("argv", sys.argv[1:])
    if cfg["mode"] == "masked":
        if not args.bank_seed:
            raise SystemExit("masked mode needs --bank-seed")
        return run_masked(cfg, args, rec)
    if cfg["mode"] == "causal":
        return run_causal(cfg, args, rec)
    raise SystemExit(f"unknown mode {cfg['mode']!r}")


if __name__ == "__main__":
    raise SystemExit(main())
