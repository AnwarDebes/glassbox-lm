"""Early full-data estimate of the CVTM-core (docs/11 section 11), BEFORE the frozen implementation
exists. EXPLORATORY: not pre-registered, tunes nothing, tests no docs/13 threshold. Validation split
only; the test split is never loaded. CPU only, one thread.

Why: to learn early whether the approach will be impressive. docs/13 A3 (H8.3 to H8.8)
pre-registers the CVTM-core; its confirmatory numbers will come from the frozen implementation in
`src/tmlm/cvtm/` (suffix clauses, `mixture.py`), NOT from this script. This script estimates the same
fixed estimator with minimal code, so a large gap to the prediction (about 1.25 to 1.28 bits per byte,
unverified) is known before the frozen implementation is finished.

Protocol (docs/11 sections 11.2 to 11.6; the estimator is fixed by the spec, nothing is tuned here):
1. Fit documents H = training documents i (0-based index in the train store) with
   splitmix64(i XOR 0xC0FFEE) mod 100 == 0. splitmix64(x) is the first output of the SplitMix64
   generator with state x: z = x + 0x9E3779B97F4A7C15, then the 64-bit finaliser of Steele, Lea and
   Flood's SplittableRandom (Stafford's variant 13): z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
   z = (z ^ (z >> 27)) * 0x94D049BB133111EB; z ^ (z >> 31), all mod 2^64. This is `mix64` of docs/11
   section 3 and `tmlm.engine.rng.mix64` (checked at start). The count of H under the reading without
   the golden-gamma increment is also logged, because "finaliser" alone could be read that way.
2. KN-5 = `KNModel(5, 8000, "modified", pad=True)` of src/tmlm/eval/baselines/ngram.py.
   (a) fitted on the training split WITHOUT H: P_KN and the longest fired suffix order k* on H (for
   the fit of the combination) and on validation (reported as a check);
   (b) fitted on ALL training documents: P_KN and k* on validation; its validation bits per byte must
   equal the logged comparator 1.3501314 (results/20260922T180823Z_M2_baselines_ngram_v8000/).
   The two fits run one after the other; the first is freed before the second starts.
3. Document clauses (docs/11 section 11.3): `tmlm.cvtm.doc_clauses.doc_features` (sort based, exact
   keys, orders 0..4, whole-document window, pointer 5..64), cross-checked against the streaming
   implementation `doc_features_stream` on H and on validation (identical outputs required).
4. Combination (docs/11 section 11.4): P(w) = sum over firing components of lambda_j(b) P_j(w), the
   weights of non-firing components dropped and the rest renormalised; bucket b = (k*, m*, c, l);
   MAP-EM, 50 iterations from uniform, float64, fixed order, Dirichlet prior of 16 pseudo-counts
   toward the pooled weights. The spec does not define "pooled weights" further; THIS SCRIPT's
   definition: pi_j = (sum over all fit rows of the responsibility of j) / (number of fit rows where j
   fires), recomputed every iteration, restricted to the firing set of the bucket and renormalised;
   M step lambda_bj = (R_bj + 16 pi_bj) / (n_b + 16). The frozen `mixture.py` may define it
   differently; the MLE variant (prior 0) is logged as a sensitivity number, not as a row.
   Fitted on H with the KN-without-H base, applied on validation with the KN-on-ALL base.
5. Rows (validation bits per byte, 282,587 tokens, 1,145,547 UTF-8 bytes, protocol CP byte accounting):
   kn5 (all), kn5+D0 (document unigram cache), kn5+D0..D4 (document n-gram clauses),
   kn5+D0..D4+P (pointer) = CVTM-core. Each row is the same estimator restricted to its components
   (bucket m* over the included orders, l = none without the pointer). Causal top-1 of every row from
   the full distribution (KN-5 dist plus the exact document successor counts plus the pointer).
6. Document-block bootstrap (60 validation documents, 2,000 resamples, one shared index matrix):
   95 percent CIs of every row minus kn5 and of CVTM-core minus the 600 s transformer's 1.2341 (a
   constant, results/20260922T172902Z_M2_baselines_transformer/, transformer_L6_W256): per resample,
   (sum of bits over the drawn documents) / (sum of their UTF-8 bytes), bytes from
   data/wikitext103/raw/validation/doc_bytes.npy.

Launch (single thread, nice 19; the s4 job holds the CPU slots):

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMLM_CPU_SLOTS=1
    source env/env.sh
    nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --limit-gib 30 -- scripts/run_logged.py --no-gpu-sampler \\
        --name cvtm_early_fulldata_exploratory --config experiments/cvtm/configs/cvtm_early_fulldata.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_early_fulldata.py

Smoke (not logged, scratch numbers only): add `--smoke` (train prefix of 3M tokens, 6 validation docs).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tmlm.cvtm.doc_clauses import doc_candidates, doc_features, doc_features_stream  # noqa: E402
from tmlm.data.corpus_store import load_split, raw_dir, split_dir  # noqa: E402
from tmlm.engine.rng import mix64 as rng_mix64  # noqa: E402
from tmlm.eval.baselines.ngram import KNModel  # noqa: E402

CORPUS = "wikitext103"
V = 8000
ORDER = 5
H_SALT = 0xC0FFEE
H_MOD = 100
EM_ITERS = 50
PRIOR = 16.0
ORDERS = (0, 1, 2, 3, 4)
PTR_MIN, PTR_MAX = 5, 64
N_BOOT = 2000
BOOT_SEED = (20260924, 1106, 1)
KN5_M2_BPB = 1.3501314033898424          # results/20260922T180823Z_M2_baselines_ngram_v8000/ kn5.validation.bits_per_byte
KN5_M2_TOP1 = 0.3512758902568059         # same run, kn5.validation.causal_top1
TRANSFORMER_BPB = 1.2340551926476289     # results/20260922T172902Z_M2_baselines_transformer/ transformer_L6_W256
KN5_M2_DIR = "results/20260922T180823Z_M2_baselines_ngram_v8000/"
TRANSFORMER_DIR = "results/20260922T172902Z_M2_baselines_transformer/"
COMPS = ("S", "D0", "D1", "D2", "D3", "D4", "P")
J = len(COMPS)
N_K, N_M, N_C, N_L = 5, 6, 5, 4          # k* 0..4; m* none,0..4; count bucket; pointer bucket
NB = N_K * N_M * N_C * N_L
ROWS = {                                  # name: (included document orders, pointer)
    "kn5": ((), False),
    "kn5_D0": ((0,), False),
    "kn5_D0to4": ((0, 1, 2, 3, 4), False),
    "cvtm_core": ((0, 1, 2, 3, 4), True),
}
GIB = 1024.0 ** 3
T0 = time.time()
METRICS: dict = {}


# ------------------------------------------------------------------ logging
def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}
    if isinstance(value, (int, float, str, bool)) or value is None:
        log(f"{name} = {value}")


def flush() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


def out_dir(smoke: bool) -> Path:
    d = os.environ.get("TMLM_RUN_DIR")
    if d and not smoke:
        return Path(d)
    p = Path(os.environ.get("TMLM_SCRATCH", "/tmp")) / "cvtm_early_fulldata_unlogged"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ memory
CG = Path("/sys/fs/cgroup")


def cgroup_bytes() -> tuple[int, int, int]:
    """(memory.current, working set = current - inactive_file, memory.max)."""
    cur = int((CG / "memory.current").read_text().strip())
    mx_txt = (CG / "memory.max").read_text().strip()
    mx = int(mx_txt) if mx_txt != "max" else 1 << 62
    inact = 0
    for line in (CG / "memory.stat").read_text().splitlines():
        k, _, v = line.partition(" ")
        if k == "inactive_file":
            inact = int(v)
            break
    return cur, max(0, cur - inact), mx


def own_rss_gib() -> tuple[float, float]:
    """(current RSS, peak RSS) of this process in GiB."""
    cur = peak = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            cur = int(line.split()[1]) / 1024.0 ** 2
        elif line.startswith("VmHWM:"):
            peak = int(line.split()[1]) / 1024.0 ** 2
    return cur, peak


class CgroupSampler:
    """Peak of the cgroup working set (all processes of the container) sampled every 2 s."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, cgroup_bytes()[1])
            except (OSError, ValueError):
                pass
            self._stop.wait(2.0)

    def stop(self) -> int:
        self._stop.set()
        self._t.join()
        return self.peak


def mem(tag: str) -> None:
    cur, ws, mx = cgroup_bytes()
    r, pk = own_rss_gib()
    log(f"memory at {tag}: rss {r:.2f} GiB, peak rss {pk:.2f} GiB, cgroup current {cur / GIB:.1f} GiB, "
        f"working set {ws / GIB:.1f} GiB, max {mx / GIB:.0f} GiB")


def wait_for_memory(expected_gib: float, tag: str, guard_gib: float, poll_s: int = 120,
                    max_wait_s: int = 3 * 3600) -> None:
    """Proceed only when memory.current + expected peak < 0.80 memory.max AND the cgroup working set
    + expected peak < 0.97 of the mem_guard limit (the guard counts every process of the container,
    so a fit started above it would be killed by other jobs' usage); poll every 2 minutes otherwise."""
    waited = 0
    while True:
        cur, ws, mx = cgroup_bytes()
        if cur + expected_gib * GIB < 0.80 * mx and ws + expected_gib * GIB < 0.97 * guard_gib * GIB:
            log(f"memory check before {tag}: current {cur / GIB:.1f} GiB + expected {expected_gib:.0f} GiB "
                f"< 0.80 x {mx / GIB:.0f} GiB and working set {ws / GIB:.1f} GiB + expected < 0.97 x guard "
                f"{guard_gib:.0f} GiB: proceed (waited {waited} s)")
            METRICS.setdefault("memory_waits_s", {"value": {}, "label": "measured", "exploratory": True})
            METRICS["memory_waits_s"]["value"][tag] = waited
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag} within {max_wait_s} s "
                               f"(current {cur / GIB:.1f} GiB, expected {expected_gib} GiB)")
        log(f"memory check before {tag}: current {cur / GIB:.1f} GiB, working set {ws / GIB:.1f} GiB, "
            f"expected {expected_gib:.0f} GiB, max {mx / GIB:.0f} GiB, guard {guard_gib:.0f} GiB: waiting {poll_s} s")
        time.sleep(poll_s)
        waited += poll_s


# ------------------------------------------------------------------ fit documents
def splitmix64(x: np.ndarray, increment: bool = True) -> np.ndarray:
    """First output of SplitMix64 with state x (docs/11 section 3 mix64), vectorized, mod 2^64.
    increment=False gives the bare finaliser (logged only, to expose the alternative reading)."""
    z = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        if increment:
            z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def fit_doc_mask(n_docs: int, increment: bool = True) -> np.ndarray:
    i = np.arange(n_docs, dtype=np.uint64)
    return (splitmix64(i ^ np.uint64(H_SALT), increment) % np.uint64(H_MOD)) == np.uint64(0)


def select_docs(tok: np.ndarray, ds: np.ndarray, keep_doc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.diff(ds.astype(np.int64))
    keep_tok = np.repeat(keep_doc, lengths)
    sub = np.ascontiguousarray(np.asarray(tok)[keep_tok])
    dsub = np.zeros(int(keep_doc.sum()) + 1, dtype=np.int64)
    dsub[1:] = np.cumsum(lengths[keep_doc])
    return sub, dsub


# ------------------------------------------------------------------ KN-5 gold probabilities and k*
def kn_gold(model: KNModel, tok: np.ndarray, ds: np.ndarray, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """P_KN(tok[p] | left context) (KNModel.prob, the reference) and k* = the longest k in 1..4 whose
    suffix clause fires (context of length k seen with a positive follower total at level k+1),
    0 when none (docs/11 section 11.2: an unseen context fires no clause)."""
    tok = np.asarray(tok).astype(np.int64)
    n = tok.shape[0]
    p = np.zeros(n, dtype=np.float64)
    ks = np.zeros(n, dtype=np.int8)
    for i in range(0, n, chunk):
        pos = np.arange(i, min(i + chunk, n))
        ctx, valid = model.contexts(tok, ds, pos, None, ORDER)
        p[pos] = model.prob(ctx, tok[pos], valid, ORDER)
        cids = model.context_ids(ctx, valid, ORDER)
        k = np.zeros(pos.shape[0], dtype=np.int8)
        for lev in range(2, ORDER + 1):
            cid = cids[lev]
            tot = model.level(lev, ORDER).total
            fired = (cid >= 0) & (tot[np.where(cid >= 0, cid, 0)] > 0)
            k = np.where(fired, lev - 1, k).astype(np.int8)
        ks[pos] = k
    return p, ks


def fit_kn(tok, ds, tag: str) -> KNModel:
    t0 = time.time()
    m = KNModel(ORDER, V, discount_mode="modified", pad=True).fit(tok, ds)
    M(f"{tag}.fit_wall_s", round(time.time() - t0, 1), unit="s", note="one KNModel.fit, single thread, nice 19, contended host")
    M(f"{tag}.n_types_per_level", m.n_types[1:])
    M(f"{tag}.n_train_tokens", int(m.n_train_tokens))
    mem(f"after {tag} fit")
    return m


# ------------------------------------------------------------------ components, buckets, EM
def components(p_kn: np.ndarray, feats, tok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gold probability [N, J] and fire mask [N, J] of S, D0..D4, P."""
    n = tok.shape[0]
    pc = np.zeros((n, J), dtype=np.float64)
    fire = np.zeros((n, J), dtype=bool)
    pc[:, 0] = p_kn
    fire[:, 0] = True
    for m in ORDERS:
        nm = feats.count(m).astype(np.int64)
        gm = feats.gold_count(m).astype(np.int64)
        f = nm >= 1
        fire[:, 1 + m] = f
        pc[f, 1 + m] = gm[f] / nm[f]
    fp = feats.ptr_len >= PTR_MIN
    fire[:, 6] = fp
    pc[fp, 6] = (feats.ptr_cand[fp].astype(np.int64) == tok[fp].astype(np.int64)).astype(np.float64)
    return pc, fire


def count_bucket(n: np.ndarray) -> np.ndarray:
    """{1, 2-3, 4-7, 8-15, 16+} -> 0..4 (0 also for n = 0, i.e. no firing order)."""
    n = np.maximum(np.asarray(n, dtype=np.int64), 1)
    return np.minimum(np.floor(np.log2(n.astype(np.float64))).astype(np.int64), 4)


def ptr_bucket(length: np.ndarray) -> np.ndarray:
    L = np.asarray(length, dtype=np.int64)
    return np.where(L < PTR_MIN, 0, np.where(L <= 8, 1, np.where(L <= 16, 2, 3)))


def buckets(kstar: np.ndarray, fire: np.ndarray, feats, orders: tuple, use_ptr: bool) -> np.ndarray:
    mstar = np.full(kstar.shape[0], -1, dtype=np.int64)
    nstar = np.zeros(kstar.shape[0], dtype=np.int64)
    for m in orders:                                   # increasing, so the highest firing order wins
        f = fire[:, 1 + m]
        mstar = np.where(f, m, mstar)
        nstar = np.where(f, feats.count(m).astype(np.int64), nstar)
    c = np.where(mstar >= 0, count_bucket(nstar), 0)
    l = ptr_bucket(feats.ptr_len) if use_ptr else np.zeros(kstar.shape[0], dtype=np.int64)
    return ((kstar.astype(np.int64) * N_M + (mstar + 1)) * N_C + c) * N_L + l


def bucket_fireset(orders: tuple, use_ptr: bool) -> np.ndarray:
    """[NB, J] components that fire in bucket b, from the bucket definition alone (S always;
    D_0..D_m* by nesting, restricted to the included orders; P iff l > 0)."""
    fs = np.zeros((NB, J), dtype=bool)
    for b in range(NB):
        l = b % N_L
        mstar = (b // (N_L * N_C)) % N_M - 1
        fs[b, 0] = True
        for m in orders:
            if m <= mstar:
                fs[b, 1 + m] = True
        if use_ptr and l > 0:
            fs[b, 6] = True
    return fs


def restrict(fire: np.ndarray, orders: tuple, use_ptr: bool) -> np.ndarray:
    keep = np.zeros(J, dtype=bool)
    keep[0] = True
    for m in orders:
        keep[1 + m] = True
    keep[6] = use_ptr
    return fire & keep[None, :]


def map_em(pc: np.ndarray, fire: np.ndarray, b: np.ndarray, fs: np.ndarray, iters: int, prior: float):
    """MAP-EM of the bucket weights (module docstring, step 4). Returns (lam [NB, J], trace)."""
    if not np.array_equal(fire, fs[b]):
        raise AssertionError("row fire masks disagree with the bucket definition (nesting violated)")
    lam = fs / fs.sum(axis=1, keepdims=True)
    n_b = np.bincount(b, minlength=NB).astype(np.float64)
    fired_rows = fire.sum(axis=0).astype(np.float64)
    trace = []
    for it in range(iters):
        w = lam[b] * pc
        tot = w.sum(axis=1)
        if not np.all(tot > 0):
            raise AssertionError("a fit row has zero mixture probability")
        trace.append(float(np.log2(tot).sum()))
        r = w / tot[:, None]
        R = np.zeros((NB, J), dtype=np.float64)
        for j in range(J):
            R[:, j] = np.bincount(b, weights=r[:, j], minlength=NB)
        pi = np.where(fired_rows > 0, R.sum(axis=0) / np.maximum(fired_rows, 1.0), 0.0)
        pb = fs * pi[None, :]
        s = pb.sum(axis=1, keepdims=True)
        pb = np.where(s > 0, pb / np.where(s > 0, s, 1.0), fs / fs.sum(axis=1, keepdims=True))
        if prior > 0:
            lam = (R + prior * pb) / (n_b + prior)[:, None]
        else:
            lam = np.where(n_b[:, None] > 0, R / np.maximum(n_b, 1.0)[:, None], pb)
    w = lam[b] * pc
    trace.append(float(np.log2(w.sum(axis=1)).sum()))
    return lam, trace


def mixture_log2(pc: np.ndarray, lam_rows: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum((lam_rows * pc).sum(axis=1), np.finfo(np.float64).tiny))


# ------------------------------------------------------------------ bootstrap
def doc_bootstrap(bits_a_doc: np.ndarray, bits_b_doc, bytes_doc: np.ndarray, idx: np.ndarray, const: float = 0.0):
    """Point estimate and 95 percent percentile CI of (sum bits_a - sum bits_b) / sum bytes over
    resampled documents (bits_b_doc None: bits_a alone, minus the constant `const`)."""
    d = bits_a_doc - (0.0 if bits_b_doc is None else bits_b_doc)
    point = float(d.sum() / bytes_doc.sum()) - const
    stats = d[idx].sum(axis=1) / bytes_doc[idx].sum(axis=1) - const
    return point, [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ------------------------------------------------------------------ causal top-1 (full distributions)
def top1_rows(model: KNModel, tok, ds, feats, lam_rows_of: dict, chunk: int, gold_log2: dict) -> dict:
    """Causal top-1 of every row from its full distribution. lam_rows_of[row] = [N, J] weights per
    position (zero for non-firing or excluded components). Checks on the fly: every combined row sums
    to one and its value at the gold token equals the gold probability used for bits per byte."""
    tok64 = np.asarray(tok).astype(np.int64)
    n = tok64.shape[0]
    correct = {r: 0 for r in lam_rows_of}
    max_sum_err = 0.0
    max_gold_err = 0.0
    for i in range(0, n, chunk):
        pos = np.arange(i, min(i + chunk, n))
        q = pos.shape[0]
        ctx, valid = model.contexts(tok64, ds, pos, None, ORDER)
        dist = model.dist(ctx, valid, ORDER)                      # [q, V]
        cands = doc_candidates(tok64, ds, pos, orders=ORDERS, window=None)
        gold = tok64[pos]
        for r, lam_all in lam_rows_of.items():
            lam = lam_all[pos]
            score = dist * lam[:, 0:1]
            for m in ORDERS:
                if not np.any(lam[:, 1 + m] > 0):
                    continue
                csr = cands[m]
                lens = np.diff(csr.indptr)
                rowrep = np.repeat(np.arange(q), lens)
                nm = feats.count(m).astype(np.float64)[pos]
                contrib = lam[rowrep, 1 + m] * csr.cnt / np.where(nm[rowrep] > 0, nm[rowrep], 1.0)
                score[rowrep, csr.tok] += contrib                  # (row, token) pairs are unique per order
            fp = np.flatnonzero(lam[:, 6] > 0)
            if fp.shape[0]:
                score[fp, feats.ptr_cand[pos[fp]].astype(np.int64)] += lam[fp, 6]
            pred = score.argmax(axis=1)
            correct[r] += int((pred == gold).sum())
            max_sum_err = max(max_sum_err, float(np.abs(score.sum(axis=1) - 1.0).max()))
            pg = np.log2(np.maximum(score[np.arange(q), gold], np.finfo(np.float64).tiny))
            max_gold_err = max(max_gold_err, float(np.abs(pg - gold_log2[r][pos]).max()))
        if (i // chunk) % 20 == 0:
            log(f"  top-1 pass: {min(i + chunk, n)}/{n} positions")
    return {"top1": {r: correct[r] / n for r in correct}, "max_row_sum_error": max_sum_err,
            "max_gold_log2_error": max_gold_err}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3M-token train prefix, 6 validation docs, not logged")
    ap.add_argument("--smoke-train-tokens", type=int, default=3_000_000)
    ap.add_argument("--smoke-val-docs", type=int, default=6)
    ap.add_argument("--top1", type=int, default=1)
    ap.add_argument("--top1-chunk", type=int, default=2048)
    ap.add_argument("--kn-peak-gib", type=float, default=20.0,
                    help="expected peak of one full KN-5 fit (scratch probe: 3.68 GiB at 27M tokens, linear)")
    ap.add_argument("--guard-gib", type=float, default=30.0, help="the --limit-gib of the enclosing mem_guard.py")
    args = ap.parse_args()
    smoke = bool(args.smoke)
    if smoke and os.environ.get("TMLM_RUN_DIR"):
        raise SystemExit("--smoke must not run under run_logged.py (scratch numbers only)")
    lab = "unverified" if smoke else "measured"
    M("status", "EXPLORATORY early estimate; not pre-registered; the docs/13 A3 numbers will come from the frozen "
      "implementation; this run tunes nothing (the estimator is fixed by docs/11 section 11)", label="unverified")
    M("args", vars(args), label=lab)
    sampler = CgroupSampler()

    # ---- the splitmix64 definition equals the repository's mix64 (docs/11 section 3)
    probe = [0, 1, 0xC0FFEE, 12345 ^ 0xC0FFEE, (1 << 64) - 1]
    assert all(int(splitmix64(np.uint64(x))) == rng_mix64(x) for x in probe)
    assert int(splitmix64(np.uint64(0))) == 0xE220A8397B1DCDAF

    # ---- data
    t0 = time.time()
    tr = load_split(CORPUS, V, "train", check=not smoke)
    va = load_split(CORPUS, V, "validation", check=True)
    tr_meta = json.loads((split_dir(CORPUS, V, "train") / "meta.json").read_text())
    va_meta = json.loads((split_dir(CORPUS, V, "validation") / "meta.json").read_text())
    raw_meta = json.loads((raw_dir(CORPUS, "validation") / "meta.json").read_text())
    doc_bytes_off = np.load(raw_dir(CORPUS, "validation") / "doc_bytes.npy").astype(np.int64)
    n_bytes = int(raw_meta["n_bytes"])
    assert n_bytes == int(va_meta["n_bytes_utf8"]) == int(doc_bytes_off[-1])
    tok_tr = np.asarray(tr.tok)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    tok_va = np.ascontiguousarray(np.asarray(va.tok))
    ds_va = np.asarray(va.doc_start).astype(np.int64)
    bytes_doc = np.diff(doc_bytes_off).astype(np.float64)
    if smoke:
        d = int(np.searchsorted(ds_tr, args.smoke_train_tokens, side="right")) - 1
        ds_tr = ds_tr[:d + 1]
        tok_tr = tok_tr[:int(ds_tr[-1])]
        kd = args.smoke_val_docs
        ds_va = ds_va[:kd + 1]
        tok_va = tok_va[:int(ds_va[-1])]
        bytes_doc = bytes_doc[:kd]
        n_bytes = int(bytes_doc.sum())
    else:
        from tmlm.common import runlog
        if runlog.current_run_dir() is not None:
            runlog.write_dataset_hashes({f"train_tok_V{V}": tr_meta["sha256"]["tok.npy"],
                                         f"validation_tok_V{V}": va_meta["sha256"]["tok.npy"],
                                         "validation_text_utf8": raw_meta["sha256"]["text.utf8"],
                                         "validation_doc_bytes": raw_meta["sha256"]["doc_bytes.npy"]})
    tok_tr = np.ascontiguousarray(tok_tr)
    n_docs_tr = ds_tr.shape[0] - 1
    n_va = int(tok_va.shape[0])
    n_docs_va = ds_va.shape[0] - 1
    M("data.train_tokens", int(tok_tr.shape[0]), label=lab)
    M("data.train_docs", n_docs_tr, label=lab)
    M("data.validation_tokens", n_va, label=lab)
    M("data.validation_bytes_utf8", n_bytes, label=lab, note="protocol CP byte accounting")
    M("data.validation_docs", n_docs_va, label=lab)
    M("data.load_wall_s", round(time.time() - t0, 1), label=lab, unit="s")

    # ---- fit documents H
    hmask = fit_doc_mask(n_docs_tr, increment=True)
    hmask_alt = fit_doc_mask(n_docs_tr, increment=False)
    h_idx = np.flatnonzero(hmask)
    lengths = np.diff(ds_tr)
    M("H.n_docs", int(hmask.sum()), label=lab)
    M("H.n_tokens", int(lengths[hmask].sum()), label=lab)
    M("H.fraction_of_train_tokens", float(lengths[hmask].sum() / lengths.sum()), label="derived" if not smoke else lab)
    M("H.doc_index_sha256", hashlib.sha256(h_idx.astype("<i8").tobytes()).hexdigest(), label=lab,
      note="sha256 of the int64 little-endian array of the H document indices")
    M("H.first_indices", [int(x) for x in h_idx[:10]], label=lab)
    M("H.alt_reading_no_increment.n_docs", int(hmask_alt.sum()), label=lab,
      note="bare finaliser without the golden-gamma increment; NOT used")
    M("H.alt_reading_no_increment.overlap_docs", int((hmask & hmask_alt).sum()), label=lab)
    tok_h, ds_h = select_docs(tok_tr, ds_tr, hmask)
    tok_noh, ds_noh = select_docs(tok_tr, ds_tr, ~hmask)
    flush()

    # ---- document clauses on H and validation (batch vs streaming cross-check)
    t0 = time.time()
    feats_h = doc_features(tok_h, ds_h, orders=ORDERS, window=None, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
    feats_va = doc_features(tok_va, ds_va, orders=ORDERS, window=None, ptr_min=PTR_MIN, ptr_max=PTR_MAX)
    M("doc_clauses.batch_wall_s", round(time.time() - t0, 1), label=lab, unit="s", note="H and validation")
    t0 = time.time()
    same_h = feats_h.same_values(doc_features_stream(tok_h, ds_h, ORDERS, None, PTR_MIN, PTR_MAX))
    same_va = feats_va.same_values(doc_features_stream(tok_va, ds_va, ORDERS, None, PTR_MIN, PTR_MAX))
    M("doc_clauses.stream_equals_batch", {"H": bool(same_h), "validation": bool(same_va)}, label=lab)
    M("doc_clauses.stream_wall_s", round(time.time() - t0, 1), label=lab, unit="s", note="includes numba JIT")
    M("doc_clauses.n_fallback", {"H": feats_h.n_fallback, "validation": feats_va.n_fallback}, label=lab)
    if not (same_h and same_va):
        flush()
        raise AssertionError("streaming and batch document features differ")
    flush()

    # ---- KN-5 without H: P_KN on H (fit of the combination) and on validation (check)
    wait_for_memory(args.kn_peak_gib, "kn5_noH fit", args.guard_gib)
    kn = fit_kn(tok_noh, ds_noh, "kn5_noH")
    t0 = time.time()
    pkn_h, ks_h = kn_gold(kn, tok_h, ds_h)
    pkn_va_noh, ks_va_noh = kn_gold(kn, tok_va, ds_va)
    M("kn5_noH.score_wall_s", round(time.time() - t0, 1), label=lab, unit="s")
    M("kn5_noH.H.bits_per_token", float(-np.log2(pkn_h).mean()), label=lab, note="held-out H documents")
    M("kn5_noH.validation.bits_per_byte", float(-np.log2(pkn_va_noh).sum() / n_bytes), label=lab)
    del kn
    gc.collect()
    mem("after freeing kn5_noH")
    flush()

    # ---- KN-5 on ALL training documents: validation
    pc_h, fire_h = components(pkn_h, feats_h, tok_h)          # H is no longer needed as tokens
    del tok_noh, ds_noh, tok_h
    gc.collect()
    wait_for_memory(args.kn_peak_gib, "kn5_all fit", args.guard_gib)
    kn = fit_kn(tok_tr, ds_tr, "kn5_all")
    t0 = time.time()
    pkn_va, ks_va = kn_gold(kn, tok_va, ds_va)
    lp_ref = kn.logprob(tok_va, ds_va, order=ORDER)
    assert np.array_equal(lp_ref, np.log2(np.maximum(pkn_va, np.finfo(np.float64).tiny)))
    bpb_kn = float(-lp_ref.sum() / n_bytes)
    M("kn5_all.score_wall_s", round(time.time() - t0, 1), label=lab, unit="s")
    M("kn5_all.validation.bits_per_byte", bpb_kn, label=lab)
    if not smoke:
        M("kn5_all.check_vs_M2", {"M2_value": KN5_M2_BPB, "abs_diff": abs(bpb_kn - KN5_M2_BPB),
                                  "equal_within_1e-9": abs(bpb_kn - KN5_M2_BPB) <= 1e-9, "M2_run": KN5_M2_DIR},
          label="derived")
    M("kstar_fraction.validation", [float((ks_va == k).mean()) for k in range(5)], label=lab,
      note="fraction of validation positions with longest fired suffix order k* = 0..4 (KN on all)")
    M("kstar_fraction.H", [float((ks_h == k).mean()) for k in range(5)], label=lab, note="KN without H")
    flush()

    # ---- combination: fit on H (KN without H), apply on validation (KN on all)
    pc_va, fire_va = components(pkn_va, feats_va, tok_va)
    pc_va_noh, _ = components(pkn_va_noh, feats_va, tok_va)
    M("fire_fraction.validation", {c: float(fire_va[:, j].mean()) for j, c in enumerate(COMPS)}, label=lab)
    M("fire_fraction.H", {c: float(fire_h[:, j].mean()) for j, c in enumerate(COMPS)}, label=lab)
    fp = fire_va[:, 6]
    M("pointer.validation.gold_hit_rate_when_fired", float(pc_va[fp, 6].mean()) if fp.any() else None, label=lab)
    gold_log2: dict = {}
    lam_rows_of: dict = {}
    tables: dict = {}
    doc_of_va = np.repeat(np.arange(n_docs_va), np.diff(ds_va))
    for row, (orders, use_ptr) in ROWS.items():
        t0 = time.time()
        f_h = restrict(fire_h, orders, use_ptr)
        f_va = restrict(fire_va, orders, use_ptr)
        b_h = buckets(ks_h, f_h, feats_h, orders, use_ptr)
        b_va = buckets(ks_va, f_va, feats_va, orders, use_ptr)
        fs = bucket_fireset(orders, use_ptr)
        assert np.array_equal(f_va, fs[b_va])
        lam, trace = map_em(pc_h, f_h, b_h, fs, EM_ITERS, PRIOR)
        lam_va = lam[b_va]
        lg = mixture_log2(pc_va, lam_va)
        gold_log2[row] = lg
        lam_rows_of[row] = lam_va
        tables[row] = lam
        bpb = float(-lg.sum() / n_bytes)
        bpt = float(-lg.mean())
        M(f"{row}.validation.bits_per_byte", bpb, label=lab)
        M(f"{row}.validation.bits_per_token", bpt, label=lab)
        M(f"{row}.validation.token_perplexity", float(2.0 ** bpt), label="derived" if not smoke else lab)
        M(f"{row}.H.bits_per_token_in_sample", float(-mixture_log2(pc_h, lam[b_h]).mean()), label=lab,
          note="the fit rows themselves (in sample)")
        M(f"{row}.em.log2_likelihood_trace", [trace[0], trace[1], trace[9], trace[EM_ITERS - 1], trace[EM_ITERS]],
          label=lab, note="sum log2 P over the fit rows at iterations 0, 1, 9, 49 and after the last M step")
        M(f"{row}.buckets_populated", {"H": int(np.unique(b_h).shape[0]), "validation": int(np.unique(b_va).shape[0]),
                                       "validation_rows_in_buckets_empty_on_H": int((np.bincount(b_h, minlength=NB)[b_va] == 0).sum())},
          label=lab)
        M(f"{row}.mean_weights.validation", {c: float(lam_va[:, j].mean()) for j, c in enumerate(COMPS)}, label=lab)
        # sensitivity 1: the MLE variant (prior 0), not a row
        lam0, _ = map_em(pc_h, f_h, b_h, fs, EM_ITERS, 0.0)
        M(f"{row}.sensitivity.mle_prior0.validation.bits_per_byte", float(-mixture_log2(pc_va, lam0[b_va]).sum() / n_bytes),
          label=lab, note="same estimator with prior 0; sensitivity to the pooled-prior definition, not a row")
        # sensitivity 2: the KN-without-H base on validation, not a row
        b_va2 = buckets(ks_va_noh, f_va, feats_va, orders, use_ptr)
        M(f"{row}.sensitivity.kn_noH_base.validation.bits_per_byte", float(-mixture_log2(pc_va_noh, lam[b_va2]).sum() / n_bytes),
          label=lab, note="weights applied with the KN-without-H base (docs/11 11.6 prescribes KN on all); not a row")
        M(f"{row}.fit_and_score_wall_s", round(time.time() - t0, 1), label=lab, unit="s")
    np.savez_compressed(out_dir(smoke) / "per_token_log2.npz", **{r: gold_log2[r] for r in gold_log2},
                        doc_of=doc_of_va, kstar=ks_va)
    np.savez_compressed(out_dir(smoke) / "bucket_weights.npz", comps=np.array(COMPS), **tables)
    assert np.array_equal(gold_log2["kn5"], lp_ref)          # the kn5 row is exactly KN-5 (lambda_S = 1)
    flush()

    # ---- bootstrap
    idx = np.random.default_rng(np.random.SeedSequence(list(BOOT_SEED))).integers(0, n_docs_va, size=(N_BOOT, n_docs_va))
    bits_doc = {r: np.bincount(doc_of_va, weights=-gold_log2[r], minlength=n_docs_va) for r in gold_log2}
    blab = "derived" if not smoke else lab
    for r in ROWS:
        if r == "kn5":
            continue
        pt, ci = doc_bootstrap(bits_doc[r], bits_doc["kn5"], bytes_doc, idx)
        M(f"{r}.minus_kn5.bits_per_byte", pt, label=blab, ci95=ci,
          note=f"document-block bootstrap, {n_docs_va} documents, {N_BOOT} resamples, seed {list(BOOT_SEED)}")
    for a, bname in (("kn5_D0to4", "kn5_D0"), ("cvtm_core", "kn5_D0to4")):
        pt, ci = doc_bootstrap(bits_doc[a], bits_doc[bname], bytes_doc, idx)
        M(f"{a}.minus_{bname}.bits_per_byte", pt, label=blab, ci95=ci, note="increment of the added components")
    pt, ci = doc_bootstrap(bits_doc["cvtm_core"], None, bytes_doc, idx, const=TRANSFORMER_BPB)
    M("cvtm_core.minus_transformer600s.bits_per_byte", pt, label=blab, ci95=ci,
      note=f"transformer 1.2341 treated as a constant ({TRANSFORMER_DIR} transformer_L6_W256); the CI covers only "
           "the CVTM-core's document sampling, not the transformer's own variability")
    flush()

    # ---- causal top-1 from the full distributions
    if args.top1:
        t0 = time.time()
        res = top1_rows(kn, tok_va, ds_va, feats_va, lam_rows_of, args.top1_chunk, gold_log2)
        for r, v in res["top1"].items():
            M(f"{r}.validation.causal_top1", v, label=lab, note="argmax of the full next-token distribution, every validation position, ties to the lower id")
        M("top1.max_row_sum_error", res["max_row_sum_error"], label=lab)
        M("top1.max_gold_log2_error", res["max_gold_log2_error"], label=lab,
          note="full-distribution value at the gold token vs the gold probability of the bits-per-byte pass")
        if not smoke:
            M("kn5.top1_check_vs_M2", {"M2_value": KN5_M2_TOP1, "abs_diff": abs(res["top1"]["kn5"] - KN5_M2_TOP1)}, label="derived")
        M("top1.wall_s", round(time.time() - t0, 1), label=lab, unit="s")
        flush()

    # ---- memory and wall
    r, pk = own_rss_gib()
    M("peak_rss_gib", round(pk, 2), label=lab, note="VmHWM of this process")
    M("peak_cgroup_working_set_gib", round(sampler.stop() / GIB, 2), label=lab,
      note="all processes of the container, sampled every 2 s (includes other jobs)")
    M("wall_s", round(time.time() - T0, 1), label=lab, unit="s")
    flush()
    if smoke:
        (out_dir(smoke) / "metrics_smoke.json").write_text(json.dumps(METRICS, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
