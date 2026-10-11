"""Feasibility numbers for TM-native levers that could raise the Tsetlin clauses' share of the
causal hybrid (EXPLORATORY: not pre-registered, no docs/13 threshold is tested; validation split
only, the test split is never loaded). CPU only, one thread.

Inputs: the WikiText-103 BPE-8000 train prefix and split of the causal probe
(results/20260924T012544Z_ckm_probe_causal_exploratory: first 5,000,000 tokens cut at a document
boundary, 5 percent of its documents held out with SeedSequence([20260923, 715, 1]), counts from the
other documents), the validation split, and that probe's saved per-token bits (per_token_bits.npz),
which give the gold probability of every base row at every validation position.

Questions (all on validation, causal, every position):
  A. Copy headroom. How often is the gold token present earlier in the same document (within
     distance D), and how often is it among the candidates of n-gram-keyed induction (the probe's
     version: most recent earlier position with the same left-k context, k = 1..5, only contexts
     seen in the counting documents; and an unrestricted version k = 1..8)? Plus a gold-dependent
     ORACLE bound: bits if a copy component put mass lam on the gold whenever it is copyable.
  B. What a learned pair-feature copy matcher (the "logical attention" lever) could add. A lookup
     table m(f) = P(tok[q] == tok[p] | f(p, q)) estimated on training pairs is the Bayes-optimal
     predictor over the pair features f = (token suffix match length L, 256-cluster suffix match
     length Lc, log2 distance, frequency class of tok[q], frequency class of tok[p-1]); a 2-class
     TM over the same literals can at best approximate it. P_attn(y) = sum_{q: tok[q]=y} m^beta /
     sum_q m^beta over every earlier in-document position q. Compared with the same table over L
     only (n-gram-keyed aggregated copy) and with a classical document n-gram cache (orders 0..4).
  C. Where the current TM share comes from: per-token bits of the best row minus the same row
     without TM clauses, split by longest left context seen in the counting documents, by whether
     the gold is copyable, and by gold frequency.
  D. Headroom of class-based and skip contexts (the "generalisation" lever): fixed templates
     (cluster bigram/trigram, word+cluster, skip-grams) counted on the counting documents and
     mixed in, against the TM share.
Every added component enters a normalized linear mixture with the saved base row,
P = (P_base + sum_i w_b(i) P_i) / (1 + sum_i w_b(i)), bucket weights fitted by L-BFGS on one half of
the validation documents (even or odd index) and scored on the other half (two-fold cross-fitting),
so every reported bit is out-of-fold. Validation is used for tuning (protocol rule 14); the test
split is never read.

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1
    source env/env.sh
    nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py --no-gpu-sampler \\
        --name tm_levers_feasibility_causal_exploratory --config <scratch json> -- \\
        .venv-tm/bin/python -u experiments/tm_levers_feasibility_causal.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

PROBE = ROOT / "results" / "20260924T012544Z_ckm_probe_causal_exploratory"
CORPUS = "wikitext103"
V = 8000
EDGE_T, EDGE_C2, EDGE_C3 = 8000, 256, 4096      # symbol for "before the document start"
LMAX, LCMAX, DBMAX, NFC = 6, 4, 14, 5
LN2 = math.log(2.0)
T0 = time.time()
METRICS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def M(name: str, value, label: str = "measured", **extra) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    METRICS[name] = {"value": value, "label": label, "exploratory": True, **extra}


def flush() -> None:
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(METRICS, on_conflict="replace")


# ------------------------------------------------------------------ helpers
def doc_lo_of(ds: np.ndarray, n: int) -> np.ndarray:
    d = np.searchsorted(ds, np.arange(n), side="right") - 1
    return ds[d].astype(np.int64)


def shifted(arr: np.ndarray, lo: np.ndarray, k: int, edge: int) -> np.ndarray:
    """arr[p - k] if p - k >= doc_lo(p) else edge."""
    n = arr.shape[0]
    out = np.full(n, edge, np.int64)
    idx = np.arange(n) - k
    ok = idx >= lo
    out[ok] = arr[idx[ok]]
    return out


def mix64(z: np.ndarray) -> np.ndarray:
    z = (z + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return z ^ (z >> np.uint64(31))


def ctx_hash(tok: np.ndarray, lo: np.ndarray, k: int) -> np.ndarray:
    """64-bit hash of the left-k token context (EDGE-padded); k = 0 gives a constant."""
    h = np.full(tok.shape[0], np.uint64(k * 1000003 + 17), np.uint64)
    with np.errstate(over="ignore"):
        for i in range(1, k + 1):
            h = mix64(h ^ shifted(tok, lo, i, EDGE_T).astype(np.uint64))
    return h


def group_rank_prev(key: np.ndarray, pos: np.ndarray):
    """For positions grouped by key (ascending position inside a group): rank inside the group and
    the previous position of the group (-1 if first)."""
    order = np.lexsort((pos, key))
    ks = key[order]
    start = np.ones(ks.shape[0], bool)
    start[1:] = ks[1:] != ks[:-1]
    gstart = np.maximum.accumulate(np.where(start, np.arange(ks.shape[0]), 0))
    rank = np.empty(ks.shape[0], np.int64)
    rank[order] = np.arange(ks.shape[0]) - gstart
    prev = np.full(ks.shape[0], -1, np.int64)
    ps = pos[order]
    prev_s = np.where(start, -1, np.concatenate([[-1], ps[:-1]]))
    prev[order] = prev_s
    return rank, prev


# ------------------------------------------------------------------ pair kernels
@njit(cache=False)
def k_pair_cell(tok, c2, fcls, dbt, p, q, dl):
    L = 0
    while L < LMAX and p - 1 - L >= dl and q - 1 - L >= dl and tok[p - 1 - L] == tok[q - 1 - L]:
        L += 1
    Lc = 0
    while Lc < LCMAX and p - 1 - Lc >= dl and q - 1 - Lc >= dl and c2[tok[p - 1 - Lc]] == c2[tok[q - 1 - Lc]]:
        Lc += 1
    db = dbt[p - q]
    fc = fcls[tok[q]]
    rc = fcls[tok[p - 1]] if p - 1 >= dl else NFC
    cell = (((L * (LCMAX + 1) + Lc) * (DBMAX + 1) + db) * NFC + fc) * (NFC + 1) + rc
    return cell, L


@njit(cache=False)
def k_train_table(tok, c2, fcls, dbt, lo, sample, n_cells):
    hits = np.zeros(n_cells, np.int64)
    tries = np.zeros(n_cells, np.int64)
    for a in range(sample.shape[0]):
        p = sample[a]
        dl = lo[p]
        y = tok[p]
        for q in range(dl, p):
            cell, L = k_pair_cell(tok, c2, fcls, dbt, p, q, dl)
            tries[cell] += 1
            if tok[q] == y:
                hits[cell] += 1
    return hits, tries


@njit(cache=False)
def k_train_hist(tok, c2, fcls, dbt, lo, sample, n_cells):
    """Per sampled position with the gold in its document history: the (cell, n_all, n_match) histogram
    over every earlier in-document q. Two passes (count, fill)."""
    mark = np.full(n_cells, -1, np.int64)
    n_ent = 0
    n_pos = 0
    for a in range(sample.shape[0]):
        p = sample[a]
        dl = lo[p]
        y = tok[p]
        has = False
        for q in range(dl, p):
            if tok[q] == y:
                has = True
                break
        if not has:
            continue
        n_pos += 1
        for q in range(dl, p):
            cell, L = k_pair_cell(tok, c2, fcls, dbt, p, q, dl)
            if mark[cell] != a:
                mark[cell] = a
                n_ent += 1
    pos_i = np.empty(n_ent, np.int64)
    cells = np.empty(n_ent, np.int64)
    na = np.zeros(n_ent)
    nm = np.zeros(n_ent)
    slot = np.full(n_cells, -1, np.int64)
    mark[:] = -1
    e = 0
    r = 0
    for a in range(sample.shape[0]):
        p = sample[a]
        dl = lo[p]
        y = tok[p]
        has = False
        for q in range(dl, p):
            if tok[q] == y:
                has = True
                break
        if not has:
            continue
        for q in range(dl, p):
            cell, L = k_pair_cell(tok, c2, fcls, dbt, p, q, dl)
            if mark[cell] != a:
                mark[cell] = a
                slot[cell] = e
                pos_i[e] = r
                cells[e] = cell
                e += 1
            k = slot[cell]
            na[k] += 1.0
            if tok[q] == y:
                nm[k] += 1.0
        r += 1
    return pos_i, cells, na, nm, n_pos


@njit(cache=False)
def k_val_attn(tok, c2, fcls, dbt, lo, svals):
    """For every position and pointer variant v (svals[v, cell] = exp(theta_v[cell])):
    P_ptr(gold) = sum over earlier q with tok[q] = gold of s / sum over all earlier q of s, the top
    candidate's share of the mass, and the suffix match length L of the top candidate."""
    n = tok.shape[0]
    nv = svals.shape[0]
    pg = np.zeros((n, nv))
    share = np.zeros((n, nv))
    ltop = np.zeros((n, nv), np.int64)
    ok = np.zeros(n, np.uint8)
    num = np.zeros(nv)
    den = np.zeros(nv)
    best = np.zeros(nv)
    bestL = np.zeros(nv, np.int64)
    for p in range(n):
        dl = lo[p]
        if p == dl:
            continue
        ok[p] = 1
        y = tok[p]
        for v in range(nv):
            num[v] = 0.0
            den[v] = 0.0
            best[v] = 0.0
            bestL[v] = 0
        for q in range(dl, p):
            cell, L = k_pair_cell(tok, c2, fcls, dbt, p, q, dl)
            hit = tok[q] == y
            for v in range(nv):
                sv = svals[v, cell]
                den[v] += sv
                if hit:
                    num[v] += sv
                if sv > best[v]:
                    best[v] = sv
                    bestL[v] = L
        for v in range(nv):
            pg[p, v] = num[v] / den[v]
            share[p, v] = best[v] / den[v]
            ltop[p, v] = bestL[v]
    return pg, share, ltop, ok


def cell_groups(groups):
    """Parameter index of every cell for each factor group; groups are tuples of axis names."""
    L, Lc, db, fc, rc = np.meshgrid(np.arange(LMAX + 1), np.arange(LCMAX + 1), np.arange(DBMAX + 1),
                                    np.arange(NFC), np.arange(NFC + 1), indexing="ij")
    ax = {"L": (L.ravel(), LMAX + 1), "Lc": (Lc.ravel(), LCMAX + 1), "db": (db.ravel(), DBMAX + 1),
          "fc": (fc.ravel(), NFC), "rc": (rc.ravel(), NFC + 1)}
    idx, off = [], 0
    for g in groups:
        k = np.zeros(L.size, np.int64)
        size = 1
        for a in g:
            v, n = ax[a]
            k = k * n + v
            size *= n
        idx.append(k + off)
        off += size
    return idx, off


def fit_pointer(pos_i, cells, na, nm, n_pos, groups, ridge=1.0, iters=300):
    """Log-linear pointer over pair-feature cells: s(q) = exp(theta[cell(p, q)]), theta = sum of factor
    parameters; maximise sum_p log(sum_match s / sum_all s) over training positions whose gold is in
    their document history, minus ridge * |params|^2 / 2."""
    from scipy.optimize import minimize
    gidx, n_par = cell_groups(groups)
    lna = np.log(na)
    m_sel = nm > 0
    lnm = np.log(np.where(m_sel, nm, 1.0))

    def theta_of(par):
        th = np.zeros(gidx[0].shape[0])
        for g in gidx:
            th += par[g]
        return th

    def obj(par):
        th = np.clip(theta_of(par), -40.0, 40.0)
        a = np.exp(lna + th[cells])
        A = np.bincount(pos_i, weights=a, minlength=n_pos)
        b = np.where(m_sel, np.exp(lnm + th[cells]), 0.0)
        B = np.bincount(pos_i, weights=b, minlength=n_pos)
        f = -np.sum(np.log(B) - np.log(A)) + 0.5 * ridge * np.sum(par * par)
        gcell_e = a / A[pos_i] - b / B[pos_i]
        gcell = np.bincount(cells, weights=gcell_e, minlength=gidx[0].shape[0])
        g = ridge * par
        for gi in gidx:
            g = g + np.bincount(gi, weights=gcell, minlength=n_par)
        return f, g

    res = minimize(obj, np.zeros(n_par), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-6})
    th = np.clip(theta_of(res.x), -40.0, 40.0)
    return th, res


# ------------------------------------------------------------------ mixture fit
def fit_mix(pb, comp_rows, comp_b, comp_p, n_b, fit_mask, iters=400):
    """Normalized linear mixture: P_t = (pb_t + sum_i w_b(i) p_i) / (1 + sum_i w_b(i)).
    Returns theta = log w (buckets never seen on fit rows stay at -30)."""
    from scipy.optimize import minimize
    sel = fit_mask[comp_rows]
    r, b, p = comp_rows[sel], comp_b[sel], comp_p[sel]
    rows = np.flatnonzero(fit_mask)
    remap = np.full(pb.shape[0], -1, np.int64)
    remap[rows] = np.arange(rows.shape[0])
    r = remap[r]
    base = pb[rows]
    seen = np.bincount(b, minlength=n_b) > 0

    def obj(theta):
        w = np.exp(theta)[b]
        W = 1.0 + np.bincount(r, weights=w, minlength=rows.shape[0])
        N = base + np.bincount(r, weights=w * p, minlength=rows.shape[0])
        f = -np.sum(np.log(N) - np.log(W))
        g_e = w * (1.0 / W[r] - p / N[r])
        g = np.bincount(b, weights=g_e, minlength=n_b)
        return f, g

    th0 = np.where(seen, -3.0, -30.0)
    res = minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=[(-30.0, 10.0)] * n_b,
                   options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-7})
    return res.x, res


def eval_mix(pb, comp_rows, comp_b, comp_p, theta):
    w = np.exp(theta)[comp_b]
    n = pb.shape[0]
    W = 1.0 + np.bincount(comp_rows, weights=w, minlength=n)
    N = pb + np.bincount(comp_rows, weights=w * comp_p, minlength=n)
    return -(np.log(N) - np.log(W)) / LN2


def crossfit(pb, comps, fold_of_row, iters=400):
    """comps: list of (rows, bucket, p, n_buckets). Returns out-of-fold bits per token and fit info."""
    rows = np.concatenate([c[0] for c in comps]) if comps else np.zeros(0, np.int64)
    offs = np.cumsum([0] + [c[3] for c in comps])
    b = np.concatenate([c[1] + offs[i] for i, c in enumerate(comps)]) if comps else np.zeros(0, np.int64)
    p = np.concatenate([c[2] for c in comps]) if comps else np.zeros(0)
    n_b = int(offs[-1])
    bits = np.empty(pb.shape[0])
    info = []
    for f in (0, 1):
        fit_mask = fold_of_row == f
        if n_b == 0:
            bits[~fit_mask] = -np.log2(pb[~fit_mask])
            continue
        th, res = fit_mix(pb, rows, b, p, n_b, fit_mask, iters)
        e = ~fit_mask
        ev = e[rows]
        bb = eval_mix(pb, rows[ev], b[ev], p[ev], th)
        bits[e] = bb[e]
        info.append({"fold_fit": f, "nit": int(res.nit), "converged": bool(res.success), "message": str(res.message)[:80]})
    return bits, info


def doc_boot(diff, doc, n_docs, B=2000, seed=0):
    s = np.bincount(doc, weights=diff, minlength=n_docs)
    c = np.bincount(doc, minlength=n_docs).astype(np.float64)
    rng = np.random.default_rng(seed)
    present = np.flatnonzero(c > 0)
    idx = present[rng.integers(0, present.shape[0], size=(B, present.shape[0]))]
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=5_000_000)
    ap.add_argument("--train-pairs-positions", type=int, default=60000)
    ap.add_argument("--val-docs", type=int, default=0, help="smoke only: first k validation documents")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--ptr-iters", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    M("args", vars(args))
    M("mode_smoke", bool(args.smoke))

    from tmlm.data.corpus_store import load_split
    from m3_tiny_lm import slice_train

    full = load_split(CORPUS, V, "train")
    tr = slice_train(full, args.n_tokens)
    train_counts_full = np.bincount(np.asarray(full.tok).astype(np.int64), minlength=V)
    km = np.asarray(full.kind_map).astype(np.int64)
    del full
    va = load_split(CORPUS, V, "validation")
    from cp_causal_perplexity import split_meta
    val_meta, _ = split_meta(V, "validation")
    n_bytes = int(val_meta["n_bytes_utf8"])
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    tok_va = np.asarray(va.tok).astype(np.int64)
    ds_va = np.asarray(va.doc_start).astype(np.int64)
    n_tr, n_va = tok_tr.shape[0], tok_va.shape[0]
    lo_tr, lo_va = doc_lo_of(ds_tr, n_tr), doc_lo_of(ds_va, n_va)
    n_doc = ds_tr.shape[0] - 1
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, 1]))
    n_ho = max(1, int(round(0.05 * n_doc)))
    ho_docs = np.sort(rng.choice(n_doc, size=n_ho, replace=False))
    doc_of_tr = np.repeat(np.arange(n_doc), np.diff(ds_tr))
    is_ho = np.isin(doc_of_tr, ho_docs)
    cnt_pos = np.flatnonzero(~is_ho)
    M("data.prefix_tokens", int(n_tr))
    M("data.counting_positions", int(cnt_pos.shape[0]), note="same split as the causal probe")
    M("data.validation_tokens", int(n_va))
    M("data.validation_bytes_utf8", n_bytes)

    probe_rows = json.loads((PROBE / "rows.json").read_text())
    z = np.load(PROBE / "per_token_bits.npz")
    assert z["positions"].shape[0] == n_va and np.array_equal(z["positions"], np.arange(n_va))
    doc_va = z["doc"].astype(np.int64)
    n_vdocs = int(ds_va.shape[0] - 1)
    assert np.array_equal(doc_va, np.repeat(np.arange(n_vdocs), np.diff(ds_va)))
    base_names = {"kn5": "comparator.kn5", "kn5_cache": "comparator.kn5+doc_cache",
                  "noTM": "kn5+ngram.lin+induction+doc_cache", "TM": "kn5+union.lin+induction+doc_cache",
                  "noTM_noind": "kn5+ngram.lin", "TM_noind": "kn5+union.lin",
                  "noTM_ind": "kn5+ngram.lin+induction", "TM_ind": "kn5+union.lin+induction"}
    bits_base = {k: z[v].astype(np.float64) for k, v in base_names.items()}
    for k, v in base_names.items():
        bpb = bits_base[k].sum() / n_bytes
        M(f"base.{k}.bpb_recomputed", float(bpb), source=f"{PROBE.relative_to(ROOT)}/per_token_bits.npz[{v}]")
    log("bases: " + ", ".join(f"{k} {bits_base[k].sum() / n_bytes:.4f}" for k in base_names))

    # smoke subset: first k validation documents
    keep_va = np.ones(n_va, bool)
    if args.val_docs > 0:
        keep_va = doc_va < args.val_docs
    fold = (doc_va % 2).astype(np.int64)

    # static tables
    c2 = km[2] - (V + 16)
    c3 = km[3] - (V + 272)
    assert c2.min() >= 0 and c2.max() < 256 and c3.min() >= 0 and c3.max() < 4096
    rank = np.empty(V, np.int64)
    rank[np.argsort(-train_counts_full, kind="stable")] = np.arange(V)
    fcls = np.select([rank < 64, rank < 256, rank < 1024, rank < 4096], [0, 1, 2, 3], 4).astype(np.int64)
    maxd = int(max(np.diff(ds_tr).max(), np.diff(ds_va).max())) + 2
    dbt = np.minimum(np.floor(np.log2(np.maximum(np.arange(maxd), 1))).astype(np.int64), DBMAX)

    # ------------------------------------------------------------------ A. copy coverage
    t0 = time.time()
    posv = np.arange(n_va)
    _, prev_gold = group_rank_prev(doc_va * V + tok_va, posv)
    dist_gold = np.where(prev_gold >= 0, posv - prev_gold, np.iinfo(np.int64).max)
    for D in (8, 32, 128, 512, 2048, 8192):
        M(f"A.gold_in_doc_history_within_{D}", float((dist_gold <= D)[keep_va].mean()))
    copyable = prev_gold >= 0
    M("A.gold_in_doc_history_any", float(copyable[keep_va].mean()))
    nonstop = fcls[tok_va] > 0
    M("A.gold_in_doc_history_any_nonstop_gold", float(copyable[keep_va & nonstop].mean()),
      note="gold outside the 64 most frequent tokens")
    M("A.share_nonstop_gold", float(nonstop[keep_va].mean()))
    # n-gram-keyed induction candidates (most recent earlier in-document position with the same left-k context)
    # training context sets for k = 1..5 (counting positions), same EDGE padding
    hv, htr = {}, {}
    cand_eq = {}
    for k in range(1, 9):
        hv[k] = ctx_hash(tok_va, lo_va, k)
        _, prev = group_rank_prev((hv[k] ^ mix64(doc_va.astype(np.uint64))).astype(np.uint64), posv)
        cand_eq[k] = (prev >= 0) & (tok_va[np.maximum(prev, 0)] == tok_va)
        cand_eq[k] &= prev >= 0
        M(f"A.induction_unrestricted_k{k}_has_candidate", float((prev >= 0)[keep_va].mean()))
        M(f"A.induction_unrestricted_k{k}_candidate_is_gold", float(cand_eq[k][keep_va].mean()))
        if k <= 5:
            htr[k] = np.unique(ctx_hash(tok_tr, lo_tr, k)[cnt_pos])
    seen = {k: np.isin(hv[k], htr[k]) for k in range(1, 6)}
    any_probe = np.zeros(n_va, bool)
    any_unres5 = np.zeros(n_va, bool)
    any_unres8 = np.zeros(n_va, bool)
    for k in range(1, 9):
        any_unres8 |= cand_eq[k]
        if k <= 5:
            any_unres5 |= cand_eq[k]
            any_probe |= cand_eq[k] & seen[k]
    M("A.induction_probe_style_any_candidate_is_gold", float(any_probe[keep_va].mean()),
      note="k = 1..5, context seen in the counting documents (as in the causal probe's n-gram arm)")
    M("A.induction_unrestricted_k1to5_any_candidate_is_gold", float(any_unres5[keep_va].mean()))
    M("A.induction_unrestricted_k1to8_any_candidate_is_gold", float(any_unres8[keep_va].mean()))
    longest = np.zeros(n_va, np.int64)
    for k in range(1, 6):
        longest = np.where(seen[k], k, longest)
    M("A.longest_seen_left_context_histogram", np.bincount(longest[keep_va], minlength=6).tolist(),
      note="longest k in 0..5 whose left-k context occurs in the counting documents")
    log(f"A coverage ({time.time() - t0:.0f} s)")
    flush()

    # ------------------------------------------------------------------ B. document n-gram cache at the gold
    t0 = time.time()
    cache_p = {}
    cache_n = {}
    for k in range(0, 5):
        hk = ctx_hash(tok_va, lo_va, k) if k > 0 else np.zeros(n_va, np.uint64)
        key_c = (hk ^ mix64(doc_va.astype(np.uint64) + np.uint64(7))).astype(np.uint64)
        r_c, _ = group_rank_prev(key_c, posv)
        with np.errstate(over="ignore"):
            key_cy = mix64(key_c ^ mix64(tok_va.astype(np.uint64) + np.uint64(99)))
        r_cy, _ = group_rank_prev(key_cy, posv)
        cache_n[k] = r_c
        cache_p[k] = np.where(r_c > 0, r_cy / np.maximum(r_c, 1), 0.0)
    log(f"B document n-gram cache ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ B. pair table on training pairs
    t0 = time.time()
    n_cells = (LMAX + 1) * (LCMAX + 1) * (DBMAX + 1) * NFC * (NFC + 1)
    rs = np.random.default_rng(np.random.SeedSequence([20260924, 1]))
    sample = np.sort(rs.choice(cnt_pos, size=min(args.train_pairs_positions, cnt_pos.shape[0]), replace=False))
    hits, tries = k_train_table(tok_tr, c2, fcls, dbt, lo_tr, sample, n_cells)
    M("B.train_pairs", int(tries.sum()), note=f"all earlier in-document q for {sample.shape[0]} sampled counting positions")
    M("B.train_pair_table_s", time.time() - t0)
    shp = (LMAX + 1, LCMAX + 1, DBMAX + 1, NFC, NFC + 1)
    H, Tt = hits.reshape(shp).astype(np.float64), tries.reshape(shp).astype(np.float64)
    # hierarchical smoothing: full cell -> (L, Lc, db) -> (L)
    HL, TL = H.sum(axis=(1, 2, 3, 4)), Tt.sum(axis=(1, 2, 3, 4))
    mL = (HL + 0.5) / (TL + 1.0)
    H3, T3 = H.sum(axis=(3, 4)), Tt.sum(axis=(3, 4))
    m3 = (H3 + 8.0 * mL[:, None, None]) / (T3 + 8.0)
    mfull = (H + 8.0 * m3[:, :, :, None, None]) / (Tt + 8.0)
    tabs = mfull.reshape(-1).copy()
    M("B.match_rate_by_L", mL.tolist(), note="P(tok[q] == tok[p]) by token suffix match length L = 0..6, training pairs")
    M("B.pairs_by_L", TL.tolist())
    # a few interpretable cells: L = 1 by distance bucket; L = 0 by frequency class of tok[q]
    M("B.match_rate_L1_by_log2_distance", ((H[1].sum(axis=(0, 2, 3)) + 0.5) / (Tt[1].sum(axis=(0, 2, 3)) + 1)).tolist())
    M("B.match_rate_L0_by_fclass_q", ((H[0].sum(axis=(0, 1, 3)) + 0.5) / (Tt[0].sum(axis=(0, 1, 3)) + 1)).tolist())
    M("B.match_rate_L1_by_fclass_prev", ((H[1].sum(axis=(0, 1, 2)) + 0.5) / (Tt[1].sum(axis=(0, 1, 2)) + 1)).tolist(),
      note="L = 1, by frequency class of tok[p-1] (0 = 64 most frequent ... 4 = rank 4096+, 5 = edge)")
    log(f"B pair table: {tries.sum()} pairs ({time.time() - t0:.0f} s)")
    flush()

    # ------------------------------------------------------------------ B. learned pointers (training pairs), then validation
    t0 = time.time()
    pos_i, cells_e, na_e, nm_e, n_pos = k_train_hist(tok_tr, c2, fcls, dbt, lo_tr, sample, n_cells)
    M("B.pointer_train_positions_with_gold_in_history", int(n_pos), note=f"of {sample.shape[0]} sampled counting positions")
    M("B.pointer_train_hist_entries", int(pos_i.shape[0]))
    log(f"B pointer histograms: {n_pos} positions, {pos_i.shape[0]} entries ({time.time() - t0:.0f} s)")
    variants = {"ptrL": [("L",)],
                "ptrLdb": [("L",), ("L", "db")],
                "ptrFull": [("L", "Lc"), ("L", "db"), ("L", "fc"), ("L", "rc"), ("L", "db", "fc")]}
    svals = np.zeros((len(variants), n_cells))
    for vi, (vname, groups) in enumerate(variants.items()):
        t1 = time.time()
        th, res = fit_pointer(pos_i, cells_e, na_e, nm_e, n_pos, groups, ridge=1.0, iters=args.ptr_iters)
        svals[vi] = np.exp(th)
        ll = -res.fun
        M(f"B.pointer.{vname}", {"train_mean_log2_ptr_likelihood": float(ll / n_pos / LN2), "nit": int(res.nit),
                                  "converged": bool(res.success), "n_params": int(res.x.shape[0]),
                                  "theta_by_L_mean": [float(th.reshape(shp)[l].mean()) for l in range(LMAX + 1)]},
          note="mean log2 of the pointer probability of the gold on training positions whose gold is in the history (ridge 1.0 included)")
        log(f"B pointer {vname}: {res.x.shape[0]} params, nit {res.nit}, conv {res.success}, "
            f"mean log2 P_ptr(gold) {ll / n_pos / LN2:.3f} ({time.time() - t1:.0f} s)")
    flush()
    t0 = time.time()
    tok_run = tok_va.copy()
    lo_run = lo_va.copy()
    if args.val_docs > 0:
        nkeep = int(ds_va[args.val_docs])
        tok_run, lo_run = tok_va[:nkeep], lo_va[:nkeep]
    pg, shr, ltop, okp = k_val_attn(tok_run, c2, fcls, dbt, lo_run, svals)
    if args.val_docs > 0:
        pad = n_va - tok_run.shape[0]
        pg = np.vstack([pg, np.zeros((pad, pg.shape[1]))])
        shr = np.vstack([shr, np.zeros((pad, shr.shape[1]))])
        ltop = np.vstack([ltop, np.zeros((pad, ltop.shape[1]), np.int64)])
        okp = np.concatenate([okp, np.zeros(pad, np.uint8)])
    for vi, vname in enumerate(variants):
        sel = (okp == 1) & keep_va & copyable
        M(f"B.pointer.{vname}.val_mean_log2_ptr_likelihood_on_copyable", float(np.log2(np.maximum(pg[sel, vi], 1e-300)).mean()))
        M(f"B.pointer.{vname}.val_top_candidate_is_gold", float(((pg[:, vi] >= shr[:, vi]) & (okp == 1))[keep_va].mean()),
          label="unverified", note="proxy: P_ptr(gold) >= top share; exact only when the top cell is unique")
    M("B.val_pointer_pass_s", time.time() - t0)
    log(f"B pointer pass on validation ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ D. class and skip templates on the counting positions
    t0 = time.time()

    def feats(tok, lo):
        t = {i: shifted(tok, lo, i, EDGE_T) for i in (1, 2, 3)}
        g2 = {i: shifted(c2[tok], lo, i, EDGE_C2) for i in (1, 2, 3)}
        g3 = {i: shifted(c3[tok], lo, i, EDGE_C3) for i in (1, 2, 3)}
        return {
            "c2_1": g2[1],
            "c2_21": g2[2] * 512 + g2[1],
            "c3_321": (g3[3] * 8192 + g3[2]) * 8192 + g3[1],
            "t1_c2_2": t[1] * 512 + g2[2],
            "t2_skip": t[2],
            "t32_skip": t[3] * 8192 + t[2],
            "t21_c2_3": (t[2] * 8192 + t[1]) * 512 + g2[3],
        }

    ftr = feats(tok_tr, lo_tr)
    fva = feats(tok_va, lo_va)
    tmpl_p, tmpl_n = {}, {}
    for name in ftr:
        ktr = ftr[name][cnt_pos] * 8192 + tok_tr[cnt_pos]
        u, cnt = np.unique(ktr, return_counts=True)
        cu, cN = np.unique(ftr[name][cnt_pos], return_counts=True)
        kv = fva[name] * 8192 + tok_va
        j = np.searchsorted(u, kv)
        jc = np.minimum(j, u.shape[0] - 1)
        cy = np.where(u[jc] == kv, cnt[jc], 0)
        j2 = np.searchsorted(cu, fva[name])
        j2c = np.minimum(j2, cu.shape[0] - 1)
        Nc = np.where(cu[j2c] == fva[name], cN[j2c], 0)
        tmpl_p[name] = np.where(Nc > 0, cy / np.maximum(Nc, 1), 0.0)
        tmpl_n[name] = Nc
        M(f"D.template.{name}.contexts", int(cu.shape[0]))
    log(f"D templates ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ component builders
    def comp_cache():
        rows, b, p = [], [], []
        for k in range(5):
            act = (cache_n[k] > 0) & keep_va
            r = np.flatnonzero(act)
            bk = k * 13 + np.minimum(np.floor(np.log2(np.maximum(cache_n[k][r], 1))).astype(np.int64), 12)
            rows.append(r); b.append(bk); p.append(cache_p[k][r])
        return (np.concatenate(rows), np.concatenate(b), np.concatenate(p), 5 * 13)

    def comp_attn(v):
        act = (okp == 1) & keep_va
        r = np.flatnonzero(act)
        bk = np.minimum((shr[r, v] * 10).astype(np.int64), 9) * (LMAX + 1) + ltop[r, v]
        return (r, bk, pg[r, v], 10 * (LMAX + 1))

    def comp_tmpl(names=None):
        rows, b, p = [], [], []
        names = list(tmpl_p) if names is None else names
        for i, name in enumerate(names):
            act = (tmpl_n[name] > 0) & keep_va
            r = np.flatnonzero(act)
            bk = i * 24 + np.minimum(np.floor(np.log2(np.maximum(tmpl_n[name][r], 1))).astype(np.int64), 23)
            rows.append(r); b.append(bk); p.append(tmpl_p[name][r])
        return (np.concatenate(rows), np.concatenate(b), np.concatenate(p), len(names) * 24)

    def comp_oracle(ind):
        r = np.flatnonzero(keep_va)
        return (r, np.zeros(r.shape[0], np.int64), ind[r].astype(np.float64), 1)

    pass
    tokens = int(keep_va.sum())
    bytes_scale = n_bytes / n_va    # bytes per token of the whole split (smoke runs: approximate)
    per_tok: dict[str, np.ndarray] = {}

    def run(name, base_key, comps):
        t1 = time.time()
        pb = np.exp2(-bits_base[base_key])
        bits, info = crossfit(pb, comps, fold, args.iters)
        per_tok[name] = bits
        bt = float(bits[keep_va].mean())
        bpb = float(bits[keep_va].sum() / (tokens * bytes_scale))
        base_bpb = float(bits_base[base_key][keep_va].sum() / (tokens * bytes_scale))
        M(f"row.{name}.bits_per_token", bt, note="out-of-fold (two-fold cross-fitting over even/odd validation documents)")
        M(f"row.{name}.bits_per_byte", bpb, label="derived" if args.val_docs > 0 else "measured")
        M(f"row.{name}.gain_bpb_over_base", base_bpb - bpb, label="derived", base=base_names[base_key], fit=info)
        log(f"row {name:<40} bpb {bpb:.4f} (base {base_bpb:.4f}, gain {base_bpb - bpb:+.4f}) {time.time() - t1:.0f} s "
            + " ".join(f"f{d['fold_fit']}:{d['nit']}{'' if d['converged'] else '!'}" for d in info))
        return bpb

    V_L, V_LDB, V_FULL = 0, 1, 2

    for bk in ("kn5", "noTM", "TM"):
        run(f"{bk}", bk, [])
        run(f"{bk}+ngcache", bk, [comp_cache()])
        run(f"{bk}+attnL", bk, [comp_attn(V_L)])
        run(f"{bk}+attnLdb", bk, [comp_attn(V_LDB)])
        run(f"{bk}+attn", bk, [comp_attn(V_FULL)])
        run(f"{bk}+ngcache+attnL", bk, [comp_cache(), comp_attn(V_L)])
        run(f"{bk}+ngcache+attnLdb", bk, [comp_cache(), comp_attn(V_LDB)])
        run(f"{bk}+ngcache+attn", bk, [comp_cache(), comp_attn(V_FULL)])
        run(f"{bk}+tmpl", bk, [comp_tmpl()])
        run(f"{bk}+ngcache+attn+tmpl", bk, [comp_cache(), comp_attn(V_FULL), comp_tmpl()])
    for bk in ("noTM_noind", "TM_noind"):
        run(f"{bk}+tmpl", bk, [comp_tmpl()])
    flush()
    # gold-dependent oracle bounds (NOT models): mass lam on the gold whenever the indicator holds
    for bk in ("noTM", "TM"):
        for D in (64, 512, 4096):
            run(f"ORACLE.{bk}+copy_if_gold_within_{D}", bk, [comp_oracle(dist_gold <= D)])
        run(f"ORACLE.{bk}+copy_if_gold_in_history", bk, [comp_oracle(copyable)])
        run(f"ORACLE.{bk}+copy_if_gold_is_probe_induction_candidate", bk, [comp_oracle(any_probe)])
        run(f"ORACLE.{bk}+copy_if_gold_is_unrestricted_k1to8_candidate", bk, [comp_oracle(any_unres8)])
    flush()

    # ------------------------------------------------------------------ contrasts with document-block bootstrap
    def contrast(label, a, b):
        d = (per_tok[a] - per_tok[b])[keep_va]
        lo_, hi_ = doc_boot(d, doc_va[keep_va], n_vdocs)
        M(f"contrast.{label}", {"bits_per_token": float(d.mean()), "ci95_bits_per_token": [lo_, hi_],
                                "bits_per_byte": float(d.sum() / (tokens * bytes_scale)), "a": a, "b": b},
          label="derived", note="a minus b, negative = a better; document-block bootstrap, 2000 resamples")
        log(f"contrast {label:<50} {d.mean():+.4f} bits/token [{lo_:+.4f}, {hi_:+.4f}], {d.sum() / (tokens * bytes_scale):+.4f} bpb")

    contrast("TM_share_probe_rows", "TM", "noTM")
    contrast("TM_share_after_ngcache_attn", "TM+ngcache+attn", "noTM+ngcache+attn")
    contrast("TM_share_after_ngcache_attn_tmpl", "TM+ngcache+attn+tmpl", "noTM+ngcache+attn+tmpl")
    contrast("TM_share_after_tmpl", "TM+tmpl", "noTM+tmpl")
    contrast("tmpl_gain_on_noTM", "noTM+tmpl", "noTM")
    contrast("ngcache_gain_on_noTM", "noTM+ngcache", "noTM")
    contrast("attn_gain_on_noTM", "noTM+attn", "noTM")
    contrast("attn_minus_attnL_on_noTM", "noTM+attn", "noTM+attnL")
    contrast("attn_gain_over_ngcache_on_noTM", "noTM+ngcache+attn", "noTM+ngcache")
    contrast("attn_minus_attnL_over_ngcache_on_noTM", "noTM+ngcache+attn", "noTM+ngcache+attnL")
    contrast("attnLdb_minus_attnL_over_ngcache_on_noTM", "noTM+ngcache+attnLdb", "noTM+ngcache+attnL")
    contrast("attn_minus_attnLdb_over_ngcache_on_noTM", "noTM+ngcache+attn", "noTM+ngcache+attnLdb")
    contrast("attnL_gain_over_ngcache_on_noTM", "noTM+ngcache+attnL", "noTM+ngcache")
    contrast("attn_gain_over_ngcache_on_TM", "TM+ngcache+attn", "TM+ngcache")
    contrast("all_on_noTM_vs_TM_probe_best", "noTM+ngcache+attn+tmpl", "TM")
    contrast("all_on_kn5", "kn5+ngcache+attn+tmpl", "kn5")
    flush()

    # ------------------------------------------------------------------ C. where the TM share comes from (saved rows)
    for tag, a, b in (("with_induction_cache", "TM", "noTM"), ("lin_only", "TM_noind", "noTM_noind"),
                      ("with_induction", "TM_ind", "noTM_ind")):
        d = (bits_base[b] - bits_base[a])     # positive = TM clauses help
        tot = float(d[keep_va].sum() / (tokens * bytes_scale))
        out = {"total_gain_bpb": tot}
        for gname, gv, ng in (("longest_seen_context", longest, 6), ("gold_fclass", fcls[tok_va], NFC),
                              ("gold_copyable", copyable.astype(np.int64), 2)):
            s = np.bincount(gv[keep_va], weights=d[keep_va], minlength=ng)
            c = np.bincount(gv[keep_va], minlength=ng)
            out[gname] = {"tokens_share": (c / c.sum()).tolist(),
                          "gain_bpb_contribution": (s / (tokens * bytes_scale)).tolist(),
                          "gain_bits_per_token_in_group": (s / np.maximum(c, 1)).tolist()}
        M(f"C.tm_share_breakdown.{tag}", out, label="derived",
          note=f"per-token bits({base_names[b]}) minus bits({base_names[a]}); groups: longest left context 0..5 seen in the counting docs; gold frequency class 0 = 64 most frequent .. 4 = rank 4096+; gold earlier in the document")
        log(f"C {tag}: total {tot:+.4f} bpb; by longest context "
            + str(np.round(np.array(out['longest_seen_context']['gain_bpb_contribution']), 4).tolist()))
    flush()
    M("wall_s", time.time() - T0)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
