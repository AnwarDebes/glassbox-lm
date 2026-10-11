"""Companion of experiments/tm_levers_feasibility_causal.py (EXPLORATORY, validation only, the test
split is never loaded; CPU, one thread): headroom of long-range TRIGGER information, the part of the
TM's bag literals that a document cache cannot supply.

A trigger expert for token x is the next-token distribution over training positions at which x (a
non-stop token) occurred in the window [p - W, p - 3] of the same document: P_x(y) = C[x, y] / C[x].
This is exactly the count read-out of a one-literal bag clause "x in the left window". The trigger
component averages the experts of the distinct tokens in the window (uniform or IDF weights), with a
cross-only variant that removes each expert's self mass (y = x, the cache part). It is mixed into the
saved base rows of the causal probe together with the document n-gram cache, with the same two-fold
cross-fitted normalized linear mixture as the companion script.

    nice -n 19 .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py --no-gpu-sampler \\
        --name tm_levers_trigger_control_causal_exploratory --config <scratch json> -- \\
        .venv-tm/bin/python -u experiments/tm_levers_trigger_control_causal.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from numba import njit

import tm_levers_feasibility_causal as F
from tm_levers_feasibility_causal import M, PROBE, V, log, flush

W_DEFAULT = 512


@njit(cache=False)
def k_trigger_counts(tok, lo, is_cnt, nonstop, W, V_):
    C = np.zeros((V_, V_), np.int32)
    Cx = np.zeros(V_, np.int64)
    cw = np.zeros(V_, np.int64)
    act = np.empty(V_, np.int64)
    where = np.full(V_, -1, np.int64)
    n_act = 0
    cur_doc_lo = -1
    n = tok.shape[0]
    for p in range(n):
        dl = lo[p]
        if dl != cur_doc_lo:
            for i in range(n_act):
                x = act[i]
                cw[x] = 0
                where[x] = -1
            n_act = 0
            cur_doc_lo = dl
        # window [p - W, p - 3]: add q = p - 3, drop q = p - W - 1
        qa = p - 3
        if qa >= dl and nonstop[tok[qa]] == 1:
            x = tok[qa]
            if cw[x] == 0:
                where[x] = n_act
                act[n_act] = x
                n_act += 1
            cw[x] += 1
        qd = p - W - 1
        if qd >= dl and nonstop[tok[qd]] == 1:
            x = tok[qd]
            cw[x] -= 1
            if cw[x] == 0:
                i = where[x]
                last = act[n_act - 1]
                act[i] = last
                where[last] = i
                n_act -= 1
                where[x] = -1
        if is_cnt[p] == 1:
            y = tok[p]
            for i in range(n_act):
                x = act[i]
                C[x, y] += 1
                Cx[x] += 1
    return C, Cx


@njit(cache=False)
def k_trigger_gold(tok, lo, nonstop, W, C, Cx, kl, topm):
    """Per position: P_trig(gold) for 4 variants: (0) all active experts weighted by KL(P_x || P_uni),
    (1) weighted by KL^4, (2) uniform average of the top-m experts by KL, (3) the same, cross-only (self
    mass removed); plus the number of active trigger tokens. Weights never depend on the gold."""
    n = tok.shape[0]
    V_ = C.shape[0]
    out = np.zeros((n, 4))
    nact_out = np.zeros(n, np.int64)
    cw = np.zeros(V_, np.int64)
    act = np.empty(V_, np.int64)
    where = np.full(V_, -1, np.int64)
    top = np.empty(topm, np.int64)
    topk = np.empty(topm)
    n_act = 0
    cur_doc_lo = -1
    for p in range(n):
        dl = lo[p]
        if dl != cur_doc_lo:
            for i in range(n_act):
                x = act[i]
                cw[x] = 0
                where[x] = -1
            n_act = 0
            cur_doc_lo = dl
        qa = p - 3
        if qa >= dl and nonstop[tok[qa]] == 1:
            x = tok[qa]
            if cw[x] == 0:
                where[x] = n_act
                act[n_act] = x
                n_act += 1
            cw[x] += 1
        qd = p - W - 1
        if qd >= dl and nonstop[tok[qd]] == 1:
            x = tok[qd]
            cw[x] -= 1
            if cw[x] == 0:
                i = where[x]
                last = act[n_act - 1]
                act[i] = last
                where[last] = i
                n_act -= 1
                where[x] = -1
        y = tok[p]
        s1 = 0.0
        w1 = 0.0
        s4 = 0.0
        w4 = 0.0
        nt = 0
        for i in range(n_act):
            x = act[i]
            if Cx[x] == 0:
                continue
            pa = C[x, y] / Cx[x]
            w = kl[x]
            w1 += w
            s1 += w * pa
            w4 += w ** 4
            s4 += (w ** 4) * pa
            # keep the top-m by KL (ties toward the earlier active slot)
            if nt < topm:
                top[nt] = x
                topk[nt] = w
                nt += 1
            else:
                jmin = 0
                for jj in range(1, topm):
                    if topk[jj] < topk[jmin]:
                        jmin = jj
                if w > topk[jmin]:
                    top[jmin] = x
                    topk[jmin] = w
        st = 0.0
        stc = 0.0
        nc = 0
        for jj in range(nt):
            x = top[jj]
            st += C[x, y] / Cx[x]
            den = Cx[x] - C[x, x]
            if den > 0:
                stc += 0.0 if y == x else C[x, y] / den
                nc += 1
        nact_out[p] = n_act
        out[p, 0] = s1 / w1 if w1 > 0 else 0.0
        out[p, 1] = s4 / w4 if w4 > 0 else 0.0
        out[p, 2] = st / nt if nt > 0 else 0.0
        out[p, 3] = stc / nc if nc > 0 else 0.0
    return out, nact_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=5_000_000)
    ap.add_argument("--window", type=int, default=W_DEFAULT)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--topm", type=int, default=8)
    args = ap.parse_args()
    M("args", vars(args))
    from tmlm.data.corpus_store import load_split
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import split_meta

    full = load_split(F.CORPUS, V, "train")
    tr = slice_train(full, args.n_tokens)
    counts_full = np.bincount(np.asarray(full.tok).astype(np.int64), minlength=V)
    del full
    va = load_split(F.CORPUS, V, "validation")
    n_bytes = int(split_meta(V, "validation")[0]["n_bytes_utf8"])
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    tok_va = np.asarray(va.tok).astype(np.int64)
    ds_va = np.asarray(va.doc_start).astype(np.int64)
    n_tr, n_va = tok_tr.shape[0], tok_va.shape[0]
    lo_tr, lo_va = F.doc_lo_of(ds_tr, n_tr), F.doc_lo_of(ds_va, n_va)
    n_doc = ds_tr.shape[0] - 1
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, 1]))
    ho_docs = np.sort(rng.choice(n_doc, size=max(1, int(round(0.05 * n_doc))), replace=False))
    is_cnt = (~np.isin(np.repeat(np.arange(n_doc), np.diff(ds_tr)), ho_docs)).astype(np.uint8)
    rank = np.empty(V, np.int64)
    rank[np.argsort(-counts_full, kind="stable")] = np.arange(V)
    nonstop = (rank >= 64).astype(np.uint8)

    z = np.load(PROBE / "per_token_bits.npz")
    doc_va = z["doc"].astype(np.int64)
    n_vdocs = int(ds_va.shape[0] - 1)
    fold = doc_va % 2
    names = {"noTM": "kn5+ngram.lin+induction+doc_cache", "TM": "kn5+union.lin+induction+doc_cache"}
    bits_base = {k: z[v].astype(np.float64) for k, v in names.items()}

    t0 = time.time()
    C, Cx = k_trigger_counts(tok_tr, lo_tr, is_cnt, nonstop, args.window, V)
    n_cnt = int(is_cnt.sum())
    uni = np.bincount(tok_tr[is_cnt == 1], minlength=V).astype(np.float64)
    pu = (uni + 1.0) / (uni.sum() + V)
    Cf = C.astype(np.float64)
    Px = Cf / np.maximum(Cx, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(Px > 0, Px * np.log2(Px / pu[None, :]), 0.0).sum(axis=1)
    del Cf, Px
    kl = np.where(Cx >= 20, kl, 0.0)
    M("trigger.kl_median_bits_active_vocab", float(np.median(kl[Cx >= 20])),
      note="KL(P(y | x in window) || P_uni) in bits, trigger tokens with at least 20 events")
    M("trigger.count_events", int(Cx.sum()), note=f"(x in window [p-{args.window}, p-3], y) events over the counting positions")
    M("trigger.count_s", time.time() - t0)
    log(f"trigger counts: {Cx.sum()} events ({time.time() - t0:.0f} s)")
    t0 = time.time()
    pt, nact = k_trigger_gold(tok_va, lo_va, nonstop, args.window, C, Cx, kl, args.topm)
    M("trigger.val_mean_active", float(nact.mean()))
    log(f"trigger gold probabilities ({time.time() - t0:.0f} s)")
    del C

    # document n-gram cache orders 0..4 (same as the companion script)
    posv = np.arange(n_va)
    cache_p, cache_n = {}, {}
    for k in range(5):
        hk = F.ctx_hash(tok_va, lo_va, k) if k > 0 else np.zeros(n_va, np.uint64)
        key_c = (hk ^ F.mix64(doc_va.astype(np.uint64) + np.uint64(7))).astype(np.uint64)
        r_c, _ = F.group_rank_prev(key_c, posv)
        with np.errstate(over="ignore"):
            key_cy = F.mix64(key_c ^ F.mix64(tok_va.astype(np.uint64) + np.uint64(99)))
        r_cy, _ = F.group_rank_prev(key_cy, posv)
        cache_n[k] = r_c
        cache_p[k] = np.where(r_c > 0, r_cy / np.maximum(r_c, 1), 0.0)

    def comp_cache():
        rows, b, p = [], [], []
        for k in range(5):
            r = np.flatnonzero(cache_n[k] > 0)
            rows.append(r)
            b.append(k * 13 + np.minimum(np.floor(np.log2(np.maximum(cache_n[k][r], 1))).astype(np.int64), 12))
            p.append(cache_p[k][r])
        return (np.concatenate(rows), np.concatenate(b), np.concatenate(p), 65)

    def comp_trig(v):
        r = np.flatnonzero(nact > 0)
        b = np.minimum(np.floor(np.log2(np.maximum(nact[r], 1))).astype(np.int64), 11)
        return (r, b, pt[r, v], 12)

    per_tok = {}

    def run(name, base, comps):
        pb = np.exp2(-bits_base[base])
        bits, info = F.crossfit(pb, comps, fold, args.iters)
        per_tok[name] = bits
        bpb = float(bits.sum() / n_bytes)
        M(f"row.{name}.bits_per_byte", bpb, note="out-of-fold, two-fold cross-fitting over even/odd validation documents", fit=info)
        log(f"row {name:<32} bpb {bpb:.4f} " + " ".join(f"f{d['fold_fit']}:{d['nit']}{'' if d['converged'] else '!'}" for d in info))

    vnames = ["trig_kl", "trig_kl4", "trig_top8", "trigX_top8"]
    for base in ("noTM", "TM"):
        run(f"{base}", base, [])
        run(f"{base}+ngcache", base, [comp_cache()])
        for v, vn in enumerate(vnames):
            run(f"{base}+ngcache+{vn}", base, [comp_cache(), comp_trig(v)])
        for v, vn in enumerate(vnames):
            run(f"{base}+{vn}", base, [comp_trig(v)])
    flush()

    def contrast(label, a, b):
        d = per_tok[a] - per_tok[b]
        lo_, hi_ = F.doc_boot(d, doc_va, n_vdocs)
        M(f"contrast.{label}", {"bits_per_token": float(d.mean()), "ci95_bits_per_token": [lo_, hi_],
                                "bits_per_byte": float(d.sum() / n_bytes), "a": a, "b": b},
          label="derived", note="a minus b, negative = a better; document-block bootstrap, 2000 resamples")
        log(f"contrast {label:<44} {d.mean():+.4f} bits/token [{lo_:+.4f}, {hi_:+.4f}], {d.sum() / n_bytes:+.4f} bpb")

    for vn in vnames:
        contrast(f"{vn}_gain_over_noTM_ngcache", f"noTM+ngcache+{vn}", "noTM+ngcache")
    contrast("TM_share_after_ngcache", "TM+ngcache", "noTM+ngcache")
    for vn in vnames:
        contrast(f"{vn}_gain_over_noTM", f"noTM+{vn}", "noTM")
        contrast(f"TM_share_after_ngcache_{vn}", f"TM+ngcache+{vn}", f"noTM+ngcache+{vn}")
        contrast(f"TM_share_after_{vn}", f"TM+{vn}", f"noTM+{vn}")
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
