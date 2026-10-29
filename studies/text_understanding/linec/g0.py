#!/usr/bin/env python
"""Line C G0, the arbiter gate (EXPLORATORY, validation only; plan LINE_C_NOTE.md cf36f27a30d5353a, approved by the
decided 13:38Z; rules in LOG.md before any run). CPU only; no Tsetlin machine.

    g0.py --selftest            # synthetic corpus, invariants only (seconds)
    g0.py --smoke               # real data, small: KN on 300 counting documents, few held-out documents
    g0.py                       # the registered G0

Floor (the recipe logged before the run):
- KN-5 = tmlm KNModel(order 5, vocab 8000, modified discounts, unk_interp, pad), fitted with tmlm.cvtm.kn_gpu.fit_kn
  (backend numpy: the same arrays as KNModel.fit) on the train documents whose residue is not in H' = {0, 1, 2}
  (tmlm.cvtm.protocol: residue(i) = mix64(i XOR 0xC0FFEE) mod 100; the thesis study's rule "KN-5 fitted WITHOUT H'").
- Document unigram cache over all earlier tokens of the same document: P_c(w) = n_doc(w) / t (t = the position in the
  document; no cache at t = 0).
- P_f = (1 - lam[m]) P_KN + lam[m] P_c, m = the longest matched order of the position (1-5); lam by EM on all
  positions with t >= 1 of the residue-0 documents.
Candidates: the floor's top 8 (ties: the smaller token id). Rows: one per (position, candidate).
Arbiter rows: 500,000 positions of the residue-1 documents (numpy default_rng(2026), without replacement);
calibration rows: 200,000 positions of the residue-2 documents (default_rng(2027)); scoring: every validation
position. The test split is never loaded.
Arms on the same facts (139 bits per row, names in FACTS): the count twin (keys fixed in the plan), logistic
regression (C in 0.1 / 1 / 10 by calibration log-loss), boosted trees (sklearn HistGradientBoosting, early stopping
on an internal 10% of the arbiter rows); every arm's score is mapped by isotonic regression fitted on calibration.
Mixing: for w in the candidate set C, P'(w) = P_f(C) [(1 - b) P_f(w) / P_f(C) + b q(w) / sum_C q], q = max(trust, 1e-4),
b in 0, 0.05, ..., 1 per arm by calibration bits; P' = P_f outside C.
G0 words: the trees mix at least 2% lower in perplexity than the floor on validation (bits per token) -> G0 PASSED
(the TM arm may run, by its own registration); else STOP (nothing to arbitrate; no TM runs).
Outputs (never overwritten): linec/g0/ (smoke: linec/g0_smoke_<utc>/) (results.json, words.txt, rows_*.npz for the TM stage,
val_bits_per_position.npz with every arm's validation bits),
linec/results_linec.jsonl (one row).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

TM_STUDY = Path(f"{_REPO_ROOT}/studies/tm_language_model")
sys.path.insert(0, str(TM_STUDY / "src"))
HERE = Path(__file__).resolve().parent
TOKENIZER = TM_STUDY / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json"
KIND_MAP = TM_STUDY / "data" / "wikitext103" / "bpe8000" / "train" / "kind_map.npy"
K = 8
ORDER = 5
V = 8000
N_ARB, N_CAL = 500_000, 200_000
GATE = 0.02  # perplexity share
T_P1 = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
T_H = (1, 2, 3, 4, 5, 6, 7, 8)
T_T = (1, 4, 16, 64, 256, 1024, 4096)
T_LP = (-1, -2, -3, -4, -5, -6, -7, -8, -10, -12)
T_RATIO = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)
T_CNT = (2, 3, 5, 10, 30, 100, 1000)
T_DOC = (2, 3, 5, 10, 20)
T_DIST = (1, 2, 5, 10, 20, 50, 100, 500, 2000)
T_UNI = (10, 100, 1000, 10_000, 100_000, 1_000_000)
T_PB = (-1, -2, -3, -4, -6, -8, -11)  # count twin: 8 floor-probability buckets
N_TYPES, N_CLUST = 5, 16


def facts_names() -> list:
    n = [f"top1_p>={x}" for x in T_P1] + [f"entropy>={x}" for x in T_H] + [f"match_order={k}" for k in range(1, 6)]
    n += [f"t>={x}" for x in T_T] + [f"prev_type={k}" for k in range(N_TYPES)] + [f"prev_class={k}" for k in range(N_CLUST)]
    n += [f"rank={k}" for k in range(1, K + 1)] + [f"log2p>={x}" for x in T_LP] + [f"p/top1>={x}" for x in T_RATIO]
    n += [f"seen_order{k}" for k in range(2, 6)] + [f"highest_seen={k}" for k in range(6)]
    n += [f"count_at_highest>={x}" for x in T_CNT] + [f"top_in_C_order{k}" for k in range(2, 6)]
    n += ["in_doc"] + [f"doc_count>={x}" for x in T_DOC] + [f"dist<={x}" for x in T_DIST]
    n += [f"type={k}" for k in range(N_TYPES)] + ["same_as_prev"] + [f"unigram>={x}" for x in T_UNI] + ["cache_raised"]
    n += [f"class={k}" for k in range(N_CLUST)]
    return n


FACTS = facts_names()


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def therm_ge(x: np.ndarray, th) -> np.ndarray:
    return np.stack([x >= t for t in th], -1)


def therm_le(x: np.ndarray, th) -> np.ndarray:
    return np.stack([x <= t for t in th], -1)


def onehot(x: np.ndarray, n: int) -> np.ndarray:
    return x[..., None] == np.arange(n)


def token_types() -> np.ndarray:
    voc = json.loads(TOKENIZER.read_text())["model"]["vocab"]
    s = [""] * V
    for k, i in voc.items():
        s[i] = k
    out = np.full(V, 4, dtype=np.int8)
    for i, x in enumerate(s):
        body = x[1:] if x.startswith("Ġ") else x
        if body.isalpha():
            out[i] = 0 if x.startswith("Ġ") else 1  # word start / continuation
        elif body.isdigit():
            out[i] = 2
        elif body and all(not ch.isalnum() for ch in body) and "Ċ" not in body:
            out[i] = 3  # punctuation
    return out


class Floor:
    """KN-5 + document unigram cache; the per-position work of G0."""

    def __init__(self, kn, lam: np.ndarray, types: np.ndarray, clusters: np.ndarray):
        from tmlm.eval.baselines.ngram import _lookup
        self.kn, self.lam, self.types, self.clusters, self._lookup = kn, lam, types, clusters, _lookup
        self.order = kn.order
        self.uni_raw = kn.raw[1].cnt[: kn.vocab].astype(np.int64)

    def match_order(self, cids: list) -> np.ndarray:
        m = np.ones(cids[2].shape[0] if self.order >= 2 else 0, dtype=np.int64)
        for k in range(2, self.order + 1):
            lv = self.kn.level(k, self.order)
            c = cids[k]
            ok = (c >= 0) & (lv.total[np.where(c >= 0, c, 0)] > 0)
            m = np.where(ok, k, m)
        return m

    def walk(self, tok, doc_start, docs, want, mode, chunk=1024, keep_full=None):
        """Every position of the documents `docs` in order; rows where want[pos] are processed.
        mode 'em': (m, t, pk_true, pc_true); mode 'cand': candidate rows and facts."""
        kn, Vv = self.kn, self.kn.vocab
        out = {k: [] for k in ("pos", "m", "t", "pk", "pc", "C", "pfC", "pft", "pfsum", "keys", "bits", "full")}
        for d in docs:
            s, e = int(doc_start[d]), int(doc_start[d + 1])
            L = e - s
            dt = np.asarray(tok[s:e]).astype(np.int64)
            base = np.zeros(Vv, dtype=np.int32)
            srt = np.argsort(dt, kind="stable")
            dsorted = dt[srt]
            skey = dsorted * (L + 1) + srt
            for t0 in range(0, L, chunk):
                t1 = min(L, t0 + chunk)
                n = t1 - t0
                oh = np.zeros((n, Vv), dtype=np.int32)
                oh[np.arange(n), dt[t0:t1]] = 1
                cum = np.cumsum(oh, axis=0) - oh + base
                base = base + oh.sum(0, dtype=np.int32)
                pos = np.arange(s + t0, s + t1)
                sel = np.flatnonzero(want[pos])
                if sel.shape[0] == 0:
                    continue
                p = pos[sel]
                t = (p - s).astype(np.int64)
                y = dt[t]
                ctx, valid = kn.contexts(tok, doc_start, p)
                cids = kn.context_ids(ctx, valid)
                m = self.match_order(cids)
                tt = np.maximum(t, 1).astype(np.float64)
                lam = np.where(t > 0, self.lam[m], 0.0)
                cnt_rows = cum[sel]
                if mode == "em":
                    pk = kn.prob(ctx, y, valid)
                    pc = np.where(t > 0, cnt_rows[np.arange(sel.shape[0]), y] / tt, 0.0)
                    for k_, v_ in (("pos", p), ("m", m), ("t", t), ("pk", pk), ("pc", pc)):
                        out[k_].append(v_)
                    continue
                D = kn.dist(ctx, valid)
                PF = (1.0 - lam)[:, None] * D + (lam / tt)[:, None] * cnt_rows
                r = np.arange(sel.shape[0])[:, None]
                part = np.sort(np.argpartition(-PF, K - 1, axis=1)[:, :K], axis=1)
                o = np.argsort(-PF[r, part], axis=1, kind="stable")
                C = np.take_along_axis(part, o, axis=1)
                pfC = PF[r, C]
                pft = PF[np.arange(sel.shape[0]), y]
                pkC = D[r, C]
                H = -(PF * np.log2(PF)).sum(1)
                # per-order facts of the candidates
                seen = np.zeros((sel.shape[0], K, self.order + 1), dtype=bool)
                cntk = np.zeros((sel.shape[0], K, self.order + 1), dtype=np.int64)
                topin = np.zeros((sel.shape[0], K, self.order + 1), dtype=bool)
                lv1 = kn.level(1, self.order)
                cntk[:, :, 1] = lv1.cnt[C]
                seen[:, :, 1] = cntk[:, :, 1] > 0
                for k in range(2, self.order + 1):
                    lv = kn.level(k, self.order)
                    c = cids[k]
                    tot = np.where(c >= 0, lv.total[np.where(c >= 0, c, 0)], 0.0)
                    key = (np.where(c >= 0, c, 0).astype(np.uint64)[:, None] << np.uint64(16)) | C.astype(np.uint64)
                    tid = self._lookup(kn.keys[k], key.reshape(-1)).reshape(C.shape)
                    ck = np.where((tid >= 0) & (tot[:, None] > 0), lv.cnt[np.where(tid >= 0, tid, 0)], 0).astype(np.int64)
                    cntk[:, :, k] = ck
                    seen[:, :, k] = ck > 0
                    alpha = np.maximum(ck - lv.discount_of(ck, None), 0.0) / np.where(tot > 0, tot, 1.0)[:, None]
                    topin[:, :, k] = (alpha > 0) & (alpha == alpha.max(1, keepdims=True))
                hs = np.where(seen[:, :, 1], 1, 0)
                for k in range(2, self.order + 1):
                    hs = np.where(seen[:, :, k], k, hs)
                c_at = np.take_along_axis(cntk, hs[:, :, None], axis=2)[:, :, 0]
                # cache facts of the candidates
                dcnt = cnt_rows[r, C]
                q = C * (L + 1) + t[:, None]
                kk = np.searchsorted(skey, q.reshape(-1)).reshape(C.shape)
                prev_ok = (kk > 0) & (dsorted[np.maximum(kk - 1, 0)] == C)
                last = np.where(prev_ok, srt[np.maximum(kk - 1, 0)], -1)
                dist = np.where(last >= 0, t[:, None] - last, 10**9)
                prev = np.where(t > 0, dt[np.maximum(t - 1, 0)], -1)
                logp = np.log2(pfC)
                bits = self.facts(pfC, H, m, t, prev, C, logp, seen, hs, c_at, topin, dcnt, dist, pkC)
                pb = np.zeros(C.shape, dtype=np.int64)
                for thr in T_PB:
                    pb += logp < thr
                cb = np.where(dcnt == 0, 0, np.where(dist <= 64, 1, np.where(dist <= 1024, 2, 3)))
                rank = np.broadcast_to(np.arange(K), C.shape)
                key1 = ((rank * 6 + hs) * 4 + cb) * 8 + pb
                key2 = rank * 8 + pb
                for k_, v_ in (("pos", p), ("m", m), ("t", t), ("C", C), ("pfC", pfC), ("pft", pft),
                               ("pfsum", pfC.sum(1)), ("bits", bits), ("keys", np.stack([key1, key2, rank], -1))):
                    out[k_].append(v_)
                if keep_full is not None:
                    for i in np.flatnonzero(keep_full[p]):
                        out["full"].append((int(p[i]), PF[i].copy()))
        res = {}
        for k_, v_ in out.items():
            if k_ == "full":
                res[k_] = v_
            elif v_:
                res[k_] = np.concatenate(v_)
        return res

    def facts(self, pfC, H, m, t, prev, C, logp, seen, hs, c_at, topin, dcnt, dist, pkC) -> np.ndarray:
        n = C.shape[0]
        P = np.concatenate([therm_ge(pfC[:, 0], T_P1), therm_ge(H, T_H), onehot(m - 1, 5), therm_ge(t, T_T),
                            onehot(np.where(prev >= 0, self.types[np.maximum(prev, 0)], -1), N_TYPES),
                            onehot(np.where(prev >= 0, self.clusters[np.maximum(prev, 0)], -1), N_CLUST)], 1)
        P = np.broadcast_to(P[:, None, :], (n, K, P.shape[1]))
        Cf = np.concatenate([
            np.broadcast_to(np.eye(K, dtype=bool)[None], (n, K, K)),
            therm_ge(logp, T_LP), therm_ge(pfC / pfC[:, :1], T_RATIO),
            seen[:, :, 2:], onehot(hs, 6), therm_ge(c_at, T_CNT), topin[:, :, 2:],
            (dcnt > 0)[..., None], therm_ge(dcnt, T_DOC), therm_le(dist, T_DIST),
            onehot(self.types[C], N_TYPES), (C == prev[:, None])[..., None], therm_ge(self.uni_raw[C], T_UNI),
            (pfC > pkC)[..., None], onehot(self.clusters[C], N_CLUST)], 2)
        B = np.concatenate([P, Cf], 2).astype(np.uint8)
        assert B.shape[2] == len(FACTS), (B.shape, len(FACTS))
        return B.reshape(n * K, -1)


def em_lambda(m, pk, pc, t, n_buckets=5, iters=2000, tol=1e-12) -> tuple:
    lam = np.zeros(n_buckets + 1)
    info = {}
    for b in range(1, n_buckets + 1):
        s = (m == b) & (t > 0)
        if s.sum() == 0:
            info[b] = {"n": 0}
            continue
        a, c = pk[s], pc[s]
        x = 0.1
        for it in range(iters):
            r = x * c / (x * c + (1 - x) * a)
            nx = float(r.mean())
            if abs(nx - x) < tol:
                x = nx
                break
            x = nx
        lam[b] = x
        ll = float(np.log2((1 - x) * a + x * c).sum())
        info[b] = {"n": int(s.sum()), "lambda": x, "iters": it + 1, "log2_lik": ll,
                   "log2_lik_kn_only": float(np.log2(a).sum())}
    return lam, info


def twin_fit(keys, y, A=10.0) -> dict:
    p0 = float(y.mean())
    out = {"p0": p0}
    n3 = np.bincount(keys[:, 2], minlength=K).astype(float)
    t3 = np.bincount(keys[:, 2], weights=y, minlength=K)
    P3 = (t3 + A * p0) / (n3 + A)
    n2 = np.bincount(keys[:, 1], minlength=K * 8).astype(float)
    t2 = np.bincount(keys[:, 1], weights=y, minlength=K * 8)
    P2 = (t2 + A * P3[np.arange(K * 8) // 8]) / (n2 + A)
    n1 = np.bincount(keys[:, 0], minlength=K * 6 * 4 * 8).astype(float)
    t1 = np.bincount(keys[:, 0], weights=y, minlength=K * 6 * 4 * 8)
    k1 = np.arange(K * 6 * 4 * 8)
    parent = (k1 // (6 * 4 * 8)) * 8 + (k1 % 8)
    P1 = (t1 + A * P2[parent]) / (n1 + A)
    out.update({"P1": P1, "cells_seen": int((n1 > 0).sum()), "cells": int(n1.shape[0])})
    return out


def mix_bits(q, pfC, pft, pfsum, in_C, true_rank, b) -> np.ndarray:
    qq = np.maximum(np.asarray(q, dtype=np.float64), 1e-4)
    share = qq / qq.sum(1, keepdims=True)
    tr = np.where(in_C, true_rank, 0)
    st = share[np.arange(q.shape[0]), tr]
    pin = pfsum * ((1 - b) * pft / pfsum + b * st)
    p = np.where(in_C, pin, pft)
    return -np.log2(p)


def best_b(q, rows) -> tuple:
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    vals = [float(mix_bits(q, *rows, b).mean()) for b in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), vals


def full_mix(PF, C, q, b) -> np.ndarray:
    out = PF.copy()
    qq = np.maximum(np.asarray(q, dtype=np.float64), 1e-4)
    s = PF[C].sum()
    out[C] = s * ((1 - b) * PF[C] / s + b * qq / qq.sum())
    return out


def run(args) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from tmlm.cvtm import protocol as P
    from tmlm.cvtm.kn_gpu import fit_kn
    from tmlm.eval.baselines.ngram import KNModel

    t_start = time.time()
    log = []

    def say(*a):
        msg = f"[{utc()} +{time.time() - t_start:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    name = "g0_selftest" if args.selftest else ("g0_smoke_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) if args.smoke else "g0")
    outdir = HERE / name
    if outdir.exists() and not args.selftest:
        sys.exit(f"{outdir} exists; not overwritten")
    if args.selftest:
        rng = np.random.default_rng(0)
        Vv, order = 60, 5
        global V, ORDER
        V, ORDER = Vv, order
        types = rng.integers(0, N_TYPES, Vv).astype(np.int8)
        clusters = rng.integers(0, N_CLUST, Vv).astype(np.int64)
        zipf = 1.0 / np.arange(1, Vv + 1)
        zipf /= zipf.sum()

        def corpus(nd, L):
            docs = []
            for _ in range(nd):
                topic = rng.choice(Vv, 5, replace=False)
                x = rng.choice(Vv, L, p=zipf)
                rep = rng.random(L) < 0.3
                x[rep] = rng.choice(topic, rep.sum())
                docs.append(x)
            ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
            return np.concatenate(docs).astype(np.uint16), ds

        tok_c, ds_c = corpus(60, 300)
        tok_h, ds_h = corpus(30, 200)
        role = np.array([i % 3 for i in range(30)])
        val_tok, val_ds = corpus(6, 250)
        val_bytes = np.full(6, 1000, dtype=np.int64)
        kn = KNModel(order, Vv).fit(tok_c, ds_c)
        n_arb, n_cal = 2000, 1000
    else:
        types = token_types()
        clusters = (np.load(KIND_MAP)[1].astype(np.int64) - V)
        assert clusters.min() == 0 and clusters.max() == N_CLUST - 1
        tok, ds = P.load_train()
        held, rest = P.split_by_residue(tok, ds, (0, 1, 2))
        if args.smoke:
            keep = np.zeros(rest.n_docs, dtype=bool)
            keep[:300] = True
            rest = P.select_docs(rest.tok, rest.doc_start, keep, rest.doc_index)
        say("counting documents", rest.n_docs, "tokens", rest.n_tokens, "| held-out H'", held.n_docs, held.n_tokens)
        kn = fit_kn(rest.tok, rest.doc_start, order=ORDER, vocab=V, backend="numpy")
        say("KN fit", json.dumps({k: v for k, v in kn.fit_info.items() if k in ("backend", "wall_s", "phase_s")})[:300])
        say("types per level", kn.n_types, "discounts", {k: [round(x, 6) for x in v] for k, v in kn.discounts().items()})
        tok_h, ds_h = held.tok, held.doc_start
        role = held.residue
        ev = P.load_eval_split("validation")
        val_tok, val_ds, val_bytes = ev.tok, ev.doc_start, ev.doc_bytes
        n_arb, n_cal = (N_ARB, N_CAL) if not args.smoke else (20_000, 10_000)
        if args.smoke:
            sub = np.zeros(role.shape[0], dtype=bool)
            for r_ in (0, 1, 2):
                sub[np.flatnonzero(role == r_)[:8]] = True
            sel = P.select_docs(tok_h, ds_h, sub, held.doc_index)
            tok_h, ds_h, role = sel.tok, sel.doc_start, sel.residue
            vkeep = np.zeros(ev.n_docs, dtype=bool)
            vkeep[:4] = True
            vs = P.select_docs(val_tok, val_ds, vkeep)
            val_tok, val_ds, val_bytes = vs.tok, vs.doc_start, ev.doc_bytes[:4]
    docs_by_role = {r_: np.flatnonzero(role == r_) for r_ in (0, 1, 2)}
    pos_role = np.repeat(role, np.diff(ds_h))

    # (1) the cache weights by EM on the residue-0 documents
    fl = Floor(kn, np.zeros(ORDER + 1), types, clusters)
    em = fl.walk(tok_h, ds_h, docs_by_role[0], pos_role == 0, "em")
    lam, lam_info = em_lambda(em["m"], em["pk"], em["pc"], em["t"], n_buckets=ORDER)
    fl.lam = lam
    say("lambda per longest-match bucket", [round(x, 6) for x in lam[1:]])

    # (2) rows: arbiter (residue 1), calibration (residue 2), validation (all)
    def sample(r_, n, seed):
        idx = np.flatnonzero(pos_role == r_)
        if idx.shape[0] > n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
        w = np.zeros(pos_role.shape[0], dtype=bool)
        w[idx] = True
        return w

    arb = fl.walk(tok_h, ds_h, docs_by_role[1], sample(1, n_arb, 2026), "cand")
    cal = fl.walk(tok_h, ds_h, docs_by_role[2], sample(2, n_cal, 2027), "cand")
    n_val = int(val_ds[-1])
    chk = np.zeros(n_val, dtype=bool)
    chk[np.random.default_rng(7).choice(n_val, min(1000, n_val), replace=False)] = True
    val = fl.walk(val_tok, val_ds, np.arange(val_ds.shape[0] - 1), np.ones(n_val, dtype=bool), "cand", keep_full=chk)
    say("rows: arbiter", arb["bits"].shape, "calibration", cal["bits"].shape, "validation", val["bits"].shape)
    kn_info = {"n_types": [int(x) for x in kn.n_types], "n_train_tokens": int(kn.n_train_tokens),
               "discounts": {str(k): [float(x) for x in v] for k, v in kn.discounts().items()}}
    fl = None
    del kn
    import gc
    gc.collect()  # the KN arrays are not needed after the walks

    def labels(blk, tk):
        y = (blk["C"] == np.asarray(tk).astype(np.int64)[blk["pos"]][:, None])
        in_C = y.any(1)
        tr = np.where(in_C, y.argmax(1), -1)
        return y, in_C, tr

    yA, inA, trA = labels(arb, tok_h)
    yC, inC, trC = labels(cal, tok_h)
    yV, inV, trV = labels(val, val_tok)
    assert val["pos"].shape[0] == n_val and (np.diff(val["pos"]) == 1).all()

    def rowsof(blk, in_C, tr):
        return (blk["pfC"], blk["pft"], blk["pfsum"], in_C, tr)

    RA, RC, RV = rowsof(arb, inA, trA), rowsof(cal, inC, trC), rowsof(val, inV, trV)
    floor_bits = -np.log2(val["pft"])
    oracle_bits = -np.log2(np.where(inV, val["pfsum"], val["pft"]))
    res = {"study": "Line C G0 (exploratory, validation)", "utc_start": utc(), "mode": name,
           "lambda": [float(x) for x in lam[1:]], "lambda_em": {str(k): v for k, v in lam_info.items()},
           "rows": {"arbiter": int(yA.size), "calibration": int(yC.size), "validation": int(yV.size)},
           "recall_at_8": {"arbiter": float(inA.mean()), "calibration": float(inC.mean()), "validation": float(inV.mean())},
           "facts": len(FACTS), "kn": kn_info}

    # (3) the arms
    XA = arb["bits"]
    XC = cal["bits"]
    XV = val["bits"]
    ya, yc = yA.reshape(-1).astype(np.int64), yC.reshape(-1).astype(np.int64)
    raw = {}
    tw = twin_fit(arb["keys"].reshape(-1, 3), ya)
    raw["count_twin"] = (tw["P1"][cal["keys"].reshape(-1, 3)[:, 0]], tw["P1"][val["keys"].reshape(-1, 3)[:, 0]])
    res["count_twin"] = {"cells": tw["cells"], "cells_seen": tw["cells_seen"], "p0": tw["p0"], "dirichlet": 10.0}
    best = None
    for c in (0.1, 1.0, 10.0):
        lr = LogisticRegression(C=c, max_iter=3000).fit(XA.astype(np.float32), ya)
        pc_ = lr.predict_proba(XC.astype(np.float32))[:, 1]
        ll = float(-np.mean(yc * np.log(np.maximum(pc_, 1e-12)) + (1 - yc) * np.log(np.maximum(1 - pc_, 1e-12))))
        say("logistic regression C", c, "calibration log-loss", round(ll, 6))
        if best is None or ll < best[0]:
            best = (ll, c, pc_, lr.predict_proba(XV.astype(np.float32))[:, 1])
    raw["logistic"] = (best[2], best[3])
    res["logistic"] = {"C": best[1], "calibration_logloss": best[0]}
    gb = HistGradientBoostingClassifier(max_iter=500 if not (args.smoke or args.selftest) else 50, learning_rate=0.1,
                                        max_leaf_nodes=63, min_samples_leaf=100, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    gb.fit(XA, ya)
    raw["trees"] = (gb.predict_proba(XC)[:, 1], gb.predict_proba(XV)[:, 1])
    res["trees"] = {"n_iter": int(gb.n_iter_)}
    say("trees iterations", gb.n_iter_)

    # (4) isotonic calibration, b, validation bits
    out_bits = {"floor": floor_bits, "oracle": oracle_bits}
    for arm, (sc, sv) in raw.items():
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(sc, yc)
        sc, sv = np.asarray(sc, dtype=np.float64), np.asarray(sv, dtype=np.float64)
        qc, qv = iso.predict(sc).reshape(-1, K), iso.predict(sv).reshape(-1, K)
        b, curve = best_b(qc, RC)
        out_bits[arm] = mix_bits(qv, *RV, b)
        res[arm].update({"b": b, "calibration_bits_per_token_at_b": float(min(curve)),
                         "calibration_bits_floor": float(np.mean(-np.log2(cal["pft"])))})
        # the contract: full vectors sum to 1, are positive, and agree with the fast path
        worst = 0.0
        pos_index = {int(p_): i for i, p_ in enumerate(val["pos"])}
        for p_, PF in val["full"]:
            i = pos_index[p_]
            f = full_mix(PF, val["C"][i], qv[i], b)
            assert abs(f.sum() - 1.0) < 1e-6 and f.min() > 0, (arm, p_, f.sum(), f.min())
            worst = max(worst, abs(-np.log2(f[int(np.asarray(val_tok)[p_])]) - out_bits[arm][i]))
        assert worst < 1e-9, (arm, worst)
        res[arm]["contract_checked_positions"] = len(val["full"])
    n_bytes = int(np.asarray(val_bytes).sum())
    doc_of = np.repeat(np.arange(val_ds.shape[0] - 1), np.diff(val_ds))
    idx = P.bootstrap_index(val_ds.shape[0] - 1)
    summary = {}
    for arm, bv in out_bits.items():
        db = np.bincount(doc_of, weights=bv, minlength=val_ds.shape[0] - 1)
        e = {"bits_per_token": float(bv.mean()), "bits_per_byte": float(bv.sum() / n_bytes),
             "perplexity_per_token": float(2 ** bv.mean())}
        if arm != "floor":
            e["gain_vs_floor_perplexity_share"] = float(1 - 2 ** (bv.mean() - floor_bits.mean()))
            dbf = np.bincount(doc_of, weights=floor_bits, minlength=val_ds.shape[0] - 1)
            e["bpb_minus_floor"] = P.bootstrap_bpb(db, val_bytes, idx, dbf)
        summary[arm] = e
    res["validation"] = summary
    gain = summary["trees"]["gain_vs_floor_perplexity_share"]
    word = "G0 PASSED" if gain >= GATE - 1e-12 else "STOP"
    lines = [f"Line C G0 ({name}): floor {summary['floor']['bits_per_token']:.4f} bits per token "
             f"({summary['floor']['bits_per_byte']:.4f} per byte); recall@8 {res['recall_at_8']['validation']:.4f}"]
    for arm in ("oracle", "count_twin", "logistic", "trees"):
        s_ = summary[arm]
        ci = s_["bpb_minus_floor"]["ci95"]
        lines.append(f"  {arm}: {s_['bits_per_token']:.4f} bits per token, perplexity {100 * s_['gain_vs_floor_perplexity_share']:+.2f}% "
                     f"lower than the floor; bits per byte minus floor {s_['bpb_minus_floor']['point']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                     + (f"; b {res[arm]['b']}" if arm in res else ""))
    lines.append(f"words: boosted trees {100 * gain:.2f}% lower perplexity than the floor (gate {100 * GATE:.0f}%) -> {word}")
    res["words"] = word
    res["utc_end"] = utc()
    res["seconds"] = round(time.time() - t_start, 1)
    res["sha256"] = {"g0.py": sha16(Path(__file__).resolve())}
    txt = "\n".join(lines)
    print(txt, flush=True)
    if args.selftest:
        # invariants: b = 0 is the floor; a perfect arbiter with b = 1 is the oracle
        z = mix_bits(np.ones((yV.shape[0], K)), *RV, 0.0)
        assert np.allclose(z, floor_bits), "b = 0 must give the floor"
        perfect = mix_bits(yV.astype(float) + 0.0, *RV, 1.0)
        miss = ~inV
        assert np.allclose(perfect[miss], floor_bits[miss])
        # perfect q has a 1e-4 floor on the others, so it is a hair above the oracle where the true token is in C
        assert (perfect[inV] >= oracle_bits[inV] - 1e-12).all() and (perfect[inV] - oracle_bits[inV] < 1e-2).all()
        assert XA.shape[1] == len(FACTS) and set(np.unique(XA)) <= {0, 1}
        print("SELFTEST OK", flush=True)
        return
    outdir.mkdir(parents=True)
    (outdir / "results.json").write_text(json.dumps(res, indent=1))
    (outdir / "words.txt").write_text(txt + "\n")
    (outdir / "log.txt").write_text("\n".join(log) + "\n")
    for nm, blk, y, in_C, tr in (("arbiter", arb, yA, inA, trA), ("calibration", cal, yC, inC, trC),
                                 ("validation", val, yV, inV, trV)):
        np.savez_compressed(outdir / f"rows_{nm}.npz", bits=np.packbits(blk["bits"], axis=1), n_facts=len(FACTS),
                            y=y, in_C=in_C, true_rank=tr, pfC=blk["pfC"], pft=blk["pft"], pfsum=blk["pfsum"],
                            pos=blk["pos"], C=blk["C"], keys=blk["keys"])
    (outdir / "facts.json").write_text(json.dumps(FACTS))
    # per-position validation bits of every arm, for paired comparisons with a later TM arm
    np.savez_compressed(outdir / "val_bits_per_position.npz", doc_of=doc_of, **out_bits)
    with open(HERE / "results_linec.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in res.items()}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
