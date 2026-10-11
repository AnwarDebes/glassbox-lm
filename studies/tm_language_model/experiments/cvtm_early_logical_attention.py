"""Early go/no-go measurement (EXPLORATORY, not pre-registered; validation split only, the test split is
never loaded): does a Tsetlin machine "logical attention" gate, which decides which earlier token of the
document to copy from Boolean pair literals that include REFLECTIVE literals (the base model's own
expectations), add accuracy that plain count caches cannot?

Setting (the same data as experiments/tm_levers_feasibility_causal.py, results/20260924T021631Z_...):
  * the first 5,000,000 tokens of WikiText-103 train at V 8000, cut at a document boundary; 5 percent of
    its documents held out with SeedSequence([20260923, 715, 1]); KN-5 (modified, in-house
    src/tmlm/eval/baselines/ngram.py) fitted on the other ("counting") documents, as in the causal probe;
  * the held-out documents are split once more by document (SeedSequence([20260924, 4242, 1])):
    part A trains the gates (pairs), part C calibrates (LR L2 and TM T, s and epoch selection, the softmax
    temperature, the mixture weights). Validation is never used to train or select a gate.
  * base B = KN-5 + document n-gram cache (orders 0..4, P(y | left-k context) from earlier positions of the
    same document), normalized linear mixture with bucket weights (order x log2 count), the "kn5+ngcache"
    row of the feasibility run (1.4966 bpb there, weights cross-fitted on validation halves).

Candidates: for position p, every distinct token that occurred earlier in the same document, capped at the
256 most recent distinct tokens (coverage lost by the cap is reported). Label: candidate == tok[p].

Literals (237 Boolean, thermometer-coded from 29 bucketed pair features; tmu appends the negations):
  exact and cluster-level (kind_map kinds 1, 2, 3 = 16, 256, 4096 clusters) suffix-match lengths between
  the left context of p and the left context of the MOST RECENT occurrence and the BEST occurrence of the
  candidate; distance of the most recent and of the best exact-match occurrence; recency rank among the
  distinct tokens; occurrence counts (all, last 64 tokens, with suffix match >= 1 and >= 2); offset of the
  first occurrence in the document; global frequency class; BPE word-start / capitalised / continuation /
  punctuation / digit / newline flags of the candidate and of tok[p-1] (from the tokenizer vocabulary);
  document depth; and the REFLECTIVE literals: P_B(candidate), its rank under B, P_KN5(candidate), B's
  mass on the candidate's 256- and 16-cluster, P_B(candidate) / P_B(top-1), the longest KN-5 context
  order k* seen in the counting documents, P_B(top-1), and the longest document-cache order that fired.

Gates (same training pairs, same literals, same read-out):
  (a) LR: logistic regression on the literals (L2 chosen on C);
  (b) TM: tmu TMClassifier, 2 classes, CPU (tmu from upstream 5605ff07; its CUDA path is broken), weighted
      clauses, T and s (and the epoch) chosen on C.
  Pairs: every positive pair of the A positions plus uniformly sampled negatives (fixed ratio).
  Read-out: score(c) = LR logit or the TM's class-1 vote sum (weights x clause outputs, unclipped; empty
  clauses output 0), P_copy(c) = softmax(score / tau) over the candidates of p, tau fitted on C.
  P_copy enters the mixture as one more component with bucket weights (top share decile x top-score
  sextile), fitted on C together with B's cache weights (protocol P1, the primary). Protocol P2 refits the
  same mixtures by two-fold cross-fitting over even / odd validation documents (the feasibility run's
  protocol, comparable with its 1.4966).

Report: validation bits per byte with document-bootstrap CIs (2,000 resamples, ratio of summed bits to
summed bytes per resample): B, B + LR gate, B + TM gate, TM minus LR; ablations without the reflective
literals; the 10 highest-weight TM clauses as readable rules. GO criterion (not
pre-registered): TM gate gain >= 0.01 bpb over B with the CI excluding 0, AND TM gate not worse than the LR
gate by more than 0.002 bpb.

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMLM_CPU_SLOTS=1
    source env/env.sh
    nice -n 19 .venv-tm/bin/python scripts/mem_guard.py --limit-gib 30 -- scripts/run_logged.py --no-gpu-sampler \\
        --name cvtm_early_logical_attention_exploratory --config experiments/cvtm/configs/cvtm_early_logical_attention.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_early_logical_attention.py
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

import tm_levers_feasibility_causal as R1  # noqa: E402  (helpers reused unchanged: ctx_hash, mix64, group_rank_prev, fit_mix, eval_mix, crossfit)

PROBE = ROOT / "results" / "20260924T012544Z_ckm_probe_causal_exploratory"
R1_RUN = "results/20260924T021631Z_tm_levers_feasibility_causal_exploratory"
CORPUS = "wikitext103"
V = 8000
CAP_DEFAULT = 256
LMAX, LCMAX = 8, 6
LN2 = math.log(2.0)
T0 = time.time()
METRICS: dict = {}

# raw pair features (uint8 columns)
FEATS = ["L_recent", "L_best", "Lk1_recent", "Lk1_best", "Lk2_recent", "Lk2_best", "Lk3_recent", "Lk3_best",
         "dist_recent", "dist_best", "recency_rank", "n_occ", "n_occ_last64", "n_occ_L1", "n_occ_L2", "first_offset",
         "cand_fclass", "cand_type", "pB", "rankB", "pKN5", "clusterB256", "clusterB16", "ratio_top", "kstar",
         "top1B", "prev_type", "depth", "cache_order"]
F = {n: i for i, n in enumerate(FEATS)}
NF = len(FEATS)
REFLECTIVE = {"pB", "rankB", "pKN5", "clusterB256", "clusterB16", "ratio_top", "kstar", "top1B", "cache_order"}
RANK_EDGES = np.array([0, 1, 2, 4, 9, 19, 49, 99, 499], np.int64)
TYPE_BITS = ["word_start", "capitalised", "continuation", "punct", "digit", "newline"]


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


# ------------------------------------------------------------------ literal specification
def literal_spec():
    """(name, feature index, kind, threshold): literal = raw >= t ('ge'), raw <= t ('le'), bit t set ('bit')."""
    out = []

    def add(feat, kind, ts, fmt):
        for t in ts:
            out.append((fmt(t), F[feat], kind, int(t)))

    add("L_recent", "ge", range(1, LMAX + 1), lambda t: f"exact suffix match at most recent occurrence >= {t}")
    add("L_best", "ge", range(1, LMAX + 1), lambda t: f"exact suffix match at best occurrence >= {t}")
    for k, nc in ((1, 16), (2, 256), (3, 4096)):
        add(f"Lk{k}_recent", "ge", range(1, LCMAX + 1), lambda t, nc=nc: f"{nc}-cluster suffix match at most recent occurrence >= {t}")
        add(f"Lk{k}_best", "ge", range(1, LCMAX + 1), lambda t, nc=nc: f"{nc}-cluster suffix match at best occurrence >= {t}")
    add("dist_recent", "le", range(0, 14), lambda t: f"distance to most recent occurrence < {2 ** (t + 1)}")
    add("dist_best", "le", range(0, 14), lambda t: f"distance to best exact-match occurrence < {2 ** (t + 1)}")
    add("recency_rank", "le", range(0, 8), lambda t: f"among the {2 ** (t + 1) - 1} most recent distinct tokens")
    add("n_occ", "ge", range(1, 8), lambda t: f"occurrences in document >= {2 ** t}")
    add("n_occ_last64", "ge", range(1, 5), lambda t: f"occurrences in last 64 tokens >= {[0, 1, 2, 3, 5][t]}")
    add("n_occ_L1", "ge", range(1, 6), lambda t: f"occurrences after the same previous token >= {2 ** t - 1}")
    add("n_occ_L2", "ge", range(1, 5), lambda t: f"occurrences after the same previous 2 tokens >= {2 ** t - 1}")
    add("first_offset", "le", range(2, 14), lambda t: f"first occurrence within the first {2 ** (t + 1) - 1} document tokens")
    add("cand_fclass", "le", range(0, 5), lambda t: f"candidate among the {[16, 64, 256, 1024, 4096][t]} most frequent train tokens")
    add("cand_type", "bit", range(0, 6), lambda t: f"candidate is {TYPE_BITS[t]}")
    add("pB", "le", range(0, 16), lambda t: f"P_B(candidate) > 2^-{t + 1}")
    add("rankB", "le", range(0, 9), lambda t: f"candidate in B's top {RANK_EDGES[t] + 1}")
    add("pKN5", "le", range(0, 16), lambda t: f"P_KN5(candidate) > 2^-{t + 1}")
    add("clusterB256", "le", range(0, 12), lambda t: f"B mass on candidate's 256-cluster > 2^-{t + 1}")
    add("clusterB16", "le", range(0, 8), lambda t: f"B mass on candidate's 16-cluster > 2^-{t + 1}")
    add("ratio_top", "le", range(0, 12), lambda t: f"P_B(candidate) > 2^-{t + 1} x P_B(top-1)")
    add("kstar", "ge", range(1, 5), lambda t: f"KN-5 matched a context of >= {t} tokens")
    add("top1B", "le", range(0, 8), lambda t: f"P_B(top-1) > 2^-{t + 1}")
    add("prev_type", "bit", range(0, 6), lambda t: f"previous token is {TYPE_BITS[t]}")
    add("depth", "le", range(3, 14), lambda t: f"position within the first {2 ** (t + 1) - 1} document tokens")
    add("cache_order", "ge", range(1, 5), lambda t: f"document cache matched a context of >= {t} tokens")
    return out


def literal_tables(spec):
    n = len(spec)
    lit_feat = np.array([s[1] for s in spec], np.int64)
    LT = np.zeros((n, 256), np.bool_)
    v = np.arange(256)
    for j, (_, f, kind, t) in enumerate(spec):
        if kind == "ge":
            LT[j] = v >= t
        elif kind == "le":
            LT[j] = v <= t
        else:
            LT[j] = ((v >> t) & 1) == 1
    return lit_feat, LT


def token_types(vocab_path: Path) -> np.ndarray:
    """uint8 [V] bit field from the byte-level BPE vocabulary (G-dot prefix = leading space, C-dot = newline)."""
    t = json.loads(vocab_path.read_text())
    voc = t["model"]["vocab"]
    assert len(voc) == V
    out = np.zeros(V, np.uint8)
    for s, i in voc.items():
        ws = s.startswith("Ġ")
        body = s[1:] if ws else s
        bits = 0
        if ws:
            bits |= 1
        first_alpha = next((ch for ch in body if ch.isalpha()), None)
        if first_alpha is not None and body[:1].isalpha() and first_alpha.isupper():
            bits |= 2
        if (not ws) and body[:1].isalnum():
            bits |= 4
        if not any(ch.isalnum() for ch in body) and "Ċ" not in body:
            bits |= 8
        if any(ch.isdigit() for ch in body):
            bits |= 16
        if "Ċ" in s:
            bits |= 32
        out[i] = bits
    return out


# ------------------------------------------------------------------ pair kernel
@njit(cache=False)
def _bl(x):
    """floor(log2(x)) for integer x >= 1."""
    b = 0
    while (x >> (b + 1)) > 0:
        b += 1
    return b


@njit(cache=False)
def _nlog2b(x, cap):
    """min(floor(-log2(x)), cap) for 0 < x (values above 1 by rounding give 0)."""
    if x >= 1.0:
        return 0
    v = int(math.floor(-math.log2(x)))
    if v < 0:
        v = 0
    return v if v < cap else cap


@njit(cache=False)
def k_pairs(P, pos, tok, lo, prevs, ncache, wts, c1, c2, c3, fcls, ctype, kstar, rank_edges, cap,
            feat, cand, lab, ncand, pstat):
    """For every position of the chunk: build the full base row B (KN-5 row P[i] + document cache orders
    0..4 with bucket weights, normalized), enumerate the <= cap most recent distinct earlier in-document
    tokens and write their raw pair features at rows i * cap + s."""
    Vv = P.shape[1]
    nch = pos.shape[0]
    b = np.empty(Vv)
    slot = np.full(Vv, -1, np.int64)
    cl = np.empty(cap, np.int64)
    Lr = np.zeros(cap, np.int64)
    Lb = np.zeros(cap, np.int64)
    Lr1 = np.zeros(cap, np.int64)
    Lb1 = np.zeros(cap, np.int64)
    Lr2 = np.zeros(cap, np.int64)
    Lb2 = np.zeros(cap, np.int64)
    Lr3 = np.zeros(cap, np.int64)
    Lb3 = np.zeros(cap, np.int64)
    dr = np.zeros(cap, np.int64)
    dbs = np.zeros(cap, np.int64)
    nocc = np.zeros(cap, np.int64)
    n64 = np.zeros(cap, np.int64)
    nl1 = np.zeros(cap, np.int64)
    nl2 = np.zeros(cap, np.int64)
    fo = np.zeros(cap, np.int64)
    cm1 = np.zeros(16)
    cm2 = np.zeros(256)
    for i in range(nch):
        p = pos[i]
        dl = lo[p]
        y = tok[p]
        for v in range(Vv):
            b[v] = P[i, v]
        wsum = 0.0
        n0 = p - dl
        inc0 = 0.0
        if n0 > 0:
            bk = _bl(n0)
            if bk > 12:
                bk = 12
            w0 = wts[bk]
            wsum += w0
            inc0 = w0 / n0
        nc = 0
        ghist = 0
        for q in range(p - 1, dl - 1, -1):
            t = tok[q]
            b[t] += inc0
            if t == y:
                ghist = 1
            s = slot[t]
            first = False
            if s < 0:
                if nc >= cap:
                    continue
                s = nc
                slot[t] = s
                cl[s] = t
                nc += 1
                first = True
            L = 0
            while L < LMAX and q - 1 - L >= dl and tok[p - 1 - L] == tok[q - 1 - L]:
                L += 1
            L1 = 0
            while L1 < LCMAX and q - 1 - L1 >= dl and c1[tok[p - 1 - L1]] == c1[tok[q - 1 - L1]]:
                L1 += 1
            L2 = 0
            while L2 < LCMAX and q - 1 - L2 >= dl and c2[tok[p - 1 - L2]] == c2[tok[q - 1 - L2]]:
                L2 += 1
            L3 = 0
            while L3 < LCMAX and q - 1 - L3 >= dl and c3[tok[p - 1 - L3]] == c3[tok[q - 1 - L3]]:
                L3 += 1
            d = p - q
            if first:
                Lr[s] = L
                Lb[s] = L
                Lr1[s] = L1
                Lb1[s] = L1
                Lr2[s] = L2
                Lb2[s] = L2
                Lr3[s] = L3
                Lb3[s] = L3
                dr[s] = d
                dbs[s] = d
                nocc[s] = 0
                n64[s] = 0
                nl1[s] = 0
                nl2[s] = 0
            else:
                if L > Lb[s]:
                    Lb[s] = L
                    dbs[s] = d
                if L1 > Lb1[s]:
                    Lb1[s] = L1
                if L2 > Lb2[s]:
                    Lb2[s] = L2
                if L3 > Lb3[s]:
                    Lb3[s] = L3
            nocc[s] += 1
            if d <= 64:
                n64[s] += 1
            if L >= 1:
                nl1[s] += 1
            if L >= 2:
                nl2[s] += 1
            fo[s] = q - dl
        kc = 0
        chain_err = 0.0
        for k in range(1, 5):
            nk = ncache[k, p]
            if nk > 0:
                kc = k
                bk = _bl(nk)
                if bk > 12:
                    bk = 12
                wk = wts[k * 13 + bk]
                wsum += wk
                inc = wk / nk
                q = prevs[k - 1, p]
                cnt = 0
                while q >= 0:
                    b[tok[q]] += inc
                    q = prevs[k - 1, q]
                    cnt += 1
                if cnt != nk:
                    chain_err = 1.0
        norm = 1.0 + wsum
        tot = 0.0
        top = 0.0
        for v in range(Vv):
            b[v] /= norm
            tot += b[v]
            if b[v] > top:
                top = b[v]
        srt = np.sort(b)
        cm1[:] = 0.0
        cm2[:] = 0.0
        for v in range(Vv):
            cm1[c1[v]] += b[v]
            cm2[c2[v]] += b[v]
        pstat[i, 0] = b[y]
        pstat[i, 1] = ghist
        pstat[i, 2] = 1.0 if slot[y] >= 0 else 0.0
        pstat[i, 3] = tot
        pstat[i, 4] = top
        pstat[i, 5] = chain_err
        ncand[i] = nc
        base = i * cap
        prevtype = ctype[tok[p - 1]] if p - 1 >= dl else 0
        depthb = _bl(p - dl) if p > dl else 0
        if depthb > 14:
            depthb = 14
        top1b = _nlog2b(top, 8)
        for s in range(nc):
            c = cl[s]
            r = base + s
            cand[r] = c
            lab[r] = 1 if c == y else 0
            feat[r, 0] = Lr[s]
            feat[r, 1] = Lb[s]
            feat[r, 2] = Lr1[s]
            feat[r, 3] = Lb1[s]
            feat[r, 4] = Lr2[s]
            feat[r, 5] = Lb2[s]
            feat[r, 6] = Lr3[s]
            feat[r, 7] = Lb3[s]
            x = _bl(dr[s])
            feat[r, 8] = x if x < 14 else 14
            x = _bl(dbs[s])
            feat[r, 9] = x if x < 14 else 14
            x = _bl(s + 1)
            feat[r, 10] = x if x < 8 else 8
            x = _bl(nocc[s])
            feat[r, 11] = x if x < 7 else 7
            m64 = n64[s]
            feat[r, 12] = m64 if m64 <= 2 else (3 if m64 <= 4 else 4)
            x = _bl(nl1[s] + 1)
            feat[r, 13] = x if x < 5 else 5
            x = _bl(nl2[s] + 1)
            feat[r, 14] = x if x < 4 else 4
            x = _bl(fo[s] + 1)
            feat[r, 15] = x if x < 14 else 14
            feat[r, 16] = fcls[c]
            feat[r, 17] = ctype[c]
            pb = b[c]
            feat[r, 18] = _nlog2b(pb, 20)
            # rank = number of tokens with B strictly above pb (binary search on the sorted row)
            lo_i = 0
            hi_i = Vv
            while lo_i < hi_i:
                mid = (lo_i + hi_i) // 2
                if srt[mid] <= pb:
                    lo_i = mid + 1
                else:
                    hi_i = mid
            rank = Vv - lo_i
            rb = rank_edges.shape[0]
            for e in range(rank_edges.shape[0]):
                if rank <= rank_edges[e]:
                    rb = e
                    break
            feat[r, 19] = rb
            feat[r, 20] = _nlog2b(P[i, c], 20)
            feat[r, 21] = _nlog2b(cm2[c2[c]], 12)
            feat[r, 22] = _nlog2b(cm1[c1[c]], 8)
            feat[r, 23] = _nlog2b(pb / top, 12)
            feat[r, 24] = kstar[i]
            feat[r, 25] = top1b
            feat[r, 26] = prevtype
            feat[r, 27] = depthb
            feat[r, 28] = kc
            slot[c] = -1


# ------------------------------------------------------------------ scoring kernels
@njit(cache=False)
def k_pack(feat, lit_feat, LT):
    """Literal-major bit packing: cols[block, j] has bit r set when literal j holds for pair 64 * block + r."""
    N = feat.shape[0]
    nl = lit_feat.shape[0]
    nb = (N + 63) // 64
    cols = np.zeros((nb, nl), np.uint64)
    one = np.uint64(1)
    for blk in range(nb):
        r0 = blk * 64
        m = min(64, N - r0)
        for j in range(nl):
            f = lit_feat[j]
            w = np.uint64(0)
            for r in range(m):
                if LT[j, feat[r0 + r, f]]:
                    w |= one << np.uint64(r)
            cols[blk, j] = w
    return cols


@njit(cache=False)
def k_tm_score(cols, N, cl_ptr, cl_lit, cl_neg, cl_w, out):
    """Class sum = sum over non-empty clauses of weight x AND(included literals), 64 pairs per word."""
    nb = cols.shape[0]
    ncl = cl_w.shape[0]
    one = np.uint64(1)
    allb = ~np.uint64(0)
    byte = np.uint64(255)
    for blk in range(nb):
        r0 = blk * 64
        m = min(64, N - r0)
        full = allb if m == 64 else (one << np.uint64(m)) - one
        for c in range(ncl):
            a = full
            for e in range(cl_ptr[c], cl_ptr[c + 1]):
                j = cl_lit[e]
                if cl_neg[e]:
                    a &= ~cols[blk, j]
                else:
                    a &= cols[blk, j]
                if a == np.uint64(0):
                    break
            if a != np.uint64(0):
                wc = cl_w[c]
                for by in range(8):
                    bb = (a >> np.uint64(8 * by)) & byte
                    if bb != np.uint64(0):
                        for bit in range(8):
                            if (bb >> np.uint64(bit)) & one:
                                out[r0 + 8 * by + bit] += wc


@njit(cache=False)
def k_lr_score(feat, contrib, bias, out):
    N = feat.shape[0]
    nf = feat.shape[1]
    for r in range(N):
        s = bias
        for f in range(nf):
            s += contrib[f, feat[r, f]]
        out[r] = s


@njit(cache=False)
def k_copy_ll(score, offs, lab, beta):
    """Sum over positions with the gold among the candidates of log softmax(beta * score)[gold], its
    derivative in beta, and the number of such positions."""
    n = offs.shape[0] - 1
    ll = 0.0
    g = 0.0
    cnt = 0
    for i in range(n):
        a = offs[i]
        bnd = offs[i + 1]
        if bnd <= a:
            continue
        gi = -1
        mx = -1e300
        for r in range(a, bnd):
            if lab[r] == 1:
                gi = r
            v = beta * score[r]
            if v > mx:
                mx = v
        if gi < 0:
            continue
        z = 0.0
        es = 0.0
        for r in range(a, bnd):
            e = math.exp(beta * score[r] - mx)
            z += e
            es += e * score[r]
        ll += beta * score[gi] - mx - math.log(z)
        g += score[gi] - es / z
        cnt += 1
    return ll, g, cnt


@njit(cache=False)
def k_copy_comp(score, offs, lab, beta, pgold, share, topscore, top_is_gold):
    n = offs.shape[0] - 1
    for i in range(n):
        a = offs[i]
        bnd = offs[i + 1]
        if bnd <= a:
            pgold[i] = 0.0
            share[i] = 0.0
            topscore[i] = 0.0
            top_is_gold[i] = 0
            continue
        mx = -1e300
        am = a
        for r in range(a, bnd):
            v = beta * score[r]
            if v > mx:
                mx = v
                am = r
        z = 0.0
        eg = 0.0
        for r in range(a, bnd):
            e = math.exp(beta * score[r] - mx)
            z += e
            if lab[r] == 1:
                eg = e
        pgold[i] = eg / z
        share[i] = 1.0 / z
        topscore[i] = score[am]
        top_is_gold[i] = lab[am]


# ------------------------------------------------------------------ statistics
def boot_bpb(diff_bits, doc, doc_bytes, B=2000, seed=0):
    """Document-block bootstrap of sum(diff bits) / sum(bytes) over resampled documents."""
    nd = doc_bytes.shape[0]
    s = np.bincount(doc, weights=diff_bits, minlength=nd)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(B, nd))
    m = s[idx].sum(1) / doc_bytes[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fit_beta(score, offs, lab):
    from scipy.optimize import minimize_scalar

    def f(lb):
        ll, _, cnt = k_copy_ll(score, offs, lab, math.exp(lb))
        return -ll / max(cnt, 1)
    sd = float(np.std(score)) if score.shape[0] else 1.0
    c0 = -math.log(max(sd, 1e-6))
    res = minimize_scalar(f, bounds=(c0 - 9.0, c0 + 5.0), method="bounded", options={"xatol": 1e-4, "maxiter": 200})
    beta = math.exp(res.x)
    ll, _, cnt = k_copy_ll(score, offs, lab, beta)
    return beta, ll / max(cnt, 1) / LN2, cnt


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=5_000_000)
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--a-frac", type=float, default=0.55, help="share of held-out documents in part A (gate training)")
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="sampled negatives per positive training pair")
    ap.add_argument("--max-train-pos", type=int, default=120000, help="cap on positive training pairs")
    ap.add_argument("--lr-l2", default="0.1,1,10,100")
    ap.add_argument("--tm-clauses", type=int, default=2000, help="clauses per class")
    ap.add_argument("--tm-grid", default="600:10,600:25,1200:15,2400:10,2400:25", help="T:s pairs")
    ap.add_argument("--tm-epochs", type=int, default=4)
    ap.add_argument("--tm-seed", type=int, default=20260924)
    ap.add_argument("--tm-noreflective", type=int, default=1, help="1: also train the chosen TM config without the reflective literals")
    ap.add_argument("--mix-iters", type=int, default=1000)
    ap.add_argument("--wall-cap-s", type=float, default=14400.0)
    ap.add_argument("--val-docs", type=int, default=0, help="smoke only: first k validation documents")
    ap.add_argument("--ho-docs", type=int, default=0, help="smoke only: first k held-out documents")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-after-grid", action="store_true", help="pilot: stop after the TM grid on C (validation is never scored)")
    args = ap.parse_args()
    M("args", vars(args))
    M("status", "EXPLORATORY, not pre-registered; validation split only, the test split is never loaded", label="unverified")

    from tmlm.data.corpus_store import load_split
    from m3_tiny_lm import slice_train
    from cp_causal_perplexity import split_meta
    from tmlm.eval.baselines.ngram import KNModel

    # ------------------------------------------------------------ data (identical to the feasibility run)
    full = load_split(CORPUS, V, "train")
    tr = slice_train(full, args.n_tokens)
    train_counts_full = np.bincount(np.asarray(full.tok).astype(np.int64), minlength=V)
    km = np.asarray(full.kind_map).astype(np.int64)
    del full
    va = load_split(CORPUS, V, "validation")
    val_meta, _ = split_meta(V, "validation")
    n_bytes_all = int(val_meta["n_bytes_utf8"])
    tok_tr = np.asarray(tr.tok).astype(np.int64)
    ds_tr = np.asarray(tr.doc_start).astype(np.int64)
    tok_va_all = np.asarray(va.tok).astype(np.int64)
    ds_va_all = np.asarray(va.doc_start).astype(np.int64)
    n_doc = ds_tr.shape[0] - 1
    rng = np.random.default_rng(np.random.SeedSequence([20260923, 715, 1]))
    n_ho = max(1, int(round(0.05 * n_doc)))
    ho_docs = np.sort(rng.choice(n_doc, size=n_ho, replace=False))
    keep_docs = np.setdiff1d(np.arange(n_doc), ho_docs)
    M("data.prefix_tokens", int(tok_tr.shape[0]))
    M("data.prefix_documents", int(n_doc))
    M("data.heldout_documents", int(n_ho), note="SeedSequence([20260923, 715, 1]), 5 percent, as in the causal probe and the feasibility run")

    def docs_stream(tok, ds, docs):
        seg = [tok[ds[d]:ds[d + 1]] for d in docs]
        dsn = np.zeros(len(seg) + 1, np.int64)
        dsn[1:] = np.cumsum([x.shape[0] for x in seg])
        return np.concatenate(seg).astype(np.int64), dsn

    tk, dk = docs_stream(tok_tr, ds_tr, keep_docs)
    M("data.counting_positions", int(tk.shape[0]))
    t0 = time.time()
    kn = KNModel(5, V, discount_mode="modified").fit(tk, dk)
    del tk, dk
    M("kn5.fit_s", time.time() - t0)
    log(f"KN-5 fitted on the counting documents ({time.time() - t0:.0f} s)")

    # held-out stream, split into A (gate training) and C (calibration) by document
    ho_use = ho_docs if args.ho_docs <= 0 else ho_docs[: args.ho_docs]
    th, dh = docs_stream(tok_tr, ds_tr, ho_use)
    n_hd = dh.shape[0] - 1
    rs = np.random.default_rng(np.random.SeedSequence([20260924, 4242, 1]))
    perm = rs.permutation(n_hd)
    n_a = int(round(args.a_frac * n_hd))
    is_a_doc = np.zeros(n_hd, bool)
    is_a_doc[perm[:n_a]] = True
    doc_h = np.repeat(np.arange(n_hd), np.diff(dh))
    is_a = is_a_doc[doc_h]
    M("data.heldout_A_documents", int(n_a))
    M("data.heldout_C_documents", int(n_hd - n_a))
    M("data.heldout_A_positions", int(is_a.sum()))
    M("data.heldout_C_positions", int((~is_a).sum()))

    # validation (smoke: first k documents)
    n_vd = ds_va_all.shape[0] - 1 if args.val_docs <= 0 else args.val_docs
    n_va = int(ds_va_all[n_vd])
    tok_va = tok_va_all[:n_va]
    ds_va = ds_va_all[: n_vd + 1]
    doc_va = np.repeat(np.arange(n_vd), np.diff(ds_va))
    doc_bytes_all = np.diff(np.load(ROOT / "data" / CORPUS / "raw" / "validation" / "doc_bytes.npy").astype(np.int64))
    assert doc_bytes_all.shape[0] == ds_va_all.shape[0] - 1 and int(doc_bytes_all.sum()) == n_bytes_all
    doc_bytes = doc_bytes_all[:n_vd].astype(np.float64)
    n_bytes = int(doc_bytes.sum())
    M("data.validation_tokens", n_va)
    M("data.validation_documents", int(n_vd))
    M("data.validation_bytes_utf8", n_bytes)
    # check that raw article d is BPE document d: decoded UTF-8 byte length of every validation document
    try:
        from tokenizers import Tokenizer
        tkz = Tokenizer.from_file(str(ROOT / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json"))
        dec_bytes = np.array([len(tkz.decode(tok_va[ds_va[d]:ds_va[d + 1]].tolist()).encode("utf-8")) for d in range(n_vd)])
        M("verify.decoded_doc_bytes_equal_raw_doc_bytes", bool(np.array_equal(dec_bytes, doc_bytes.astype(np.int64))),
          max_abs_diff=int(np.abs(dec_bytes - doc_bytes.astype(np.int64)).max()),
          note="byte-level BPE decode of every validation document vs raw doc_bytes.npy (document alignment for the bootstrap)")
    except Exception as exc:  # pragma: no cover - reported, not hidden
        M("verify.decoded_doc_bytes_equal_raw_doc_bytes", f"not checked: {exc!r}", label="unverified")

    # static tables
    c1 = (km[1] - V).astype(np.int64)
    c2 = (km[2] - (V + 16)).astype(np.int64)
    c3 = (km[3] - (V + 272)).astype(np.int64)
    assert c1.min() >= 0 and c1.max() < 16 and c2.min() >= 0 and c2.max() < 256 and c3.min() >= 0 and c3.max() < 4096
    rank = np.empty(V, np.int64)
    rank[np.argsort(-train_counts_full, kind="stable")] = np.arange(V)
    fcls = np.select([rank < 16, rank < 64, rank < 256, rank < 1024, rank < 4096], [0, 1, 2, 3, 4], 5).astype(np.int64)
    ctype = token_types(ROOT / "data" / "tokenizers" / "bpe_8000" / "tokenizer.json").astype(np.int64)
    M("tokenizer.type_bit_counts", {TYPE_BITS[k]: int(((ctype >> k) & 1).sum()) for k in range(6)},
      note="derived from data/tokenizers/bpe_8000/tokenizer.json (byte-level BPE, G-dot = leading space)")
    spec = literal_spec()
    lit_feat, LT = literal_tables(spec)
    n_lit = len(spec)
    refl_mask = np.array([FEATS[s[1]] in REFLECTIVE for s in spec])
    M("literals.n", n_lit)
    M("literals.n_reflective", int(refl_mask.sum()))
    M("literals.names", [s[0] for s in spec])

    # ------------------------------------------------------------ per-stream gold-level cache features (R1 method)
    def stream_prep(tok, ds):
        n = tok.shape[0]
        nd = ds.shape[0] - 1
        doc = np.repeat(np.arange(nd), np.diff(ds))
        lo = ds[doc].astype(np.int64)
        posv = np.arange(n)
        cache_p, cache_n, prevs = {}, np.zeros((5, n), np.int64), np.full((4, n), -1, np.int64)
        for k in range(0, 5):
            hk = R1.ctx_hash(tok, lo, k) if k > 0 else np.zeros(n, np.uint64)
            key_c = (hk ^ R1.mix64(doc.astype(np.uint64) + np.uint64(7))).astype(np.uint64)
            r_c, prev_c = R1.group_rank_prev(key_c, posv)
            with np.errstate(over="ignore"):
                key_cy = R1.mix64(key_c ^ R1.mix64(tok.astype(np.uint64) + np.uint64(99)))
            r_cy, _ = R1.group_rank_prev(key_cy, posv)
            cache_n[k] = r_c
            cache_p[k] = np.where(r_c > 0, r_cy / np.maximum(r_c, 1), 0.0)
            if k > 0:
                prevs[k - 1] = prev_c
        return {"tok": tok, "ds": ds, "doc": doc, "lo": lo, "cache_p": cache_p, "cache_n": cache_n, "prevs": prevs, "n": n}

    t0 = time.time()
    SH = stream_prep(th, dh)
    SV = stream_prep(tok_va, ds_va)
    SH["kn_lp2"] = kn.logprob(th, dh, order=5)
    SV["kn_lp2"] = kn.logprob(tok_va, ds_va, order=5)
    log(f"stream prep + KN-5 gold log-probs ({time.time() - t0:.0f} s)")
    # verification: KN-5 per-token bits equal the causal probe's saved comparator.kn5
    if args.val_docs <= 0 and not args.stop_after_grid:
        z = np.load(PROBE / "per_token_bits.npz")
        saved = z["comparator.kn5"].astype(np.float64)
        dif = np.abs(saved - (-SV["kn_lp2"]))
        M("verify.kn5_bits_vs_probe_max_abs_diff", float(dif.max()), note=f"{PROBE.relative_to(ROOT)}/per_token_bits.npz[comparator.kn5] (float32)")
        M("verify.kn5_bpb_recomputed", float(-SV["kn_lp2"].sum() / n_bytes))
        assert dif.max() < 1e-3, "KN-5 does not reproduce the probe's comparator"

    def comp_cache(S, rows_mask=None):
        rows, b, p = [], [], []
        for k in range(5):
            act = S["cache_n"][k] > 0
            if rows_mask is not None:
                act &= rows_mask
            r = np.flatnonzero(act)
            bk = k * 13 + np.minimum(np.floor(np.log2(np.maximum(S["cache_n"][k][r], 1))).astype(np.int64), 12)
            rows.append(r); b.append(bk); p.append(S["cache_p"][k][r])
        return (np.concatenate(rows), np.concatenate(b), np.concatenate(p), 5 * 13)

    def join(comps):
        rows = np.concatenate([c[0] for c in comps])
        offs = np.cumsum([0] + [c[3] for c in comps])
        b = np.concatenate([c[1] + offs[i] for i, c in enumerate(comps)])
        p = np.concatenate([c[2] for c in comps])
        return rows, b, p, int(offs[-1])

    fitC = ~is_a                               # calibration rows of the held-out stream
    pbH = np.exp2(SH["kn_lp2"])
    pbV = np.exp2(SV["kn_lp2"])

    def fit_P1(compsH, compsV):
        """Weights fitted on the C rows of the held-out stream, bits on every validation position."""
        rH, bH, pH, nb = join(compsH)
        th_, res = R1.fit_mix(pbH, rH, bH, pH, nb, fitC, args.mix_iters)
        rV, bV, pV, _ = join(compsV)
        bits = R1.eval_mix(pbV, rV, bV, pV, th_)
        bitsC = R1.eval_mix(pbH, rH, bH, pH, th_)[fitC]
        return bits, th_, {"nit": int(res.nit), "converged": bool(res.success), "C_bits_per_token": float(bitsC.mean())}

    fold = (doc_va % 2).astype(np.int64)

    def fit_P2(compsV):
        bits, info = R1.crossfit(pbV, compsV, fold, args.mix_iters)
        return bits, info

    # base B: weights on C (P1) and the R1 protocol (P2) as a reproduction check
    t0 = time.time()
    if args.stop_after_grid:
        rH_, bH_, pH_, nb_ = join([comp_cache(SH)])
        thetaB, _ = R1.fit_mix(pbH, rH_, bH_, pH_, nb_, fitC, args.mix_iters)
        bitsB1 = bitsB2 = None
    else:
        bitsB1, thetaB, infoB1 = fit_P1([comp_cache(SH)], [comp_cache(SV)])
        bitsB2, infoB2 = fit_P2([comp_cache(SV)])
        M("B.fit_P1", infoB1)
        M("B.fit_P2", infoB2)
        log(f"B: P1 {bitsB1.sum() / n_bytes:.4f} bpb, P2 {bitsB2.sum() / n_bytes:.4f} bpb (R1 row 1.4966) ({time.time() - t0:.0f} s)")
    wts = np.exp(thetaB)
    M("B.theta", thetaB.tolist(), note="log bucket weights of the 5 cache orders x 13 log2-count buckets, fitted on C")

    # ------------------------------------------------------------ pair features
    rank_edges = RANK_EDGES

    def kstar_of(ctx, valid):
        cids = kn.context_ids(ctx, valid, 5)
        ks = np.zeros(ctx.shape[0], np.int64)
        for k in range(2, 6):
            lv = kn.level(k, 5)
            cid = cids[k]
            seen = cid >= 0
            seen[seen] = lv.total[cid[seen]] > 0
            ks = np.where(seen, k - 1, ks)
        return ks

    def run_pairs(S, name, handler):
        """Stream every position of S in chunks through the pair kernel; `handler(chunk_index, posc, nc, feat, lab)`
        receives the compacted pairs of the chunk (position-major, candidate order = recency) and keeps what it
        needs, so the full pair table is never held in memory. Returns per-position arrays."""
        t1 = time.time()
        n = S["n"]
        tok, ds, lo = S["tok"], S["ds"], S["lo"]
        cap = args.cap
        ncand = np.zeros(n, np.int64)
        pstat = np.zeros((n, 6))
        n_pairs = 0
        for ci, a in enumerate(range(0, n, args.chunk)):
            posc = np.arange(a, min(n, a + args.chunk))
            ctx, valid = kn.contexts(tok, ds, posc, None, 5)
            P = kn.dist(ctx, valid, 5)
            ks = kstar_of(ctx, valid)
            m = posc.shape[0]
            fb = np.zeros((m * cap, NF), np.uint8)
            cb = np.zeros(m * cap, np.uint16)
            lb = np.zeros(m * cap, np.uint8)
            nc = np.zeros(m, np.int64)
            ps = np.zeros((m, 6))
            k_pairs(P, posc, tok, lo, S["prevs"], S["cache_n"], wts, c1, c2, c3, fcls, ctype, ks, rank_edges, cap,
                    fb, cb, lb, nc, ps)
            del P
            keep = (np.arange(cap)[None, :] < nc[:, None]).reshape(-1)
            handler(ci, posc, nc, fb[keep], lb[keep])
            ncand[posc] = nc
            pstat[posc] = ps
            n_pairs += int(nc.sum())
            if ci % 50 == 0:
                log(f"  pairs {name}: {a + m}/{n} positions, {n_pairs} pairs ({time.time() - t1:.0f} s)")
        M(f"pairs.{name}.n_pairs", int(n_pairs))
        M(f"pairs.{name}.build_s", time.time() - t1)
        log(f"pairs {name}: {n_pairs} pairs, mean {ncand.mean():.1f} candidates per position ({time.time() - t1:.0f} s)")
        return {"ncand": ncand, "pstat": pstat}

    def check_rows(nm, PP, bits):
        ps = PP["pstat"]
        M(f"verify.{nm}.B_row_sum_max_abs_err", float(np.abs(ps[:, 3] - 1.0).max()))
        M(f"verify.{nm}.cache_chain_errors", int(ps[:, 5].sum()))
        if bits is not None:
            M(f"verify.{nm}.B_gold_full_row_vs_mixture_max_abs_log2_diff", float(np.abs(np.log2(ps[:, 0]) + bits).max()),
              note="B(gold) from the full-vocabulary row of the pair kernel vs the P1 mixture formula at the gold")
        gh, gc = ps[:, 1] > 0, ps[:, 2] > 0
        M(f"coverage.{nm}.gold_in_document_history", float(gh.mean()))
        M(f"coverage.{nm}.gold_among_candidates", float(gc.mean()))
        M(f"coverage.{nm}.lost_by_cap", float((gh & ~gc).mean()), note=f"gold earlier in the document but not among the {args.cap} most recent distinct tokens")
        M(f"coverage.{nm}.mean_candidates", float(PP["ncand"].mean()))
        M(f"coverage.{nm}.share_positions_at_cap", float((PP["ncand"] >= args.cap).mean()))
        M(f"coverage.{nm}.B_top1_is_gold", float((ps[:, 0] >= ps[:, 4]).mean()), label="unverified", note="proxy: B(gold) equals the row maximum (ties counted as hits)")
        assert float(np.abs(ps[:, 3] - 1.0).max()) < 1e-9 and int(ps[:, 5].sum()) == 0
        if bits is not None:
            assert float(np.abs(np.log2(ps[:, 0]) + bits).max()) < 1e-6
        flush()

    # held-out stream: keep every pair of the C positions; of the A positions keep every positive pair and a
    # uniform Bernoulli pool of negatives (rate 0.02, deterministic per chunk), from which the training
    # negatives are drawn uniformly
    POOL_RATE = 0.02
    keepC = {"feat": [], "lab": []}
    keepA = {"pos": [], "neg": [], "n_neg_total": 0}

    def handler_H(ci, posc, nc, feat, lab):
        pa = np.repeat(is_a[posc], nc)
        keepC["feat"].append(feat[~pa]); keepC["lab"].append(lab[~pa])
        fa, la = feat[pa], lab[pa]
        keepA["pos"].append(fa[la == 1])
        neg = fa[la == 0]
        keepA["n_neg_total"] += int(neg.shape[0])
        u = np.random.default_rng(np.random.SeedSequence([20260924, 98, ci])).random(neg.shape[0])
        keepA["neg"].append(neg[u < POOL_RATE])

    PH = run_pairs(SH, "heldout", handler_H)
    check_rows("heldout", PH, None)
    posC = np.flatnonzero(fitC)
    featC = np.concatenate(keepC["feat"]); labC = np.concatenate(keepC["lab"])
    del keepC
    offsC = np.zeros(posC.shape[0] + 1, np.int64)
    offsC[1:] = np.cumsum(PH["ncand"][posC])
    assert offsC[-1] == featC.shape[0]
    featPos = np.concatenate(keepA["pos"]); featPool = np.concatenate(keepA["neg"])
    rsamp = np.random.default_rng(np.random.SeedSequence([20260924, 99, 1]))
    if featPos.shape[0] > args.max_train_pos:
        featPos = featPos[np.sort(rsamp.choice(featPos.shape[0], size=args.max_train_pos, replace=False))]
    n_neg = int(round(args.neg_ratio * featPos.shape[0]))
    if n_neg > featPool.shape[0]:
        log(f"  negative pool {featPool.shape[0]} smaller than {n_neg}: all pooled negatives used")
        n_neg = featPool.shape[0]
    featNeg = featPool[np.sort(rsamp.choice(featPool.shape[0], size=n_neg, replace=False))]
    featT = np.concatenate([featPos, featNeg])
    yT = np.concatenate([np.ones(featPos.shape[0], np.uint32), np.zeros(n_neg, np.uint32)])
    M("train.pairs", int(featT.shape[0]))
    M("train.positives", int(featPos.shape[0]))
    M("train.negatives", int(n_neg))
    M("train.negative_pairs_in_A", int(keepA["n_neg_total"]))
    M("train.negative_sampling_rate", float(n_neg / max(keepA["n_neg_total"], 1)), label="derived",
      note="uniform over every negative pair of the A positions (Bernoulli pool at 0.02, then uniform without replacement)")
    del keepA, featPos, featPool, featNeg
    XT = LT[np.arange(n_lit)[None, :], featT[:, lit_feat]].astype(np.uint8)   # [n_train, n_lit] 0/1
    del featT
    log(f"training pairs: {XT.shape[0]} ({int(yT.sum())} positive), {n_lit} literals; C pairs {featC.shape[0]}")
    flush()

    # ------------------------------------------------------------ gate (a): logistic regression
    from scipy.optimize import minimize

    def fit_lr(X01, y, l2, lit_sel, iters=800):
        Xf = X01[:, lit_sel].astype(np.float64)
        yf = y.astype(np.float64)
        d = Xf.shape[1]

        def obj(w):
            z = Xf @ w[:d] + w[d]
            lse = np.logaddexp(0.0, z)
            f = float(np.sum(lse - yf * z)) + 0.5 * l2 * float(w[:d] @ w[:d])
            r = 0.5 * (1.0 + np.tanh(0.5 * z)) - yf
            g = np.empty(d + 1)
            g[:d] = Xf.T @ r + l2 * w[:d]
            g[d] = float(r.sum())
            return f, g

        res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-6})
        return res.x, res

    def lr_contrib(w, lit_sel):
        contrib = np.zeros((NF, 256))
        lits = np.flatnonzero(lit_sel)
        for k, j in enumerate(lits):
            contrib[lit_feat[j]] += w[k] * LT[j]
        return contrib, float(w[-1])

    def lr_scores(feat, contrib, bias):
        out = np.empty(feat.shape[0])
        k_lr_score(feat, contrib, bias, out)
        return out

    LRres = {}
    for tag, sel in (("LR", np.ones(n_lit, bool)), ("LR_noreflective", ~refl_mask)):
        best = None
        grid = [float(x) for x in args.lr_l2.split(",")] if tag == "LR" else [LRres["LR"]["l2"]]
        for l2 in grid:
            t1 = time.time()
            w, res = fit_lr(XT, yT, l2, sel)
            contrib, bias = lr_contrib(w, sel)
            sC = lr_scores(featC, contrib, bias)
            beta, ll2, cnt = fit_beta(sC, offsC, labC)
            M(f"{tag}.grid.l2_{l2:g}", {"C_mean_log2_copy_likelihood": ll2, "beta": beta, "nit": int(res.nit),
                                        "converged": bool(res.success), "C_positions_with_gold": int(cnt), "fit_s": time.time() - t1})
            log(f"{tag} l2 {l2:g}: C copy log2-lik {ll2:.4f} (beta {beta:.3f}, nit {res.nit}) {time.time() - t1:.0f} s")
            if best is None or ll2 > best["ll"]:
                best = {"ll": ll2, "l2": l2, "contrib": contrib, "bias": bias, "beta": beta, "w": w}
        LRres[tag] = best
        M(f"{tag}.chosen", {"l2": best["l2"], "C_mean_log2_copy_likelihood": best["ll"], "beta": best["beta"]})
        flush()
    # the 10 largest LR literal weights (for comparison with the TM rules)
    wl = LRres["LR"]["w"][:-1]
    top = np.argsort(-np.abs(wl))[:10]
    M("LR.top_literal_weights", [f"{wl[j]:+.3f}  {spec[j][0]}" for j in top])

    # ------------------------------------------------------------ gate (b): Tsetlin machine (tmu, CPU)
    from tmu.models.classification.vanilla_classifier import TMClassifier

    colsC = k_pack(featC, lit_feat, LT)
    log(f"packed C literals: {colsC.shape}")

    def clauses_of(tm, lit_idx):
        """CSR of the non-empty class-1 clauses in terms of the FULL literal index; weights."""
        inc = tm.clause_banks[1].get_literals()          # [clauses, 2 * n_used]
        w = tm.weight_banks[1].get_weights().astype(np.int64).copy()
        nu = lit_idx.shape[0]
        ptr, lits, negs, ws, keep = [0], [], [], [], []
        for c in range(inc.shape[0]):
            pj = np.flatnonzero(inc[c, :nu])
            nj = np.flatnonzero(inc[c, nu:])
            if pj.shape[0] + nj.shape[0] == 0:
                continue
            keep.append(c)
            lits.extend(lit_idx[pj].tolist()); negs.extend([0] * pj.shape[0])
            lits.extend(lit_idx[nj].tolist()); negs.extend([1] * nj.shape[0])
            ptr.append(len(lits))
            ws.append(w[c])
        return (np.array(ptr, np.int64), np.array(lits, np.int64), np.array(negs, np.uint8), np.array(ws, np.int64),
                np.array(keep, np.int64), inc, w)

    def tm_scores_cols(cols, N, cl):
        out = np.zeros(N, np.int64)
        k_tm_score(cols, N, cl[0], cl[1], cl[2], cl[3], out)
        return out.astype(np.float64)

    def train_tm(T, s, lit_sel, tag, epochs):
        lit_idx = np.flatnonzero(lit_sel)
        X = np.ascontiguousarray(XT[:, lit_idx].astype(np.uint32))
        tm = TMClassifier(args.tm_clauses, T=int(T), s=float(s), platform="CPU", weighted_clauses=True, seed=args.tm_seed)
        hist, best = [], None
        for ep in range(1, epochs + 1):
            if time.time() - T0 > args.wall_cap_s:
                log(f"  {tag}: wall cap reached before epoch {ep}")
                M(f"{tag}.wall_cap_hit_before_epoch", ep)
                break
            t1 = time.time()
            tm.fit(X, yT)
            fit_s = time.time() - t1
            cl = clauses_of(tm, lit_idx)
            sC = tm_scores_cols(colsC, featC.shape[0], cl)
            beta, ll2, cnt = fit_beta(sC, offsC, labC)
            n_inc = np.array([cl[0][i + 1] - cl[0][i] for i in range(cl[3].shape[0])])
            rec = {"epoch": ep, "C_mean_log2_copy_likelihood": ll2, "beta": beta, "fit_s": fit_s, "eval_s": time.time() - t1 - fit_s,
                   "nonempty_clauses": int(cl[3].shape[0]), "mean_literals_per_nonempty_clause": float(n_inc.mean()) if n_inc.shape[0] else 0.0,
                   "abs_weight_sum": int(np.abs(cl[3]).sum())}
            hist.append(rec)
            log(f"  {tag} T {T} s {s} epoch {ep}: C copy log2-lik {ll2:.4f} (beta {beta:.4g}), fit {fit_s:.0f} s, "
                f"{cl[3].shape[0]} non-empty clauses, {rec['mean_literals_per_nonempty_clause']:.1f} literals each")
            if best is None or ll2 > best["ll"]:
                best = {"ll": ll2, "T": T, "s": s, "epoch": ep, "beta": beta, "cl": cl, "lit_idx": lit_idx}
            last_cl = cl
        if hist:
            d = verify_eval(tm, lit_sel, last_cl, f"{tag}.T{T}_s{s:g}")
            assert d == 0, "bit-parallel clause evaluator disagrees with tmu"
        return tm, hist, best

    # verification of the bit-parallel evaluator against tmu's own class sums
    def verify_eval(tm, lit_sel, cl, tag):
        lit_idx = np.flatnonzero(lit_sel)
        rv = np.random.default_rng(7)
        idx = np.sort(rv.choice(featC.shape[0], size=min(3000, featC.shape[0]), replace=False))
        f = featC[idx]
        X = np.ascontiguousarray(LT[lit_idx[None, :], f[:, lit_feat[lit_idx]]].astype(np.uint32))
        _, cs = tm.predict(X, return_class_sums=True)
        mine = np.zeros(idx.shape[0], np.int64)
        cols = k_pack(f, lit_feat, LT)
        k_tm_score(cols, idx.shape[0], cl[0], cl[1], cl[2], cl[3], mine)
        inc, w = cl[5], cl[6]
        empty_w = int(w[(inc.sum(1) == 0)].sum())
        d1 = int(np.abs(cs[:, 1].astype(np.int64) - mine).max())
        d2 = int(np.abs(cs[:, 1].astype(np.int64) - (mine + empty_w)).max())
        M(f"verify.{tag}.evaluator_vs_tmu_class1_sum_max_abs_diff", min(d1, d2),
          note=f"3000 C pairs; tmu predict (incremental evaluation) counts empty clauses as firing (their weight sum {empty_w} is a constant per pair, cancels in the softmax); diff without / with that constant: {d1} / {d2}")
        return min(d1, d2)

    grid = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in args.tm_grid.split(",")]
    tm_best = None
    for (T, s) in grid:
        if time.time() - T0 > args.wall_cap_s:
            M(f"TM.grid_skipped.T{T}_s{s:g}", "wall cap", label="unverified")
            continue
        tm, hist, best = train_tm(T, s, np.ones(n_lit, bool), "TM", args.tm_epochs)
        M(f"TM.grid.T{T}_s{s:g}", hist)
        flush()
        if best is not None and (tm_best is None or best["ll"] > tm_best["ll"]):
            tm_best = best
        del tm
    assert tm_best is not None, "no TM trained within the wall cap"
    M("TM.chosen", {"T": tm_best["T"], "s": tm_best["s"], "epoch": tm_best["epoch"], "C_mean_log2_copy_likelihood": tm_best["ll"],
                    "beta": tm_best["beta"], "clauses_per_class": args.tm_clauses})
    flush()
    if args.stop_after_grid:
        M("wall_s", time.time() - T0)
        flush()
        log("pilot: stopped after the TM grid (validation not scored)")
        return 0
    TMres = {"TM": tm_best}
    if args.tm_noreflective and time.time() - T0 < args.wall_cap_s:
        tmn, histn, bestn = train_tm(tm_best["T"], tm_best["s"], ~refl_mask, "TM_noreflective", tm_best["epoch"])
        M("TM_noreflective.grid", histn)
        if bestn is not None:
            TMres["TM_noreflective"] = bestn
            M("TM_noreflective.chosen", {"T": bestn["T"], "s": bestn["s"], "epoch": bestn["epoch"], "C_mean_log2_copy_likelihood": bestn["ll"], "beta": bestn["beta"]})
        flush()

    # readable rules: the 10 highest-weight class-1 clauses (positive polarity = vote FOR copying) and the 5 most negative
    cl = tm_best["cl"]
    ptr, lits, negs, ws, keep = cl[0], cl[1], cl[2], cl[3], cl[4]
    one_c = np.zeros(featC.shape[0], np.int64)

    name_of = {(sp[1], sp[2], sp[3]): sp[0] for sp in spec}

    def rule_text(ci):
        """Conjunction reduced to the tightest bound per feature (thermometer literals imply each other)."""
        lo_b, hi_b, bits_on, bits_off = {}, {}, {}, {}
        for e in range(ptr[ci], ptr[ci + 1]):
            _, f, kind, t = spec[lits[e]]
            neg = bool(negs[e])
            if kind == "bit":
                (bits_off if neg else bits_on).setdefault(f, set()).add(t)
                continue
            if kind == "ge":
                lo, hi = (None, t - 1) if neg else (t, None)
            else:
                lo, hi = (t + 1, None) if neg else (None, t)
            if lo is not None:
                lo_b[f] = max(lo_b.get(f, -1), lo)
            if hi is not None:
                hi_b[f] = min(hi_b.get(f, 999), hi)
        parts = []
        for f in sorted(set(lo_b) | set(hi_b) | set(bits_on) | set(bits_off)):
            kinds = {sp[2] for sp in spec if sp[1] == f}
            if "ge" in kinds:
                if f in lo_b:
                    parts.append(name_of[(f, "ge", lo_b[f])])
                if f in hi_b:
                    parts.append("NOT " + name_of[(f, "ge", hi_b[f] + 1)])
            elif "le" in kinds:
                if f in hi_b:
                    parts.append(name_of[(f, "le", hi_b[f])])
                if f in lo_b:
                    parts.append("NOT " + name_of[(f, "le", lo_b[f] - 1)])
            for t in sorted(bits_on.get(f, ())):
                parts.append(name_of[(f, "bit", t)])
            for t in sorted(bits_off.get(f, ())):
                parts.append("NOT " + name_of[(f, "bit", t)])
            if f in lo_b and f in hi_b and lo_b[f] > hi_b[f]:
                parts.append(f"[contradiction on {FEATS[f]}: never fires]")
        return " AND ".join(parts) if parts else "(empty)"

    def rule_stats(ci):
        sub = (np.array([0, ptr[ci + 1] - ptr[ci]], np.int64), lits[ptr[ci]:ptr[ci + 1]], negs[ptr[ci]:ptr[ci + 1]], np.array([1], np.int64))
        one_c[:] = 0
        k_tm_score(colsC, featC.shape[0], sub[0], sub[1], sub[2], sub[3], one_c)
        fire = one_c > 0
        return float(fire.mean()), float(labC[fire].mean()) if fire.any() else float("nan")

    order_pos = np.argsort(-ws, kind="stable")
    rules = []
    for ci in order_pos[:10]:
        fr, prec = rule_stats(ci)
        rules.append({"weight": int(ws[ci]), "rule": rule_text(ci), "n_literals": int(ptr[ci + 1] - ptr[ci]),
                      "C_pair_firing_rate": fr, "C_precision_label1": prec})
    order_neg = np.argsort(ws, kind="stable")
    vetoes = []
    for ci in order_neg[:5]:
        fr, prec = rule_stats(ci)
        vetoes.append({"weight": int(ws[ci]), "rule": rule_text(ci), "n_literals": int(ptr[ci + 1] - ptr[ci]),
                       "C_pair_firing_rate": fr, "C_precision_label1": prec})
    M("TM.top10_clauses_for_copy", rules, note="class-1 clauses of the chosen TM with the largest weights; firing rate and share of label-1 pairs among firing pairs on all C pairs")
    M("TM.top5_clauses_against_copy", vetoes)
    log("TM top-10 clauses (class 1, largest weights):")
    for r in rules:
        log(f"  w {r['weight']:+5d}  fires {r['C_pair_firing_rate']:.4f}  prec {r['C_precision_label1']:.3f}  IF {r['rule']}")
    log("TM 5 most negative clauses:")
    for r in vetoes:
        log(f"  w {r['weight']:+5d}  fires {r['C_pair_firing_rate']:.4f}  prec {r['C_precision_label1']:.3f}  IF {r['rule']}")
    flush()
    del colsC

    # ------------------------------------------------------------ copy components and mixtures
    NSH, NSC = 10, 6

    def copy_comp_arrays(score, offs, lab, beta):
        n = offs.shape[0] - 1
        pg, sh, ts, tg = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
        k_copy_comp(score, offs, lab, beta, pg, sh, ts, tg)
        return pg, sh, ts, tg

    def scores_for(gate, feat):
        if gate.startswith("LR"):
            r = LRres[gate]
            return lr_scores(feat, r["contrib"], r["bias"]), r["beta"]
        r = TMres[gate]
        out = np.zeros(feat.shape[0], np.int64)
        blk = 64 * 65536
        for a in range(0, feat.shape[0], blk):
            fb = feat[a:a + blk]
            cols = k_pack(fb, lit_feat, LT)
            o = np.zeros(fb.shape[0], np.int64)
            k_tm_score(cols, fb.shape[0], r["cl"][0], r["cl"][1], r["cl"][2], r["cl"][3], o)
            out[a:a + blk] = o
        return out.astype(np.float64), r["beta"]

    gates = ["LR", "LR_noreflective"] + [g for g in ("TM", "TM_noreflective") if g in TMres]
    per_tok = {"B.P1": bitsB1, "B.P2": bitsB2}
    # C positions of the held-out stream (the only held-out rows the mixture fit uses)
    cpC = {}
    for g in gates:
        sC, beta = scores_for(g, featC)
        cpC[g] = copy_comp_arrays(sC, offsC, labC, beta)
        del sC
    del featC
    # validation: stream the pairs once more, score every gate chunk by chunk, keep per-position outputs only
    n_vpos = SV["n"]
    cpV = {g: (np.zeros(n_vpos), np.zeros(n_vpos), np.zeros(n_vpos), np.zeros(n_vpos, np.int64)) for g in gates}

    def handler_V(ci, posc, nc, feat, lab):
        offs = np.zeros(nc.shape[0] + 1, np.int64)
        offs[1:] = np.cumsum(nc)
        for g in gates:
            s_, beta_ = scores_for(g, feat)
            pg, sh, ts, tg = copy_comp_arrays(s_, offs, lab, beta_)
            cpV[g][0][posc] = pg; cpV[g][1][posc] = sh; cpV[g][2][posc] = ts; cpV[g][3][posc] = tg

    t1 = time.time()
    PV = run_pairs(SV, "validation", handler_V)
    check_rows("validation", PV, bitsB1)
    log(f"validation pairs scored by {len(gates)} gates ({time.time() - t1:.0f} s)")
    compH, compV = {}, {}
    actC = PH["ncand"][posC] > 0
    actV = PV["ncand"] > 0
    cop = PV["pstat"][:, 2] > 0
    for g in gates:
        pgC, shC, tsC, tgC = cpC[g]
        pgV, shV, tsV, tgV = cpV[g]
        edges = np.quantile(tsC[actC], np.arange(1, NSC) / NSC)

        def comp(rows, pg, sh, ts, act):
            r = np.flatnonzero(act)
            bsh = np.minimum((sh[r] * NSH).astype(np.int64), NSH - 1)
            bsc = np.searchsorted(edges, ts[r], side="right")
            return (rows[r], bsh * NSC + bsc, pg[r], NSH * NSC)
        compH[g] = comp(posC, pgC, shC, tsC, actC)
        compV[g] = comp(np.arange(n_vpos), pgV, shV, tsV, actV)
        M(f"gate.{g}.validation_copy_diagnostics", {
            "mean_log2_Pcopy_gold_when_gold_is_candidate": float(np.log2(np.maximum(pgV[cop], 1e-300)).mean()),
            "copy_top1_is_gold_share_of_all_positions": float(tgV[actV].sum() / tsV.shape[0]),
            "copy_top1_is_gold_share_when_gold_is_candidate": float(tgV[cop].mean()),
            "beta": LRres[g]["beta"] if g.startswith("LR") else TMres[g]["beta"], "score_bin_edges_on_C": edges.tolist()}, label="measured")
        log(f"gate {g}: val copy top1-is-gold {tgV[cop].mean():.4f} when gold is a candidate, mean log2 Pcopy(gold) {np.log2(np.maximum(pgV[cop], 1e-300)).mean():.4f}")
    flush()

    rows_def = {"B": [], "B+LR": ["LR"], "B+TM": ["TM"], "B+LR_noreflective": ["LR_noreflective"],
                "B+TM_noreflective": ["TM_noreflective"], "B+LR+TM": ["LR", "TM"]}
    bpb = {}
    for name, gs in rows_def.items():
        if any(g not in compH for g in gs):
            continue
        t1 = time.time()
        cH = [comp_cache(SH)] + [compH[g] for g in gs]
        cV = [comp_cache(SV)] + [compV[g] for g in gs]
        b1, _, info1 = fit_P1(cH, cV)
        b2, info2 = fit_P2(cV)
        per_tok[f"{name}.P1"] = b1
        per_tok[f"{name}.P2"] = b2
        for proto, bits, info in (("P1", b1, info1), ("P2", b2, info2)):
            v = float(bits.sum() / n_bytes)
            bpb[(name, proto)] = v
            M(f"row.{proto}.{name}.bits_per_byte", v, fit=info,
              note=("mixture weights fitted on the C documents of the held-out training split" if proto == "P1"
                    else "mixture weights by two-fold cross-fitting over even/odd validation documents (feasibility-run protocol)"))
            M(f"row.{proto}.{name}.bits_per_token", float(bits.mean()))
        log(f"row {name:<22} P1 {bpb[(name, 'P1')]:.4f}  P2 {bpb[(name, 'P2')]:.4f} bpb ({time.time() - t1:.0f} s)")
    flush()

    def contrast(proto, a, b, label):
        if f"{a}.{proto}" not in per_tok or f"{b}.{proto}" not in per_tok:
            return None
        d = per_tok[f"{a}.{proto}"] - per_tok[f"{b}.{proto}"]
        lo_, hi_ = boot_bpb(d, doc_va, doc_bytes)
        v = float(d.sum() / n_bytes)
        M(f"contrast.{proto}.{label}", {"bpb": v, "ci95_bpb": [lo_, hi_], "a": a, "b": b}, label="derived",
          note="a minus b in bits per byte (negative = a better); document-block bootstrap of summed bits / summed bytes, 2000 resamples")
        log(f"contrast {proto} {label:<40} {v:+.4f} bpb [{lo_:+.4f}, {hi_:+.4f}]")
        return v, lo_, hi_

    res = {}
    for proto in ("P1", "P2"):
        res[(proto, "TM")] = contrast(proto, "B+TM", "B", "TM_gate_minus_B")
        res[(proto, "LR")] = contrast(proto, "B+LR", "B", "LR_gate_minus_B")
        res[(proto, "TMLR")] = contrast(proto, "B+TM", "B+LR", "TM_gate_minus_LR_gate")
        contrast(proto, "B+LR_noreflective", "B", "LR_noreflective_gate_minus_B")
        contrast(proto, "B+LR", "B+LR_noreflective", "reflective_literals_in_LR")
        contrast(proto, "B+TM_noreflective", "B", "TM_noreflective_gate_minus_B")
        contrast(proto, "B+TM", "B+TM_noreflective", "reflective_literals_in_TM")
        contrast(proto, "B+LR+TM", "B+LR", "TM_on_top_of_LR")
    for proto in ("P1", "P2"):
        tmv, tmlr = res[(proto, "TM")], res[(proto, "TMLR")]
        gain, gain_hi = -tmv[0], -tmv[1]         # gain = B minus B+TM; CI upper bound of the difference gives the lower bound of the gain
        crit1 = gain >= 0.01 and tmv[2] < 0.0
        crit2 = tmlr[0] <= 0.002
        M(f"GO.{proto}", {"met": bool(crit1 and crit2), "TM_gain_bpb": gain, "TM_gain_ci95_bpb": [-tmv[2], -tmv[1]],
                          "criterion_1_gain_ge_0.01_and_ci_excludes_0": bool(crit1),
                          "TM_minus_LR_bpb": tmlr[0], "criterion_2_TM_not_worse_than_LR_by_more_than_0.002": bool(crit2)},
          label="derived", note="criterion not pre-registered; P1 is the primary protocol")
        log(f"GO ({proto}): {'MET' if crit1 and crit2 else 'NOT MET'}  (TM gain {gain:+.4f} bpb, CI [{-tmv[2]:+.4f}, {-tmv[1]:+.4f}]; TM minus LR {tmlr[0]:+.4f})")
    if os.environ.get("TMLM_RUN_DIR"):
        np.savez_compressed(Path(os.environ["TMLM_RUN_DIR"]) / "per_token_bits.npz", doc=doc_va.astype(np.int32),
                            **{k: v.astype(np.float32) for k, v in per_tok.items()})
    M("wall_s", time.time() - T0)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
