"""CR-TM: TA learning aimed at the CVTM-core's errors, pilot P1 of docs/13 amendment A5 (EXPLORATORY;
seed 100 first, seeds 101 and 102 only in the extension). The registered model is H9.3 of A4; P1 runs it
on the frozen cache of `results/20260924T085728Z_cvtm_cache_build/` (lane L0).

Model. At a position p the candidates are the top-64 classes of the core, P_core(. | p) (cache group
"hprime": the frozen EM weights over the KN-5 fitted without H'). A gate g scores every (p, c) pair from
Boolean literals; the read-out is the exponential tilt of the core with the exact sparse normaliser of
docs/11 section 11.5 (the same read-out as the copy selector, `tmlm.cvtm.copy_selector.k_readout`):

    P(w | p) = P_core(w | p) exp(t(w)) / Z(p),  t(c) = beta_b s'(c) + a_b for c in the top-64, else 0,
    Z(p) = (1 - sum_top64 P_core(c)) + sum_top64 P_core(c) exp(t(c)),

s'(c) = (s(c) - mu) / sigma the gate score standardised on the selection pairs, b = b(p) the core's
bucket (k*, m*, c, l) of docs/11 11.4 (520 buckets). (beta_b, a_b) are fitted on residue 2a: pooled
maximum likelihood first, then per bucket by MAP towards the pooled pair (`copy_selector.fit_buckets`).
A5 names "beta per bucket"; a_b is the bucket offset of the same read-out in lane B1 (H9.1), kept so
that P1 and H9.1 share one read-out.

Data roles (docs/13 A4 H' and A5 P1): residue 1 trains (250,000 positions whose gold is among the
top-64, each with the gold and K = 4 negatives drawn from P_core over the top-64 without the gold,
without replacement); residue 2 is split BY DOCUMENT into 2a (selection: configurations, standardisation,
(beta_b, a_b)) and 2b (judgement). The split is fixed (independent of the seed): the 130 residue-2
documents with the smallest mix64(doc_id XOR SALT_2AB) form 2a, the other 131 form 2b. Validation and
test are not used.

Literals (one raw uint8 feature per row per family member; every literal is a function of ONE raw
feature, so the LR controls can be evaluated exactly through lookup tables):
- context: the 16-cluster and the 243-cluster of the tokens at offsets -1 to -8. The kind_map levels
  are nested (level-2 cluster = 16 x level-1 cluster + sub-index, checked), so each offset carries a
  one-hot of the level-1 cluster (16) and a one-hot of the sub-index (16): the 243-cluster is the
  conjunction of two literals. Plus the BPE type bits of tok[p-1] and the document depth.
- candidate: one-hot of its 16-cluster and of its 243-cluster (243 literals), its BPE type bits and
  its frequency class.
- relational: eq_k (the candidate's 16-cluster / 243-cluster / token equals that at offset -k,
  k = 1..8) and cluster-bag membership (counts of the candidate's 16- and 243-cluster among the last 32
  tokens and of its 243-cluster in the document so far).
- reflective (the core's own state): P_core(c), its rank in the top-64, P_core(c) / P_core(top-1),
  P_S(c) and its rank in the KN top-64, the core's mass (within the top-64) on c's clusters,
  P_core(top-1), the mass outside the top-64, and the bucket parts k*, m*, count and pointer buckets,
  "c is the pointer candidate".
- copy (classical): occurrences of c earlier in the document, distance to the most recent one, exact
  suffix match at the most recent occurrence and the best over the 64 most recent ones, counts with
  suffix match >= 1, 2, 3 and within the last 64 tokens.
- candidate id (ablation only): the hierarchical code of the token id (level-1 cluster, the level-2
  and level-3 sub-indices, the index inside its level-3 cluster; a 4-literal conjunction names a token).

Literal sets: MAIN (everything but the candidate id), NOREL (MAIN without the relational family:
ablation "no relational literals"), CTXID (context plus the candidate id: ablation "context-only
literals plus the candidate id", the NCE-bank objective without the engine change).

Gates (same pairs, same literals): TM (tmu TMClassifier, 2 classes, weighted clauses, Type I and II
feedback, CPU; score s = class-1 minus class-0 vote sum of the non-empty clauses, evaluated by a
bit-parallel evaluator checked against tmu), LR and LR2 (logistic regression on the literals and on all
pairwise literal products, exact table evaluation), HGB (scikit-learn HistGradientBoostingClassifier on
the raw features: numeric columns for thermometer families, categorical columns for one-hot families,
one 0/1 column per bit literal; information-equivalent to the literal set, and every literal split is
one of its splits), C0 (a TM of the same size trained on 1 percent of the pairs).

Everything is deterministic: positions, negatives and subsamples are keyed by counter-based hashes or
seeded numpy generators; tmu is seeded and single-threaded.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numba import njit

from . import copy_selector as CS
from .mixture import BucketLayout

V = 8000
TOPK = 64
K_NEG = 4
N_OFF = 8
WIN = 32
OCC_SCAN = 64
SALT_2AB = 0x2AB2AB
SALT_POS = 0x5A11B0
SALT_NEG = 0x9E6A7E
N_2A_DOCS = 130
LN2 = math.log(2.0)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)

# ------------------------------------------------------------------ raw features
RAW = ([f"ctx_c16_{k}" for k in range(1, N_OFF + 1)] + [f"ctx_sub_{k}" for k in range(1, N_OFF + 1)]
       + ["prev_type", "depth",
          "cand_c16", "cand_c2", "cand_type", "cand_fcls",
          "eq16", "eq243", "eqtok", "bag16_w32", "bag243_w32", "bag243_doc",
          "pcore", "rank", "ratio_top", "knrank", "ps", "mass16", "mass243", "top1", "tail", "kstar", "mstar",
          "cbucket", "lbucket", "is_ptr",
          "nocc", "dist", "lrec", "lbest", "nl1", "nl2", "nl3", "n64",
          "cand_sub2", "cand_sub3", "cand_idx4"])
R = {n: i for i, n in enumerate(RAW)}
NRAW = len(RAW)
# largest value the kernel writes per raw feature (table sizes of the LR controls; asserted in the build)
RAW_MAX = np.zeros(NRAW, np.int64)
for _k in range(1, N_OFF + 1):
    RAW_MAX[R[f"ctx_c16_{_k}"]] = 16
    RAW_MAX[R[f"ctx_sub_{_k}"]] = 16
for _n, _m in {"prev_type": 63, "depth": 14, "cand_c16": 15, "cand_c2": 255, "cand_type": 63, "cand_fcls": 5,
               "eq16": 255, "eq243": 255, "eqtok": 255, "bag16_w32": 4, "bag243_w32": 4, "bag243_doc": 5,
               "pcore": 20, "rank": 6, "ratio_top": 12, "knrank": 7, "ps": 20, "mass16": 8, "mass243": 12,
               "top1": 8, "tail": 12, "kstar": 4, "mstar": 5, "cbucket": 5, "lbucket": 3, "is_ptr": 1,
               "nocc": 6, "dist": 14, "lrec": 8, "lbest": 8, "nl1": 4, "nl2": 4, "nl3": 4, "n64": 4,
               "cand_sub2": 15, "cand_sub3": 15, "cand_idx4": 63}.items():
    RAW_MAX[R[_n]] = _m
TYPE_BITS = ["word_start", "capitalised", "continuation", "punct", "digit", "newline"]
RAW_FAMILY = {}
for _n in RAW:
    if _n.startswith("ctx_") or _n in ("prev_type", "depth"):
        RAW_FAMILY[_n] = "context"
    elif _n in ("cand_c16", "cand_c2", "cand_type", "cand_fcls"):
        RAW_FAMILY[_n] = "candidate"
    elif _n in ("eq16", "eq243", "eqtok", "bag16_w32", "bag243_w32", "bag243_doc"):
        RAW_FAMILY[_n] = "relational"
    elif _n in ("nocc", "dist", "lrec", "lbest", "nl1", "nl2", "nl3", "n64"):
        RAW_FAMILY[_n] = "copy"
    elif _n in ("cand_sub2", "cand_sub3", "cand_idx4"):
        RAW_FAMILY[_n] = "candidate_id"
    else:
        RAW_FAMILY[_n] = "reflective"


# ------------------------------------------------------------------ token tables
def token_tables(kind_map: np.ndarray, counts: np.ndarray, vocab_path: Path | None) -> dict[str, np.ndarray]:
    """c1 (16 clusters), c2 (0..255, 243 used), sub2 = c2 % 16, sub3 = c3 % 16, idx4 (rank of the token
    inside its level-3 cluster by id), ctype (BPE type bits), fcls (frequency class from `counts`),
    c2_used (sorted used level-2 values). Raises when the levels are not nested."""
    km = np.asarray(kind_map).astype(np.int64)
    c1 = km[1] - V
    c2 = km[2] - (V + 16)
    c3 = km[3] - (V + 272)
    if not (c1.min() >= 0 and c1.max() < 16 and c2.min() >= 0 and c2.max() < 256 and c3.min() >= 0 and c3.max() < 4096):
        raise ValueError("kind_map does not hold the 16 / 256 / 4096 cluster levels")
    if not (np.all(c2 // 16 == c1) and np.all(c3 // 16 == c2)):
        raise ValueError("kind_map levels are not nested (level k+1 = 16 x level k + sub-index)")
    idx4 = np.zeros(V, np.int64)
    order = np.lexsort((np.arange(V), c3))
    c3s = c3[order]
    start = np.r_[0, np.flatnonzero(np.diff(c3s)) + 1]
    run = np.arange(V) - np.repeat(start, np.diff(np.r_[start, V]))
    idx4[order] = run
    if idx4.max() > RAW_MAX[R["cand_idx4"]]:
        raise ValueError(f"a level-3 cluster has {idx4.max() + 1} members, more than the literal table allows")
    counts = np.asarray(counts).astype(np.int64)[:V]
    rank = np.empty(V, np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(V)
    fcls = np.select([rank < 16, rank < 64, rank < 256, rank < 1024, rank < 4096], [0, 1, 2, 3, 4], 5).astype(np.int64)
    if vocab_path is not None:
        ctype = CS.exploratory_module().token_types(Path(vocab_path)).astype(np.int64)
    else:
        ctype = np.zeros(V, np.int64)
    return {"c1": c1, "c2": c2, "sub2": c2 % 16, "sub3": c3 % 16, "idx4": idx4, "ctype": ctype, "fcls": fcls,
            "c2_used": np.unique(c2)}


# ------------------------------------------------------------------ literal specification
@dataclass
class LiteralSet:
    """Literals over the raw features: name, raw feature, kind ('eq' raw == t, 'ge' raw >= t,
    'le' raw <= t, 'bit' bit t of raw set), threshold, family; LT[literal, raw value] truth table."""

    names: list
    feat: np.ndarray
    kind: list
    thr: np.ndarray
    family: list
    LT: np.ndarray

    @property
    def n(self) -> int:
        return int(self.feat.shape[0])

    def subset(self, idx: np.ndarray) -> "LiteralSet":
        idx = np.asarray(idx, np.int64)
        return LiteralSet(names=[self.names[i] for i in idx], feat=self.feat[idx].copy(),
                          kind=[self.kind[i] for i in idx], thr=self.thr[idx].copy(),
                          family=[self.family[i] for i in idx], LT=self.LT[idx].copy())

    def select(self, families: Sequence[str], extra_raw: Sequence[str] = ()) -> np.ndarray:
        fam = set(families)
        ex = {R[n] for n in extra_raw}
        return np.array([i for i in range(self.n) if self.family[i] in fam or int(self.feat[i]) in ex], np.int64)

    def raw_used(self) -> np.ndarray:
        return np.unique(self.feat)

    def matrix(self, feat: np.ndarray, dtype=np.uint8) -> np.ndarray:
        """[n, L] 0/1 literal values of raw features [n, NRAW]."""
        f = np.asarray(feat)
        return self.LT[np.arange(self.n)[None, :], f[:, self.feat]].astype(dtype)


def build_literals(c2_used: np.ndarray) -> LiteralSet:
    names, feat, kind, thr = [], [], [], []

    def add(raw, k, ts, fmt):
        for t in ts:
            names.append(fmt(int(t)))
            feat.append(R[raw])
            kind.append(k)
            thr.append(int(t))

    for k in range(1, N_OFF + 1):
        add(f"ctx_c16_{k}", "eq", range(16), lambda t, k=k: f"16-cluster at offset -{k} is {t}")
    for k in range(1, N_OFF + 1):
        add(f"ctx_sub_{k}", "eq", range(16), lambda t, k=k: f"243-cluster sub-index at offset -{k} is {t}")
    add("prev_type", "bit", range(6), lambda t: f"previous token is {TYPE_BITS[t]}")
    add("depth", "le", range(3, 14), lambda t: f"position within the first {2 ** (t + 1) - 1} document tokens")
    add("cand_c16", "eq", range(16), lambda t: f"candidate 16-cluster is {t}")
    add("cand_c2", "eq", [int(x) for x in c2_used], lambda t: f"candidate 243-cluster is {t // 16}.{t % 16}")
    add("cand_type", "bit", range(6), lambda t: f"candidate is {TYPE_BITS[t]}")
    add("cand_fcls", "le", range(5), lambda t: f"candidate among the {[16, 64, 256, 1024, 4096][t]} most frequent tokens")
    add("eq16", "bit", range(N_OFF), lambda t: f"candidate 16-cluster equals that at offset -{t + 1}")
    add("eq243", "bit", range(N_OFF), lambda t: f"candidate 243-cluster equals that at offset -{t + 1}")
    add("eqtok", "bit", range(N_OFF), lambda t: f"candidate equals the token at offset -{t + 1}")
    add("bag16_w32", "ge", range(1, 5), lambda t: f"candidate 16-cluster among the last 32 tokens >= {[0, 1, 2, 4, 8][t]} times")
    add("bag243_w32", "ge", range(1, 5), lambda t: f"candidate 243-cluster among the last 32 tokens >= {[0, 1, 2, 4, 8][t]} times")
    add("bag243_doc", "ge", range(1, 6), lambda t: f"candidate 243-cluster earlier in the document >= {[0, 1, 2, 4, 8, 16][t]} times")
    add("pcore", "le", range(16), lambda t: f"P_core(candidate) > 2^-{t + 1}")
    add("rank", "le", range(6), lambda t: f"candidate in the core's top {2 ** (t + 1) - 1}")
    add("ratio_top", "le", range(12), lambda t: f"P_core(candidate) > 2^-{t + 1} x P_core(top-1)")
    add("knrank", "le", range(7), lambda t: f"candidate in the KN top {2 ** (t + 1) - 1}")
    add("ps", "le", range(16), lambda t: f"P_S(candidate) > 2^-{t + 1}")
    add("mass16", "le", range(8), lambda t: f"core top-64 mass on the candidate's 16-cluster > 2^-{t + 1}")
    add("mass243", "le", range(12), lambda t: f"core top-64 mass on the candidate's 243-cluster > 2^-{t + 1}")
    add("top1", "le", range(8), lambda t: f"P_core(top-1) > 2^-{t + 1}")
    add("tail", "le", range(12), lambda t: f"core mass outside the top-64 > 2^-{t + 1}")
    add("kstar", "ge", range(1, 5), lambda t: f"longest fired suffix context >= {t} tokens")
    add("mstar", "ge", range(1, 6), lambda t: f"a document clause of order >= {t - 1} fired")
    add("cbucket", "ge", range(2, 6), lambda t: f"document count at m* >= {[0, 1, 2, 4, 8, 16][t]}")
    add("lbucket", "ge", range(1, 4), lambda t: f"pointer match length >= {[0, 5, 9, 17][t]}")
    add("is_ptr", "ge", [1], lambda t: "candidate is the pointer candidate")
    add("nocc", "ge", range(1, 7), lambda t: f"candidate earlier in the document >= {[0, 1, 2, 4, 8, 16, 32][t]} times")
    add("dist", "le", range(14), lambda t: f"distance to the candidate's most recent occurrence < {2 ** (t + 1)}")
    add("lrec", "ge", range(1, 9), lambda t: f"suffix match at the most recent occurrence >= {t}")
    add("lbest", "ge", range(1, 9), lambda t: f"best suffix match over the 64 most recent occurrences >= {t}")
    add("nl1", "ge", range(1, 5), lambda t: f"occurrences after the same previous token >= {[0, 1, 2, 4, 8][t]}")
    add("nl2", "ge", range(1, 5), lambda t: f"occurrences after the same previous 2 tokens >= {[0, 1, 2, 4, 8][t]}")
    add("nl3", "ge", range(1, 5), lambda t: f"occurrences after the same previous 3 tokens >= {[0, 1, 2, 4, 8][t]}")
    add("n64", "ge", range(1, 5), lambda t: f"occurrences within the last 64 tokens >= {[0, 1, 2, 3, 5][t]}")
    add("cand_sub2", "eq", range(16), lambda t: f"candidate level-2 sub-index is {t}")
    add("cand_sub3", "eq", range(16), lambda t: f"candidate level-3 sub-index is {t}")
    add("cand_idx4", "eq", range(64), lambda t: f"candidate index inside its level-3 cluster is {t}")
    feat_a = np.array(feat, np.int64)
    thr_a = np.array(thr, np.int64)
    LT = np.zeros((len(names), 256), np.bool_)
    v = np.arange(256)
    for j in range(len(names)):
        k, t = kind[j], thr_a[j]
        if k == "eq":
            LT[j] = v == t
        elif k == "ge":
            LT[j] = v >= t
        elif k == "le":
            LT[j] = v <= t
        else:
            LT[j] = ((v >> t) & 1) == 1
    family = [RAW_FAMILY[RAW[f]] for f in feat_a]
    return LiteralSet(names=names, feat=feat_a, kind=kind, thr=thr_a, family=family, LT=LT)


def literal_sets(all_lits: LiteralSet, idx4_max: int) -> dict[str, LiteralSet]:
    """MAIN, NOREL and CTXID (module docstring). The candidate-id index literals stop at the largest
    level-3 cluster (idx4_max + 1 values)."""
    main = all_lits.select(["context", "candidate", "relational", "reflective", "copy"])
    norel = all_lits.select(["context", "candidate", "reflective", "copy"])
    ctxid = [i for i in all_lits.select(["context", "candidate_id"], extra_raw=["cand_c16"])
             if not (all_lits.feat[i] == R["cand_idx4"] and all_lits.thr[i] > idx4_max)]
    return {"MAIN": all_lits.subset(main), "NOREL": all_lits.subset(norel),
            "CTXID": all_lits.subset(np.array(ctxid, np.int64))}


# ------------------------------------------------------------------ deterministic hashing
def mix64_np(z) -> np.ndarray:
    z = np.asarray(z).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
    return z ^ (z >> np.uint64(31))


def split_2ab(doc_index: np.ndarray, residue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bool [d] masks of the 2a and 2b documents: the N_2A_DOCS residue-2 documents with the smallest
    mix64(doc_id XOR SALT_2AB) (ties by doc_id) are 2a, the other residue-2 documents 2b."""
    doc_index = np.asarray(doc_index).astype(np.int64)
    r2 = np.flatnonzero(np.asarray(residue) == 2)
    h = mix64_np(doc_index[r2].astype(np.uint64) ^ np.uint64(SALT_2AB))
    order = r2[np.lexsort((doc_index[r2], h))]
    a = np.zeros(doc_index.shape[0], bool)
    b = np.zeros(doc_index.shape[0], bool)
    a[order[:N_2A_DOCS]] = True
    b[order[N_2A_DOCS:]] = True
    return a, b


def doc_positions(doc_start: np.ndarray, doc_mask: np.ndarray) -> np.ndarray:
    """int64 positions of the documents with doc_mask True (ascending)."""
    ds = np.asarray(doc_start).astype(np.int64)
    parts = [np.arange(ds[d], ds[d + 1], dtype=np.int64) for d in np.flatnonzero(doc_mask)]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def gold_slots(top_ids: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """int64 [q]: index of the gold among the top-64 (-1 when absent)."""
    hit = np.asarray(top_ids).astype(np.int64) == np.asarray(gold).astype(np.int64)[:, None]
    any_ = hit.any(axis=1)
    return np.where(any_, hit.argmax(axis=1), -1).astype(np.int64)


def sample_positions(eligible: np.ndarray, n: int, seed: int, tag: int = 1) -> np.ndarray:
    """Sorted sample of `n` positions from `eligible` without replacement (numpy SeedSequence([seed,
    SALT_POS, tag]); all of them if fewer)."""
    eligible = np.asarray(eligible).astype(np.int64)
    if eligible.shape[0] <= n:
        return eligible.copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SALT_POS, int(tag)]))
    return np.sort(rng.choice(eligible, size=int(n), replace=False))


@njit(cache=False)
def _mix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=False)
def k_negatives(top_p, gslot, doc_id, offset, seed, K, out):
    """K distinct negative slots per position, drawn without replacement with probabilities proportional
    to P_core over the top-64 without the gold (Gumbel top-K; the uniforms are mix64 of (seed, stable
    document ID, in-document offset, slot), so a draw does not depend on chunking or order)."""
    q = top_p.shape[0]
    kk = top_p.shape[1]
    key = np.empty(kk)
    for i in range(q):
        h0 = _mix64(_mix64(np.uint64(seed) ^ np.uint64(SALT_NEG)) + np.uint64(doc_id[i]))
        h0 = _mix64(h0 + np.uint64(offset[i]))
        for j in range(kk):
            if j == gslot[i] or top_p[i, j] <= 0.0:
                key[j] = -1e300
                continue
            h = _mix64(h0 + np.uint64(j + 1))
            u = (float(h >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)
            key[j] = math.log(top_p[i, j]) - math.log(-math.log(u))
        for k in range(K):
            best = -1
            bv = -1e301
            for j in range(kk):
                if key[j] > bv:
                    bv = key[j]
                    best = j
            out[i, k] = best
            key[best] = -1e302


def draw_negatives(top_p: np.ndarray, gslot: np.ndarray, doc_id: np.ndarray, offset: np.ndarray, seed: int,
                   K: int = K_NEG) -> np.ndarray:
    out = np.zeros((top_p.shape[0], K), np.int64)
    k_negatives(np.ascontiguousarray(top_p, dtype=np.float64), np.asarray(gslot, np.int64),
                np.asarray(doc_id, np.int64), np.asarray(offset, np.int64), int(seed), int(K), out)
    return out


# ------------------------------------------------------------------ bucket parts
def bucket_parts(bucket: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mstar 0 none / m + 1, cbucket 0 none / c + 1, lbucket) of full-component bucket ids (520)."""
    lay = BucketLayout.of(None)
    b = np.asarray(bucket).astype(np.int64)
    if lay.n_buckets != 520 or b.min() < 0 or b.max() >= 520:
        raise ValueError("bucket ids of the full component set (520) expected")
    l = b % lay.n_l
    rest = b // lay.n_l
    mc = rest % lay.n_mc
    j = np.where(mc > 0, (mc - 1) // 5, -1)
    c = np.where(mc > 0, (mc - 1) % 5, -1)
    orders = np.array(lay.d_orders, np.int64)
    mstar = np.where(j >= 0, orders[np.maximum(j, 0)] + 1, 0)
    return mstar, np.where(c >= 0, c + 1, 0), l


# ------------------------------------------------------------------ pair features
@njit(cache=False)
def _bl(x):
    b = 0
    while (x >> (b + 1)) > 0:
        b += 1
    return b


@njit(cache=False)
def _nlog2b(x, cap):
    """min(floor(-log2(x)), cap) for x > 0 (0 for x >= 1; cap for x <= 0)."""
    if x <= 0.0:
        return cap
    if x >= 1.0:
        return 0
    v = int(math.floor(-math.log2(x)))
    if v < 0:
        v = 0
    return v if v < cap else cap


@njit(cache=False)
def _b5(n):
    """0, 1, 2-3, 4-7, 8+ -> 0..4."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    if n <= 7:
        return 3
    return 4


@njit(cache=False)
def k_pair_features(tok, doc_start, pos, top_ids, top_p, kn_ids, kn_p, kstar, mstar, cb, lb, ptr_len, ptr_cand,
                    slots, c1, c2, sub3, idx4, ctype, fcls, feat, lab, cand):
    """Raw features of the pairs (pos[i], top_ids[i, slots[i, s]]) at rows i * S + s (module docstring;
    the column order is RAW). pos ascending; tok is the group's stream, doc_start its boundaries."""
    q = pos.shape[0]
    S = slots.shape[1]
    Vv = c1.shape[0]
    occ_start = np.zeros(Vv + 1, np.int64)
    fill = np.zeros(Vv, np.int64)
    occ = np.zeros(1, np.int64)
    cnt2d = np.zeros(256, np.int64)
    cnt16w = np.zeros(16, np.int64)
    cnt2w = np.zeros(256, np.int64)
    knpos = np.full(Vv, -1, np.int64)
    m16 = np.zeros(16)
    m2 = np.zeros(256)
    ctx16 = np.zeros(N_OFF, np.int64)
    ctxsub = np.zeros(N_OFF, np.int64)
    ctxtok = np.zeros(N_OFF, np.int64)
    ctx2 = np.zeros(N_OFF, np.int64)
    cur = -1
    a = 0
    b = 0
    cursor = 0
    d = 0
    nd = doc_start.shape[0] - 1
    for i in range(q):
        p = pos[i]
        while d < nd and doc_start[d + 1] <= p:
            d += 1
        if d != cur:
            cur = d
            a = doc_start[d]
            b = doc_start[d + 1]
            occ_start[:] = 0
            for x in range(a, b):
                occ_start[tok[x] + 1] += 1
            for v in range(Vv):
                occ_start[v + 1] += occ_start[v]
            occ = np.zeros(b - a, np.int64)
            for v in range(Vv):
                fill[v] = occ_start[v]
            for x in range(a, b):
                t = tok[x]
                occ[fill[t]] = x
                fill[t] += 1
            cnt2d[:] = 0
            cnt16w[:] = 0
            cnt2w[:] = 0
            cursor = a
        while cursor < p:
            t = tok[cursor]
            cnt2d[c2[t]] += 1
            cnt16w[c1[t]] += 1
            cnt2w[c2[t]] += 1
            if cursor - WIN >= a:
                u = tok[cursor - WIN]
                cnt16w[c1[u]] -= 1
                cnt2w[c2[u]] -= 1
            cursor += 1
        y = tok[p]
        for k in range(N_OFF):
            x = p - 1 - k
            if x >= a:
                tt = tok[x]
                ctxtok[k] = tt
                ctx16[k] = c1[tt]
                ctx2[k] = c2[tt]
                ctxsub[k] = c2[tt] % 16
            else:
                ctxtok[k] = -1
                ctx16[k] = 16
                ctx2[k] = -1
                ctxsub[k] = 16
        prevtype = ctype[tok[p - 1]] if p - 1 >= a else 0
        depthb = _bl(p - a) if p > a else 0
        if depthb > 14:
            depthb = 14
        top1 = top_p[i, 0]
        s64 = 0.0
        for j in range(TOPK):
            pj = top_p[i, j]
            s64 += pj
            cj = top_ids[i, j]
            m16[c1[cj]] += pj
            m2[c2[cj]] += pj
        tail = 1.0 - s64
        for j in range(TOPK):
            knpos[kn_ids[i, j]] = j
        pfire = ptr_len[i] >= 5
        for s in range(S):
            j = slots[i, s]
            c = top_ids[i, j]
            r = i * S + s
            cand[r] = c
            lab[r] = 1 if c == y else 0
            for k in range(N_OFF):
                feat[r, k] = ctx16[k]
                feat[r, N_OFF + k] = ctxsub[k]
            feat[r, 16] = prevtype
            feat[r, 17] = depthb
            feat[r, 18] = c1[c]
            feat[r, 19] = c2[c]
            feat[r, 20] = ctype[c]
            feat[r, 21] = fcls[c]
            e16 = 0
            e2 = 0
            et = 0
            for k in range(N_OFF):
                if ctxtok[k] >= 0:
                    if ctx16[k] == c1[c]:
                        e16 |= 1 << k
                    if ctx2[k] == c2[c]:
                        e2 |= 1 << k
                    if ctxtok[k] == c:
                        et |= 1 << k
            feat[r, 22] = e16
            feat[r, 23] = e2
            feat[r, 24] = et
            feat[r, 25] = _b5(cnt16w[c1[c]])
            feat[r, 26] = _b5(cnt2w[c2[c]])
            nb = cnt2d[c2[c]]
            feat[r, 27] = 5 if nb >= 16 else _b5(nb)
            pc = top_p[i, j]
            feat[r, 28] = _nlog2b(pc, 20)
            feat[r, 29] = _bl(j + 1)
            feat[r, 30] = _nlog2b(pc / top1, 12) if top1 > 0.0 else 12
            kr = knpos[c]
            if kr >= 0:
                feat[r, 31] = _bl(kr + 1)
                feat[r, 32] = _nlog2b(kn_p[i, kr], 20)
            else:
                feat[r, 31] = 7
                feat[r, 32] = 20
            feat[r, 33] = _nlog2b(m16[c1[c]], 8)
            feat[r, 34] = _nlog2b(m2[c2[c]], 12)
            feat[r, 35] = _nlog2b(top1, 8)
            feat[r, 36] = _nlog2b(tail, 12)
            feat[r, 37] = kstar[i]
            feat[r, 38] = mstar[i]
            feat[r, 39] = cb[i]
            feat[r, 40] = lb[i]
            feat[r, 41] = 1 if (pfire and ptr_cand[i] == c) else 0
            # copy features: occurrences of c in [a, p)
            st = occ_start[c]
            en = occ_start[c + 1]
            lo_i = st
            hi_i = en
            while lo_i < hi_i:
                mid = (lo_i + hi_i) // 2
                if occ[mid] < p:
                    lo_i = mid + 1
                else:
                    hi_i = mid
            k_occ = lo_i - st
            n_b = 0
            if k_occ >= 32:
                n_b = 6
            elif k_occ >= 16:
                n_b = 5
            elif k_occ >= 8:
                n_b = 4
            elif k_occ >= 4:
                n_b = 3
            elif k_occ >= 2:
                n_b = 2
            elif k_occ >= 1:
                n_b = 1
            feat[r, 42] = n_b
            lrec = 0
            lbest = 0
            nl1 = 0
            nl2 = 0
            nl3 = 0
            n64 = 0
            distb = 14
            if k_occ > 0:
                qr = occ[st + k_occ - 1]
                db = _bl(p - qr)
                distb = db if db < 14 else 14
                stop = k_occ - OCC_SCAN
                if stop < 0:
                    stop = 0
                for jj in range(k_occ - 1, stop - 1, -1):
                    qq = occ[st + jj]
                    L = 0
                    while L < 8 and qq - 1 - L >= a and tok[p - 1 - L] == tok[qq - 1 - L]:
                        L += 1
                    if jj == k_occ - 1:
                        lrec = L
                    if L > lbest:
                        lbest = L
                    if L >= 1:
                        nl1 += 1
                    if L >= 2:
                        nl2 += 1
                    if L >= 3:
                        nl3 += 1
                    if p - qq <= 64:
                        n64 += 1
            feat[r, 43] = distb
            feat[r, 44] = lrec
            feat[r, 45] = lbest
            feat[r, 46] = _b5(nl1)
            feat[r, 47] = _b5(nl2)
            feat[r, 48] = _b5(nl3)
            feat[r, 49] = n64 if n64 <= 2 else (3 if n64 <= 4 else 4)
            feat[r, 50] = c2[c] % 16
            feat[r, 51] = sub3[c]
            feat[r, 52] = idx4[c]
        for j in range(TOPK):
            cj = top_ids[i, j]
            m16[c1[cj]] = 0.0
            m2[c2[cj]] = 0.0
            knpos[kn_ids[i, j]] = -1


@dataclass
class PosArrays:
    """Per-position inputs of the pair kernel and the read-out, gathered for a sorted position list of
    one stream (hprime or BLiMP)."""

    pos: np.ndarray            # int64 [q] positions in the stream
    top_ids: np.ndarray        # int64 [q, 64] core top-64
    top_p: np.ndarray          # float64 [q, 64]
    kn_ids: np.ndarray         # int64 [q, 64] KN top-64
    kn_p: np.ndarray           # float64 [q, 64]
    kstar: np.ndarray          # int64 [q]
    bucket: np.ndarray         # int64 [q]
    ptr_len: np.ndarray        # int64 [q]
    ptr_cand: np.ndarray       # int64 [q]
    p_gold: np.ndarray         # float64 [q] core P(gold)
    gold: np.ndarray           # int64 [q]


def gather(arrs: dict, pos: np.ndarray, kstar_name: str = "kstar_noHp") -> PosArrays:
    pos = np.asarray(pos).astype(np.int64)
    return PosArrays(pos=pos, top_ids=np.asarray(arrs["core_top_ids"][pos]).astype(np.int64),
                     top_p=np.asarray(arrs["core_top_p"][pos], dtype=np.float64),
                     kn_ids=np.asarray(arrs["kn_top_ids"][pos]).astype(np.int64),
                     kn_p=np.asarray(arrs["kn_top_p"][pos], dtype=np.float64),
                     kstar=np.asarray(arrs[kstar_name][pos]).astype(np.int64),
                     bucket=np.asarray(arrs["bucket"][pos]).astype(np.int64),
                     ptr_len=np.asarray(arrs["ptr_len"][pos]).astype(np.int64),
                     ptr_cand=np.asarray(arrs["ptr_cand"][pos]).astype(np.int64),
                     p_gold=np.asarray(arrs["core_p_gold"][pos], dtype=np.float64),
                     gold=np.asarray(arrs["gold"][pos]).astype(np.int64))


def pair_features(tok: np.ndarray, doc_start: np.ndarray, pa: PosArrays, slots: np.ndarray, tabs: dict
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feat uint8 [q * S, NRAW], lab uint8 [q * S], cand uint16 [q * S]) for the slots [q, S]."""
    q, S = slots.shape
    feat = np.zeros((q * S, NRAW), np.uint8)
    lab = np.zeros(q * S, np.uint8)
    cand = np.zeros(q * S, np.uint16)
    if q == 0:
        return feat, lab, cand
    mstar, cb, lb = bucket_parts(pa.bucket)
    k_pair_features(np.asarray(tok).astype(np.int64), np.asarray(doc_start).astype(np.int64), pa.pos, pa.top_ids,
                    pa.top_p, pa.kn_ids, pa.kn_p, pa.kstar, mstar, cb, lb, pa.ptr_len, pa.ptr_cand,
                    np.ascontiguousarray(slots, dtype=np.int64), tabs["c1"], tabs["c2"], tabs["sub3"], tabs["idx4"],
                    tabs["ctype"], tabs["fcls"], feat, lab, cand)
    return feat, lab, cand


# ------------------------------------------------------------------ bit-parallel TM evaluation
def literal_csr(ls: LiteralSet) -> tuple[np.ndarray, np.ndarray]:
    """CSR over (raw feature f, raw value v) -> literal indices true for that value (row f * 256 + v)."""
    rows: list[list[int]] = [[] for _ in range(NRAW * 256)]
    for j in range(ls.n):
        f = int(ls.feat[j])
        for v in np.flatnonzero(ls.LT[j]):
            rows[f * 256 + int(v)].append(j)
    ptr = np.zeros(NRAW * 256 + 1, np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    idx = np.array([j for r in rows for j in r], np.int64)
    return ptr, idx


_DEBRUIJN = np.array([0, 47, 1, 56, 48, 27, 2, 60, 57, 49, 41, 37, 28, 16, 3, 61, 54, 58, 35, 52, 50, 42, 21, 44,
                      38, 32, 29, 23, 17, 11, 4, 62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
                      25, 39, 14, 33, 19, 30, 9, 24, 13, 18, 8, 12, 7, 6, 5, 63], np.int64)


@njit(cache=False)
def _bitscan(x, table):
    """index of the lowest set bit of x (x != 0), de Bruijn method on x ^ (x - 1)."""
    return table[((x ^ (x - np.uint64(1))) * np.uint64(0x03F79D71B4CB0A89)) >> np.uint64(58)]


@njit(cache=False)
def k_tm_score(feat, csr_ptr, csr_idx, n_lit, cl_ptr, cl_lit, cl_neg, cl_w, table, out):
    """out[r] += sum of the weights of the clauses that fire on pair r (64 pairs per word)."""
    n = feat.shape[0]
    nf = feat.shape[1]
    nb = (n + 63) // 64
    ncl = cl_w.shape[0]
    cols = np.zeros(n_lit, np.uint64)
    one = np.uint64(1)
    for blk in range(nb):
        r0 = blk * 64
        m = min(64, n - r0)
        cols[:] = 0
        for r in range(m):
            bit = one << np.uint64(r)
            for f in range(nf):
                row = f * 256 + feat[r0 + r, f]
                for e in range(csr_ptr[row], csr_ptr[row + 1]):
                    cols[csr_idx[e]] |= bit
        full = ~np.uint64(0) if m == 64 else (one << np.uint64(m)) - one
        for c in range(ncl):
            a = full
            for e in range(cl_ptr[c], cl_ptr[c + 1]):
                if cl_neg[e]:
                    a &= ~cols[cl_lit[e]]
                else:
                    a &= cols[cl_lit[e]]
                if a == np.uint64(0):
                    break
            if a != np.uint64(0):
                w = cl_w[c]
                while a != np.uint64(0):
                    out[r0 + _bitscan(a, table)] += w
                    a &= a - one


class TMEval:
    """Scores of a clause set over raw pair features (streamed; no literal matrix is materialised)."""

    def __init__(self, ls: LiteralSet, cs: CS.ClauseSet) -> None:
        self.ls = ls
        self.cs = cs
        self.csr = literal_csr(ls)

    def score(self, feat: np.ndarray) -> np.ndarray:
        out = np.zeros(feat.shape[0], np.int64)
        k_tm_score(np.ascontiguousarray(feat), self.csr[0], self.csr[1], self.ls.n, self.cs.ptr, self.cs.lits,
                   self.cs.negs, self.cs.w, _DEBRUIJN, out)
        return out.astype(np.float64)


def train_tm_epoch(tm, feat: np.ndarray, y: np.ndarray, ls: LiteralSet, seed: int, epoch: int,
                   chunk: int = 65536) -> float:
    """One epoch of tmu training over (feat, y): a global permutation keyed by (seed, epoch), then
    tm.fit on consecutive chunks (tmu shuffles inside a chunk with its own seeded generator). Returns
    the wall seconds of the fit calls."""
    n = feat.shape[0]
    perm = np.random.default_rng(np.random.SeedSequence([int(seed), 0x7E, int(epoch)])).permutation(n)
    t = 0.0
    yy = np.asarray(y).astype(np.uint32)
    for lo in range(0, n, chunk):
        idx = perm[lo:lo + chunk]
        X = np.ascontiguousarray(ls.matrix(feat[idx], np.uint32))
        t0 = time.perf_counter()
        tm.fit(X, np.ascontiguousarray(yy[idx]))
        t += time.perf_counter() - t0
        del X
    return t


# ------------------------------------------------------------------ table logistic regression (LR, LR2)
class TableLR:
    """Logistic regression on the literals of a set (pairwise=False) or on all pairwise literal products
    x_i x_j, i <= j (pairwise=True; x_i x_i = x_i). Exact evaluation through tables over the raw
    features: T_f[a] = sum_{i <= j in f} W_ij L_i(a) L_j(a); T_fg[a, b] = sum_{i in f, j in g} W_ij L_i(a) L_j(b).
    Loss: sum over pairs of log(1 + e^s) - y s + (l2 / 2) ||W||^2 (bias not penalised). Generalises
    `copy_selector.TableLR` to any literal set of this module."""

    def __init__(self, ls: LiteralSet, pairwise: bool) -> None:
        self.ls = ls
        self.pairwise = bool(pairwise)
        L = ls.n
        self.L = L
        raws = [int(f) for f in ls.raw_used()]
        self.raws = raws
        self.nf = len(raws)
        self.nv = np.array([int(RAW_MAX[f]) + 1 for f in raws], np.int64)
        self.col = np.array(raws, np.int64)
        self.groups = [np.flatnonzero(ls.feat == f) for f in raws]
        self.LTf = [ls.LT[g][:, : self.nv[k]].astype(np.float64) for k, g in enumerate(self.groups)]
        self.iu = np.triu_indices(L) if self.pairwise else (np.arange(L), np.arange(L))
        self.n_params = int(self.iu[0].shape[0]) + 1
        off = 0
        self.diag_off = np.zeros(self.nf, np.int64)
        for k in range(self.nf):
            self.diag_off[k] = off
            off += int(self.nv[k])
        pf, pg, poff, png = [], [], [], []
        if self.pairwise:
            for f in range(self.nf):
                for g in range(f + 1, self.nf):
                    pf.append(f)
                    pg.append(g)
                    poff.append(off)
                    png.append(int(self.nv[g]))
                    off += int(self.nv[f] * self.nv[g])
        self.pair_f = np.array(pf, np.int64)
        self.pair_g = np.array(pg, np.int64)
        self.pair_off = np.array(poff, np.int64)
        self.pair_ng = np.array(png, np.int64)
        self.n_table = int(off)

    def W_of(self, theta: np.ndarray) -> np.ndarray:
        W = np.zeros((self.L, self.L))
        W[self.iu] = theta[:-1]
        return W

    def tables(self, theta: np.ndarray) -> np.ndarray:
        W = self.W_of(theta)
        T = np.zeros(self.n_table)
        for k in range(self.nf):
            g = self.groups[k]
            Wff = np.triu(W[np.ix_(g, g)])
            Lf = self.LTf[k]
            T[self.diag_off[k]: self.diag_off[k] + self.nv[k]] = np.einsum("ia,ij,ja->a", Lf, Wff, Lf)
        for q in range(self.pair_f.shape[0]):
            f, g = int(self.pair_f[q]), int(self.pair_g[q])
            blk = self.LTf[f].T @ W[np.ix_(self.groups[f], self.groups[g])] @ self.LTf[g]
            T[self.pair_off[q]: self.pair_off[q] + blk.size] = blk.reshape(-1)
        return T

    def grad_from_hist(self, H: np.ndarray) -> np.ndarray:
        G = np.zeros((self.L, self.L))
        for k in range(self.nf):
            g = self.groups[k]
            Lf = self.LTf[k]
            h = H[self.diag_off[k]: self.diag_off[k] + self.nv[k]]
            G[np.ix_(g, g)] = np.triu((Lf * h[None, :]) @ Lf.T)
        for q in range(self.pair_f.shape[0]):
            f, g = int(self.pair_f[q]), int(self.pair_g[q])
            h = H[self.pair_off[q]: self.pair_off[q] + self.nv[f] * self.nv[g]].reshape(self.nv[f], self.nv[g])
            G[np.ix_(self.groups[f], self.groups[g])] = self.LTf[f] @ h @ self.LTf[g].T
        return G[self.iu]

    def _cols(self, feat: np.ndarray) -> np.ndarray:
        f = np.ascontiguousarray(np.asarray(feat)[:, self.col])
        if f.size and np.any(f.max(axis=0) > self.nv - 1):
            raise ValueError("raw value above RAW_MAX")
        return f

    def score(self, feat: np.ndarray, theta: np.ndarray, T: np.ndarray | None = None) -> np.ndarray:
        T = self.tables(theta) if T is None else T
        out = np.empty(feat.shape[0])
        CS.k_table_score(self._cols(feat), self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng,
                         T, float(theta[-1]), out)
        return out

    def fit(self, feat: np.ndarray, y: np.ndarray, l2: float, theta0: np.ndarray | None = None,
            maxiter: int = 300) -> tuple[np.ndarray, dict]:
        from scipy.optimize import minimize

        fc = self._cols(feat)
        yf = np.asarray(y).astype(np.float64)
        H = np.zeros(self.n_table)
        s = np.empty(fc.shape[0])
        n_eval = [0]

        def obj(theta):
            n_eval[0] += 1
            T = self.tables(theta)
            CS.k_table_score(fc, self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng, T,
                             float(theta[-1]), s)
            lse = np.logaddexp(0.0, s)
            w = theta[:-1]
            f = float(np.sum(lse - yf * s)) + 0.5 * l2 * float(w @ w)
            r = 0.5 * (1.0 + np.tanh(0.5 * s)) - yf
            H[:] = 0.0
            CS.k_table_hist(fc, r, self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng, H)
            g = np.empty_like(theta)
            g[:-1] = self.grad_from_hist(H) + l2 * w
            g[-1] = float(r.sum())
            return f, g

        x0 = np.zeros(self.n_params) if theta0 is None else np.asarray(theta0, np.float64).copy()
        t0 = time.perf_counter()
        res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-5, "maxcor": 20})
        info = {"l2": float(l2), "nit": int(res.nit), "n_eval": int(n_eval[0]), "converged": bool(res.success),
                "stopped_at_cap": bool(int(res.nit) >= int(maxiter)), "message": str(res.message),
                "loss": float(res.fun), "fit_s": time.perf_counter() - t0, "pairwise": self.pairwise,
                "n_params": self.n_params, "maxiter": int(maxiter)}
        return res.x, info


# ------------------------------------------------------------------ gradient-boosted trees on raw features
@dataclass
class HGBLayout:
    """Columns of the HGB design matrix: (raw feature, kind, bit or None, category map or None)."""

    cols: list
    categorical: np.ndarray


def hgb_layout(ls: LiteralSet) -> HGBLayout:
    cols, cat = [], []
    for f in [int(x) for x in ls.raw_used()]:
        g = np.flatnonzero(ls.feat == f)
        kinds = {ls.kind[j] for j in g}
        if len(kinds) != 1:
            raise ValueError(f"raw feature {RAW[f]} mixes literal kinds {kinds}")
        k = kinds.pop()
        if k == "bit":
            for j in g:
                cols.append((f, "bit", int(ls.thr[j]), None))
                cat.append(False)
        elif k == "eq":
            vals = sorted(int(ls.thr[j]) for j in g)
            cmap = np.full(256, len(vals), np.int64)          # values without a literal share one category
            cmap[np.array(vals)] = np.arange(len(vals))
            if len(vals) + 1 > 254:
                raise ValueError("too many categories for HistGradientBoosting")
            cols.append((f, "cat", None, cmap))
            cat.append(True)
        else:
            cols.append((f, "num", None, None))
            cat.append(False)
    return HGBLayout(cols=cols, categorical=np.array(cat, bool))


def hgb_matrix(feat: np.ndarray, lay: HGBLayout, dtype=np.float64) -> np.ndarray:
    feat = np.asarray(feat)
    out = np.empty((feat.shape[0], len(lay.cols)), dtype)
    for k, (f, kind, bit, cmap) in enumerate(lay.cols):
        v = feat[:, f].astype(np.int64)
        if kind == "bit":
            out[:, k] = (v >> bit) & 1
        elif kind == "cat":
            out[:, k] = cmap[v]
        else:
            out[:, k] = v
    return out


@dataclass
class HGBTrees:
    """The trees of a fitted binary HistGradientBoostingClassifier as flat arrays with sklearn's raw-data
    prediction semantics (`_predictor._predict_one_from_raw_data`) AND its input preprocessing: with
    categorical columns, sklearn reorders the columns (categorical first, then numerical, each in their
    original order) and ordinal-encodes each categorical column by the sorted categories seen in training
    (unknown categories become NaN, i.e. missing). col_* describe the TRANSFORMED columns."""

    tree_off: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    is_leaf: np.ndarray
    is_cat: np.ndarray
    bitset_idx: np.ndarray
    miss_left: np.ndarray
    value: np.ndarray
    raw_left_bits: np.ndarray       # uint32 [n_cat_splits + 1, 8]
    known_bits: np.ndarray          # uint32 [n_cat_features + 1, 8]
    f_idx_map: np.ndarray           # int64 [n_columns]
    baseline: float
    col_f: np.ndarray               # raw feature of every transformed column
    col_kind: np.ndarray            # 0 numeric, 1 bit, 2 categorical
    col_bit: np.ndarray
    cat_tab: np.ndarray             # int64 [n_columns, 256]: ordinal code of a raw value, -1 unknown (missing)


def hgb_trees(model, lay: HGBLayout) -> HGBTrees:
    offs, feats, thr, left, right, leaf, iscat, bidx, mleft, val, bits = [0], [], [], [], [], [], [], [], [], [], []
    nbits = 0
    for it in model._predictors:
        if len(it) != 1:
            raise ValueError("binary classifier expected (one tree per iteration)")
        pr = it[0]
        nodes = pr.nodes
        base = offs[-1]
        feats.append(nodes["feature_idx"].astype(np.int64))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int64) + base)
        right.append(nodes["right"].astype(np.int64) + base)
        leaf.append(nodes["is_leaf"].astype(np.uint8))
        iscat.append(nodes["is_categorical"].astype(np.uint8))
        bidx.append(nodes["bitset_idx"].astype(np.int64) + nbits)
        mleft.append(nodes["missing_go_to_left"].astype(np.uint8))
        val.append(nodes["value"].astype(np.float64))
        rb = np.asarray(pr.raw_left_cat_bitsets, np.uint32).reshape(-1, 8)
        bits.append(rb)
        nbits += rb.shape[0]
        offs.append(base + nodes.shape[0])
    known, fmap = model._bin_mapper.make_known_categories_bitsets()
    known = np.asarray(known, np.uint32).reshape(-1, 8)
    rbits = np.concatenate(bits) if nbits else np.zeros((0, 8), np.uint32)
    ncol = len(lay.cols)
    pre = model._preprocessor
    cat_mask = lay.categorical.copy()
    if pre is None:
        perm = np.arange(ncol)
        cats = [None] * ncol
    else:
        if not np.array_equal(np.asarray(model.is_categorical_, bool), cat_mask):
            raise ValueError("the model's categorical mask differs from the layout")
        perm = np.r_[np.flatnonzero(cat_mask), np.flatnonzero(~cat_mask)]
        enc = pre.named_transformers_["encoder"]
        cats = [np.asarray(c, np.float64) for c in enc.categories_] + [None] * int((~cat_mask).sum())
    col_f = np.zeros(ncol, np.int64)
    col_kind = np.zeros(ncol, np.int64)
    col_bit = np.zeros(ncol, np.int64)
    cat_tab = np.full((ncol, 256), -1, np.int64)
    for k in range(ncol):
        f, kind, bit, cmap = lay.cols[int(perm[k])]
        col_f[k] = f
        col_kind[k] = {"num": 0, "bit": 1, "cat": 2}[kind]
        col_bit[k] = bit if bit is not None else 0
        if kind == "cat":
            cc = cats[k]
            if cc is None:
                cat_tab[k] = cmap
            else:
                cc = cc[~np.isnan(cc)]
                codes = cmap.astype(np.float64)
                pos = np.searchsorted(cc, codes)
                ok = (pos < cc.shape[0]) & (cc[np.minimum(pos, cc.shape[0] - 1)] == codes)
                cat_tab[k] = np.where(ok, pos, -1)
    return HGBTrees(tree_off=np.array(offs, np.int64), feature=np.concatenate(feats), threshold=np.concatenate(thr),
                    left=np.concatenate(left), right=np.concatenate(right), is_leaf=np.concatenate(leaf),
                    is_cat=np.concatenate(iscat), bitset_idx=np.concatenate(bidx), miss_left=np.concatenate(mleft),
                    value=np.concatenate(val), raw_left_bits=np.vstack([rbits, np.zeros((1, 8), np.uint32)]),
                    known_bits=np.vstack([known, np.zeros((1, 8), np.uint32)]),
                    f_idx_map=np.asarray(fmap, np.int64).reshape(-1),
                    baseline=float(np.asarray(model._baseline_prediction).reshape(-1)[0]),
                    col_f=col_f, col_kind=col_kind, col_bit=col_bit, cat_tab=cat_tab)


@njit(cache=False)
def k_hgb_raw(feat, col_f, col_kind, col_bit, cat_tab, tree_off, feature, threshold, left, right, is_leaf, is_cat,
              bitset_idx, miss_left, value, raw_left_bits, known_bits, f_idx_map, baseline, stages, out):
    """out[k, r] = baseline + the first stages[k] trees (added in tree order, as sklearn's _raw_predict),
    evaluated on the raw uint8 features through the transformed column layout."""
    n = feat.shape[0]
    ns = stages.shape[0]
    nt = tree_off.shape[0] - 1
    for r in range(n):
        acc = baseline
        k = 0
        for t in range(nt):
            node = tree_off[t]
            while is_leaf[node] == 0:
                c = feature[node]
                raw = feat[r, col_f[c]]
                kind = col_kind[c]
                if kind == 2:
                    v = cat_tab[c, raw]
                    if v < 0:
                        node = left[node] if miss_left[node] else right[node]
                    elif is_cat[node] == 0:
                        node = left[node] if float(v) <= threshold[node] else right[node]
                    else:
                        b = bitset_idx[node]
                        if (raw_left_bits[b, v // 32] >> np.uint32(v % 32)) & np.uint32(1):
                            node = left[node]
                        elif (known_bits[f_idx_map[c], v // 32] >> np.uint32(v % 32)) & np.uint32(1):
                            node = right[node]
                        else:
                            node = left[node] if miss_left[node] else right[node]
                else:
                    if kind == 0:
                        x = float(raw)
                    else:
                        x = float((raw >> col_bit[c]) & 1)
                    if x <= threshold[node]:
                        node = left[node]
                    else:
                        node = right[node]
            acc += value[node]
            while k < ns and stages[k] == t + 1:
                out[k, r] = acc
                k += 1


def hgb_staged_scores(trees: HGBTrees, feat: np.ndarray, stages: Sequence[int], chunk: int = 262144) -> np.ndarray:
    """[len(stages), n] raw decision values after the given numbers of trees (exact sklearn semantics;
    checked against decision_function by the callers)."""
    st = np.array(sorted(int(s) for s in stages), np.int64)
    if st[-1] > trees.tree_off.shape[0] - 1:
        raise ValueError(f"the model has fewer than {st[-1]} trees")
    out = np.zeros((st.shape[0], feat.shape[0]))
    for lo in range(0, feat.shape[0], chunk):
        f = np.ascontiguousarray(np.asarray(feat[lo:lo + chunk]))
        o = np.zeros((st.shape[0], f.shape[0]))
        k_hgb_raw(f, trees.col_f, trees.col_kind, trees.col_bit, trees.cat_tab, trees.tree_off, trees.feature,
                  trees.threshold, trees.left, trees.right, trees.is_leaf, trees.is_cat, trees.bitset_idx,
                  trees.miss_left, trees.value, trees.raw_left_bits, trees.known_bits, trees.f_idx_map,
                  trees.baseline, st, o)
        out[:, lo:lo + f.shape[0]] = o
    return out


# ------------------------------------------------------------------ read-out
def readout_data(pa: PosArrays, gslot: np.ndarray) -> CS.ReadoutData:
    """Read-out inputs of every top-64 pair (64 per position, offsets 64 i)."""
    q = pa.pos.shape[0]
    offs = np.arange(q + 1, dtype=np.int64) * TOPK
    return CS.ReadoutData(offs=offs, gslot=np.asarray(gslot, np.int64), lpgold=np.log(pa.p_gold),
                          bucket=np.asarray(pa.bucket, np.int64), pc=np.ascontiguousarray(pa.top_p.reshape(-1)))


def standardise(score: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(score))
    sd = float(np.std(score))
    return mu, (sd if sd > 0 else 1.0)


def fit_readout(score: np.ndarray, rd: CS.ReadoutData, n_buckets: int = 520, per_bucket: bool = True) -> dict:
    """Standardise the scores on these pairs, fit the pooled (beta, a), then (optionally) the per-bucket
    MAP pairs. Returns {"mu", "sigma", "pooled", "beta", "alpha", "info", "lp" (ln P(gold) per position)}."""
    mu, sd = standardise(score)
    z = (score - mu) / sd
    th, info_p = CS.fit_pooled(z, rd)
    out = {"mu": mu, "sigma": sd, "pooled": th.tolist(), "pooled_info": info_p}
    if per_bucket:
        be, al, info_b = CS.fit_buckets(z, rd, th, n_buckets)
    else:
        be, al, info_b = np.full(n_buckets, th[0]), np.full(n_buckets, th[1]), {}
    out.update(beta=be, alpha=al, bucket_info=info_b, lp=CS.readout_logp(z, rd, be, al))
    return out


def apply_readout(score: np.ndarray, rd: CS.ReadoutData, fit: dict) -> np.ndarray:
    z = (score - fit["mu"]) / fit["sigma"]
    return CS.readout_logp(z, rd, fit["beta"], fit["alpha"])


# ------------------------------------------------------------------ statistics and verdicts
def doc_bits(lp_nat: np.ndarray, doc_of: np.ndarray, n_docs: int) -> np.ndarray:
    return np.bincount(np.asarray(doc_of, np.int64), weights=-np.asarray(lp_nat) / LN2, minlength=int(n_docs))


def contrast(doc_bits_a: np.ndarray, doc_bits_b: np.ndarray, doc_bytes: np.ndarray, idx: np.ndarray) -> dict:
    """a minus b in bits per byte with the paired document bootstrap (percentile 95 percent interval)."""
    from .protocol import bootstrap_bpb

    return bootstrap_bpb(doc_bits_a, doc_bytes, idx, doc_bits_ref=doc_bits_b)


def verdict_arm1(core_diff: dict, best_ctrl_diff: dict, c0_diff: dict) -> dict:
    """A5 P1 arm (i). core_diff: TM minus core; best_ctrl_diff: TM minus the best of {LR, LR2, HGB};
    c0_diff: TM minus C0 (bits per byte, point and ci95). GO(i) and the NO-GO part of arm (i)."""
    gain = -core_diff["point"]
    gain_ci_excl0 = core_diff["ci95"][1] < 0.0
    ctrl = best_ctrl_diff["point"]
    ctrl_ci_excl0 = best_ctrl_diff["ci95"][1] < 0.0
    c0 = c0_diff["point"]
    go = bool(gain >= 0.012 and gain_ci_excl0 and ctrl <= -0.003 and ctrl_ci_excl0 and c0 <= -0.005)
    nogo_part = bool(gain < 0.005 or ctrl > 0.0)
    return {"GO_i": go, "arm_i_nogo_condition": nogo_part, "gain_bpb": gain, "gain_ci95": [-core_diff["ci95"][1], -core_diff["ci95"][0]],
            "gain_ci_excludes_0": bool(gain_ci_excl0), "tm_minus_best_control_bpb": ctrl,
            "tm_minus_best_control_ci95": best_ctrl_diff["ci95"], "tm_minus_best_control_ci_excludes_0": bool(ctrl_ci_excl0),
            "tm_minus_c0_bpb": c0, "tm_minus_c0_ci95": c0_diff["ci95"],
            "criteria": {"gain_ge_0.012": bool(gain >= 0.012), "gain_ci_excludes_0": bool(gain_ci_excl0),
                         "tm_minus_best_control_le_-0.003": bool(ctrl <= -0.003),
                         "tm_minus_best_control_ci_excludes_0": bool(ctrl_ci_excl0), "tm_minus_c0_le_-0.005": bool(c0 <= -0.005)},
            "nogo_rule": "arm (i) gains less than 0.005 or trails the best control (TM minus best control > 0)"}


def verdict_arm2(core_diff_pts: dict, best_ctrl_diff_pts: dict) -> dict:
    """A5 P1 arm (ii) on the BLiMP dev half, accuracy points: TM minus core and TM minus the best control."""
    g = core_diff_pts["point"]
    g_excl = core_diff_pts["ci95"][0] > 0.0
    c = best_ctrl_diff_pts["point"]
    go = bool(g >= 2.0 and g_excl and c >= 1.0)
    nogo_part = bool(g < 1.0 or c < 0.0)
    return {"GO_ii": go, "arm_ii_nogo_condition": nogo_part, "tm_minus_core_points": g,
            "tm_minus_core_ci95": core_diff_pts["ci95"], "tm_minus_core_ci_excludes_0": bool(g_excl),
            "tm_minus_best_control_points": c, "tm_minus_best_control_ci95": best_ctrl_diff_pts["ci95"],
            "nogo_rule": "arm (ii) below 1.0 point over the core or trails the best control"}


def verdict_p1(arm1: dict | None, arm2: dict | None) -> dict:
    """GO if either arm is GO; NO-GO if BOTH arms meet their NO-GO condition; grey otherwise. An arm
    that was not run is untested: it is never GO and never satisfies a NO-GO condition."""
    untested = [n for n, a in (("arm (i)", arm1), ("arm (ii)", arm2)) if a is None]
    go = bool((arm1 is not None and arm1["GO_i"]) or (arm2 is not None and arm2["GO_ii"]))
    nogo = bool(arm1 is not None and arm2 is not None and arm1["arm_i_nogo_condition"] and arm2["arm_ii_nogo_condition"])
    v = "GO" if go else ("NO-GO" if nogo else "grey")
    return {"verdict": v, "untested_arms": untested,
            "rule": "docs/13 A5 P1: GO = (i) or (ii); NO-GO = arm (i) below 0.005 or trailing AND arm (ii) below 1.0 point "
                    "or trailing; grey otherwise (one extension allowed; still grey after it counts as NO-GO for N*-early)"}


def paired_bootstrap_points(a_correct: np.ndarray, b_correct: np.ndarray, n_boot: int = 2000,
                            seed: Sequence[int] = (20260924, 0xB1, 1)) -> dict:
    """Accuracy of a minus accuracy of b in points (percent) over pairs, paired bootstrap over pairs."""
    a = np.asarray(a_correct, np.float64)
    b = np.asarray(b_correct, np.float64)
    d = a - b
    n = d.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([int(s) for s in seed]))
    stats = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[k] = d[idx].mean()
    return {"point": float(100.0 * d.mean()), "ci95": [float(100.0 * np.percentile(stats, 2.5)),
                                                        float(100.0 * np.percentile(stats, 97.5))],
            "n_boot": int(n_boot), "n_pairs": int(n)}


# ------------------------------------------------------------------ BLiMP (arm ii)
BLIMP_REPO = "nyu-mll/blimp"
BLIMP_REVISION = "877fba0801ffb7cbd8c39c1ff314a46f053f6036"
BLIMP_DEV_SALT = 0xB11A9D


def wikitext_style(s: str) -> str:
    """BLiMP sentence -> the WikiText-103 surface form of the training text: a leading space, clitics
    and punctuation split off by spaces (" isn 't", " Paul 's", " word ."), single spaces."""
    s = s.strip()
    s = re.sub(r"(\w)n't\b", r"\1 n't", s)
    s = re.sub(r"(\w)'(s|re|ve|ll|d|m)\b", r"\1 '\2", s)
    s = re.sub(r"(\w)'(\s|$)", r"\1 '\2", s)
    s = re.sub(r"([.,?!;:])", r" \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " " + s


def blimp_dev_mask(uid: Sequence[str], pair_id: np.ndarray) -> np.ndarray:
    """bool [n]: the dev half of the H9.4 split. Inside each paradigm (UID) the pairs are ordered by
    mix64(sha256(UID)[:8] XOR pair_id XOR BLIMP_DEV_SALT) (ties by pair_id; the parquet column pair_id) and the first floor(n / 2) are dev."""
    import hashlib

    uid = np.asarray(uid)
    pair_id = np.asarray(pair_id).astype(np.int64)
    dev = np.zeros(uid.shape[0], bool)
    for u in np.unique(uid):
        idx = np.flatnonzero(uid == u)
        key = np.uint64(int.from_bytes(hashlib.sha256(str(u).encode("utf-8")).digest()[:8], "little"))
        h = mix64_np((pair_id[idx].astype(np.uint64) ^ key) ^ np.uint64(BLIMP_DEV_SALT))
        order = idx[np.lexsort((pair_id[idx], h))]
        dev[order[: idx.shape[0] // 2]] = True
    return dev
