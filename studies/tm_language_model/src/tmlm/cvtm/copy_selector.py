"""TA-learned copy selector over the frozen CVTM-core (docs/13 amendment A4, hypotheses H9.1 and H9.2;
the read-out uses the exact sparse normaliser of docs/11 section 11.5).

Model. For a position p with the frozen CVTM-core distribution P_core(. | p), the candidates C(p) are
the distinct tokens that occurred earlier in the same document, the CAP = 1024 most recent ones
(candidate order: recency). A gate g scores every (p, c) pair, c in C(p), from Boolean literals; the
read-out is the exponential tilt of the core

    P(w | p) = P_core(w | p) exp(t(w)) / Z(p),   t(c) = beta_b s'(c) + a_b for c in C(p), t(w) = 0 otherwise,
    Z(p) = 1 + sum over c in C(p) of P_core(c | p) (exp(t(c)) - 1)

(docs/11 11.5: "normaliser Z = 1 + sum over classes with a vote of P_core(c)(e^s_c - 1)"), with
s'(c) = (s(c) - mu) / sigma the gate score standardised on the selection pairs (a reparametrisation
only: the fitted distribution does not depend on mu and sigma) and b = b(p) the bucket (k*, m*, c, l)
of docs/11 11.4 of the core at p (520 buckets). beta_b plays the role of the temperature of A4
("P_core x exp(beta_b x r)" with r = s' + a_b / beta_b) and a_b sets how much mass moves to the
candidates as a whole (the "mixture weight" of the tilt). Z is computed in the shifted form
Z = (1 - sum P_core(c)) + sum P_core(c) exp(t(c)), algebraically identical, safe for large t.

Fitting of (beta_b, a_b) on the selection positions (docs/13 A4: held-out training documents of
residue 2): first pooled (one pair for all buckets, maximum likelihood, L-BFGS); then per bucket by
maximum a posteriori with a Gaussian prior centred at the pooled pair whose precision is 16 times the
pooled per-position Fisher information (16 pseudo-positions, the analogue of the 16 Dirichlet
pseudo-counts of docs/11 11.4). Buckets without selection positions keep the pooled pair.

Literals: the 237 literals over 29 bucketed pair features of `experiments/cvtm_early_logical_attention.py`
(imported by `literal_construction`, never copied), with the base B of that script replaced by the
frozen core: "P_B" is P_core, "rank under B" the rank under P_core, "P_KN5" the P_S (Kneser-Ney vote
form) of the core's own suffix clauses, "B cluster masses" the core's cluster masses, k* the core's
longest fired suffix context and the document-cache order the highest firing D_m (m >= 1).

Gates (docs/13 A4 H9.1 and H9.2, same pairs and literals):
- TM: tmu TMClassifier, 2 classes, weighted clauses, Type I and Type II feedback, CPU. Score
  s = class-1 vote sum minus class-0 vote sum over the non-empty clauses (unclipped), evaluated
  64 pairs per word by the bit-parallel evaluator of the exploratory script (checked against tmu).
- LR: logistic regression on the literals (L2), score = logit.
- LR2: logistic regression on all pairwise literal products x_i x_j (i <= j, so the literals
  themselves are included), L2, evaluated exactly through per-feature-pair tables.
- HGB: scikit-learn HistGradientBoostingClassifier on the literal matrix, score = raw logit,
  evaluated by a numba tree walker that equals `decision_function` (checked).
- C0: a TM of the same size trained on 1 percent of the pairs.

Everything here is deterministic: no draw depends on thread layout or chunking.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numba import njit

from .doc_clauses import doc_candidates
from .mixture import D_ORDERS, component_mask
from .model import suffix_ln_rows

ROOT = Path(__file__).resolve().parents[3]
CAP = 1024
LN2 = math.log(2.0)
LMAX, LCMAX = 8, 6                      # suffix-match caps of the exploratory literal set
# feature order of the exploratory script (asserted equal in `literal_construction`)
FEATS = ["L_recent", "L_best", "Lk1_recent", "Lk1_best", "Lk2_recent", "Lk2_best", "Lk3_recent", "Lk3_best",
         "dist_recent", "dist_best", "recency_rank", "n_occ", "n_occ_last64", "n_occ_L1", "n_occ_L2", "first_offset",
         "cand_fclass", "cand_type", "pB", "rankB", "pKN5", "clusterB256", "clusterB16", "ratio_top", "kstar",
         "top1B", "prev_type", "depth", "cache_order"]
NF = len(FEATS)
# largest value the pair kernel can write per feature (table sizes of the LR controls; checked in the build)
FEAT_MAX = np.array([LMAX, LMAX, LCMAX, LCMAX, LCMAX, LCMAX, LCMAX, LCMAX, 14, 14, 8, 7, 4, 5, 4, 14, 5, 63, 20, 9, 20,
                     12, 8, 12, 4, 8, 63, 14, 4], dtype=np.int64)
RANK_EDGES = np.array([0, 1, 2, 4, 9, 19, 49, 99, 499], np.int64)
# deterministic subsets (docs/11 section 3 mix64 keyed by the stable document ID and the in-document offset)
S2_SALT = 0x5E1EC7102               # residue-2 selection positions: mix64(mix64(id ^ salt) + offset) % S2_DEN == 0
S2_DEN = 8
POOL_SALT = 0x9001C0DE              # residue-1 negative pool: Bernoulli(POOL_RATE) keyed by (id, offset, candidate)
POOL_RATE = 0.01
POOL_THRESH = np.uint64(int(POOL_RATE * 2.0 ** 64))
PRIOR_POSITIONS = 16.0
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)


# ------------------------------------------------------------------ literal construction (imported)
@dataclass(frozen=True)
class Literals:
    """The literal set of the exploratory script: spec [(name, feature, kind, threshold)], the
    feature of every literal, the truth table LT[literal, raw value], the reflective mask."""

    spec: list
    lit_feat: np.ndarray
    LT: np.ndarray
    reflective: np.ndarray
    names: list

    @property
    def n(self) -> int:
        return int(self.lit_feat.shape[0])

    def matrix(self, feat: np.ndarray, dtype=np.uint8) -> np.ndarray:
        """[n, n_literals] 0/1 literal values of raw pair features [n, 29]."""
        f = np.asarray(feat)
        return self.LT[np.arange(self.n)[None, :], f[:, self.lit_feat]].astype(dtype)


def exploratory_module():
    """`experiments/cvtm_early_logical_attention.py` as a module (import only, never edited)."""
    exp = str(ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import cvtm_early_logical_attention as LA  # noqa: WPS433

    return LA


def core_names(name: str) -> str:
    """Literal names of the exploratory script with its base B named as what it is here: the core."""
    return (name.replace("P_B(top-1)", "P_core(top-1)").replace("P_B(candidate)", "P_core(candidate)")
            .replace("B's top", "the core's top").replace("B mass", "core mass")
            .replace("x P_B(", "x P_core(").replace("P_KN5(candidate)", "P_S(candidate)"))


def literal_construction() -> Literals:
    LA = exploratory_module()
    if list(LA.FEATS) != FEATS:
        raise AssertionError("the exploratory feature order changed; the pair kernel writes FEATS")
    if (LA.LMAX, LA.LCMAX) != (LMAX, LCMAX) or not np.array_equal(LA.RANK_EDGES, RANK_EDGES):
        raise AssertionError("the exploratory caps changed")
    spec = LA.literal_spec()
    lit_feat, LT = LA.literal_tables(spec)
    refl = np.array([FEATS[s[1]] in LA.REFLECTIVE for s in spec])
    names = [core_names(s[0]) for s in spec]
    return Literals(spec=spec, lit_feat=np.asarray(lit_feat, np.int64), LT=np.asarray(LT, np.bool_),
                    reflective=refl, names=names)


def token_tables(counts: np.ndarray, kind_map: np.ndarray, vocab_path: Path | None, V: int) -> dict[str, np.ndarray]:
    """c1, c2, c3 (16, 256, 4096 clusters of the train store's kind_map), fcls (frequency class from
    the token `counts` of the counting documents) and ctype (BPE type bits), as in the exploratory script."""
    LA = exploratory_module()
    km = np.asarray(kind_map).astype(np.int64)
    c1 = (km[1] - V).astype(np.int64)
    c2 = (km[2] - (V + 16)).astype(np.int64)
    c3 = (km[3] - (V + 272)).astype(np.int64)
    if not (c1.min() >= 0 and c1.max() < 16 and c2.min() >= 0 and c2.max() < 256 and c3.min() >= 0 and c3.max() < 4096):
        raise ValueError("kind_map does not hold the 16 / 256 / 4096 cluster levels")
    counts = np.asarray(counts).astype(np.int64)[:V]
    rank = np.empty(V, np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(V)
    fcls = np.select([rank < 16, rank < 64, rank < 256, rank < 1024, rank < 4096], [0, 1, 2, 3, 4], 5).astype(np.int64)
    ctype = LA.token_types(Path(vocab_path)).astype(np.int64) if vocab_path is not None else np.zeros(V, np.int64)
    return {"c1": c1, "c2": c2, "c3": c3, "fcls": fcls, "ctype": ctype}


# ------------------------------------------------------------------ deterministic subsets
def mix64_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
    return z ^ (z >> np.uint64(31))


def s2_hash(doc_index_of_pos: np.ndarray, offset: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return mix64_np(mix64_np(np.asarray(doc_index_of_pos).astype(np.uint64) ^ np.uint64(S2_SALT))
                        + np.asarray(offset).astype(np.uint64))


def s2_mask(doc_index_of_pos: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """bool: the position belongs to the selection subset S2 (1 in S2_DEN positions)."""
    return (s2_hash(doc_index_of_pos, offset) % np.uint64(S2_DEN)) == 0


def s2_half(doc_index_of_pos: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """int64 0 / 1: the half of S2 (bit 3 of the same hash); half 0 (S2a) scores every configuration,
    both halves fit the chosen configuration's per-bucket read-out."""
    return ((s2_hash(doc_index_of_pos, offset) >> np.uint64(3)) & np.uint64(1)).astype(np.int64)


@njit(cache=False)
def _mix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=False)
def pool_hash(doc_id, offset, cand):
    return _mix64(_mix64(_mix64(np.uint64(doc_id) ^ np.uint64(0x9001C0DE)) + np.uint64(offset)) + np.uint64(cand))


@njit(cache=False)
def candidate_counts(tok, doc_start, positions, cap):
    """(ncand int64, n_distinct int64): the number of distinct earlier in-document tokens, capped at
    `cap` and uncapped, for every requested position (positions ascending)."""
    V = 0
    for i in range(tok.shape[0]):
        if tok[i] + 1 > V:
            V = tok[i] + 1
    seen = np.full(V, -1, np.int64)
    nq = positions.shape[0]
    ncand = np.zeros(nq, np.int64)
    ndist = np.zeros(nq, np.int64)
    j = 0
    for d in range(doc_start.shape[0] - 1):
        a = doc_start[d]
        b = doc_start[d + 1]
        if j >= nq:
            break
        if positions[j] >= b:
            continue
        nd = 0
        for p in range(a, b):
            while j < nq and positions[j] == p:
                ndist[j] = nd
                ncand[j] = nd if nd < cap else cap
                j += 1
            t = tok[p]
            if seen[t] != d:
                seen[t] = d
                nd += 1
    return ncand, ndist


# ------------------------------------------------------------------ core rows (docs/11 11.4, as dense_pass)
def core_rows(core, rec, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P_S [q, V], P_core [q, V]) at the strictly increasing positions `pos` of the stream of `rec`,
    with the float64 operations of `CVTMCore.dense_pass` in the same order, so every entry equals the
    dense pass (its gold entry equals `rec.p_gold` bit for bit; checked by the callers)."""
    V = core.vocab
    pos = np.asarray(pos).astype(np.int64)
    cm = component_mask(core.components)
    wmask = core.weights * cm[None, :]
    d_inc = [m for m in D_ORDERS if cm[1 + m]]
    ds = rec.doc_start
    tok = rec.tok
    q = pos.shape[0]
    r = np.arange(q)
    doc_of_pos = np.searchsorted(ds, pos, side="right") - 1
    ctx, valid = core.suffix.contexts(tok, ds, pos)
    lnrows = suffix_ln_rows(core.suffix, ctx, valid)
    P = np.exp(lnrows[:, :V])
    del lnrows
    wr = wmask[rec.bucket[pos]]
    act = rec.act[pos] & cm[None, :]
    row = wr[:, 0:1] * P
    den = 0.0 + wr[:, 0]
    if d_inc:
        d0, d1 = int(doc_of_pos[0]), int(doc_of_pos[-1])
        a, b = int(ds[d0]), int(ds[d1 + 1])
        cands = doc_candidates(tok[a:b], ds[d0:d1 + 2] - a, pos - a, orders=tuple(d_inc), window=core.window)
    for m in d_inc:
        csr = cands[m]
        lens = np.diff(csr.indptr)
        rowrep = np.repeat(r, lens)
        nm = rec.feats.count(m).astype(np.int64)[pos]
        if not np.array_equal(lens > 0, act[:, 1 + m]):
            raise RuntimeError(f"D{m}: candidate rows disagree with the firing mask")
        pm = csr.cnt / nm[rowrep]
        row[rowrep, csr.tok] = row[rowrep, csr.tok] + wr[rowrep, 1 + m] * pm
        den = den + np.where(act[:, 1 + m], wr[:, 1 + m], 0.0)
    if cm[6]:
        fp = np.flatnonzero(act[:, 6])
        cand = rec.feats.ptr_cand[pos[fp]].astype(np.int64)
        row[fp, cand] = row[fp, cand] + wr[fp, 6] * 1.0
        den = den + np.where(act[:, 6], wr[:, 6], 0.0)
    row = row / den[:, None]
    return P, row


def cache_order(doc_n: np.ndarray) -> np.ndarray:
    """int64 [n]: the highest document order m in 1..4 whose clause D_m fires (n_m >= 1), else 0."""
    n = np.asarray(doc_n)
    out = np.zeros(n.shape[1], np.int64)
    for m in range(1, n.shape[0]):
        out = np.where(n[m] >= 1, m, out)
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
    """min(floor(-log2(x)), cap) for x > 0 (0 for x >= 1)."""
    if x >= 1.0:
        return 0
    v = int(math.floor(-math.log2(x)))
    if v < 0:
        v = 0
    return v if v < cap else cap


@njit(cache=False)
def k_pairs(pos, tok, lo, Pc, Ps, kstar, corder, c1, c2, c3, fcls, ctype, rank_edges, cap,
            feat, cand, lab, pcand, ncand, pstat):
    """For every position of the chunk: enumerate the <= cap most recent distinct earlier in-document
    tokens and write their raw pair features (the definitions of the exploratory k_pairs, with the base
    row B replaced by the core row Pc and P_KN5 by the suffix row Ps) at rows i * cap + s.
    pstat[i]: 0 Pc(gold), 1 gold in the document history, 2 gold among the candidates, 3 row sum,
    4 row maximum, 5 distinct earlier tokens (uncapped), 6 slot of the gold among the candidates (-1)."""
    Vv = Pc.shape[1]
    nch = pos.shape[0]
    slot = np.full(Vv, -1, np.int64)
    mark = np.full(Vv, -1, np.int64)
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
        dl = lo[i]
        y = tok[p]
        tot = 0.0
        top = 0.0
        cm1[:] = 0.0
        cm2[:] = 0.0
        for v in range(Vv):
            x = Pc[i, v]
            tot += x
            if x > top:
                top = x
            cm1[c1[v]] += x
            cm2[c2[v]] += x
        srt = np.sort(Pc[i])
        nc = 0
        ghist = 0
        ndist = 0
        for q in range(p - 1, dl - 1, -1):
            t = tok[q]
            if t == y:
                ghist = 1
            if mark[t] != p:
                mark[t] = p
                ndist += 1
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
        gs = slot[y]
        pstat[i, 0] = Pc[i, y]
        pstat[i, 1] = ghist
        pstat[i, 2] = 1.0 if gs >= 0 else 0.0
        pstat[i, 3] = tot
        pstat[i, 4] = top
        pstat[i, 5] = ndist
        pstat[i, 6] = gs
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
            pb = Pc[i, c]
            pcand[r] = pb
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
            feat[r, 18] = _nlog2b(pb, 20)
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
            feat[r, 20] = _nlog2b(Ps[i, c], 20)
            feat[r, 21] = _nlog2b(cm2[c2[c]], 12)
            feat[r, 22] = _nlog2b(cm1[c1[c]], 8)
            feat[r, 23] = _nlog2b(pb / top, 12)
            feat[r, 24] = kstar[i]
            feat[r, 25] = top1b
            feat[r, 26] = prevtype
            feat[r, 27] = depthb
            feat[r, 28] = corder[i]
        for s in range(nc):
            slot[cl[s]] = -1


@dataclass
class PairChunk:
    """Compacted pairs of a chunk of positions (position-major, candidates in recency order)."""

    pos: np.ndarray            # int64 [q]
    ncand: np.ndarray          # int64 [q]
    pstat: np.ndarray          # float64 [q, 7]
    feat: np.ndarray           # uint8 [n_pairs, 29]
    cand: np.ndarray           # uint16 [n_pairs]
    lab: np.ndarray            # uint8 [n_pairs]
    pcand: np.ndarray          # float64 [n_pairs] P_core(candidate)
    core_top: tuple | None = None   # (ids, p) top-k of the core rows when asked for


def pairs_for_positions(core, rec, pos: np.ndarray, tabs: dict, corder: np.ndarray, cap: int = CAP,
                        topk: int = 0) -> PairChunk:
    """Core rows and pair features of the strictly increasing positions `pos` of the stream of `rec`
    (one chunk; memory about 2 x 8 x q x V bytes for the rows plus q x cap x 40 bytes)."""
    from .model import topk_rows

    pos = np.asarray(pos).astype(np.int64)
    P_S, Pc = core_rows(core, rec, pos)
    q = pos.shape[0]
    lo = rec.doc_start[np.searchsorted(rec.doc_start, pos, side="right") - 1].astype(np.int64)
    ks = np.asarray(rec.kstar)[pos].astype(np.int64)
    co = np.asarray(corder)[pos].astype(np.int64)
    feat = np.zeros((q * cap, NF), np.uint8)
    cand = np.zeros(q * cap, np.uint16)
    lab = np.zeros(q * cap, np.uint8)
    pcand = np.zeros(q * cap, np.float64)
    ncand = np.zeros(q, np.int64)
    pstat = np.zeros((q, 7), np.float64)
    k_pairs(pos, rec.tok, lo, Pc, P_S, ks, co, tabs["c1"], tabs["c2"], tabs["c3"], tabs["fcls"], tabs["ctype"],
            RANK_EDGES, cap, feat, cand, lab, pcand, ncand, pstat)
    top = topk_rows(Pc, topk) if topk else None
    del P_S, Pc
    keep = (np.arange(cap)[None, :] < ncand[:, None]).reshape(-1)
    return PairChunk(pos=pos, ncand=ncand, pstat=pstat, feat=feat[keep], cand=cand[keep], lab=lab[keep],
                     pcand=pcand[keep], core_top=top)


def pool_select(chunk: PairChunk, doc_id_of_pos: np.ndarray, offset_of_pos: np.ndarray) -> np.ndarray:
    """bool [n_pairs]: negatives of the Bernoulli(POOL_RATE) pool (keyed by stable document ID,
    in-document offset and candidate token; independent of chunking)."""
    rep_doc = np.repeat(np.asarray(doc_id_of_pos).astype(np.uint64), chunk.ncand)
    rep_off = np.repeat(np.asarray(offset_of_pos).astype(np.uint64), chunk.ncand)
    h = _pool_hash_vec(rep_doc, rep_off, chunk.cand.astype(np.uint64))
    return (h < POOL_THRESH) & (chunk.lab == 0)


@njit(cache=False)
def _pool_hash_vec(doc, off, cand):
    out = np.empty(doc.shape[0], np.uint64)
    for i in range(doc.shape[0]):
        out[i] = pool_hash(doc[i], off[i], cand[i])
    return out


# ------------------------------------------------------------------ bit-parallel literal packing and TM scoring
def pack_literals(feat: np.ndarray, lits: Literals) -> np.ndarray:
    """uint64 [ceil(n / 64), n_literals]: bit r of cols[block, j] = literal j of pair 64 block + r."""
    LA = exploratory_module()
    return LA.k_pack(np.ascontiguousarray(feat), lits.lit_feat, lits.LT)


def pack_labels(lab: np.ndarray) -> np.ndarray:
    """uint64 [ceil(n / 64)]: bit r of word b = label of pair 64 b + r."""
    lab = np.asarray(lab).astype(np.uint8)
    n = lab.shape[0]
    nb = (n + 63) // 64
    pad = np.zeros(nb * 64, np.uint8)
    pad[:n] = lab
    bits = np.packbits(pad.reshape(nb, 64)[:, ::-1], axis=1)       # big-endian bytes of the reversed row
    return bits.view(">u8").astype(np.uint64).reshape(nb)


@dataclass
class ClauseSet:
    """Non-empty clauses of a 2-class tmu TM as signed votes for s = class-1 sum - class-0 sum."""

    ptr: np.ndarray            # int64 [n_clauses + 1]
    lits: np.ndarray           # int64 literal index (full literal set)
    negs: np.ndarray           # uint8 1 = negated literal
    w: np.ndarray              # int64 signed weight (class 1: +w, class 0: -w)
    origin: np.ndarray         # int64 [n_clauses, 2]: (class, clause index in its bank)
    empty_const: int = 0       # sum of the signed weights of the empty clauses (tmu predict counts them)

    @property
    def n(self) -> int:
        return int(self.w.shape[0])


def tm_clause_set(tm, n_lit: int, lit_idx: np.ndarray | None = None) -> ClauseSet:
    """The non-empty clauses of both class banks; `lit_idx` maps the TM's input columns to literal
    indices of the full set (identity by default)."""
    lit_idx = np.arange(n_lit, dtype=np.int64) if lit_idx is None else np.asarray(lit_idx, np.int64)
    nu = lit_idx.shape[0]
    ptr, lits, negs, ws, origin = [0], [], [], [], []
    empty = 0
    for cls, sign in ((1, 1), (0, -1)):
        inc = tm.clause_banks[cls].get_literals()
        w = tm.weight_banks[cls].get_weights().astype(np.int64)
        for c in range(inc.shape[0]):
            pj = np.flatnonzero(inc[c, :nu])
            nj = np.flatnonzero(inc[c, nu:2 * nu])
            if pj.shape[0] + nj.shape[0] == 0:
                empty += sign * int(w[c])
                continue
            if int(w[c]) == 0:                        # a zero vote changes no score
                continue
            lits.extend(lit_idx[pj].tolist())
            negs.extend([0] * pj.shape[0])
            lits.extend(lit_idx[nj].tolist())
            negs.extend([1] * nj.shape[0])
            ptr.append(len(lits))
            ws.append(sign * int(w[c]))
            origin.append((cls, c))
    return ClauseSet(ptr=np.array(ptr, np.int64), lits=np.array(lits, np.int64), negs=np.array(negs, np.uint8),
                     w=np.array(ws, np.int64), origin=np.array(origin, np.int64).reshape(-1, 2), empty_const=int(empty))


_BITCNT = np.array([bin(v).count("1") for v in range(256)], np.int64)
_BITIDX = np.array([[b for b in range(8) if (v >> b) & 1] + [0] * (8 - bin(v).count("1")) for v in range(256)], np.int64)


@njit(cache=False)
def k_tm_score_fast(cols, N, cl_ptr, cl_lit, cl_neg, cl_w, out, bitcnt, bitidx):
    """The class sum of the exploratory k_tm_score (sum over non-empty clauses of weight x AND of the
    included literals, 64 pairs per word) with a byte-table scatter; identical output (tested)."""
    nb = cols.shape[0]
    ncl = cl_w.shape[0]
    one = np.uint64(1)
    allb = ~np.uint64(0)
    byte = np.uint64(255)
    acc = np.zeros(64, np.int64)
    for blk in range(nb):
        r0 = blk * 64
        m = min(64, N - r0)
        if m <= 0:
            break
        full = allb if m == 64 else (one << np.uint64(m)) - one
        acc[:] = 0
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
                    bb = np.int64((a >> np.uint64(8 * by)) & byte)
                    base = 8 * by
                    for k in range(bitcnt[bb]):
                        acc[base + bitidx[bb, k]] += wc
        for r in range(m):
            out[r0 + r] += acc[r]


def tm_scores(cols: np.ndarray, lo: int, hi: int, cs: ClauseSet) -> np.ndarray:
    """float64 [hi - lo]: s of the pairs lo..hi-1 of a packed store (block-aligned internally)."""
    b0, b1 = lo // 64, (hi + 63) // 64
    n_blk_pairs = (b1 - b0) * 64
    out = np.zeros(n_blk_pairs, np.int64)
    k_tm_score_fast(np.ascontiguousarray(cols[b0:b1]), n_blk_pairs, cs.ptr, cs.lits, cs.negs, cs.w, out, _BITCNT, _BITIDX)
    return out[lo - b0 * 64: hi - b0 * 64].astype(np.float64)


def tm_scores_reference(cols: np.ndarray, lo: int, hi: int, cs: ClauseSet) -> np.ndarray:
    """The same with the exploratory evaluator (tests and checks)."""
    LA = exploratory_module()
    b0, b1 = lo // 64, (hi + 63) // 64
    n_blk_pairs = (b1 - b0) * 64
    out = np.zeros(n_blk_pairs, np.int64)
    LA.k_tm_score(np.ascontiguousarray(cols[b0:b1]), n_blk_pairs, cs.ptr, cs.lits, cs.negs, cs.w, out)
    return out[lo - b0 * 64: hi - b0 * 64].astype(np.float64)


@njit(cache=False)
def k_clause_stats(cols, labbits, n, cl_ptr, cl_lit, cl_neg, fire, fire_pos):
    """Per clause: number of the first n pairs where it fires and how many of those have label 1."""
    nb = cols.shape[0]
    one = np.uint64(1)
    allb = ~np.uint64(0)
    ncl = cl_ptr.shape[0] - 1
    for blk in range(nb):
        r0 = blk * 64
        if r0 >= n:
            break
        m = min(64, n - r0)
        full = allb if m == 64 else (one << np.uint64(m)) - one
        lb = labbits[blk]
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
                fire[c] += _popcount(a)
                fire_pos[c] += _popcount(a & lb)


@njit(cache=False)
def _popcount(x):
    c = 0
    while x != np.uint64(0):
        x &= x - np.uint64(1)
        c += 1
    return c


# ------------------------------------------------------------------ logistic regression on literals and on literal pairs
class TableLR:
    """Logistic regression on the literals (pairwise=False) or on all pairwise literal products
    x_i x_j, i <= j (pairwise=True; x_i x_i = x_i, so the literals are included). The model is
    evaluated exactly through per-feature (and per feature-pair) lookup tables over the raw pair
    features: T_ff[a] = sum_{i <= j in f} W_ij L_i(a) L_j(a), T_fg[a, b] = sum_{i in f, j in g} W_ij L_i(a) L_j(b).
    Loss: sum of log(1 + e^s) - y s over the pairs + (l2 / 2) ||W||^2 (the bias is not penalised)."""

    def __init__(self, lits: Literals, pairwise: bool) -> None:
        self.lits = lits
        self.pairwise = bool(pairwise)
        L = lits.n
        self.L = L
        self.nv = (FEAT_MAX + 1).astype(np.int64)
        self.groups = [np.flatnonzero(lits.lit_feat == f) for f in range(NF)]
        for f, g in enumerate(self.groups):          # literals of a feature are contiguous and ordered by feature
            if g.shape[0] and not np.array_equal(g, np.arange(g[0], g[0] + g.shape[0])):
                raise AssertionError("literals of one feature are not contiguous")
        self.LTf = [lits.LT[self.groups[f]][:, : self.nv[f]].astype(np.float64) for f in range(NF)]
        if self.pairwise:
            iu = np.triu_indices(L)
        else:
            iu = (np.arange(L), np.arange(L))
        self.iu = iu
        self.n_params = int(iu[0].shape[0]) + 1
        diag_off = np.zeros(NF, np.int64)
        off = 0
        for f in range(NF):
            diag_off[f] = off
            off += int(self.nv[f])
        pf, pg, poff, png = [], [], [], []
        if self.pairwise:
            for f in range(NF):
                for g in range(f + 1, NF):
                    if self.groups[f].shape[0] == 0 or self.groups[g].shape[0] == 0:
                        continue
                    pf.append(f)
                    pg.append(g)
                    poff.append(off)
                    png.append(int(self.nv[g]))
                    off += int(self.nv[f] * self.nv[g])
        self.diag_off = diag_off
        self.pair_f = np.array(pf, np.int64)
        self.pair_g = np.array(pg, np.int64)
        self.pair_off = np.array(poff, np.int64)
        self.pair_ng = np.array(png, np.int64)
        self.n_table = int(off)

    # -------------------------------------------------------------- parameters and tables
    def W_of(self, theta: np.ndarray) -> np.ndarray:
        W = np.zeros((self.L, self.L))
        W[self.iu] = theta[:-1]
        return W

    def tables(self, theta: np.ndarray) -> np.ndarray:
        W = self.W_of(theta)
        T = np.zeros(self.n_table)
        for f in range(NF):
            g = self.groups[f]
            if g.shape[0] == 0:
                continue
            Wff = np.triu(W[np.ix_(g, g)])
            Lf = self.LTf[f]
            T[self.diag_off[f]: self.diag_off[f] + self.nv[f]] = np.einsum("ia,ij,ja->a", Lf, Wff, Lf)
        for k in range(self.pair_f.shape[0]):
            f, g = int(self.pair_f[k]), int(self.pair_g[k])
            blk = self.LTf[f].T @ W[np.ix_(self.groups[f], self.groups[g])] @ self.LTf[g]
            T[self.pair_off[k]: self.pair_off[k] + blk.size] = blk.reshape(-1)
        return T

    def grad_from_hist(self, H: np.ndarray) -> np.ndarray:
        """d(sum r_n s_n) / dW_ij for i <= j from the residual histogram H (layout of `tables`)."""
        G = np.zeros((self.L, self.L))
        for f in range(NF):
            g = self.groups[f]
            if g.shape[0] == 0:
                continue
            Lf = self.LTf[f]
            h = H[self.diag_off[f]: self.diag_off[f] + self.nv[f]]
            M = (Lf * h[None, :]) @ Lf.T
            G[np.ix_(g, g)] = np.triu(M)
        for k in range(self.pair_f.shape[0]):
            f, g = int(self.pair_f[k]), int(self.pair_g[k])
            h = H[self.pair_off[k]: self.pair_off[k] + self.nv[f] * self.nv[g]].reshape(self.nv[f], self.nv[g])
            G[np.ix_(self.groups[f], self.groups[g])] = self.LTf[f] @ h @ self.LTf[g].T
        return G[self.iu]

    # -------------------------------------------------------------- scoring and fitting
    def score(self, feat: np.ndarray, theta: np.ndarray, T: np.ndarray | None = None) -> np.ndarray:
        T = self.tables(theta) if T is None else T
        out = np.empty(feat.shape[0])
        k_table_score(np.ascontiguousarray(feat), self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng,
                      T, float(theta[-1]), out)
        return out

    def fit(self, feat: np.ndarray, y: np.ndarray, l2: float, theta0: np.ndarray | None = None,
            maxiter: int = 500) -> tuple[np.ndarray, dict]:
        from scipy.optimize import minimize

        feat = np.ascontiguousarray(feat)
        yf = np.asarray(y).astype(np.float64)
        H = np.zeros(self.n_table)
        s = np.empty(feat.shape[0])
        n_eval = [0]

        def obj(theta):
            n_eval[0] += 1
            T = self.tables(theta)
            k_table_score(feat, self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng, T,
                          float(theta[-1]), s)
            lse = np.logaddexp(0.0, s)
            w = theta[:-1]
            f = float(np.sum(lse - yf * s)) + 0.5 * l2 * float(w @ w)
            r = 0.5 * (1.0 + np.tanh(0.5 * s)) - yf
            H[:] = 0.0
            k_table_hist(feat, r, self.diag_off, self.pair_f, self.pair_g, self.pair_off, self.pair_ng, H)
            g = np.empty_like(theta)
            g[:-1] = self.grad_from_hist(H) + l2 * w
            g[-1] = float(r.sum())
            return f, g

        x0 = np.zeros(self.n_params) if theta0 is None else np.asarray(theta0, np.float64).copy()
        t0 = time.perf_counter()
        res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-5, "maxcor": 20})
        info = {"l2": float(l2), "nit": int(res.nit), "n_eval": int(n_eval[0]), "converged": bool(res.success),
                "message": str(res.message), "loss": float(res.fun), "fit_s": time.perf_counter() - t0,
                "pairwise": self.pairwise, "n_params": self.n_params}
        return res.x, info


@njit(cache=False)
def k_table_score(feat, diag_off, pair_f, pair_g, pair_off, pair_ng, T, bias, out):
    n = feat.shape[0]
    nf = feat.shape[1]
    npair = pair_f.shape[0]
    for r in range(n):
        s = bias
        for f in range(nf):
            s += T[diag_off[f] + feat[r, f]]
        for k in range(npair):
            s += T[pair_off[k] + feat[r, pair_f[k]] * pair_ng[k] + feat[r, pair_g[k]]]
        out[r] = s


@njit(cache=False)
def k_table_hist(feat, resid, diag_off, pair_f, pair_g, pair_off, pair_ng, H):
    n = feat.shape[0]
    nf = feat.shape[1]
    npair = pair_f.shape[0]
    for r in range(n):
        w = resid[r]
        for f in range(nf):
            H[diag_off[f] + feat[r, f]] += w
        for k in range(npair):
            H[pair_off[k] + feat[r, pair_f[k]] * pair_ng[k] + feat[r, pair_g[k]]] += w


# ------------------------------------------------------------------ histogram gradient boosting evaluator
@dataclass
class TreeEnsemble:
    """The trees of a fitted binary HistGradientBoostingClassifier as flat node arrays."""

    tree_off: np.ndarray       # int64 [n_trees + 1]
    feature: np.ndarray        # int64 literal index
    threshold: np.ndarray      # float64
    left: np.ndarray           # int64 (absolute node index)
    right: np.ndarray
    is_leaf: np.ndarray        # uint8
    value: np.ndarray          # float64
    baseline: float

    @property
    def n_trees(self) -> int:
        return int(self.tree_off.shape[0] - 1)


def hgb_ensemble(model) -> TreeEnsemble:
    offs, feats, thr, left, right, leaf, val = [0], [], [], [], [], [], []
    for it in model._predictors:
        if len(it) != 1:
            raise ValueError("binary classifier expected (one tree per iteration)")
        nodes = it[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        base = offs[-1]
        feats.append(nodes["feature_idx"].astype(np.int64))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int64) + base)
        right.append(nodes["right"].astype(np.int64) + base)
        leaf.append(nodes["is_leaf"].astype(np.uint8))
        val.append(nodes["value"].astype(np.float64))
        offs.append(base + nodes.shape[0])
    return TreeEnsemble(tree_off=np.array(offs, np.int64), feature=np.concatenate(feats), threshold=np.concatenate(thr),
                        left=np.concatenate(left), right=np.concatenate(right), is_leaf=np.concatenate(leaf),
                        value=np.concatenate(val), baseline=float(np.asarray(model._baseline_prediction).reshape(-1)[0]))


@njit(cache=False)
def k_hgb_score(feat, lit_feat, LT, tree_off, feature, threshold, left, right, is_leaf, value, baseline,
                stages, out):
    """out[k, r]: baseline + the first stages[k] trees (added in tree order, as sklearn's _raw_predict)."""
    n = feat.shape[0]
    ns = stages.shape[0]
    nt = tree_off.shape[0] - 1
    for r in range(n):
        acc = baseline
        k = 0
        for t in range(nt):
            node = tree_off[t]
            while is_leaf[node] == 0:
                j = feature[node]
                x = 1.0 if LT[j, feat[r, lit_feat[j]]] else 0.0
                if x <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            acc += value[node]
            while k < ns and stages[k] == t + 1:
                out[k, r] = acc
                k += 1


def hgb_scores_walk(feat: np.ndarray, lits: Literals, ens: TreeEnsemble, stages: Sequence[int]) -> np.ndarray:
    """Raw logits after the first `stages[k]` trees by walking every tree (reference evaluator)."""
    st = np.asarray(sorted(int(s) for s in stages), np.int64)
    out = np.zeros((st.shape[0], feat.shape[0]))
    k_hgb_score(np.ascontiguousarray(feat), lits.lit_feat, lits.LT, ens.tree_off, ens.feature, ens.threshold,
                ens.left, ens.right, ens.is_leaf, ens.value, ens.baseline, st, out)
    return out


def hgb_scores(feat: np.ndarray, lits: Literals, ens: TreeEnsemble, stages: Sequence[int]) -> np.ndarray:
    """Raw logits after the first `stages[k]` trees (the tree walk; equal to sklearn, tested)."""
    return hgb_scores_walk(feat, lits, ens, stages)


# ------------------------------------------------------------------ the read-out
@njit(cache=False)
def k_readout(score, pc, offs, gslot, lpgold, bucket, beta, alpha, want_grad, gb, ga, lp_out):
    """Total negative log-likelihood (nats) of the tilted read-out over the positions; per-position
    ln P(gold) in lp_out; with want_grad the per-bucket gradients are ADDED to gb, ga."""
    n = offs.shape[0] - 1
    tot = 0.0
    for i in range(n):
        a0 = offs[i]
        a1 = offs[i + 1]
        if a1 == a0:
            lp_out[i] = lpgold[i]
            tot -= lpgold[i]
            continue
        b = bucket[i]
        be = beta[b]
        al = alpha[b]
        m = 0.0
        for r in range(a0, a1):
            t = be * score[r] + al
            if t > m:
                m = t
        rest = 1.0
        S = 0.0
        Ss = 0.0
        for r in range(a0, a1):
            rest -= pc[r]
            w = pc[r] * math.exp(be * score[r] + al - m)
            S += w
            Ss += w * score[r]
        if rest < 0.0:
            rest = 0.0
        Zs = rest * math.exp(-m) + S
        lz = m + math.log(Zs)
        g = gslot[i]
        sg = 0.0
        tg = 0.0
        if g >= 0:
            sg = score[a0 + g]
            tg = be * sg + al
        lp = lpgold[i] + tg - lz
        lp_out[i] = lp
        tot -= lp
        if want_grad:
            if g >= 0:
                gb[b] += Ss / Zs - sg
                ga[b] += S / Zs - 1.0
            else:
                gb[b] += Ss / Zs
                ga[b] += S / Zs
    return tot


@njit(cache=False)
def k_readout_row(score, pc, cand, core_row, beta, alpha):
    """The full read-out distribution of ONE position from its full core row (tests and checks)."""
    V = core_row.shape[0]
    out = core_row.copy()
    rest = 1.0
    S = 0.0
    for r in range(score.shape[0]):
        rest -= pc[r]
        S += pc[r] * math.exp(beta * score[r] + alpha)
    Z = rest + S
    for r in range(score.shape[0]):
        out[cand[r]] = core_row[cand[r]] * math.exp(beta * score[r] + alpha)
    for v in range(V):
        out[v] = out[v] / Z
    return out


@dataclass
class ReadoutData:
    """Per-position inputs of the read-out: offsets into the pair arrays, slot of the gold among the
    candidates (-1), ln P_core(gold), bucket; per pair: P_core(candidate)."""

    offs: np.ndarray
    gslot: np.ndarray
    lpgold: np.ndarray
    bucket: np.ndarray
    pc: np.ndarray

    @property
    def n(self) -> int:
        return int(self.offs.shape[0] - 1)


def readout_logp(score: np.ndarray, rd: ReadoutData, beta: np.ndarray, alpha: np.ndarray,
                 bucket: np.ndarray | None = None) -> np.ndarray:
    """float64 [n]: ln P(gold) of the tilted read-out."""
    lp = np.empty(rd.n)
    b = rd.bucket if bucket is None else bucket
    k_readout(score, rd.pc, rd.offs, rd.gslot, rd.lpgold, b, np.asarray(beta, np.float64), np.asarray(alpha, np.float64),
              False, np.zeros(1), np.zeros(1), lp)
    return lp


def fit_pooled(score: np.ndarray, rd: ReadoutData, x0=(1.0, 0.0), maxiter: int = 200) -> tuple[np.ndarray, dict]:
    """Maximum-likelihood (beta, a) shared by all positions; returns theta [2] and info (bits per position)."""
    from scipy.optimize import minimize

    zb = np.zeros(rd.n, np.int64)
    lp = np.empty(rd.n)
    gb, ga = np.zeros(1), np.zeros(1)
    n_eval = [0]

    def obj(th):
        n_eval[0] += 1
        gb[:] = 0.0
        ga[:] = 0.0
        f = k_readout(score, rd.pc, rd.offs, rd.gslot, rd.lpgold, zb, th[:1].copy(), th[1:].copy(), True, gb, ga, lp)
        return f / rd.n, np.array([gb[0], ga[0]]) / rd.n

    t0 = time.perf_counter()
    res = minimize(obj, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-9})
    f0 = obj(np.zeros(2))[0]
    info = {"beta": float(res.x[0]), "a": float(res.x[1]), "bits_per_position": float(res.fun / LN2),
            "core_bits_per_position": float(f0 / LN2), "nit": int(res.nit), "n_eval": int(n_eval[0]),
            "converged": bool(res.success), "fit_s": time.perf_counter() - t0}
    return res.x.copy(), info


def pooled_hessian(score: np.ndarray, rd: ReadoutData, theta0: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Per-position average Hessian (2 x 2) of the pooled negative log-likelihood at theta0 (central
    differences of the analytic gradient), symmetrised."""
    zb = np.zeros(rd.n, np.int64)
    lp = np.empty(rd.n)

    def grad(th):
        gb, ga = np.zeros(1), np.zeros(1)
        k_readout(score, rd.pc, rd.offs, rd.gslot, rd.lpgold, zb, th[:1].copy(), th[1:].copy(), True, gb, ga, lp)
        return np.array([gb[0], ga[0]]) / rd.n

    Hm = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        Hm[:, k] = (grad(theta0 + e) - grad(theta0 - e)) / (2 * h)
    return 0.5 * (Hm + Hm.T)


def fit_buckets(score: np.ndarray, rd: ReadoutData, theta0: np.ndarray, n_buckets: int,
                prior: float = PRIOR_POSITIONS, maxiter: int = 300) -> tuple[np.ndarray, np.ndarray, dict]:
    """MAP (beta_b, a_b) per bucket: negative log-likelihood + (prior / 2) sum_b (th_b - th0)' H (th_b - th0),
    H the pooled per-position Hessian at th0 (module docstring). Returns beta [nb], a [nb], info."""
    from scipy.optimize import minimize

    Hm = pooled_hessian(score, rd, theta0)
    ev = np.linalg.eigvalsh(Hm)
    if not np.all(ev > 0):
        raise RuntimeError(f"pooled Hessian is not positive definite: eigenvalues {ev}")
    P = prior * Hm
    lp = np.empty(rd.n)
    counts = np.bincount(rd.bucket, minlength=n_buckets)
    n_eval = [0]

    def obj(x):
        n_eval[0] += 1
        be, al = x[:n_buckets].copy(), x[n_buckets:].copy()
        gb, ga = np.zeros(n_buckets), np.zeros(n_buckets)
        f = k_readout(score, rd.pc, rd.offs, rd.gslot, rd.lpgold, rd.bucket, be, al, True, gb, ga, lp)
        d = np.stack([be - theta0[0], al - theta0[1]], axis=1)       # [nb, 2]
        pd = d @ P
        f += 0.5 * float(np.sum(pd * d))
        gb += pd[:, 0]
        ga += pd[:, 1]
        return f / rd.n, np.concatenate([gb, ga]) / rd.n

    x0 = np.concatenate([np.full(n_buckets, theta0[0]), np.full(n_buckets, theta0[1])])
    t0 = time.perf_counter()
    res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-10, "maxcor": 20})
    be, al = res.x[:n_buckets].copy(), res.x[n_buckets:].copy()
    lpv = readout_logp(score, rd, be, al)
    info = {"nit": int(res.nit), "n_eval": int(n_eval[0]), "converged": bool(res.success), "message": str(res.message),
            "objective_per_position": float(res.fun), "bits_per_position_fitted": float(-lpv.mean() / LN2),
            "pooled_hessian_per_position": Hm.tolist(), "prior_positions": float(prior),
            "buckets_with_positions": int((counts > 0).sum()), "fit_s": time.perf_counter() - t0}
    return be, al, info


# ------------------------------------------------------------------ readable clauses
def rule_text(lits: Literals, clause_lits: np.ndarray, clause_negs: np.ndarray) -> str:
    """A conjunction reduced to the tightest bound per feature (thermometer literals imply each other),
    in the literal names of this module (the core instead of B)."""
    spec = lits.spec
    name_of = {(sp[1], sp[2], sp[3]): core_names(sp[0]) for sp in spec}
    lo_b, hi_b, bits_on, bits_off = {}, {}, {}, {}
    for j, ng in zip(np.asarray(clause_lits).tolist(), np.asarray(clause_negs).tolist()):
        _, f, kind, t = spec[j]
        neg = bool(ng)
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
            if f in lo_b and (f, "ge", lo_b[f]) in name_of:
                parts.append(name_of[(f, "ge", lo_b[f])])
            if f in hi_b and (f, "ge", hi_b[f] + 1) in name_of:
                parts.append("NOT " + name_of[(f, "ge", hi_b[f] + 1)])
        elif "le" in kinds:
            if f in hi_b and (f, "le", hi_b[f]) in name_of:
                parts.append(name_of[(f, "le", hi_b[f])])
            if f in lo_b and (f, "le", lo_b[f] - 1) in name_of:
                parts.append("NOT " + name_of[(f, "le", lo_b[f] - 1)])
        for t in sorted(bits_on.get(f, ())):
            parts.append(name_of[(f, "bit", t)])
        for t in sorted(bits_off.get(f, ())):
            parts.append("NOT " + name_of[(f, "bit", t)])
        if f in lo_b and f in hi_b and lo_b[f] > hi_b[f]:
            parts.append(f"[contradiction on {FEATS[f]}: never fires]")
    return " AND ".join(parts) if parts else "(empty)"


# ------------------------------------------------------------------ verdict rules (docs/13 A4)
def h91_verdict(diff_bpb: float, ci95: Sequence[float]) -> dict:
    """H9.1 on one seed: core+selector minus core at most -0.010 bits per byte AND the CI excludes 0."""
    ok = diff_bpb <= -0.010 and ci95[1] < 0.0
    return {"met": bool(ok), "gain_bpb": -float(diff_bpb), "threshold_gain_bpb": 0.010,
            "ci95_of_difference": [float(ci95[0]), float(ci95[1])], "ci_excludes_0": bool(ci95[1] < 0.0 or ci95[0] > 0.0)}


def h92_verdict(tm_minus_best: float, ci95: Sequence[float]) -> str:
    """"beats" if at most -0.002 with the CI excluding 0; "matches" if at most 0.000; else "trails"."""
    if tm_minus_best <= -0.002 and ci95[1] < 0.0:
        return "beats"
    if tm_minus_best <= 0.0:
        return "matches"
    return "trails"


def g1_t2(gains: Sequence[float], ci_excl0: Sequence[bool]) -> dict:
    """Gate G1 T2 of docs/13 A4 with the H9.1 bands of docs/13 A5 D0.1, from the per-seed gains (core
    minus core+selector, bits per byte) and, per seed, whether the document-bootstrap CI of the
    difference lies entirely below 0 (`ci_excl0`).

    - band A: every seed gains at least 0.010, each with its CI excluding 0 -> G1 T2 open (H9.3 build);
    - band B: every seed gains at least 0.005 with its CI excluding 0, and not band A -> the copy
      selector stays a row, H9.3 runs only as a 5M pilot (A5: P1 may substitute only with a recorded decision);
    - band C: otherwise -> no T2 engine work, T2 reported negative after H8.10.
    The band is set by the smallest per-seed gain; the seed mean is reported alongside it (D0.1)."""
    g = np.asarray(gains, np.float64)
    ex = np.asarray(list(ci_excl0), bool)
    if g.shape[0] == 0 or ex.shape[0] != g.shape[0]:
        raise ValueError("one gain and one CI flag per seed")
    if np.all(g >= 0.010) and np.all(ex):
        band, gate = "A", "open: the CR-TM build (H9.3) opens"
    elif np.all(g >= 0.005) and np.all(ex):
        band, gate = "B", "middle: the copy selector stays a row; H9.3 runs only as a 5M pilot"
    else:
        band, gate = "C", "closed: no T2 engine work; T2 is reported negative after H8.10"
    return {"band": band, "gate": gate, "gains_bpb": g.tolist(), "ci_excludes_0": ex.tolist(),
            "min_gain_bpb": float(g.min()), "mean_gain_bpb": float(g.mean()),
            "rule": "docs/13 A5 D0.1: A = every seed >= 0.010 with its CI excluding 0; B = every seed >= 0.005 with its "
                    "CI excluding 0 and not A; C = otherwise (the smallest per-seed gain decides)"}


def credit_d03(band: str, tm_minus_c0: float, ci95: Sequence[float]) -> dict:
    """docs/13 A5 D0.3: the copy selector has learning credit when it is in band A AND TM minus C0 is at
    most -0.005 bits per byte with the CI excluding 0 (3-seed mean, paired document bootstrap)."""
    c0_ok = tm_minus_c0 <= -0.005 and ci95[1] < 0.0
    return {"credit": bool(band == "A" and c0_ok), "band": band, "tm_minus_c0_bpb": float(tm_minus_c0),
            "tm_minus_c0_ci95": [float(ci95[0]), float(ci95[1])], "tm_minus_c0_meets_rule": bool(c0_ok),
            "rule": "band A AND TM minus C0 <= -0.005 bits per byte with the CI excluding 0"}


def lbfgs_hit_cap(info: dict, maxiter: int) -> bool:
    """True when an L-BFGS-B fit stopped at its iteration (or evaluation) limit instead of a convergence
    criterion (docs/13 A5 D0.2: such a control is refitted once with the cap raised four-fold)."""
    msg = str(info.get("message", "")).upper()
    return bool(int(info.get("nit", 0)) >= int(maxiter) or "ITERATIONS REACHED LIMIT" in msg
                or "EVALUATIONS EXCEEDS LIMIT" in msg)
