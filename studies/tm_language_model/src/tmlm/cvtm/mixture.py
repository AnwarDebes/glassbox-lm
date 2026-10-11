"""Fixed combination of the CVTM (normative text: docs/11 section 11.4).

    P(w | p) = lambda_S(b) P_KN(w | h) + sum over firing D_m of lambda_m(b) P_Dm(w | p) + lambda_P(b) P_P(w | p),

with the weights of non-firing components dropped and the rest renormalised to sum to one.

Components, in this fixed order (column index in every [n, 7] array): S, D0, D1, D2, D3, D4, P.
Active mask per position: S always; D_m iff n_m >= 1; P iff L >= ptr_min (5).

Bucket b(p) = (k*, m*, c, l):
- k* in 0..4: the longest fired suffix context (`kn_gold_and_depth`), 0 when only the bias fires;
- m* in {none, 0..4}: the highest firing document order among the included D components;
- c: count bucket of n_(m*) in {1, 2-3, 4-7, 8-15, 16+} (absent when m* is none);
- l: pointer bucket in {none, 5-8, 9-16, 17+} (only when P is included).
Bucket id = (k* * n_mc + mc) * n_l + l with mc = 0 for none, else 1 + 5 j + c (j the index of m*
among the included D orders), n_mc = 1 + 5 * (number of included D components), n_l = 4 with P and
1 without: 5 * 26 * 4 = 520 buckets for the full model, 5 * 6 * 1 = 30 for the H8.4 comparator
"KN-5 plus a document unigram cache" (components S and D0). `BucketLayout` encodes and decodes.

The bucket fixes the active set: D_m firing implies D_(m-1) firing and P firing implies D_4 firing
(`tmlm.cvtm.doc_clauses`), so an included D_m is active iff m <= m*, and P iff l is not none.
`fit_map_em` checks this on its input (every bucket must have one active set) and refuses otherwise.

Fitting (`fit_map_em`), float64, fixed operation order, deterministic, no early stop:
1. Pooled weights pi over all rows by plain EM from uniform over the components active in at least
   one row. Rows have different active sets and the model renormalises over them, so the update is
   the minorise-maximise step of that likelihood: w_i <- n_i / c_i with n_i the summed E-step
   responsibilities and c_i = sum over rows where i is active of 1 / (sum of the active weights of
   the row), then scaled to the simplex (the likelihood is scale free). Jensen on the numerator and
   the tangent of the convex -log on the normaliser make it a minoriser, so the log-likelihood never
   decreases (logged per iteration, checked to 1e-12 by the tests). With every component active in
   every row it is the classical EM update.
2. Per bucket b with rows: MAP-EM over the bucket's active set A_b, starting uniform over A_b, E-step
   responsibilities, M-step v_i = (n_bi + a_bi) / (rows_b + prior) with Dirichlet pseudo-counts
   a_bi = prior * pi_i / pi(A_b) (16 pseudo-counts toward the pooled weights restricted to the
   components that can fire in the bucket). This is exact MAP-EM for the objective
   log-likelihood + sum a_bi log v_bi (logged per iteration, non-decreasing). Weights outside A_b
   are 0 (they never enter the renormalised mixture).
3. A bucket without fit rows gets the pooled weights pi.
Returned: weights [n_buckets, 7], every row on the simplex.

Scoring: `gold_prob` (vectorised, the fixed component order) and `full_distribution` (one
position, [V], sums to one) perform the same float64 operations in the same order, so the gold entry
of the full distribution equals `gold_prob` bit for bit when both see the same inputs.
`residual_votes` is the vote form of section 11.4: r(w) = ln(l_S + sum_i l_i P_i(w) / P_KN(w | h))
with l the renormalised weights, so P(w) = P_KN(w | h) exp(r(w)) exactly; classes outside the
document and pointer candidates all get the default vote ln(l_S).

Also here: `kn_gold_and_depth` (P_KN of the gold token and k* from a fitted
`tmlm.eval.baselines.ngram.KNModel`, read only) and the plumbing smoke
(`python -m tmlm.cvtm.mixture --smoke CONFIG`, results through scripts/run_logged.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

COMPONENTS = ("S", "D0", "D1", "D2", "D3", "D4", "P")
N_COMP = len(COMPONENTS)
D_ORDERS = (0, 1, 2, 3, 4)
K_MAX = 4                                    # K - 1 for the order-5 KN
N_COUNT_BUCKETS = 5                          # {1, 2-3, 4-7, 8-15, 16+}
N_PTR_BUCKETS = 4                            # {none, 5-8, 9-16, 17+}
PTR_MIN = 5
CACHE_COMPONENTS = ("S", "D0")               # the H8.4 comparator


# ------------------------------------------------------------------ components and buckets
def component_mask(components: Sequence[str] | None = None) -> np.ndarray:
    """bool [7]: which of (S, D0..D4, P) are included. S is required (it is always active)."""
    if components is None:
        return np.ones(N_COMP, dtype=bool)
    comps = tuple(components)
    unknown = set(comps) - set(COMPONENTS)
    if unknown or len(set(comps)) != len(comps):
        raise ValueError(f"components must be distinct names from {COMPONENTS}")
    if "S" not in comps:
        raise ValueError("component S (the suffix clauses) is required")
    return np.array([c in comps for c in COMPONENTS], dtype=bool)


def count_bucket(n) -> np.ndarray:
    """0 for n = 1, 1 for 2-3, 2 for 4-7, 3 for 8-15, 4 for 16+, -1 for n <= 0."""
    n = np.asarray(n).astype(np.int64)
    return np.select([n <= 0, n == 1, n <= 3, n <= 7, n <= 15], [-1, 0, 1, 2, 3], 4).astype(np.int64)


def pointer_bucket(length, ptr_min: int = PTR_MIN) -> np.ndarray:
    """0 none (L < ptr_min), 1 for 5-8, 2 for 9-16, 3 for 17+."""
    length = np.asarray(length).astype(np.int64)
    return np.where(length < ptr_min, 0, np.where(length <= 8, 1, np.where(length <= 16, 2, 3)))


@dataclass(frozen=True)
class BucketLayout:
    """Bucket ids for one component set (see the module docstring)."""

    components: tuple[str, ...]

    @classmethod
    def of(cls, components: Sequence[str] | None = None) -> "BucketLayout":
        mask = component_mask(components)
        return cls(tuple(c for c, keep in zip(COMPONENTS, mask) if keep))

    @property
    def d_orders(self) -> tuple[int, ...]:
        return tuple(m for m in D_ORDERS if f"D{m}" in self.components)

    @property
    def has_pointer(self) -> bool:
        return "P" in self.components

    @property
    def n_mc(self) -> int:
        return 1 + N_COUNT_BUCKETS * len(self.d_orders)

    @property
    def n_l(self) -> int:
        return N_PTR_BUCKETS if self.has_pointer else 1

    @property
    def n_buckets(self) -> int:
        return (K_MAX + 1) * self.n_mc * self.n_l

    def encode(self, k_star, m_star, c, l) -> np.ndarray:
        """m_star: document order (-1 for none), c: count bucket (ignored when m_star is -1),
        l: pointer bucket (must be 0 without P)."""
        k_star = np.asarray(k_star).astype(np.int64)
        m_star = np.asarray(m_star).astype(np.int64)
        c = np.asarray(c).astype(np.int64)
        l = np.asarray(l).astype(np.int64)
        if np.any((k_star < 0) | (k_star > K_MAX)):
            raise ValueError("k* must lie in 0..4")
        j = np.full(m_star.shape, -1, dtype=np.int64)
        for idx, m in enumerate(self.d_orders):
            j = np.where(m_star == m, idx, j)
        if np.any((m_star >= 0) & (j < 0)):
            raise ValueError("m* is not an included document order")
        if np.any((m_star >= 0) & ((c < 0) | (c >= N_COUNT_BUCKETS))):
            raise ValueError("count bucket out of range")
        if np.any((l < 0) | (l >= self.n_l)):
            raise ValueError("pointer bucket out of range for this component set")
        mc = np.where(m_star >= 0, 1 + N_COUNT_BUCKETS * j + np.maximum(c, 0), 0)
        return (k_star * self.n_mc + mc) * self.n_l + l

    def decode(self, bucket: int) -> tuple[int, int | None, int | None, int]:
        """(k*, m* or None, c or None, l) of one bucket id."""
        b = int(bucket)
        if not 0 <= b < self.n_buckets:
            raise ValueError("bucket id out of range")
        l = b % self.n_l
        rest = b // self.n_l
        mc = rest % self.n_mc
        k = rest // self.n_mc
        if mc == 0:
            return k, None, None, l
        j, c = divmod(mc - 1, N_COUNT_BUCKETS)
        return k, self.d_orders[j], c, l

    def active_components(self, bucket: int) -> np.ndarray:
        """bool [7]: the active set that the bucket implies (module docstring)."""
        _, m_star, _, l = self.decode(bucket)
        act = np.zeros(N_COMP, dtype=bool)
        act[0] = True
        for m in self.d_orders:
            if m_star is not None and m <= m_star:
                act[1 + m] = True
        act[6] = self.has_pointer and l > 0
        return act


def counts_matrix(feats) -> np.ndarray:
    """int64 [5, n]: n_m for m = 0..4 from a `DocFeatures` (zeros for orders it does not hold)."""
    n = feats.ptr_len.shape[0]
    out = np.zeros((len(D_ORDERS), n), dtype=np.int64)
    for m in feats.orders:
        out[m] = feats.count(m)
    return out


def active_mask(n_counts, ptr_len, components: Sequence[str] | None = None, ptr_min: int = PTR_MIN) -> np.ndarray:
    """bool [n, 7] in the order (S, D0..D4, P); n_counts is [5, n] (row m = n_m)."""
    n_counts = np.asarray(n_counts)
    ptr_len = np.asarray(ptr_len)
    cm = component_mask(components)
    act = np.zeros((ptr_len.shape[0], N_COMP), dtype=bool)
    act[:, 0] = True
    for m in D_ORDERS:
        act[:, 1 + m] = n_counts[m] >= 1
    act[:, 6] = ptr_len >= ptr_min
    return act & cm[None, :]


def bucket_ids(k_star, n_counts, ptr_len, components: Sequence[str] | None = None,
               ptr_min: int = PTR_MIN) -> tuple[np.ndarray, int]:
    """(bucket [n] int64, n_buckets) of section 11.4 for the given component set."""
    lay = BucketLayout.of(components)
    n_counts = np.asarray(n_counts).astype(np.int64)
    ptr_len = np.asarray(ptr_len)
    n = ptr_len.shape[0]
    m_star = np.full(n, -1, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    for m in lay.d_orders:                         # ascending: the last firing order wins
        fire = n_counts[m] >= 1
        m_star = np.where(fire, m, m_star)
        c = np.where(fire, count_bucket(n_counts[m]), c)
    l = pointer_bucket(ptr_len, ptr_min) if lay.has_pointer else np.zeros(n, dtype=np.int64)
    return lay.encode(k_star, m_star, c, l), lay.n_buckets


def gold_components(p_kn_gold, feats, gold_tok) -> tuple[np.ndarray, np.ndarray]:
    """(p [n, 7], active [n, 7]) from P_KN(gold) and a `DocFeatures`: P_Dm(gold) = g_m / n_m where
    D_m fires, P_P(gold) = 1 if the pointer candidate is the gold token; inactive entries are 0."""
    p_kn_gold = np.asarray(p_kn_gold, dtype=np.float64)
    gold_tok = np.asarray(gold_tok).astype(np.int64)
    n = p_kn_gold.shape[0]
    p = np.zeros((n, N_COMP), dtype=np.float64)
    act = np.zeros((n, N_COMP), dtype=bool)
    p[:, 0] = p_kn_gold
    act[:, 0] = True
    for m in feats.orders:
        nm = feats.count(m).astype(np.int64)
        gm = feats.gold_count(m).astype(np.int64)
        fire = nm >= 1
        act[:, 1 + m] = fire
        p[:, 1 + m] = np.where(fire, gm.astype(np.float64) / np.maximum(nm, 1), 0.0)
    fire_p = feats.ptr_len >= feats.ptr_min
    act[:, 6] = fire_p
    p[:, 6] = np.where(fire_p & (feats.ptr_cand.astype(np.int64) == gold_tok), 1.0, 0.0)
    return p, act


# ------------------------------------------------------------------ fitting
def _prepare(p_comp_gold, active, components):
    p_comp_gold = np.asarray(p_comp_gold, dtype=np.float64)
    active = np.asarray(active).astype(bool)
    if p_comp_gold.ndim != 2 or p_comp_gold.shape[1] != N_COMP or active.shape != p_comp_gold.shape:
        raise ValueError("p_comp_gold and active must both be [n, 7]")
    cm = component_mask(components)
    act = active & cm[None, :]
    if not act.any(axis=1).all():
        raise ValueError("every row needs an active component")
    p = np.where(act, p_comp_gold, 0.0)
    if np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("active probabilities must be finite and >= 0")
    if np.any(p.sum(axis=1) <= 0.0):
        raise ValueError("a row gives probability 0 to its gold token under every active component")
    return p, act, cm


def pooled_em(p_comp_gold, active, iters: int = 50, components: Sequence[str] | None = None,
              trace: dict | None = None) -> np.ndarray:
    """Plain EM (minorise-maximise, module docstring) of one weight vector over all rows; [7] on the
    simplex, 0 for components never active. trace["pooled_ll"] gets the mean log-likelihood per
    row (nats) at the start and after every iteration (iters + 1 values)."""
    p, act, _ = _prepare(p_comp_gold, active, components)
    return _pooled_em(p, act, iters, trace)


def _pooled_em(p, act, iters, trace):
    ever = act.any(axis=0)
    w = np.where(ever, 1.0 / ever.sum(), 0.0)
    lls = []
    actf = act.astype(np.float64)
    for it in range(iters + 1):
        num = np.where(act, w[None, :] * p, 0.0)
        tot = num.sum(axis=1)
        s = (actf * w[None, :]).sum(axis=1)
        lls.append(math.fsum((np.log(tot) - np.log(s)).tolist()) / p.shape[0])
        if it == iters:
            break
        gam = num / tot[:, None]
        nk = gam.sum(axis=0)
        ck = (actf / s[:, None]).sum(axis=0)
        w = np.where(ck > 0, nk / np.where(ck > 0, ck, 1.0), 0.0)
        w = w / w.sum()
    if trace is not None:
        trace["pooled_ll"] = lls
        trace["pooled_weights"] = w.tolist()
    return w


def fit_map_em(p_comp_gold, active, bucket, iters: int = 50, prior: float = 16.0, *,
               n_buckets: int | None = None, components: Sequence[str] | None = None,
               trace: dict | None = None) -> np.ndarray:
    """MAP-EM bucket weights of section 11.4 (module docstring): weights [n_buckets, 7], every row
    on the simplex. p_comp_gold [n, 7] float64 (inactive entries ignored), active [n, 7] bool,
    bucket [n] int. `components` restricts the model (e.g. CACHE_COMPONENTS for H8.4; the bucket
    ids must then come from `bucket_ids` with the same components). With `trace` (a dict) the fit
    records pooled_ll, pooled_weights, bucket_ll, bucket_map_objective (mean per row, nats, iters + 1
    values each), rows_per_bucket and the active set per bucket."""
    p, act, cm = _prepare(p_comp_gold, active, components)
    bucket = np.asarray(bucket).astype(np.int64)
    n = p.shape[0]
    if bucket.shape != (n,) or (n and bucket.min() < 0):
        raise ValueError("bucket must be [n] with ids >= 0")
    nb = int(n_buckets) if n_buckets is not None else int(bucket.max()) + 1 if n else 0
    if n and bucket.max() >= nb:
        raise ValueError("bucket id >= n_buckets")
    if prior < 0:
        raise ValueError("prior must be >= 0")
    tr: dict = {}
    pooled = _pooled_em(p, act, iters, tr)
    rows_b = np.bincount(bucket, minlength=nb).astype(np.int64)
    # the active set of every bucket (must be unique)
    code = (act.astype(np.int64) << np.arange(N_COMP, dtype=np.int64)[None, :]).sum(axis=1)
    first = np.full(nb, -1, dtype=np.int64)
    first[bucket[::-1]] = code[::-1]
    if np.any(code != first[bucket]):
        bad = np.unique(bucket[code != first[bucket]])
        raise ValueError(f"buckets {bad[:10].tolist()} mix different active sets; the bucket must fix the active set")
    has = rows_b > 0
    A = np.zeros((nb, N_COMP), dtype=bool)
    A[has] = ((first[has][:, None] >> np.arange(N_COMP)[None, :]) & 1).astype(bool)
    pi_a = (np.where(A, pooled[None, :], 0.0)).sum(axis=1)
    n_a = A.sum(axis=1)
    a = np.where(A, prior * pooled[None, :] / np.where(pi_a > 0, pi_a, 1.0)[:, None], 0.0)
    a = np.where((pi_a > 0)[:, None], a, np.where(A, prior / np.maximum(n_a, 1)[:, None], 0.0))
    v = np.where(A, 1.0 / np.maximum(n_a, 1)[:, None], 0.0)
    lls, objs = [], []
    a_pos = a > 0
    for it in range(iters + 1):
        num = np.where(act, v[bucket] * p, 0.0)
        tot = num.sum(axis=1)
        ll_sum = math.fsum(np.log(tot).tolist())
        lp = math.fsum((a[a_pos] * np.log(v[a_pos])).tolist())
        lls.append(ll_sum / max(n, 1))
        objs.append((ll_sum + lp) / max(n, 1))
        if it == iters:
            break
        gam = num / tot[:, None]
        nbk = np.stack([np.bincount(bucket, weights=gam[:, i], minlength=nb) for i in range(N_COMP)], axis=1)
        den_b = np.where(has, rows_b + prior, 1.0).astype(np.float64)
        v = np.where(has[:, None], (nbk + a) / den_b[:, None], v)
    weights = np.where(has[:, None], v, pooled[None, :])
    weights = np.where(cm[None, :], weights, 0.0)
    if trace is not None:
        trace.update(tr)
        trace["bucket_ll"] = lls
        trace["bucket_map_objective"] = objs
        trace["rows_per_bucket"] = rows_b.tolist()
        trace["active_sets"] = {int(b): [COMPONENTS[i] for i in range(N_COMP) if A[b, i]] for b in np.flatnonzero(has)}
        trace["prior_pseudo_counts"] = float(prior)
        trace["iters"] = int(iters)
    return weights


# ------------------------------------------------------------------ scoring
def gold_prob(p_comp_gold, active, bucket, weights, components: Sequence[str] | None = None) -> np.ndarray:
    """P(gold) of every row: sum over active i of w_bi p_i / sum over active i of w_bi (fixed order)."""
    p_comp_gold = np.asarray(p_comp_gold, dtype=np.float64)
    act = np.asarray(active).astype(bool) & component_mask(components)[None, :]
    w = np.asarray(weights, dtype=np.float64)[np.asarray(bucket).astype(np.int64)]
    num = np.zeros(p_comp_gold.shape[0], dtype=np.float64)
    den = np.zeros(p_comp_gold.shape[0], dtype=np.float64)
    for i in range(N_COMP):
        a = act[:, i]
        num = num + np.where(a, w[:, i] * np.where(a, p_comp_gold[:, i], 0.0), 0.0)
        den = den + np.where(a, w[:, i], 0.0)
    if np.any(den <= 0.0):
        raise ValueError("a row has zero total weight over its active components")
    return num / den


def _doc_row(doc_rows, m):
    if doc_rows is None:
        return None
    if isinstance(doc_rows, Mapping):
        r = doc_rows.get(m)
    else:
        r = doc_rows[m] if m < len(doc_rows) else None
    if r is None:
        return None
    t, c = np.asarray(r[0]).astype(np.int64), np.asarray(r[1]).astype(np.int64)
    if t.shape[0] == 0 or int(c.sum()) <= 0:
        return None
    return t, c


def full_distribution(base_row, doc_rows, pointer_candidate: int, weights_row,
                      components: Sequence[str] | None = None) -> np.ndarray:
    """The CVTM distribution [V] of one position. base_row: P_KN(. | h) [V] (sums to one);
    doc_rows: for m = 0..4 (sequence or {m: ...}) None or (tokens, counts) of the earlier successors
    (`doc_candidates` rows; D_m is active iff the row is non-empty); pointer_candidate: token or -1;
    weights_row: the [7] weights of the position's bucket."""
    base_row = np.asarray(base_row, dtype=np.float64)
    w = np.asarray(weights_row, dtype=np.float64)
    cm = component_mask(components)
    out = w[0] * base_row
    den = 0.0 + w[0]
    for m in D_ORDERS:
        if not cm[1 + m]:
            continue
        r = _doc_row(doc_rows, m)
        if r is None:
            continue
        t, c = r
        pm = c / int(c.sum())
        out[t] = out[t] + w[1 + m] * pm
        den = den + w[1 + m]
    if cm[6] and pointer_candidate is not None and int(pointer_candidate) >= 0:
        k = int(pointer_candidate)
        out[k] = out[k] + w[6] * 1.0
        den = den + w[6]
    if den <= 0.0:
        raise ValueError("zero total weight over the active components")
    return out / den


def residual_votes(base_row, doc_rows, pointer_candidate: int, weights_row,
                   components: Sequence[str] | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Vote form of section 11.4: (default vote ln l_S, candidate tokens, their votes r(w)) with
    P(w) = P_KN(w | h) exp(r(w)); a class that is no document or pointer candidate gets the default."""
    base_row = np.asarray(base_row, dtype=np.float64)
    full = full_distribution(base_row, doc_rows, pointer_candidate, weights_row, components)
    w = np.asarray(weights_row, dtype=np.float64)
    cm = component_mask(components)
    den = w[0]
    cands = []
    for m in D_ORDERS:
        r = _doc_row(doc_rows, m) if cm[1 + m] else None
        if r is not None:
            den = den + w[1 + m]
            cands.append(r[0])
    if cm[6] and pointer_candidate is not None and int(pointer_candidate) >= 0:
        den = den + w[6]
        cands.append(np.array([int(pointer_candidate)], dtype=np.int64))
    tok = np.unique(np.concatenate(cands)) if cands else np.zeros(0, dtype=np.int64)
    default = math.log(w[0] / den)
    votes = np.log(full[tok]) - np.log(base_row[tok])
    return default, tok, votes


# ------------------------------------------------------------------ KN read-out helpers
def kn_gold_and_depth(kn, tok, doc_start, positions=None, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """(P_KN(tok[p] | left in-document context) float64, k* int64) for every position of `positions`
    (default: all) under a fitted `KNModel` of order K <= 5. k* is the longest fired suffix context:
    the largest k <= K - 1 whose length-k context has followers at level k + 1 (KNModel's `seen`),
    0 when only the bias fires."""
    tok = np.asarray(tok).astype(np.int64)
    doc_start = np.asarray(doc_start).astype(np.int64)
    order = kn.order
    if order - 1 > K_MAX:
        raise ValueError("k* is defined for orders up to 5")
    pos_all = np.arange(tok.shape[0], dtype=np.int64) if positions is None else np.asarray(positions).astype(np.int64)
    p_out = np.zeros(pos_all.shape[0], dtype=np.float64)
    k_out = np.zeros(pos_all.shape[0], dtype=np.int64)
    for i in range(0, pos_all.shape[0], chunk):
        pos = pos_all[i:i + chunk]
        ctx, valid = kn.contexts(tok, doc_start, pos, None, order)
        p_out[i:i + chunk] = kn.prob(ctx, tok[pos], valid, order)
        cids = kn.context_ids(ctx, valid, order)
        ks = np.zeros(pos.shape[0], dtype=np.int64)
        for k in range(2, order + 1):
            cid = cids[k]
            total = kn.level(k, order).total
            fired = (cid >= 0) & (total[np.where(cid >= 0, cid, 0)] > 0)
            ks = np.where(fired, np.maximum(ks, k - 1), ks)
        k_out[i:i + chunk] = ks
    return p_out, k_out


# ------------------------------------------------------------------ plumbing smoke
def _smoke(config_path: str) -> int:
    """Plumbing smoke on the V 8000 validation split (labelled smoke; NOT the protocol of section
    11.6, whose fit documents are held-out TRAINING documents): KN-5 on the first `train_tokens`
    training tokens (cut at a document boundary), document features of every validation position,
    one EM fit on the first half of the validation DOCUMENTS, bits per byte on the second half for
    KN-5 alone, KN-5 plus a document unigram cache (S, D0) and KN-5 plus all document clauses,
    streaming vs batch equality, full-distribution checks, and ns per token of the streaming pass."""
    import json
    import os
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    from tmlm.common import runlog
    from tmlm.cvtm import doc_clauses as dc
    from tmlm.data.corpus_store import load_raw_split, load_split, split_dir
    from tmlm.eval.baselines.ngram import KNModel

    cfg = json.loads(Path(config_path).read_text())
    t_start = time.time()
    metrics: dict = {}

    def log(msg: str) -> None:
        print(f"[{time.time() - t_start:8.1f} s] {msg}", flush=True)

    def put(name: str, value, label: str = "measured", **extra) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        metrics[name] = {"value": value, "label": label, "smoke": True, **extra}
        if os.environ.get("TMLM_RUN_DIR"):
            runlog.write_metrics({name: metrics[name]}, on_conflict="replace")

    corpus, vocab = cfg["corpus"], int(cfg["vocab"])
    tr = load_split(corpus, vocab, "train")
    ds_full = np.asarray(tr.doc_start).astype(np.int64)
    d = int(np.searchsorted(ds_full, int(cfg["train_tokens"]), side="right")) - 1
    end = int(ds_full[max(1, d)])
    tok_tr = np.asarray(tr.tok[:end]).astype(np.int64)
    ds_tr = ds_full[: int(np.searchsorted(ds_full, end, side="right"))]
    del tr
    put("data.kn_train_tokens", int(tok_tr.shape[0]), note="first training documents, cut at a document boundary")
    put("data.kn_train_documents", int(ds_tr.shape[0] - 1))
    t0 = time.time()
    kn = KNModel(int(cfg["kn_order"]), vocab).fit(tok_tr, ds_tr)
    put("time.kn_fit_s", time.time() - t0)
    log(f"KN-{cfg['kn_order']} fitted on {tok_tr.shape[0]} tokens")
    del tok_tr

    va = load_split(corpus, vocab, "validation")
    tok = np.asarray(va.tok).astype(np.int64)
    ds = np.asarray(va.doc_start).astype(np.int64)
    meta = json.loads((split_dir(corpus, vocab, "validation") / "meta.json").read_text())
    raw = load_raw_split(corpus, "validation")
    doc_bytes = np.asarray(raw.doc_bytes).astype(np.int64)
    n_docs = ds.shape[0] - 1
    if doc_bytes.shape[0] != n_docs + 1 or int(doc_bytes[-1]) != int(meta["n_bytes_utf8"]):
        raise RuntimeError("validation byte offsets do not match the token documents")
    n = tok.shape[0]
    doc_of = np.repeat(np.arange(n_docs), np.diff(ds))
    half = n_docs // 2
    fit_rows = doc_of < half
    score_rows = ~fit_rows
    score_bytes = int(doc_bytes[-1] - doc_bytes[half])
    put("data.validation_tokens", int(n))
    put("data.fit_documents", list(range(half)), note="first half of the validation documents (smoke only)")
    put("data.score_documents", list(range(half, n_docs)))
    put("data.score_tokens", int(score_rows.sum()))
    put("data.score_bytes_utf8", score_bytes)

    t0 = time.time()
    p_kn, k_star = kn_gold_and_depth(kn, tok, ds)
    put("time.kn_gold_s", time.time() - t0)
    put("kn.k_star_histogram", np.bincount(k_star, minlength=K_MAX + 1).tolist())

    t0 = time.time()
    feats = dc.doc_features(tok, ds)
    put("time.doc_features_batch_s", time.time() - t0)
    t0 = time.time()
    feats_s = dc.doc_features_stream(tok, ds)
    put("time.doc_features_stream_s", time.time() - t0, note="includes numba compilation")
    put("check.stream_equals_batch.whole_document", bool(feats_s.same_values(feats)))
    put("check.n_fallback", {"batch": feats.n_fallback, "stream": feats_s.n_fallback})
    for w in cfg["timing_windows"]:
        fw = dc.doc_features(tok, ds, window=w)
        put(f"check.stream_equals_batch.window_{w}", bool(dc.doc_features_stream(tok, ds, window=w).same_values(fw)))
    for m in range(5):
        put(f"doc.fire_rate.D{m}", float((feats.count(m) >= 1).mean()))
    put("doc.fire_rate.P", float((feats.ptr_len >= feats.ptr_min).mean()))
    put("doc.pointer_candidate_is_gold_when_firing",
        float((feats.ptr_cand == tok)[feats.ptr_len >= feats.ptr_min].mean()))

    p_comp, act = gold_components(p_kn, feats, tok)
    n_counts = counts_matrix(feats)
    ln2 = math.log(2.0)
    kn_bits = -np.log(p_kn) / ln2
    put("bpb.score_half.kn5", float(kn_bits[score_rows].sum() / score_bytes),
        note="KN-5 alone on the second half of the validation documents")
    for tag, comps in (("kn5+doc_unigram_cache", CACHE_COMPONENTS), ("kn5+doc_clauses", None)):
        bucket, nb = bucket_ids(k_star, n_counts, feats.ptr_len, comps)
        trace: dict = {}
        t0 = time.time()
        w = fit_map_em(p_comp[fit_rows], act[fit_rows], bucket[fit_rows], iters=int(cfg["em_iters"]),
                       prior=float(cfg["prior"]), n_buckets=nb, components=comps, trace=trace)
        put(f"time.em_fit_s.{tag}", time.time() - t0)
        dll = np.diff(np.asarray(trace["pooled_ll"]))
        dob = np.diff(np.asarray(trace["bucket_map_objective"]))
        put(f"em.{tag}.pooled_ll_min_step", float(dll.min()), note="must be >= -1e-12")
        put(f"em.{tag}.bucket_map_objective_min_step", float(dob.min()))
        put(f"em.{tag}.pooled_weights", dict(zip(BucketLayout.of(comps).components,
                                                 [x for x, keep in zip(trace["pooled_weights"], component_mask(comps)) if keep])))
        put(f"em.{tag}.buckets_with_fit_rows", int(np.count_nonzero(trace["rows_per_bucket"])), note=f"of {nb}")
        put(f"em.{tag}.fit_ll_first_last", [trace["bucket_ll"][0], trace["bucket_ll"][-1]], note="mean nats per fit row")
        pg = gold_prob(p_comp, act, bucket, w, comps)
        bits = -np.log(pg) / ln2
        put(f"bpb.score_half.{tag}", float(bits[score_rows].sum() / score_bytes))
        put(f"bpb.fit_half.{tag}", float(bits[fit_rows].sum() / int(doc_bytes[half])),
            label="derived", note="in-sample (the fit half), for plumbing only")
        log(f"{tag}: score-half bpb {bits[score_rows].sum() / score_bytes:.4f}")
        if comps is None:
            # full distributions on a deterministic sample of score positions
            sample = np.flatnonzero(score_rows)[:: max(1, int(score_rows.sum()) // int(cfg["full_dist_positions"]))]
            ctx, valid = kn.contexts(tok, ds, sample, None, kn.order)
            base = kn.dist(ctx, valid)
            cands = dc.doc_candidates(tok, ds, sample)
            sums, gdiff, vdiff, pdiff = [], [], [], []
            for r, pidx in enumerate(sample):
                rows = [cands[m].row(r) for m in range(5)]
                full = full_distribution(base[r], rows, int(feats.ptr_cand[pidx]) if feats.ptr_len[pidx] >= 5 else -1,
                                         w[bucket[pidx]])
                sums.append(abs(float(full.sum()) - 1.0))
                gdiff.append(abs(float(full[tok[pidx]]) - float(pg[pidx])))
                pdiff.append(abs(float(base[r, tok[pidx]]) - float(p_kn[pidx])))
                dv, ct, cv = residual_votes(base[r], rows, int(feats.ptr_cand[pidx]) if feats.ptr_len[pidx] >= 5 else -1,
                                            w[bucket[pidx]])
                rv = np.full(vocab, dv)
                rv[ct] = cv
                vdiff.append(float(np.max(np.abs(base[r] * np.exp(rv) - full))))
            put("check.full_distribution.positions", int(sample.shape[0]))
            put("check.full_distribution.max_abs_sum_minus_1", max(sums))
            put("check.full_distribution.max_abs_gold_minus_gold_prob", max(gdiff),
                note="nonzero only through KNModel.dist vs KNModel.prob rounding (see max_abs_kn_dist_minus_prob)")
            put("check.full_distribution.max_abs_kn_dist_minus_prob", max(pdiff))
            put("check.residual_votes.max_abs_error", max(vdiff))
    # streaming cost per token (H8.7c style, contended host)
    for w_ in cfg["timing_windows"]:
        med, per = dc.ns_per_token(tok, ds, w_, repeats=int(cfg["timing_repeats"]), return_all=True)
        put(f"time.stream_ns_per_token.window_{w_}", med, per_pass=per,
            note="median of the timed passes, single thread, nice 19, host shared with other jobs (contended)")
        log(f"stream ns/token at N={w_}: {med:.0f}")
    put("time.wall_s", time.time() - t_start)
    print(json.dumps({k: v["value"] for k, v in metrics.items()}, indent=1, default=str), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--smoke", metavar="CONFIG", required=True, help="config JSON of the plumbing smoke")
    args = ap.parse_args(argv)
    return _smoke(args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
