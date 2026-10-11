"""Class and skip templates for pilot P2 of docs/13 amendment A5 (EXPLORATORY, counting only, no TM).

P2 asks for the CEILING of class-based and skip contexts at full data: fixed templates are counted over
the training split WITHOUT H' (docs/13 A4: residues 0, 1, 2 held out), each template's successor
distribution enters a normalised linear mixture with the frozen CVTM-core, the mixture weights are
fitted on residue 2a of H' and the mixture is judged on residue 2b. Validation and test are not used.

Templates
---------
The names follow `experiments/tm_levers_feasibility_causal.py` (part D, `feats`), where six of the
eight come from: `t` is the exact token, `c2` the COARSE class (in that script the 256-slot exchange
level, kind_map row 2, 243 classes in use at V 8000), `c3` the FINE class (the 4096-slot level,
kind_map row 3, 2,497 classes in use); the digits after a symbol are the left offsets it is read at
(1 = the token just before the predicted position p, 2 = two before, ...). The context of template T
at position p is the tuple below; offsets before the start of p's document read a per-kind EDGE
symbol (the `shifted(..., lo, i, EDGE)` rule of the source). The template predicts tok[p]:

| name       | context tuple at p                                   | source |
|------------|------------------------------------------------------|--------|
| c2_1       | (c2[t[p-1]])                                         | feasibility script |
| c2_21      | (c2[t[p-2]], c2[t[p-1]])                             | feasibility script |
| t1_c2_2    | (t[p-1], c2[t[p-2]])                                 | feasibility script |
| t1_c3_2    | (t[p-1], c3[t[p-2]])                                 | new in A5: t1_c2_2 with the fine class |
| c3_321     | (c3[t[p-3]], c3[t[p-2]], c3[t[p-1]])                 | feasibility script |
| t21_c2_3   | (t[p-2], t[p-1], c2[t[p-3]])                         | feasibility script |
| t32_skip   | (t[p-3], t[p-2])  (skips p-1; no class)              | feasibility script |
| t321_c2_4  | (t[p-3], t[p-2], t[p-1], c2[t[p-4]])                 | new in A5: t21_c2_3 one order up |

The two templates that are not in the source file (t1_c3_2, t321_c2_4) are read by the same naming
rule; this reading is a definition made here, not a registered text. `t2_skip` of the source is not
among the A5 templates and is not used.

Every context tuple is packed EXACTLY into an int64 (mixed radix: token base V + 1, coarse base K2 + 1,
fine base K3 + 1, the EDGE symbol being the last digit value); the (context, next token) key is
context * V + token. `check_key_range` asserts that no key can overflow. There is no hashing of
contexts, so there are no collisions: the counts are exact integers.

Estimator of one template at an evaluation position p (docs/11 11.2 style maximum likelihood, the
source's rule): N(p) = number of counting positions with the same context, P_T(tok[p] | p) =
#(context, tok[p]) / N(p); the template FIRES only when N(p) > 0.

Class maps (A5: "the repository's exchange clusters (243 and 2,497), PPMI-SVD k-means (256 and 2,048),
and a random partition of equal size"); each map family supplies the coarse (c2) and the fine (c3) role:
- exchange: kind_map rows 2 and 3 of the train store (`tmlm.data.clusters`, fitted on the whole train
  split, H' included: a class-level, not a count-level, use of H'; stated in the run log);
- ppmi: positional PPMI (offsets -2, -1, +1, +2 inside documents, counted over the training split
  without H'; context distribution smoothing 0.75, Levy, Goldberg and Dagan 2015), truncated SVD
  (k = 128, ARPACK with a seeded start vector), rows U * sqrt(S) normalised to unit length, k-means
  (Lloyd, one initialisation, k-means++ seeding, 300 iterations, one thread) with 256 and 2,048
  clusters;
- random: for each real map family, the SAME labels permuted over the vocabulary with a seeded
  generator (`random_like`). The random map therefore has exactly the real map's number of classes and
  its class-size histogram (equal size in both readings of the phrase).

Mixture (the source's `fit_mix` / `eval_mix`, reimplemented here and tested equal):
P(p) = (P_core(p) + sum over firing templates i of w_b(i, p) P_i(p)) / (1 + sum over firing i of w_b(i, p)),
w = exp(theta), bucket b(i, p) = (template i, k*(p), count bucket min(floor(log2 N_i(p)), 23)), k* the
longest fired suffix context of the core's base at p (A5: "k*-by-count-bucket weights"); theta fitted by
L-BFGS-B (bounds [-30, 10], start -3 for buckets seen in the fit rows, -30 otherwise) on residue 2a.

Residue 2a / 2b (A5 P1: "half of residue 2's 261 documents, split by document"): the residue-2
documents in ascending stable document ID; even ranks form 2a, odd ranks form 2b (131 and 130).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

# kinds of a template item
TOK, COARSE, FINE = 0, 1, 2

# (kind, left offset) items, farthest first (the packing order of the source's `feats`)
TEMPLATES: dict[str, tuple[tuple[int, int], ...]] = {
    "c2_1": ((COARSE, 1),),
    "c2_21": ((COARSE, 2), (COARSE, 1)),
    "t1_c2_2": ((TOK, 1), (COARSE, 2)),
    "t1_c3_2": ((TOK, 1), (FINE, 2)),
    "c3_321": ((FINE, 3), (FINE, 2), (FINE, 1)),
    "t21_c2_3": ((TOK, 2), (TOK, 1), (COARSE, 3)),
    "t32_skip": ((TOK, 3), (TOK, 2)),
    "t321_c2_4": ((TOK, 3), (TOK, 2), (TOK, 1), (COARSE, 4)),
}
TEMPLATE_NAMES = tuple(TEMPLATES)
N_CB = 24                    # count buckets floor(log2 N) capped at 23 (the source's 24)
N_KSTAR = 5                  # k* in 0..4
LN2 = math.log(2.0)


def uses_map(name: str) -> bool:
    """True when the template reads a class (so its counts depend on the map family)."""
    return any(k != TOK for k, _ in TEMPLATES[name])


# ------------------------------------------------------------------ class maps
@dataclass(frozen=True)
class ClassMap:
    """Coarse and fine class of every token (contiguous ids 0..K-1)."""

    name: str
    coarse: np.ndarray       # int64 [V]
    fine: np.ndarray         # int64 [V]

    @property
    def k_coarse(self) -> int:
        return int(self.coarse.max()) + 1

    @property
    def k_fine(self) -> int:
        return int(self.fine.max()) + 1


def contiguous(labels: np.ndarray) -> np.ndarray:
    """Labels renumbered 0..K-1 in ascending order of the original label (K = distinct labels)."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64).reshape(-1)


def exchange_map(kind_map: np.ndarray) -> ClassMap:
    """The repository's exchange clusters: kind_map rows 2 (coarse) and 3 (fine), renumbered."""
    km = np.asarray(kind_map).astype(np.int64)
    return ClassMap("exchange", contiguous(km[2]), contiguous(km[3]))


def random_like(cm: ClassMap, seed: Sequence[int]) -> ClassMap:
    """The labels of `cm` permuted over the vocabulary (numpy SeedSequence(seed); one permutation for
    the coarse and an independent one for the fine map): same class count and class-size histogram."""
    ss = np.random.SeedSequence([int(s) for s in seed])
    rc, rf = (np.random.default_rng(s) for s in ss.spawn(2))
    return ClassMap(f"random_{cm.name}", rc.permutation(cm.coarse).astype(np.int64),
                    rf.permutation(cm.fine).astype(np.int64))


@njit(cache=False)
def _cooccur(tok, doc_start, keep_doc, off, vocab):
    """int64 [V, V]: counts of (tok[p], tok[p + off]) with p and p + off in the same kept document."""
    c = np.zeros((vocab, vocab), np.int64)
    for d in range(doc_start.shape[0] - 1):
        if not keep_doc[d]:
            continue
        lo = doc_start[d]
        hi = doc_start[d + 1]
        for p in range(lo, hi - off):
            c[tok[p], tok[p + off]] += 1
    return c


def ppmi_matrix(tok: np.ndarray, doc_start: np.ndarray, keep_doc: np.ndarray, vocab: int,
                offsets: Sequence[int] = (-2, -1, 1, 2), cds: float = 0.75):
    """Sparse CSR [V, len(offsets) * V] positive PMI of (word, (offset, context word)) inside documents,
    context distribution smoothing `cds`. Returns (matrix, info)."""
    import scipy.sparse as sp

    tok = np.asarray(tok)
    if tok.dtype not in (np.uint16, np.int64):
        tok = tok.astype(np.int64)
    tok = np.ascontiguousarray(tok)
    ds = np.ascontiguousarray(np.asarray(doc_start).astype(np.int64))
    keep = np.ascontiguousarray(np.asarray(keep_doc).astype(np.bool_))
    fwd = {o: _cooccur(tok, ds, keep, o, vocab) for o in sorted({abs(int(o)) for o in offsets})}
    blocks = []
    for o in offsets:
        o = int(o)
        m = fwd[abs(o)] if o > 0 else fwd[abs(o)].T      # (w at p, c at p + o)
        blocks.append(sp.csr_matrix(m))
    del fwd
    C = sp.hstack(blocks, format="csr").astype(np.float64)
    total = float(C.sum())
    rw = np.asarray(C.sum(axis=1)).ravel()
    cc = np.asarray(C.sum(axis=0)).ravel()
    pc = cc ** cds
    pc = pc / pc.sum()
    coo = C.tocoo()
    pmi = np.log(coo.data * total) - np.log(rw[coo.row]) - np.log(total * pc[coo.col])
    keep_e = pmi > 0
    M = sp.csr_matrix((pmi[keep_e], (coo.row[keep_e], coo.col[keep_e])), shape=C.shape)
    info = {"pairs": total, "nnz_counts": int(C.nnz), "nnz_ppmi": int(M.nnz), "offsets": [int(o) for o in offsets],
            "cds": cds, "tokens_without_context": int((rw == 0).sum())}
    return M, info


def svd_embed(M, dims: int, seed: Sequence[int]) -> tuple[np.ndarray, dict]:
    """U * sqrt(S) of the rank-`dims` truncated SVD (ARPACK, seeded start vector), rows L2-normalised
    (zero rows stay zero)."""
    from scipy.sparse.linalg import svds

    rng = np.random.default_rng(np.random.SeedSequence([int(s) for s in seed]))
    v0 = rng.standard_normal(min(M.shape))
    u, s, _ = svds(M, k=int(dims), v0=v0, solver="arpack")
    order = np.argsort(-s, kind="stable")
    u, s = u[:, order], s[order]
    X = u * np.sqrt(s)[None, :]
    nrm = np.linalg.norm(X, axis=1)
    X = X / np.where(nrm > 0, nrm, 1.0)[:, None]
    return X, {"dims": int(dims), "singular_values_top5": s[:5].tolist(), "singular_value_min": float(s[-1])}


def kmeans_labels(X: np.ndarray, k: int, seed: int, max_iter: int = 300) -> tuple[np.ndarray, dict]:
    """Lloyd k-means (scikit-learn, k-means++ seeding, one initialisation, one thread); contiguous labels."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=int(k), n_init=1, max_iter=int(max_iter), random_state=int(seed),
                    algorithm="lloyd").fit(X)
    lab = contiguous(km.labels_)
    return lab, {"k": int(k), "non_empty": int(lab.max()) + 1, "n_iter": int(km.n_iter_),
                 "inertia": float(km.inertia_)}


def ppmi_svd_map(tok, doc_start, keep_doc, vocab: int, sizes=(256, 2048), dims: int = 128,
                 seed: int = 100) -> tuple[ClassMap, dict]:
    """The PPMI-SVD k-means map family (coarse = sizes[0], fine = sizes[1] clusters)."""
    M, info = ppmi_matrix(tok, doc_start, keep_doc, vocab)
    X, sinfo = svd_embed(M, dims, (seed, 1))
    del M
    coarse, ci = kmeans_labels(X, sizes[0], seed)
    fine, fi = kmeans_labels(X, sizes[1], seed)
    return ClassMap("ppmi", coarse, fine), {"ppmi": info, "svd": sinfo, "kmeans_coarse": ci, "kmeans_fine": fi}


# ------------------------------------------------------------------ exact template keys
def template_arrays(name: str, cm: ClassMap | None, vocab: int):
    """(kinds, offsets, maps [3, V], bases [3], edges [3]) of a template under a class map."""
    items = TEMPLATES[name]
    kinds = np.array([k for k, _ in items], np.int64)
    offs = np.array([o for _, o in items], np.int64)
    ident = np.arange(vocab, dtype=np.int64)
    if cm is None:
        if uses_map(name):
            raise ValueError(f"{name} needs a class map")
        coarse = fine = np.zeros(vocab, np.int64)
        kc = kf = 1
    else:
        coarse, fine, kc, kf = cm.coarse, cm.fine, cm.k_coarse, cm.k_fine
    maps = np.ascontiguousarray(np.stack([ident, coarse, fine]).astype(np.int64))
    edges = np.array([vocab, kc, kf], np.int64)
    bases = edges + 1
    check_key_range(kinds, bases, vocab)
    return kinds, offs, maps, bases, edges


def check_key_range(kinds: np.ndarray, bases: np.ndarray, vocab: int) -> int:
    """Largest possible (context, token) key + 1; raises when it does not fit in int64."""
    top = 1
    for k in kinds:
        top *= int(bases[int(k)])
    top *= int(vocab)
    if top >= 2 ** 63:
        raise OverflowError(f"template key range {top} does not fit in int64")
    return top


@njit(cache=False)
def _ctx_key(tok, p, lo, kinds, offs, maps, bases, edges):
    key = 0
    for j in range(kinds.shape[0]):
        kd = kinds[j]
        q = p - offs[j]
        if q < lo:
            v = edges[kd]
        else:
            v = maps[kd, tok[q]]
        key = key * bases[kd] + v
    return key


@njit(cache=False)
def stream_keys(tok, doc_start, kinds, offs, maps, bases, edges):
    """int64 [n]: the context key of every position of a stream of whole documents."""
    n = tok.shape[0]
    out = np.empty(n, np.int64)
    for d in range(doc_start.shape[0] - 1):
        lo = doc_start[d]
        for p in range(lo, doc_start[d + 1]):
            out[p] = _ctx_key(tok, p, lo, kinds, offs, maps, bases, edges)
    return out


# ------------------------------------------------------------------ exact counting through an open-addressing table
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


@njit(cache=False)
def _slot(key, mask):
    z = np.uint64(key) + _GOLDEN
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    z = z ^ (z >> _S31)
    return np.int64(z & mask)


@njit(cache=False)
def build_table(keys):
    """Open-addressing table of DISTINCT non-negative int64 keys: (slot keys, slot index, mask)."""
    n = keys.shape[0]
    size = 16
    while size < 2 * n:
        size *= 2
    tk = np.full(size, -1, np.int64)
    ti = np.full(size, -1, np.int64)
    mask = np.uint64(size - 1)
    for i in range(n):
        h = _slot(keys[i], mask)
        while tk[h] != -1:
            h = np.int64((np.uint64(h) + np.uint64(1)) & mask)
        tk[h] = keys[i]
        ti[h] = i
    return tk, ti, mask


@njit(cache=False)
def lookup(tk, ti, mask, key):
    """Index of `key` in the table's key list, -1 when absent."""
    h = _slot(key, mask)
    while True:
        v = tk[h]
        if v == key:
            return ti[h]
        if v == -1:
            return -1
        h = np.int64((np.uint64(h) + np.uint64(1)) & mask)


@njit(cache=False)
def _count_pass(tok, doc_start, keep_doc, kinds, offs, maps, bases, edges, vocab,
                tkc, tic, mc, tkf, tif, mf, cnt_c, cnt_f):
    n_pos = 0
    for d in range(doc_start.shape[0] - 1):
        if not keep_doc[d]:
            continue
        lo = doc_start[d]
        for p in range(lo, doc_start[d + 1]):
            n_pos += 1
            c = _ctx_key(tok, p, lo, kinds, offs, maps, bases, edges)
            i = lookup(tkc, tic, mc, c)
            if i >= 0:
                cnt_c[i] += 1
                j = lookup(tkf, tif, mf, c * vocab + tok[p])
                if j >= 0:
                    cnt_f[j] += 1
    return n_pos


@dataclass(frozen=True)
class TemplateCounts:
    """Per evaluation position: N (count of the context over the counting positions) and C (count of
    (context, gold)); p = C / N where N > 0."""

    name: str
    n: np.ndarray            # int64 [m]
    c: np.ndarray            # int64 [m]
    n_contexts_eval: int
    counting_positions: int

    @property
    def fires(self) -> np.ndarray:
        return self.n > 0

    @property
    def p(self) -> np.ndarray:
        return np.where(self.n > 0, self.c / np.maximum(self.n, 1), 0.0)


def count_template(name: str, cm: ClassMap | None, vocab: int,
                   eval_tok: np.ndarray, eval_doc_start: np.ndarray, eval_rows: np.ndarray,
                   cnt_tok: np.ndarray, cnt_doc_start: np.ndarray, cnt_keep_doc: np.ndarray) -> TemplateCounts:
    """Exact counts of template `name` for the evaluation rows `eval_rows` (positions of the evaluation
    stream, whole documents given by eval_doc_start) over the kept documents of the counting stream."""
    kinds, offs, maps, bases, edges = template_arrays(name, cm, vocab)
    etok = np.ascontiguousarray(np.asarray(eval_tok).astype(np.int64))
    eds = np.ascontiguousarray(np.asarray(eval_doc_start).astype(np.int64))
    rows = np.asarray(eval_rows).astype(np.int64)
    ctx = stream_keys(etok, eds, kinds, offs, maps, bases, edges)[rows]
    full = ctx * vocab + etok[rows]
    uc, ic = np.unique(ctx, return_inverse=True)
    uf, iff = np.unique(full, return_inverse=True)
    tkc, tic, mc = build_table(uc)
    tkf, tif, mf = build_table(uf)
    cnt_c = np.zeros(uc.shape[0], np.int64)
    cnt_f = np.zeros(uf.shape[0], np.int64)
    ctok = np.asarray(cnt_tok)
    if ctok.dtype not in (np.uint16, np.int64):
        ctok = ctok.astype(np.int64)
    ctok = np.ascontiguousarray(ctok)
    npos = _count_pass(ctok, np.ascontiguousarray(np.asarray(cnt_doc_start).astype(np.int64)),
                       np.ascontiguousarray(np.asarray(cnt_keep_doc).astype(np.bool_)),
                       kinds, offs, maps, bases, edges, np.int64(vocab), tkc, tic, mc, tkf, tif, mf, cnt_c, cnt_f)
    return TemplateCounts(name=name, n=cnt_c[ic.reshape(-1)], c=cnt_f[iff.reshape(-1)],
                          n_contexts_eval=int(uc.shape[0]), counting_positions=int(npos))


# ------------------------------------------------------------------ mixture (the source's fit_mix / eval_mix)
def count_bucket(n: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(np.log2(np.maximum(n, 1))).astype(np.int64), N_CB - 1)


def components(counts: Sequence[TemplateCounts], kstar: np.ndarray):
    """(rows, bucket, p, n_buckets) of the firing templates; bucket = (template, k*, count bucket)."""
    ks = np.asarray(kstar).astype(np.int64)
    if ks.min() < 0 or ks.max() >= N_KSTAR:
        raise ValueError("k* must lie in 0..4")
    per = N_KSTAR * N_CB
    rows, b, p = [], [], []
    for i, tc in enumerate(counts):
        r = np.flatnonzero(tc.fires)
        rows.append(r)
        b.append(i * per + ks[r] * N_CB + count_bucket(tc.n[r]))
        p.append(tc.p[r])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 0
    return np.concatenate(rows), np.concatenate(b), np.concatenate(p), len(counts) * per


def fit_mix(pb, comp_rows, comp_b, comp_p, n_b, fit_mask, iters=400):
    """Normalized linear mixture P_t = (pb_t + sum_i w_b(i) p_i) / (1 + sum_i w_b(i)), theta = log w
    fitted by L-BFGS-B on the rows of fit_mask (identical to `fit_mix` of
    experiments/tm_levers_feasibility_causal.py; tested). Buckets without fit rows stay at -30."""
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
    """Bits (-log2 P) of the mixture at every row of pb."""
    w = np.exp(theta)[comp_b]
    n = pb.shape[0]
    W = 1.0 + np.bincount(comp_rows, weights=w, minlength=n)
    N = pb + np.bincount(comp_rows, weights=w * comp_p, minlength=n)
    return -(np.log(N) - np.log(W)) / LN2


# ------------------------------------------------------------------ residue 2a / 2b
def split_2ab(doc_index: np.ndarray, residue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_2a, is_2b) per document: residue-2 documents in ascending stable ID, even ranks 2a, odd 2b."""
    di = np.asarray(doc_index).astype(np.int64)
    r2 = np.flatnonzero(np.asarray(residue) == 2)
    order = r2[np.argsort(di[r2], kind="stable")]
    a = np.zeros(di.shape[0], bool)
    b = np.zeros(di.shape[0], bool)
    a[order[0::2]] = True
    b[order[1::2]] = True
    return a, b


# ------------------------------------------------------------------ A5 P2 verdict
GO_MIN = 0.010
NOGO_G = 0.005
NOGO_R = 0.003


def verdict(g_best: float, g_best_ci: Sequence[float], r: float) -> dict:
    """A5 P2: NO-GO when G_best < 0.005 or R < 0.003; GO when G_best >= 0.010 with its CI excluding 0;
    otherwise "in between". A5 does not order the two rules; where both hold (G_best >= 0.010 with the CI
    excluding 0 but R < 0.003) NO-GO is returned, because a random map of equal size then does as well
    (the flag `overlap` says so)."""
    nogo = (g_best < NOGO_G) or (r < NOGO_R)
    ci_excl0 = (g_best_ci[0] > 0) or (g_best_ci[1] < 0)
    go = (g_best >= GO_MIN) and ci_excl0
    if nogo:
        v = "NO-GO"
    elif go:
        v = "GO"
    else:
        v = "IN_BETWEEN"
    return {"verdict": v, "nogo_rule": bool(nogo), "go_rule": bool(go), "overlap": bool(nogo and go),
            "g_best_ci_excludes_0": bool(ci_excl0)}
