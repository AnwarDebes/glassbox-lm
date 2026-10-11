"""Learning-credit controls of docs/13 amendment A7 (every R1 arm): C0y, G+ and the count baselines.

C0y  the same run stopped at 1 percent of epoch 1: the checkpoint written after the first
     `c0y_batches(n_examples of epoch 0, B)` = ceil(0.01 x number of batches of epoch 0) batches of the arm's
     own training run (experiments/ssl_r1_pretext.py --stage train writes it; same seed, same data order).

G+   the most frequent single (offset, symbol) literals and literal pairs, equal in number to the clauses, as a
     feature map. Literal space (docs/11 sections 1.2 to 1.4, exact segments, positive literals): for a
     position p and an offset o of the arm's mode (masked and diffusion: -8..-1, +1..+8; causal: -8..-1),
     q = p + o; if q lies in the document of p the true literals are (o, kind_map[k][tok[q]]) for the kinds
     k = 0..3, else the single true literal is (o, SYM_EDGE). Nothing is hidden (the featurisation of the
     banks hides only the target position, which is never an offset). A pair is two true literals at
     DIFFERENT offsets (two kinds at one offset are nested cluster prefixes of one token, so such a pair
     would only repeat its finer member). Frequency = the number of positions of the pretraining prefix
     where the literal (pair) is true, all positions, exact (no sampling). Ranking: count descending, then
     single before pair, then literal ids ascending (packed as docs/11 lit ids, seg << 17 | sym << 1);
     the first n features (n = the arm's C) are G+; "dimension-matched" G+ is its first 8,192 features.
     The feature value of a text is the number of its positions where the feature is true (the same count
     code as the clause banks: log(1 + count), L2 rows, in the probe).
     Every G+ routine takes an array module `xp` (numpy for tests and small data, cupy on the GPU); the
     results are identical integers either way.

Count baselines (same token streams, text tokens only): bag of words (BPE-8k unigrams), cluster bag of
words (the exchange clusters of the WikiText-103 train kind_map, levels 1 to 3), and PPMI-SVD over the same
window of 8 (offsets -8..-1 and +1..+8 on the pretraining prefix; PPMI with context-distribution smoothing
0.75 and the SVD embedding U_300 S_300^0.5, the primary of src/tmlm/ssl/concepts.py), a text being the
mean vector of its tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..engine.config import LIT_EMPTY, n_sym, offset_seg, sym_edge
from . import features as F

C0Y_FRACTION = 0.01
N_KINDS = 4


# ------------------------------------------------------------------ C0y
def c0y_batches(n_examples_epoch0: int, batch: int, frac: float = C0Y_FRACTION) -> int:
    """Number of batches of epoch 0 after which the young bank C0y is saved: ceil(frac x batches)."""
    nb = (int(n_examples_epoch0) + int(batch) - 1) // int(batch)
    return max(1, int(math.ceil(frac * nb)))


# ------------------------------------------------------------------ G+ literal space
def gplus_offsets(mode: str) -> tuple[int, ...]:
    if mode == "causal":
        return tuple(range(-8, 0))
    if mode in ("masked", "diffusion"):
        return tuple(range(-8, 0)) + tuple(range(1, 9))
    raise ValueError(mode)


def columns(mode: str) -> list[tuple[int, int]]:
    """Column c = (offset, kind) in the order offsets ascending, kinds 0..3."""
    return [(o, k) for o in gplus_offsets(mode) for k in range(N_KINDS)]


def column_pairs(mode: str) -> list[tuple[int, int]]:
    """(c1, c2), c1 < c2, with different offsets (so offset(c1) < offset(c2) and lit(c1) < lit(c2))."""
    cols = columns(mode)
    return [(a, b) for a in range(len(cols)) for b in range(a + 1, len(cols)) if cols[a][0] != cols[b][0]]


def lit_of(o, sym, xp=np):
    """Packed docs/11 literal id of the positive exact literal (offset o, symbol sym) (arrays allowed)."""
    return (np.int64(offset_seg(int(o))) << np.int64(17)) | (xp.asarray(sym).astype(xp.int64) << 1)


def doc_bounds(xp, doc_start) -> tuple[Any, Any]:
    """(lo, hi) int64 xp arrays per position: the document range [lo, hi) of every position (built on the host)."""
    ds = np.asarray(doc_start, np.int64)
    lens = np.diff(ds)
    return xp.asarray(np.repeat(ds[:-1], lens)), xp.asarray(np.repeat(ds[1:], lens))


def sym_column(xp, tok, lo, hi, kind_map, pos, o: int, k: int, vocab: int):
    """int32 [len(pos)]: the symbol of column (offset o, kind k) at the positions (-1: no true literal)."""
    q = pos + int(o)
    inside = (q >= lo[pos]) & (q < hi[pos])
    tq = tok[xp.clip(q, 0, max(0, int(tok.shape[0]) - 1))].astype(xp.int64)
    s = kind_map[k][tq].astype(xp.int32)
    fill = xp.int32(int(sym_edge(vocab)) if k == 0 else -1)
    return xp.where(inside, s, fill)


def sym_matrix(xp, tok, lo, hi, kind_map, pos, mode: str, vocab: int):
    """int32 [len(pos), n_cols]: the symbol of every column (`columns(mode)` order) at the positions, -1 where
    no literal of that column is true. tok, lo, hi (from doc_bounds), kind_map, pos: xp arrays."""
    pos = pos.astype(xp.int64)
    cols = columns(mode)
    out = xp.empty((pos.shape[0], len(cols)), dtype=xp.int32)
    for c, (o, k) in enumerate(cols):
        out[:, c] = sym_column(xp, tok, lo, hi, kind_map, pos, o, k, vocab)
    return out


@dataclass
class GPlus:
    """The ranked G+ features: feature id f = rank. lit2 == LIT_EMPTY for singles."""

    mode: str
    vocab: int
    lit1: np.ndarray       # int64 [n]
    lit2: np.ndarray       # int64 [n]
    count: np.ndarray      # int64 [n]
    col1: np.ndarray       # int64 [n] column of lit1
    col2: np.ndarray       # int64 [n] column of lit2 (-1 for singles)
    sym1: np.ndarray       # int64 [n]
    sym2: np.ndarray       # int64 [n] (-1 for singles)
    n_positions: int
    info: dict

    @property
    def n(self) -> int:
        return int(self.lit1.shape[0])

    @property
    def is_pair(self) -> np.ndarray:
        return self.lit2 != LIT_EMPTY

    def save(self, path) -> None:
        import json
        np.savez(path, lit1=self.lit1, lit2=self.lit2, count=self.count, col1=self.col1, col2=self.col2,
                 sym1=self.sym1, sym2=self.sym2, mode=self.mode, vocab=np.int64(self.vocab),
                 n_positions=np.int64(self.n_positions), info=json.dumps(self.info))

    @staticmethod
    def load(path) -> "GPlus":
        import json
        z = np.load(path)
        return GPlus(str(z["mode"]), int(z["vocab"]), z["lit1"], z["lit2"], z["count"], z["col1"], z["col2"],
                     z["sym1"], z["sym2"], int(z["n_positions"]), json.loads(str(z["info"])))

    def describe(self, f: int, token_text=None) -> str:
        from ..engine.config import lit_text
        a = lit_text(int(self.lit1[f]), self.vocab, token_text)
        if self.lit2[f] == LIT_EMPTY:
            return a
        return a + " AND " + lit_text(int(self.lit2[f]), self.vocab, token_text)


def _to_host(xp, a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _rank(cnt: np.ndarray, kind: np.ndarray, lit1: np.ndarray, lit2: np.ndarray) -> np.ndarray:
    """Order: count desc, single (0) before pair (1), lit1 asc, lit2 asc."""
    return np.lexsort((lit2, lit1, kind, -cnt))


def gplus_select(xp, tok, doc_start, kind_map, mode: str, vocab: int, n_features: int,
                 positions=None, chunk: int = 1 << 22, log=None) -> GPlus:
    """Exact top-`n_features` singles and pairs over `positions` (default every position of the corpus)."""
    tok_d = xp.asarray(np.asarray(tok))
    km_d = xp.asarray(np.asarray(kind_map)).astype(xp.int64)
    lo, hi = doc_bounds(xp, np.asarray(doc_start))
    n_tok = int(tok_d.shape[0])
    pos_all = xp.arange(n_tok, dtype=xp.int64) if positions is None else xp.asarray(np.asarray(positions)).astype(xp.int64)
    n_pos = int(pos_all.shape[0])
    ns = int(n_sym(vocab))
    cols = columns(mode)
    n_cols = len(cols)
    # singles: exact counts per column (bincount per chunk of positions)
    single_cnt = np.zeros((n_cols, ns), np.int64)
    for a in range(0, n_pos, chunk):
        p = pos_all[a:a + chunk]
        for c, (o, k) in enumerate(cols):
            v = sym_column(xp, tok_d, lo, hi, km_d, p, o, k, vocab)
            v = v[v >= 0]
            if v.size:
                single_cnt[c] += _to_host(xp, xp.bincount(v.astype(xp.int64), minlength=ns))
            del v
    cc, ss = np.nonzero(single_cnt)
    cand = {"cnt": single_cnt[cc, ss], "kind": np.zeros(cc.shape[0], np.int64),
            "lit1": np.array([int(lit_of(cols[c][0], s)) for c, s in zip(cc.tolist(), ss.tolist())], np.int64),
            "lit2": np.full(cc.shape[0], LIT_EMPTY, np.int64), "col1": cc.astype(np.int64), "col2": np.full(cc.shape[0], -1, np.int64),
            "sym1": ss.astype(np.int64), "sym2": np.full(cc.shape[0], -1, np.int64)}

    def cut(c: dict) -> tuple[dict, int]:
        order = _rank(c["cnt"], c["kind"], c["lit1"], c["lit2"])[: int(n_features)]
        c = {k: v[order] for k, v in c.items()}
        tau = int(c["cnt"][-1]) if c["cnt"].shape[0] >= int(n_features) else 1
        return c, tau

    cand, tau = cut(cand)
    col_max = single_cnt.max(axis=1)
    pairs = column_pairs(mode)
    n_skipped = 0
    for pi, (c1, c2) in enumerate(pairs):
        if min(col_max[c1], col_max[c2]) < tau:
            n_skipped += 1
            continue
        keys_parts = []
        for a in range(0, n_pos, chunk):
            p = pos_all[a:a + chunk]
            s1 = sym_column(xp, tok_d, lo, hi, km_d, p, cols[c1][0], cols[c1][1], vocab).astype(xp.int64)
            s2 = sym_column(xp, tok_d, lo, hi, km_d, p, cols[c2][0], cols[c2][1], vocab).astype(xp.int64)
            ok = (s1 >= 0) & (s2 >= 0)
            keys_parts.append(s1[ok] * ns + s2[ok])
            del s1, s2, ok
        keys = xp.concatenate(keys_parts) if len(keys_parts) > 1 else keys_parts[0]
        if keys.size == 0:
            continue
        u, n = xp.unique(keys, return_counts=True)
        keep = n >= tau
        u, n = _to_host(xp, u[keep]), _to_host(xp, n[keep]).astype(np.int64)
        if u.size == 0:
            continue
        s1h, s2h = u // ns, u % ns
        o1, o2 = cols[c1][0], cols[c2][0]
        new = {"cnt": n, "kind": np.ones(u.shape[0], np.int64),
               "lit1": np.asarray(lit_of(o1, s1h), np.int64), "lit2": np.asarray(lit_of(o2, s2h), np.int64),
               "col1": np.full(u.shape[0], c1, np.int64), "col2": np.full(u.shape[0], c2, np.int64),
               "sym1": s1h.astype(np.int64), "sym2": s2h.astype(np.int64)}
        cand = {k: np.concatenate([cand[k], new[k]]) for k in cand}
        cand, tau = cut(cand)
        if log is not None and (pi + 1) % 200 == 0:
            log(f"  G+ column pair {pi + 1}/{len(pairs)}: tau {tau}, skipped {n_skipped}")
    n_single = int((cand["kind"] == 0).sum())
    info = {"n_features": int(cand["cnt"].shape[0]), "n_single": n_single, "n_pair": int(cand["cnt"].shape[0] - n_single),
            "min_count": int(cand["cnt"][-1]) if cand["cnt"].size else 0, "max_count": int(cand["cnt"][0]) if cand["cnt"].size else 0,
            "column_pairs": len(pairs), "column_pairs_skipped_by_bound": n_skipped, "offsets": list(gplus_offsets(mode))}
    return GPlus(mode, int(vocab), cand["lit1"], cand["lit2"], cand["cnt"], cand["col1"], cand["col2"],
                 cand["sym1"], cand["sym2"], n_pos, info)


def doc_chunks(doc_start: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Consecutive document ranges [d0, d1) of about `target` positions (a longer document is alone)."""
    ds = np.asarray(doc_start, np.int64)
    out, d0 = [], 0
    n_doc = ds.shape[0] - 1
    while d0 < n_doc:
        d1 = int(np.searchsorted(ds, ds[d0] + target, side="right")) - 1
        d1 = max(d0 + 1, min(d1, n_doc))
        out.append((d0, d1))
        d0 = d1
    return out


def gplus_features(xp, g: GPlus, tok, doc_start, kind_map, n_cols: int | None = None, chunk: int = 8192) -> F.SparseDocs:
    """Per document (every position of it, as the bank featurisation): count of positions where each of the
    first `n_cols` G+ features (default all) is true. Returns SparseDocs over n_cols columns."""
    n_use = g.n if n_cols is None else min(int(n_cols), g.n)
    ns = int(n_sym(g.vocab))
    mode, vocab = g.mode, g.vocab
    cols = columns(mode)
    pairs = column_pairs(mode)
    pair_index = {p: i for i, p in enumerate(pairs)}
    fid = np.arange(n_use, dtype=np.int64)
    single = g.lit2[:n_use] == LIT_EMPTY
    s_keys = g.col1[:n_use][single] * ns + g.sym1[:n_use][single]
    s_ord = np.argsort(s_keys)
    s_keys_h, s_fid_h = s_keys[s_ord], fid[single][s_ord]
    pr = ~single
    cpi = np.array([pair_index[(int(a), int(b))] for a, b in zip(g.col1[:n_use][pr], g.col2[:n_use][pr])], np.int64)
    p_keys = cpi * (ns * ns) + g.sym1[:n_use][pr] * ns + g.sym2[:n_use][pr]
    p_ord = np.argsort(p_keys)
    p_keys_h, p_fid_h = p_keys[p_ord], fid[pr][p_ord]
    active = np.unique(cpi)
    a_c1 = np.array([pairs[i][0] for i in active], np.int64)
    a_c2 = np.array([pairs[i][1] for i in active], np.int64)
    s_keys_d, s_fid_d = xp.asarray(s_keys_h), xp.asarray(s_fid_h)
    p_keys_d, p_fid_d = xp.asarray(p_keys_h), xp.asarray(p_fid_h)
    a_c1_d, a_c2_d, act_d = xp.asarray(a_c1), xp.asarray(a_c2), xp.asarray(active)
    col_ids_d = xp.arange(len(cols), dtype=xp.int64)
    tok_d = xp.asarray(np.asarray(tok))
    km_d = xp.asarray(np.asarray(kind_map)).astype(xp.int64)
    ds = np.asarray(doc_start, np.int64)
    lo, hi = doc_bounds(xp, ds)
    parts = []

    def lookup(sorted_keys, sorted_fid, q):
        if sorted_keys.shape[0] == 0 or q.size == 0:
            return xp.zeros(q.shape, dtype=bool), xp.zeros(q.shape, dtype=xp.int64)
        qf = q.reshape(-1)
        i = xp.minimum(xp.searchsorted(sorted_keys, qf), sorted_keys.shape[0] - 1)
        hit = sorted_keys[i] == qf
        return hit.reshape(q.shape), sorted_fid[i].reshape(q.shape)

    for d0, d1 in doc_chunks(ds, chunk):
        a, b = int(ds[d0]), int(ds[d1])
        pos = xp.arange(a, b, dtype=xp.int64)
        S = sym_matrix(xp, tok_d, lo, hi, km_d, pos, mode, vocab).astype(xp.int64)
        doc_rel = xp.asarray(np.repeat(np.arange(d1 - d0, dtype=np.int64), np.diff(ds[d0:d1 + 1])))
        ev_d, ev_f = [], []
        g1 = col_ids_d[None, :] * ns + S
        ok1 = S >= 0
        hit, f = lookup(s_keys_d, s_fid_d, g1)
        m = hit & ok1
        rr = xp.nonzero(m)
        ev_d.append(doc_rel[rr[0]])
        ev_f.append(f[m])
        if active.size:
            A1, A2 = S[:, a_c1_d], S[:, a_c2_d]
            ok2 = (A1 >= 0) & (A2 >= 0)
            g2 = act_d[None, :] * (ns * ns) + A1 * ns + A2
            hit2, f2 = lookup(p_keys_d, p_fid_d, g2)
            m2 = hit2 & ok2
            rr2 = xp.nonzero(m2)
            ev_d.append(doc_rel[rr2[0]])
            ev_f.append(f2[m2])
            del A1, A2, g2, hit2, f2, m2
        dd = xp.concatenate(ev_d).astype(xp.int64)
        ff = xp.concatenate(ev_f).astype(xp.int64)
        key = (dd << 32) | ff
        u, cnt = xp.unique(key, return_counts=True)
        uh, ch = _to_host(xp, u), _to_host(xp, cnt)
        drel = (uh >> 32).astype(np.int64)
        ip = np.zeros(d1 - d0 + 1, np.int64)
        ip[1:] = np.cumsum(np.bincount(drel, minlength=d1 - d0))
        if ch.size and int(ch.max()) > 0xFFFFFFFF:
            raise OverflowError("count exceeds u32")
        parts.append(F.SparseDocs(ip, (uh & 0xFFFFFFFF).astype(np.uint32), ch.astype(np.uint32), int(n_use)))
        del S, g1, hit, f, m, dd, ff, key, u, cnt
    if not parts:
        return F.SparseDocs(np.zeros(1, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint32), int(n_use))
    return F.merge_rows(parts)


def gplus_bruteforce(tok, doc_start, kind_map, mode: str, vocab: int, positions=None) -> dict[tuple[int, int], int]:
    """Reference (pure Python loops; tiny corpora only): {(lit1, lit2 or LIT_EMPTY): count}."""
    tok = [int(t) for t in np.asarray(tok)]
    ds = [int(x) for x in np.asarray(doc_start)]
    km = np.asarray(kind_map)
    edge = int(sym_edge(vocab))
    out: dict[tuple[int, int], int] = {}
    doc_of = np.repeat(np.arange(len(ds) - 1), np.diff(ds))
    pos = range(len(tok)) if positions is None else [int(p) for p in positions]
    for p in pos:
        d = int(doc_of[p])
        lits = []
        for o in gplus_offsets(mode):
            q = p + o
            if ds[d] <= q < ds[d + 1]:
                lits += [(o, int(lit_of(o, int(km[k][tok[q]]))) ) for k in range(N_KINDS)]
            else:
                lits.append((o, int(lit_of(o, edge))))
        for i, (oa, la) in enumerate(lits):
            out[(la, LIT_EMPTY)] = out.get((la, LIT_EMPTY), 0) + 1
            for ob, lb in lits[i + 1:]:
                if oa != ob:
                    key = (min(la, lb), max(la, lb))
                    out[key] = out.get(key, 0) + 1
    return out


# ------------------------------------------------------------------ count baselines
def bag_of_words(docs: F.TokenDocs, vocab: int) -> F.SparseDocs:
    return F.bag_of_tokens(docs, vocab, bigrams=False)


def cluster_bag_of_words(docs: F.TokenDocs, kind_map: np.ndarray) -> F.SparseDocs:
    return F.cluster_bags(docs, kind_map, levels=(1, 2, 3))[0]


def ppmi_svd(tok: np.ndarray, doc_start: np.ndarray, vocab: int, window: int = 8, dim: int = 300,
             power: float = 0.5) -> tuple[np.ndarray, dict]:
    """float32 [V, dim] token vectors: co-occurrence counts over offsets -window..-1, +1..+window inside a
    document, PPMI with context-distribution smoothing 0.75 (src/tmlm/ssl/concepts.py weight_values 'ppmi'),
    embedding U_dim S_dim^power from the top eigenpairs of X X^T (top_eigenpairs, concepts.spectral_embedding)."""
    from . import concepts as K

    offs = [o for o in range(-window, window + 1) if o != 0]
    X = K.cooc_counts(tok, doc_start, vocab, offs)
    r, c = np.nonzero(X)
    v = X[r, c].astype(np.float64)
    del X
    m = K.marginals_from_coo(r, c, v, vocab, vocab, K.add_one_unigram(tok, vocab))
    w = K.weight_values(np, r, c, v, m.to(np), "ppmi")
    keep = w > 0
    G = K.gram_sparse(r[keep], c[keep], w[keep], vocab, vocab)
    ev, U = top_eigenpairs(G, dim)
    E = K.spectral_embedding(np, ev, U, dim, power)
    info = {"window": window, "dim": dim, "power": power, "nnz_cooc": int(r.shape[0]), "nnz_ppmi": int(keep.sum()),
            "top_eigenvalues": [float(x) for x in np.asarray(ev[:5])]}
    return np.asarray(E, np.float32), info


def top_eigenpairs(G: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k largest eigenpairs of the symmetric PSD matrix G, descending (negative round-off clipped to 0).
    LAPACK dsyevr on the index subset only (scipy.linalg.eigh subset_by_index): the same pairs as the full
    decomposition of concepts.eig_desc, without computing the other eigenvectors."""
    from scipy.linalg import eigh

    n = G.shape[0]
    k = min(int(k), n)
    w, U = eigh(G, subset_by_index=[n - k, n - 1], driver="evr")
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), U[:, order]


def ppmi_svd_docs(docs: F.TokenDocs, E: np.ndarray) -> np.ndarray:
    return F.mean_embedding(docs, E)


def binarise_counts(sd: F.SparseDocs, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """uint32 dense [len(rows), len(cols)]: 1 where the count is positive (the TM probe's input)."""
    X = sd.rows(np.asarray(rows, np.int64)).to_csr("binary", cols=np.asarray(cols, np.int64), l2=False)
    return np.asarray(X.toarray() > 0, dtype=np.uint32)


def top_columns_by(values: np.ndarray, k: int) -> np.ndarray:
    """The k columns with the largest values (ties toward the lower id), ascending ids; zero columns excluded."""
    return F.top_df_columns(np.asarray(values, np.int64), int(k))


__all__ = ["c0y_batches", "gplus_select", "gplus_features", "gplus_bruteforce", "GPlus", "columns", "column_pairs",
           "bag_of_words", "cluster_bag_of_words", "ppmi_svd", "ppmi_svd_docs", "binarise_counts", "doc_chunks"]
