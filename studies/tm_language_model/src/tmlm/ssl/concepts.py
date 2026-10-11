"""Word-concept read-outs of frozen self-supervised clause banks (EXPLORATORY; the conceptualization
challenge C1, docs/12 F20 and F23). No label of any kind is used to build a representation; the word
similarity sets and WordNet are used only to evaluate.

A representation of token t is row t of a non-negative matrix X (V rows), then a weighting, then a
reduction, then cosine similarity.

Profiles X
  count    X[t, j] = O_jt, the number of counting positions where clause j fires and the target is t
           (exactly the counts of the T1 read-out, experiments/cvtm_t1_readout.py).
  lit      X[t, (j, neg)] = 1 if clause j includes the TOKEN literal t (exact at any offset, or in a bag),
           positive (neg 0) or negated (neg 1); offset-agnostic. Cluster literals are not expanded to
           their member tokens (that would copy the exchange clustering, a baseline, into the profile).
  litoff   X[t, (j, seg, neg)] = 1: the same literal bound to its segment (offset or bag bucket).
  cooc     X[t, s] = number of times token s occurs at one of the window offsets of an occurrence of t
           (both inside one document): the classic count baseline.
Weightings (elementwise on the nonzeros x of X)
  raw      x
  ppmi     max(0, ln(x Z / (R_t N_j^0.75))), R_t the row sum, N_j the column sum, Z = sum_j N_j^0.75
           (context-distribution smoothing 0.75 of Levy, Goldberg and Dagan 2015, TACL 3:211-225)
  vote     ln(1 + x / (D_j P(t))), D_j = number of nonzero rows of column j, P the add-one unigram of
           the row tokens. For `count` this is exactly the Witten-Bell vote of the T1 read-out,
           q_jt = ln(1 + lam_j O_jt / (N_j (1 - lam_j) P(t))) with lam_j = N_j / (N_j + D_j), because
           lam_j / (1 - lam_j) = N_j / D_j (tested).
Reductions
  full     cosine of the weighted rows
  svdK_pP  rows of U_K S_K^P, where X X^T = U S^2 U^T (so U are the left singular vectors of the weighted
           X and S its singular values), then cosine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from tmlm.common.hashing import sha256_file  # noqa: F401 (re-exported)

CDS = 0.75
LIT_EMPTY = 0xFFFFFFFF
N_SEG_ALL = 16 + 2 * 13  # exact segments + bag segments (docs/11 1.4)


# ------------------------------------------------------------------ datasets
def parse_simlex(path: str | Path) -> list[tuple[str, str, float]]:
    """SimLex-999.txt: tab separated, header line; columns word1, word2, POS, SimLex999, ..."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines[0].startswith("word1\tword2\tPOS\tSimLex999"):
        raise ValueError("unexpected SimLex-999 header")
    out = []
    for ln in lines[1:]:
        if ln.strip():
            f = ln.split("\t")
            out.append((f[0], f[1], float(f[3])))
    return out


def parse_ws353(path: str | Path) -> list[tuple[str, str, float]]:
    """WordSim-353 combined.csv: header 'Word 1,Word 2,Human (mean)'."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines[0].startswith("Word 1,Word 2,Human"):
        raise ValueError("unexpected WordSim-353 header")
    out = []
    for ln in lines[1:]:
        if ln.strip():
            f = ln.split(",")
            out.append((f[0], f[1], float(f[2])))
    return out


def parse_men_lemma(path: str | Path) -> list[tuple[str, str, float]]:
    """MEN lemma form ('berry-n seed-n 37.000000'): the '-x' part-of-speech suffix is removed."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        a, b, s = ln.split()
        if a[-2] != "-" or b[-2] != "-":
            raise ValueError(f"unexpected MEN lemma entry {ln!r}")
        out.append((a[:-2], b[:-2], float(s)))
    return out


# ------------------------------------------------------------------ tokens
def single_token_id(encode: Callable[[str], Sequence[int]], word: str, try_lower: bool = True) -> int | None:
    """Token id of ' ' + word when it is ONE BPE token (the mid-sentence form); if not, and
    `try_lower`, the same for the lower-cased word; else None."""
    for w in ([word, word.lower()] if try_lower and word.lower() != word else [word]):
        ids = list(encode(" " + w))
        if len(ids) == 1:
            return int(ids[0])
    return None


def map_pairs(pairs: Sequence[tuple[str, str, float]], encode: Callable[[str], Sequence[int]]) -> dict:
    """Token ids of both words, the gold score and the coverage mask (both words single tokens)."""
    a = np.full(len(pairs), -1, np.int64)
    b = np.full(len(pairs), -1, np.int64)
    g = np.array([p[2] for p in pairs], np.float64)
    for i, (w1, w2, _) in enumerate(pairs):
        t1, t2 = single_token_id(encode, w1), single_token_id(encode, w2)
        if t1 is not None and t2 is not None:
            a[i], b[i] = t1, t2
    cov = a >= 0
    return {"a": a[cov], "b": b[cov], "gold": g[cov], "covered": cov, "n_total": len(pairs), "n_covered": int(cov.sum())}


def token_texts(token_bytes: Sequence[bytes]) -> list[str]:
    return [t.decode("utf-8", errors="backslashreplace") for t in token_bytes]


def word_like_mask(texts: Sequence[str], min_len: int = 3) -> np.ndarray:
    """Tokens that are a whole lower-case ASCII word with its leading space (' dog')."""
    out = np.zeros(len(texts), bool)
    for i, t in enumerate(texts):
        w = t[1:]
        out[i] = t.startswith(" ") and len(w) >= min_len and w.isascii() and w.isalpha() and w.islower()
    return out


# ------------------------------------------------------------------ weighting
@dataclass
class Marginals:
    """Row sums R (V), column sums N (F), nonzero rows per column D (F), row prior P (V, add-one unigram)."""
    row_sum: np.ndarray
    col_sum: np.ndarray
    col_distinct: np.ndarray
    p_row: np.ndarray

    @property
    def z_cds(self) -> float:
        n = np.asarray(self.col_sum, np.float64)
        return float(np.power(n[n > 0], CDS).sum())

    def to(self, xp) -> dict:
        return {"R": xp.asarray(np.asarray(self.row_sum, np.float64)), "N": xp.asarray(np.asarray(self.col_sum, np.float64)),
                "D": xp.asarray(np.asarray(self.col_distinct, np.float64)), "P": xp.asarray(np.asarray(self.p_row, np.float64)),
                "Z": self.z_cds}


def marginals_from_coo(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int, n_cols: int, p_row: np.ndarray) -> Marginals:
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    v = np.asarray(vals, np.float64)
    return Marginals(np.bincount(rows, weights=v, minlength=n_rows), np.bincount(cols, weights=v, minlength=n_cols),
                     np.bincount(cols, minlength=n_cols).astype(np.int64), np.asarray(p_row, np.float64))


def add_one_unigram(tokens: np.ndarray, vocab: int) -> np.ndarray:
    c = np.bincount(np.asarray(tokens, np.int64), minlength=vocab).astype(np.float64)
    return (c + 1.0) / (c.sum() + vocab)


def weight_values(xp, rows, cols, vals, m: dict, scheme: str):
    """Weighted nonzeros (float32) of a block; `rows` index m['R'] and m['P'], `cols` index m['N'] and
    m['D'] (global column ids). Every x must be > 0."""
    x = vals.astype(xp.float64)
    if scheme == "raw":
        w = x
    elif scheme == "ppmi":
        w = xp.log(x) + math.log(m["Z"]) - xp.log(m["R"][rows]) - CDS * xp.log(m["N"][cols])
        w = xp.maximum(w, 0.0)
    elif scheme == "vote":
        w = xp.log1p(x / (m["D"][cols] * m["P"][rows]))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return w.astype(xp.float32)


def gram_add_dense(xp, G, rows, cols_local, w, n_rows: int, n_cols_block: int) -> None:
    """G += B B^T for the dense block B[rows, cols_local] = w ((row, col) pairs unique)."""
    B = xp.zeros((n_rows, n_cols_block), dtype=xp.float32)
    B[rows, cols_local] = w
    G += (B @ B.T).astype(xp.float64)
    del B


def gram_sparse(rows, cols, w, n_rows: int, n_cols: int) -> np.ndarray:
    """X X^T on the CPU for a sparse X (float64 dense result)."""
    import scipy.sparse as sp
    X = sp.csr_matrix((np.asarray(w, np.float64), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))), shape=(n_rows, n_cols))
    return np.asarray((X @ X.T).toarray(), np.float64)


# ------------------------------------------------------------------ reductions and similarities
def parse_reduction(name: str) -> tuple[int, float] | None:
    if name == "full":
        return None
    k, p = name.split("_")
    return int(k[3:]), float(p[1:])


def cosine_from_gram(xp, G):
    d = xp.sqrt(xp.maximum(xp.diag(G), 0.0))
    inv = xp.where(d > 0, 1.0 / xp.where(d > 0, d, 1.0), 0.0)
    return (G * inv[:, None] * inv[None, :]).astype(xp.float32)


def eig_desc(xp, G, kmax: int):
    """Top-kmax eigenpairs of the symmetric PSD Gram matrix, descending (negative round-off clipped)."""
    w, U = xp.linalg.eigh(G)
    order = xp.argsort(w)[::-1][:kmax]
    return xp.maximum(w[order], 0.0), U[:, order]


def spectral_embedding(xp, w_desc, U_desc, k: int, p: float):
    s = xp.sqrt(w_desc[:k])
    return (U_desc[:, :k] * (s ** p)[None, :]).astype(xp.float32)


def cosine_rows(xp, E):
    n = xp.sqrt((E.astype(xp.float64) ** 2).sum(axis=1))
    inv = xp.where(n > 0, 1.0 / xp.where(n > 0, n, 1.0), 0.0)
    En = (E * inv[:, None]).astype(xp.float32)
    return En @ En.T


# ------------------------------------------------------------------ baselines
def cooc_counts(tok: np.ndarray, doc_start: np.ndarray, vocab: int, offsets: Iterable[int],
                target_mask: np.ndarray | None = None, chunk: int = 1 << 24) -> np.ndarray:
    """Dense (V, V) int64: X[t, s] = number of (position i, offset o) with tok[i] = t, tok[i + o] = s, both in
    the same document; only positions i with target_mask[i] (all when None) count as targets."""
    tok = np.asarray(tok)
    n = int(tok.shape[0])
    ds = np.asarray(doc_start, np.int64)
    doc_end = np.repeat(ds[1:], np.diff(ds))  # end (exclusive) of the document of each position
    doc_beg = np.repeat(ds[:-1], np.diff(ds))
    X = np.zeros(vocab * vocab, np.int64)
    for o in offsets:
        o = int(o)
        if o == 0:
            raise ValueError("offset 0")
        for a in range(0, n, chunk):
            i = np.arange(a, min(n, a + chunk), dtype=np.int64)
            j = i + o
            ok = (j >= doc_beg[i]) & (j < doc_end[i])
            if target_mask is not None:
                ok &= np.asarray(target_mask[a:a + i.shape[0]], bool)
            i, j = i[ok], j[ok]
            X += np.bincount(tok[i].astype(np.int64) * vocab + tok[j].astype(np.int64), minlength=vocab * vocab)
    return X.reshape(vocab, vocab)


def literal_profile(lit_id: np.ndarray, lit_state: np.ndarray, alive: np.ndarray, vocab: int, offset_bound: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Unique (token row, feature column) pairs of the included TOKEN literals of alive clauses;
    feature = 2 j + neg (offset-agnostic) or 2 (j * 42 + seg) + neg (segment-bound)."""
    C = int(lit_id.shape[0])
    inc = (lit_state >= 128) & (lit_id != LIT_EMPTY) & alive.astype(bool)[:, None]
    j = np.repeat(np.arange(C, dtype=np.int64), inc.sum(axis=1))
    L = lit_id[inc].astype(np.int64)
    seg, sym, neg = L >> 17, (L >> 1) & 0xFFFF, L & 1
    keep = sym < vocab
    j, seg, sym, neg = j[keep], seg[keep], sym[keep], neg[keep]
    if offset_bound:
        col, n_cols = 2 * (j * N_SEG_ALL + seg) + neg, 2 * C * N_SEG_ALL
    else:
        col, n_cols = 2 * j + neg, 2 * C
    key = np.unique(sym * n_cols + col)
    return key // n_cols, key % n_cols, int(n_cols)


# ------------------------------------------------------------------ evaluation
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 999, int(seed)]))
    return rng.integers(0, n, size=(int(n_boot), int(n)))


def spearman_boot(gold: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman correlation of every bootstrap resample (rows of idx)."""
    from scipy.stats import rankdata
    rg = rankdata(np.asarray(gold)[idx], axis=1)
    rp = rankdata(np.asarray(pred)[idx], axis=1)
    rg -= rg.mean(axis=1, keepdims=True)
    rp -= rp.mean(axis=1, keepdims=True)
    den = np.sqrt((rg ** 2).sum(axis=1) * (rp ** 2).sum(axis=1))
    return np.where(den > 0, (rg * rp).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)


def ci95(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def neighbour_agreement(neigh: np.ndarray, query_labels: Sequence, label_of: dict) -> float:
    """Mean over queries of the fraction of neighbours whose label equals the query's label."""
    tot = 0.0
    for q, row in zip(query_labels, neigh):
        tot += float(np.mean([label_of[int(t)] == q for t in row]))
    return tot / max(1, len(query_labels))


def synonym_hit_rate(neigh: np.ndarray, queries: np.ndarray, synsets_of: dict) -> float:
    """Fraction of queries with at least one neighbour sharing a WordNet synset with the query."""
    hits = 0
    for q, row in zip(queries, neigh):
        sq = synsets_of[int(q)]
        hits += int(any(sq & synsets_of[int(t)] for t in row))
    return hits / max(1, len(queries))


def label_chance(labels: Sequence) -> float:
    """Probability that two independently drawn members share a label (sum of squared shares)."""
    _, c = np.unique(np.asarray(labels, dtype=object).astype(str), return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


def kmeans_cosine(E: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    """Spherical k-means (unit rows, cosine), k-means++ initialisation, deterministic given the seed."""
    X = np.asarray(E, np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n > 0, n, 1.0)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1002, int(seed)]))
    cent = [X[int(rng.integers(X.shape[0]))]]
    d2 = np.full(X.shape[0], np.inf)
    for _ in range(1, k):
        d2 = np.minimum(d2, 2.0 - 2.0 * X @ cent[-1])
        p = np.maximum(d2, 0)
        cent.append(X[int(rng.choice(X.shape[0], p=p / p.sum()))] if p.sum() > 0 else X[int(rng.integers(X.shape[0]))])
    Cm = np.stack(cent)
    assign = np.zeros(X.shape[0], np.int64)
    for it in range(iters):
        new = np.argmax(X @ Cm.T, axis=1)
        if it > 0 and np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            m = assign == c
            if m.any():
                v = X[m].sum(axis=0)
                nv = np.linalg.norm(v)
                Cm[c] = v / nv if nv > 0 else Cm[c]
    return assign


def purity(assign: np.ndarray, labels: Sequence) -> float:
    lab = np.asarray(labels, dtype=object).astype(str)
    tot = 0
    for c in np.unique(assign):
        _, cnt = np.unique(lab[assign == c], return_counts=True)
        tot += int(cnt.max())
    return tot / max(1, lab.shape[0])


def permuted_purity(assign: np.ndarray, labels: Sequence, n_perm: int, seed: int) -> float:
    """Mean purity of the same cluster sizes with the labels shuffled (the chance level)."""
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 1003, int(seed)]))
    lab = np.asarray(labels, dtype=object)
    return float(np.mean([purity(assign, lab[rng.permutation(lab.shape[0])]) for _ in range(int(n_perm))]))


# ------------------------------------------------------------------ readable clauses
def clause_rule(lit_ids: np.ndarray, lit_states: np.ndarray, vocab: int, token_text: Callable[[int], str],
                cluster_members: Callable[[int], list[str]] | None = None) -> str:
    """'IF lit AND lit ...' over the included literals (ascending literal id); a cluster literal shows a
    few member tokens in braces (the cluster is the exchange clustering's, not learned by the TA)."""
    from ..engine.config import kind_of, lit_text, lit_unpack
    parts = []
    for lit, s in sorted(zip(lit_ids.tolist(), lit_states.tolist())):
        if lit == LIT_EMPTY or s < 128:
            continue
        txt = lit_text(int(lit), vocab, token_text)
        _, sym, _ = lit_unpack(int(lit))
        if cluster_members is not None and kind_of(vocab, sym) > 0:
            txt += " {" + ",".join(repr(m) for m in cluster_members(sym)) + "}"
        parts.append(txt)
    return "IF " + " AND ".join(parts) if parts else "IF (no included literal)"
