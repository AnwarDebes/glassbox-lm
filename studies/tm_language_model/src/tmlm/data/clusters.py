"""Teacher-free hierarchical word clustering for kind_map rows 1 to 3 (docs/03 "Literal space",
docs/11 section 1.2): 3 levels, 16-way, by the exchange algorithm on bigram context counts of
the TRAIN split (Kneser and Ney 1993; Martin, Liermann and Ney 1998).

Objective (maximised): F = sum_{g,h} f(N(g,h)) - sum_g f(N_left(g)) - sum_h f(N_right(h)) with
f(x) = x log x, N(g,h) the count of cluster bigrams (left cluster g, right cluster h) inside
documents, N_left and N_right the row and column sums. F is the class-bigram log-likelihood up
to a constant, so it uses the LEFT and the RIGHT context of every token at once.

Hierarchy: level 1 has 16 clusters; level 2 has 256, cluster id `16 * parent + child`, and a
token may only move between the 16 children of its level-1 cluster; level 3 has 4096 the same
way under level 2. So a level-k cluster refines its level-(k-1) parent by construction
(`id_k // 16 == id_{k-1}`).

Algorithm per level: tokens are visited in descending unigram frequency; for every token the
exact change of F for moving it to each sibling cluster is computed (removal delta plus
insertion delta, O(support of the token's cluster-context vectors) per candidate), and the
token moves to the best cluster when that strictly increases F. A pass over all tokens is one
iteration; `max_iter` caps the passes and a pass without a move stops early. F after every
iteration is logged. Deterministic given the seed: the seed only fixes the initial random
assignment (numpy Generator), everything else is a deterministic function of the counts.

At V=2000 the 4096-cluster level necessarily leaves clusters empty and most clusters with a
single token (the objective prefers singletons); the same holds in part at V=8000 (about two
tokens per cluster). That is expected and stated in every log.

Memory: two dense int64 [V, K] matrices per level (token to cluster context counts, both
directions), at most 2 x 32768 x 4096 x 8 bytes = 2 GiB for V=32768 at level 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..engine.config import CLUSTER_SIZES, sym_cluster_base

BRANCH = 16
LEVELS = 3
assert CLUSTER_SIZES == tuple(BRANCH ** (k + 1) for k in range(LEVELS))


def bigram_counts(tok: np.ndarray, doc_start: np.ndarray, vocab: int) -> sp.csr_matrix:
    """Sparse [V, V] int64 counts of (tok[i], tok[i+1]) pairs inside documents (sort-based)."""
    tok = np.asarray(tok)
    n = tok.shape[0]
    if n < 2:
        return sp.csr_matrix((vocab, vocab), dtype=np.int64)
    left = tok[:-1].astype(np.int64)
    right = tok[1:].astype(np.int64)
    valid = np.ones(n - 1, dtype=bool)
    inner = np.asarray(doc_start[1:-1], dtype=np.int64)
    inner = inner[(inner > 0) & (inner < n)]
    valid[inner - 1] = False  # the pair crossing a document boundary
    key = (left * vocab + right)[valid]
    uniq, cnt = np.unique(key, return_counts=True)
    rows = uniq // vocab
    cols = uniq % vocab
    return sp.csr_matrix((cnt.astype(np.int64), (rows, cols)), shape=(vocab, vocab), dtype=np.int64)


def _xlogx(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)
    return out


def objective(M: np.ndarray, R: np.ndarray, C: np.ndarray) -> float:
    return float(_xlogx(M).sum() - _xlogx(R).sum() - _xlogx(C).sum())


@dataclass
class ExchangeLog:
    level: int
    n_clusters: int
    n_tokens: int
    objective: list[float] = field(default_factory=list)  # after every iteration
    moves: list[int] = field(default_factory=list)
    initial_objective: float = 0.0
    converged: bool = False

    def as_dict(self) -> dict:
        return {"level": self.level, "n_clusters": self.n_clusters, "n_tokens": self.n_tokens,
                "initial_objective": self.initial_objective, "objective": self.objective,
                "moves": self.moves, "converged": self.converged}


def _cluster_matrices(B: sp.csr_matrix, assign: np.ndarray, K: int):
    V = B.shape[0]
    onehot = sp.csr_matrix((np.ones(V, dtype=np.int64), (np.arange(V), assign)), shape=(V, K))
    W2C = np.asarray((B @ onehot).todense(), dtype=np.int64)        # [V, K]: w -> cluster h
    C2W = np.asarray((B.T.tocsr() @ onehot).todense(), dtype=np.int64)  # [V, K]: cluster g -> w
    M = np.asarray((onehot.T @ (B @ onehot)).todense(), dtype=np.int64)  # [K, K]
    return W2C, C2W, M


def exchange(B: sp.csr_matrix, parent: np.ndarray, assign: np.ndarray, K: int, level: int,
             max_iter: int = 20, order: np.ndarray | None = None, eps: float = 1e-6) -> tuple[np.ndarray, ExchangeLog]:
    """One level of exchange clustering. `parent[w] * BRANCH <= assign[w] < (parent[w] + 1) * BRANCH`.
    Returns the final assignment and the per-iteration log."""
    V = B.shape[0]
    assign = np.array(assign, dtype=np.int64)
    parent = np.asarray(parent, dtype=np.int64)
    if np.any(assign // BRANCH != parent):
        raise ValueError("initial assignment is not inside the parent clusters")
    Bc = B.tocsr()
    Bt = B.T.tocsr()  # row w of Bt = column w of B
    nl = np.asarray(Bc.sum(axis=1)).ravel().astype(np.int64)  # w as left token
    nr = np.asarray(Bt.sum(axis=1)).ravel().astype(np.int64)  # w as right token
    self_cnt = np.asarray(Bc.diagonal(), dtype=np.int64)
    W2C, C2W, M = _cluster_matrices(Bc, assign, K)
    R = M.sum(axis=1)
    C = M.sum(axis=0)
    if order is None:
        order = np.lexsort((np.arange(V), -(nl + nr)))
    log = ExchangeLog(level=level, n_clusters=K, n_tokens=V, initial_objective=objective(M, R, C))
    sib = np.arange(BRANCH, dtype=np.int64)
    for _ in range(max_iter):
        moves = 0
        for w in order:
            w = int(w)
            if nl[w] + nr[w] == 0:
                continue
            a = int(assign[w])
            r = W2C[w]
            c = C2W[w]
            sw = int(self_cnt[w])
            # ---- remove w from a
            old_row = M[a].copy()
            old_col = M[:, a].copy()
            M[a] -= r
            M[:, a] -= c
            M[a, a] += sw
            R_old, C_old = int(R[a]), int(C[a])
            R[a] -= nl[w]
            C[a] -= nr[w]
            d_rem = (_xlogx(M[a]).sum() - _xlogx(old_row).sum()
                     + _xlogx(M[:, a]).sum() - _xlogx(old_col).sum()
                     - (_xlogx(M[a, a]) - _xlogx(old_row[a]))
                     - (_xlogx(R[a]) - _xlogx(R_old)) - (_xlogx(C[a]) - _xlogx(C_old)))
            # ---- insertion delta for every sibling b != a
            cands = parent[w] * BRANCH + sib
            cands = cands[cands != a]
            supp_r = np.flatnonzero(r)
            supp_c = np.flatnonzero(c)
            rows = M[cands[:, None], supp_r[None, :]]
            d_add = (_xlogx(rows + r[supp_r][None, :]) - _xlogx(rows)).sum(axis=1)
            cols = M[supp_c[:, None], cands[None, :]]
            d_add += (_xlogx(cols + c[supp_c][:, None]) - _xlogx(cols)).sum(axis=0)
            m_ba = M[cands, a]
            m_ab = M[a, cands]
            diag = M[cands, cands]
            # the generic row/column terms counted h = a and h = b; replace them
            d_add -= _xlogx(m_ba + r[a]) - _xlogx(m_ba)
            d_add -= _xlogx(m_ab + c[a]) - _xlogx(m_ab)
            d_add -= _xlogx(diag + r[cands]) - _xlogx(diag)
            d_add -= _xlogx(diag + c[cands]) - _xlogx(diag)
            d_add += _xlogx(m_ba + r[a] - sw) - _xlogx(m_ba)
            d_add += _xlogx(m_ab + c[a] - sw) - _xlogx(m_ab)
            d_add += _xlogx(diag + r[cands] + c[cands] + sw) - _xlogx(diag)
            d_add -= _xlogx(R[cands] + nl[w]) - _xlogx(R[cands])
            d_add -= _xlogx(C[cands] + nr[w]) - _xlogx(C[cands])
            best = int(np.argmax(d_add))
            b = int(cands[best]) if d_rem + d_add[best] > eps else a
            # ---- insert w into b
            M[b] += r
            M[:, b] += c
            M[b, a] -= sw
            M[a, b] -= sw
            M[b, b] += sw
            R[b] += nl[w]
            C[b] += nr[w]
            if b != a:
                assign[w] = b
                moves += 1
                lo, hi = Bt.indptr[w], Bt.indptr[w + 1]
                u = Bt.indices[lo:hi]
                vals = Bt.data[lo:hi]
                W2C[u, a] -= vals
                W2C[u, b] += vals
                lo, hi = Bc.indptr[w], Bc.indptr[w + 1]
                v = Bc.indices[lo:hi]
                vals = Bc.data[lo:hi]
                C2W[v, a] -= vals
                C2W[v, b] += vals
        log.objective.append(objective(M, R, C))
        log.moves.append(moves)
        if moves == 0:
            log.converged = True
            break
    return assign.astype(np.int64), log


def hierarchical_exchange(B: sp.csr_matrix, seed: int, max_iter: int = 20) -> tuple[list[np.ndarray], list[ExchangeLog]]:
    """Levels 1..3: assignments (int64 [V] with ids in [0, 16^k)) and one log per level."""
    V = B.shape[0]
    rng = np.random.default_rng(seed)
    parent = np.zeros(V, dtype=np.int64)
    levels: list[np.ndarray] = []
    logs: list[ExchangeLog] = []
    for k in range(1, LEVELS + 1):
        K = BRANCH ** k
        init = parent * BRANCH + rng.integers(0, BRANCH, size=V)
        assign, log = exchange(B, parent, init, K, level=k, max_iter=max_iter)
        levels.append(assign)
        logs.append(log)
        parent = assign
    return levels, logs


def kind_map_from_levels(vocab: int, levels: list[np.ndarray]) -> np.ndarray:
    """kind_map u16 [4, V] with the cluster symbols of docs/11 section 1.2."""
    km = np.zeros((4, vocab), dtype=np.uint16)
    km[0] = np.arange(vocab, dtype=np.uint16)
    for kind, assign in zip((1, 2, 3), levels):
        if assign.shape != (vocab,) or assign.min() < 0 or assign.max() >= CLUSTER_SIZES[kind - 1]:
            raise ValueError(f"level {kind} assignment out of range")
        km[kind] = sym_cluster_base(vocab, kind) + assign
    return km


def check_refinement(levels: list[np.ndarray]) -> bool:
    return all(np.array_equal(levels[k] // BRANCH, levels[k - 1]) for k in range(1, len(levels)))


def cluster_stats(assign: np.ndarray, K: int) -> dict:
    sizes = np.bincount(assign, minlength=K)
    return {"n_clusters": int(K), "empty": int((sizes == 0).sum()), "singletons": int((sizes == 1).sum()),
            "largest": int(sizes.max()), "mean_size_nonempty": float(sizes[sizes > 0].mean())}
