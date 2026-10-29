"""Codes and inputs of C-core (PLAN_part3.md section 4, "Codes"; proposal C section 4, step 2).

Tree E: exchange clustering of the train bigrams into 64 word classes (tally floor/exchange.py,
  sentence mode, <eos> and BOS fixed in classes of their own, Martin et al. start, at most 20 passes,
  stop below 0.1% moves), then a balanced binary merge tree over the 64 classes: at each of 6 levels
  the current clusters are paired greedily, always merging the unpaired pair with the least loss of
  class-bigram log-likelihood (recomputed after every merge; ties to the lowest index pair). Each class
  gets a 6-bit path (bit 1 = the top split); the child with the smaller lowest class id gets bit 0.
Tree B: recursive bisection of the C4 counting-table vectors (c4words_S1000_v1, PPMI-SVD, no neural
  network) of the PTB words, depth 8. Each node is split at the train-count-weighted median of its
  words' projections on its first principal direction (weighted by train counts; uniform weights in a
  node whose words all have count 0). Words with equal projections stay together; the cut is the
  boundary between distinct projections whose left weight is closest to half (ties to the first).
  Mapping (proposal C, fixed rule): a PTB word found as a string in words.txt gets its row; N gets the
  mean of the rows "0" to "9"; <unk> and every other word without a row get the train-count-weighted
  mean of the rows of the PTB words with a row seen 1 to 5 times in train. Every vector is
  L2-normalised. <eos> gets no B path (it is never a previous word in sentence mode).
Inputs: for each previous word t-1 .. t-4 inside the sentence, 15 bits: E path (6, top first), B path
  (8, top first), pad flag; a position before the sentence start has pad 1 and all code bits 0.
  60 bits per position. Target: the exchange class of the token (64 word classes + <eos> = 65).
"""
import numpy as np

from . import paths

paths.setup()
from tally.floor import exchange as ex  # noqa: E402

from .layout import (BITS_PER_WORD, DEPTH_B, DEPTH_E, K_E, N_BITS, N_CLASSES,  # noqa: E402,F401
                     N_PREV, PAD_B, PAD_E, decode, features, prev_words)

RARE_MAX = 5


def _g(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = x[m] * np.log(x[m])
    return out


def objective(W):
    """Class-bigram log-likelihood up to constants: sum g(W) - sum g(row sums) - sum g(col sums)."""
    return float(_g(W).sum() - _g(W.sum(1)).sum() - _g(W.sum(0)).sum())


def merge_losses(W, cand):
    """Loss (decrease of objective(W)) of merging items u < v of cand; inf on and below the diagonal."""
    cand = np.asarray(cand, dtype=np.int64)
    G = _g
    A = W[cand, :]
    B = W[:, cand].T
    GA, GB = G(A), G(B)
    rows = (G(A[:, None, :] + A[None, :, :]) - GA[:, None, :] - GA[None, :, :]).sum(2)
    cols = (G(B[:, None, :] + B[None, :, :]) - GB[:, None, :] - GB[None, :, :]).sum(2)
    Wcc = W[np.ix_(cand, cand)]
    du = np.diag(Wcc)
    Gd, Gw, GwT = G(du), G(Wcc), G(Wcc.T)
    r_u = G(du[:, None] + Wcc.T) - Gd[:, None] - GwT
    r_v = G(Wcc + du[None, :]) - Gw - Gd[None, :]
    c_u = G(du[:, None] + Wcc) - Gd[:, None] - Gw
    c_v = G(Wcc.T + du[None, :]) - GwT - Gd[None, :]
    diag = G(du[:, None] + Wcc + Wcc.T + du[None, :]) - Gd[:, None] - Gw - GwT - Gd[None, :]
    L, R = W.sum(1)[cand], W.sum(0)[cand]
    marg = (G(L[:, None] + L[None, :]) - G(L)[:, None] - G(L)[None, :]
            + G(R[:, None] + R[None, :]) - G(R)[:, None] - G(R)[None, :])
    dF = rows - r_u - r_v + cols - c_u - c_v + diag - marg
    loss = -dF
    loss[np.tril_indices(len(cand))] = np.inf
    return loss


def balanced_tree(M, K, depth):
    """Balanced merge tree over items 0..K-1 of the bigram matrix M ((K + f) x (K + f); items K.. are
    fixed context items that are never merged). Returns (paths (K,), levels): paths[i] is the depth-bit
    path of leaf i (top bit first); levels[l] lists the merges of level l as (leaves of child 0,
    leaves of child 1, loss)."""
    assert K == 2 ** depth and M.shape[0] == M.shape[1] >= K
    nfix = M.shape[0] - K
    W = np.asarray(M, dtype=np.float64).copy()
    clusters = [(i,) for i in range(K)]
    levels = []
    for _ in range(depth):
        n = len(clusters)
        Wk = W.copy()
        unpaired = list(range(n))
        merges = []
        while unpaired:
            loss = merge_losses(Wk, unpaired)
            flat = int(np.argmin(loss))
            i, j = divmod(flat, len(unpaired))
            u, v = unpaired[i], unpaired[j]
            Wk[u, :] += Wk[v, :]
            Wk[:, u] += Wk[:, v]
            Wk[v, :] = 0.0
            Wk[:, v] = 0.0
            merges.append((u, v, float(loss[i, j])))
            unpaired = [x for x in unpaired if x not in (u, v)]
        # new clusters, numbered by their lowest leaf id; child 0 = the child with the lower leaf id
        new = []
        for u, v, l in merges:
            a, b = (u, v) if min(clusters[u]) < min(clusters[v]) else (v, u)
            new.append((clusters[a], clusters[b], l, u))      # u is the row of Wk that holds the merge
        new.sort(key=lambda t: min(t[0] + t[1]))
        levels.append([(c0, c1, l) for c0, c1, l, _ in new])
        keep = [row for _, _, _, row in new] + list(range(n, n + nfix))
        W = Wk[np.ix_(keep, keep)]
        clusters = [tuple(sorted(c0 + c1)) for c0, c1, _, _ in new]
    paths_ = np.zeros(K, dtype=np.int64)
    for li, merges in enumerate(levels):           # level 0 = the finest merges = the last bit
        bit = 1 << li
        for c0, c1, _ in merges:
            for leaf in c1:
                paths_[leaf] |= bit
    return paths_, levels


def class_bigrams(train, V, eos, cls):
    """Class bigram matrix (K_E + 2) x (K_E + 2): word classes, <eos> class K_E, BOS item K_E + 1."""
    left, right, count = ex.bigrams(np.asarray(train, dtype=np.int64), eos, V + 1, V)
    item = np.append(np.asarray(cls, dtype=np.int64), K_E + 1)   # BOS (item V) -> K_E + 1
    M = np.zeros((K_E + 2, K_E + 2))
    np.add.at(M, (item[left], item[right]), count)
    return M


def tree_E(train, V, eos, log=None):
    C, info = ex.tree(np.asarray(train, dtype=np.int64), V, eos, levels=(K_E,), log=log)
    cls = C[:, 0].astype(np.int64)
    assert cls[eos] == K_E and (cls == K_E).sum() == 1
    assert set(np.unique(cls[np.arange(V) != eos])) == set(range(K_E))
    M = class_bigrams(train, V, eos, cls)
    leaf_paths, levels = balanced_tree(M, K_E, DEPTH_E)
    assert len(np.unique(leaf_paths)) == K_E
    path6 = np.zeros(V, dtype=np.int64)
    words = np.arange(V) != eos
    path6[words] = leaf_paths[cls[words]]
    meta = {"exchange": info, "merge_losses_per_level": [[m[2] for m in lv] for lv in levels],
            "objective_classes": objective(M)}
    return cls, path6, leaf_paths, meta


def c4_vectors(vocab, counts, eos):
    words = (paths.C4DIR / "words.txt").read_text(encoding="utf-8").splitlines()
    row = {w: i for i, w in enumerate(words)}
    vec = np.load(paths.C4DIR / "vectors.npy")
    assert vec.shape == (len(words), 500), vec.shape
    V = len(vocab)
    X = np.zeros((V, vec.shape[1]))
    kind = np.array(["fallback"] * V, dtype=object)
    for w in range(V):
        if w != eos and vocab[w] in row:
            X[w] = vec[row[vocab[w]]].astype(np.float64)
            kind[w] = "row"
    kind[eos] = "eos"
    n_idx = vocab.index("N")
    assert kind[n_idx] == "fallback"
    X[n_idx] = np.mean([vec[row[str(d)]].astype(np.float64) for d in range(10)], 0)
    kind[n_idx] = "digits"
    rare = np.flatnonzero((kind == "row") & (counts >= 1) & (counts <= RARE_MAX))
    fb = (counts[rare][:, None] * X[rare]).sum(0) / counts[rare].sum()
    fallback = np.flatnonzero(kind == "fallback")
    X[fallback] = fb
    nz = np.arange(V) != eos
    X[nz] /= np.linalg.norm(X[nz], axis=1, keepdims=True)
    tot = counts.sum()
    cov = {k: {"types": int((kind == k).sum()), "train_token_share": float(counts[kind == k].sum() / tot)}
           for k in ("row", "digits", "fallback", "eos")}
    cov["rare_words_in_fallback_mean"] = int(len(rare))
    cov["unk_is_fallback"] = bool(kind[vocab.index("<unk>")] == "fallback")
    return X, kind, cov


def _project(X, w, node):
    """Projections of the node's words on the node's first weighted principal direction, and weights."""
    ww = w[node].astype(np.float64)
    if ww.sum() <= 0:
        ww = np.ones(len(node))
    Z = X[node]
    mu = ww @ Z / ww.sum()
    Zc = Z - mu
    S = (Zc * ww[:, None]).T @ Zc
    _, evecs = np.linalg.eigh(S)
    v = evecs[:, -1]
    v = v * np.sign(v[int(np.argmax(np.abs(v)))])
    U, inv = np.unique(Z, axis=0, return_inverse=True)   # identical vectors get one, identical projection
    return ((U - mu) @ v)[inv.ravel()], ww


def _split(X, w, node):
    n = len(node)
    if n <= 1:
        return node, node[:0], "single"
    p, ww = _project(X, w, node)
    order = np.lexsort((node, p))
    ps = p[order]
    cw = np.cumsum(ww[order])
    bounds = np.flatnonzero(ps[1:] > ps[:-1]) + 1
    if len(bounds) == 0:
        return node, node[:0], "tied"
    k = int(bounds[int(np.argmin(np.abs(cw[bounds - 1] - cw[-1] / 2)))])
    return np.sort(node[order[:k]]), np.sort(node[order[k:]]), "split"


def tree_B(X, counts, eos, depth=DEPTH_B):
    V = X.shape[0]
    path8 = np.zeros(V, dtype=np.int64)
    nodes = [np.flatnonzero(np.arange(V) != eos)]
    notes = {"single": 0, "tied": 0, "split": 0}
    for d in range(depth):
        nxt = []
        for node in nodes:
            left, right, note = _split(X, counts, node)
            notes[note] += 1
            path8[right] |= 1 << (depth - 1 - d)
            nxt += [left, right]
        nodes = nxt
    leaves = [len(nd) for nd in nodes]
    mass = [float(counts[nd].sum()) for nd in nodes]
    return path8, {"nodes": notes, "leaf_sizes_min_median_max": [min(leaves), int(np.median(leaves)), max(leaves)],
                   "empty_leaves": int(sum(1 for x in leaves if x == 0)),
                   "leaf_mass_share_max": max(mass) / max(sum(mass), 1.0)}


def emission(train, cls, V, delta=0.5):
    """P(w | c(w)) = (n(w) + delta) / (n(c) + delta |c|), tally floor/classlm.py's emission."""
    n = np.bincount(np.asarray(train), minlength=V).astype(np.float64)
    K = int(cls.max()) + 1
    size = np.bincount(cls, minlength=K).astype(np.float64)
    ncls = np.bincount(cls, weights=n, minlength=K)
    return (n + delta) / (ncls[cls] + delta * size[cls])
