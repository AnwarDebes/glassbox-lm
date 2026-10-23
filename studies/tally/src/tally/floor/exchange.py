"""Exchange clustering (Martin, Liermann and Ney 1998) on bigram counts, and the class tree.

Objective: the class bigram log-likelihood of the training text, up to constants,
    F = sum_{c,d} g(M[c,d]) - sum_c g(L[c]) - sum_d g(R[d]),   g(x) = x log x,
with M[c,d] the number of bigrams (u, v) with u in class c and v in class d, L[c] the number of
bigrams whose left item is in c and R[d] the number whose right item is in d. Each item is moved
in turn (most frequent first) to the class that gives the largest F; a pass ends when every
movable item has been visited. Stops when a pass moves fewer than stop_share of the movable items
or after max_passes. A class is never left empty (its last item does not move).

Initialisation (Martin et al.): the K - 1 most frequent movable items each get their own class,
all other movable items go to the last class. Fixed items (sentence start BOS, <eos>) stay in
their own classes; they are only context.

The tree (bottom-up): the finest level is exchange clustering of the words; each coarser level is
exchange clustering of the classes of the level below, with the class bigram counts as the counts
of the items. Every level is therefore a union of classes of the level below, so the levels are
the prefixes of one path code.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _g(x):
    return x * np.log(x) if x > 0 else 0.0


@njit(cache=True)
def _objective(M, L, R):
    f = 0.0
    K = M.shape[0]
    for c in range(K):
        for d in range(K):
            f += _g(M[c, d])
        f -= _g(L[c]) + _g(R[c])
    return f


@njit(cache=True)
def _class_counts(cls, rptr, ridx, rcnt, K):
    n = len(cls)
    M = np.zeros((K, K))
    for u in range(n):
        for j in range(rptr[u], rptr[u + 1]):
            M[cls[u], cls[ridx[j]]] += rcnt[j]
    return M


@njit(cache=True)
def _passes(cls, order, allowed_ptr, allowed, parent, rptr, ridx, rcnt, lptr, lidx, lcnt,
            K, max_passes, stop_share):
    """Exchange passes in place on cls. allowed: candidate classes per parent group
    (allowed[allowed_ptr[g]:allowed_ptr[g + 1]]); parent[u] is the group of item u."""
    n = len(cls)
    M = _class_counts(cls, rptr, ridx, rcnt, K)
    L = M.sum(axis=1).copy()
    R = M.sum(axis=0).copy()
    size = np.zeros(K, dtype=np.int64)
    for u in range(n):
        size[cls[u]] += 1
    nR = np.zeros(K)
    nL = np.zeros(K)
    touchR = np.zeros(K, dtype=np.int64)
    touchL = np.zeros(K, dtype=np.int64)
    trace = np.zeros(max_passes + 1)
    moves = np.zeros(max_passes, dtype=np.int64)
    trace[0] = _objective(M, L, R)
    done = 0
    for p in range(max_passes):
        mv = 0
        for t in range(len(order)):
            w = order[t]
            a = cls[w]
            if size[a] <= 1:
                continue
            # neighbour class counts of w (self bigrams apart)
            nr = 0
            nl = 0
            s = 0.0
            ltot = 0.0
            rtot = 0.0
            for j in range(rptr[w], rptr[w + 1]):
                v = ridx[j]
                ltot += rcnt[j]
                if v == w:
                    s += rcnt[j]
                    continue
                d = cls[v]
                if nR[d] == 0:
                    touchR[nr] = d
                    nr += 1
                nR[d] += rcnt[j]
            for j in range(lptr[w], lptr[w + 1]):
                u = lidx[j]
                rtot += lcnt[j]
                if u == w:
                    continue
                c = cls[u]
                if nL[c] == 0:
                    touchL[nl] = c
                    nl += 1
                nL[c] += lcnt[j]
            # remove w from a
            for q in range(nr):
                d = touchR[q]
                M[a, d] -= nR[d]
            for q in range(nl):
                c = touchL[q]
                M[c, a] -= nL[c]
            M[a, a] -= s
            L[a] -= ltot
            R[a] -= rtot
            # best class among the allowed ones
            g = parent[w]
            best = a
            best_gain = -1e300
            for q0 in range(allowed_ptr[g], allowed_ptr[g + 1]):
                b = allowed[q0]
                gain = 0.0
                for q in range(nr):
                    d = touchR[q]
                    if d != b:
                        gain += _g(M[b, d] + nR[d]) - _g(M[b, d])
                for q in range(nl):
                    c = touchL[q]
                    if c != b:
                        gain += _g(M[c, b] + nL[c]) - _g(M[c, b])
                gain += _g(M[b, b] + nR[b] + nL[b] + s) - _g(M[b, b])
                gain -= _g(L[b] + ltot) - _g(L[b])
                gain -= _g(R[b] + rtot) - _g(R[b])
                if gain > best_gain + 1e-9 or (b == a and gain >= best_gain - 1e-9):
                    best_gain = gain
                    best = b
            # add w to best
            for q in range(nr):
                d = touchR[q]
                M[best, d] += nR[d]
            for q in range(nl):
                c = touchL[q]
                M[c, best] += nL[c]
            M[best, best] += s
            L[best] += ltot
            R[best] += rtot
            if best != a:
                cls[w] = best
                size[a] -= 1
                size[best] += 1
                mv += 1
            for q in range(nr):
                nR[touchR[q]] = 0.0
            for q in range(nl):
                nL[touchL[q]] = 0.0
        moves[p] = mv
        trace[p + 1] = _objective(M, L, R)
        done = p + 1
        if mv < stop_share * len(order):
            break
    return trace[:done + 1], moves[:done]


def csr(left, right, count, n):
    """Right-neighbour and left-neighbour lists of items 0..n-1 from bigram triples."""
    o = np.lexsort((right, left))
    rptr = np.searchsorted(left[o], np.arange(n + 1)).astype(np.int64)
    o2 = np.lexsort((left, right))
    lptr = np.searchsorted(right[o2], np.arange(n + 1)).astype(np.int64)
    return (rptr, right[o].astype(np.int64), count[o].astype(np.float64),
            lptr, left[o2].astype(np.int64), count[o2].astype(np.float64))


def bigrams(stream, eos, n_items, bos):
    """Sentence-mode bigram triples (left, right, count) of a word stream, with BOS before
    every sentence (BOS = item id bos)."""
    x = np.asarray(stream, dtype=np.int64)
    prev = np.empty_like(x)
    prev[0] = bos
    prev[1:] = x[:-1]
    prev[1:][x[:-1] == eos] = bos
    k, c = np.unique(prev * n_items + x, return_counts=True)
    return k // n_items, k % n_items, c


def exchange(left, right, count, n_items, K, movable, freq, fixed_class=None, init=None,
             groups=None, max_passes=20, stop_share=0.001):
    """Cluster items 0..n_items-1 into K classes.

    movable: boolean mask of items that are clustered (into classes 0..K-1); every other item
    keeps its class fixed_class[item] (>= K). init: optional starting classes of the movable
    items; default Martin et al. groups: optional group per item; an item may only move to a
    class in the same group (classes are given to groups by the init). Returns (classes of all
    items, objective trace, moves per pass)."""
    movable = np.asarray(movable, dtype=bool)
    mov = np.flatnonzero(movable)
    order = mov[np.argsort(-np.asarray(freq)[mov], kind="stable")]
    cls = np.zeros(n_items, dtype=np.int64)
    if fixed_class is not None:
        cls[~movable] = np.asarray(fixed_class)[~movable]
    if init is None:
        assert len(order) >= K
        cls[order[:K - 1]] = np.arange(K - 1)
        cls[order[K - 1:]] = K - 1
    else:
        cls[mov] = np.asarray(init)[mov]
    Ktot = int(cls.max()) + 1
    if groups is None:
        parent = np.zeros(n_items, dtype=np.int64)
        allowed_ptr = np.array([0, K], dtype=np.int64)
        allowed = np.arange(K, dtype=np.int64)
    else:
        parent = np.asarray(groups, dtype=np.int64)
        gl = [np.unique(cls[mov][parent[mov] == g]) for g in range(int(parent.max()) + 1)]
        allowed_ptr = np.concatenate([[0], np.cumsum([len(a) for a in gl])]).astype(np.int64)
        allowed = np.concatenate(gl).astype(np.int64)
    rptr, ridx, rcnt, lptr, lidx, lcnt = csr(np.asarray(left), np.asarray(right),
                                            np.asarray(count), n_items)
    trace, moves = _passes(cls, order, allowed_ptr, allowed, parent, rptr, ridx, rcnt, lptr,
                           lidx, lcnt, Ktot, max_passes, stop_share)
    return cls, trace, moves


def objective(cls, left, right, count):
    """F of a full assignment (for tests)."""
    K = int(cls.max()) + 1
    M = np.zeros((K, K))
    np.add.at(M, (cls[left], cls[right]), count)
    return _objective(M, M.sum(1), M.sum(0))


def tree(stream, V, eos, levels=(1024, 256, 64, 16), max_passes=20, stop_share=0.001, log=None):
    """Bottom-up class tree over the words 0..V-1 (eos kept in its own class at every level).

    Returns (codes, info): codes is (V, len(levels)) with codes[:, i] the class of every word at
    levels[i] (0..levels[i]-1 for words, levels[i] for <eos>)."""
    x = np.asarray(stream, dtype=np.int64)
    bos = V
    n = V + 1
    left, right, count = bigrams(x, eos, n, bos)
    freq = np.bincount(x, minlength=n).astype(np.float64)
    codes = np.zeros((V, len(levels)), dtype=np.int64)
    info = []
    # finest level: words
    K = levels[0]
    movable = np.ones(n, dtype=bool)
    movable[[eos, bos]] = False
    fixed = np.zeros(n, dtype=np.int64)
    fixed[eos], fixed[bos] = K, K + 1
    cls, trace, moves = exchange(left, right, count, n, K, movable, freq, fixed, None, None,
                                 max_passes, stop_share)
    codes[:, 0] = cls[:V]
    info.append({"level": K, "items": int(movable.sum()), "passes": len(moves),
                 "moves": moves.tolist(), "objective": trace.tolist()})
    if log:
        log(f"exchange level {K}: passes {len(moves)} moves {moves.tolist()}")
    # coarser levels: the classes of the level below are the items
    prev = cls
    prevK = K
    for li, K in enumerate(levels[1:], start=1):
        # items: classes 0..prevK-1, eos class prevK, bos class prevK + 1
        n2 = prevK + 2
        k2 = prev[left] * n2 + prev[right]
        uk, inv = np.unique(k2, return_inverse=True)
        c2 = np.bincount(inv, weights=count)
        l2, r2 = uk // n2, uk % n2
        f2 = np.bincount(prev, weights=freq, minlength=n2)
        mov2 = np.ones(n2, dtype=bool)
        mov2[[prevK, prevK + 1]] = False
        fix2 = np.zeros(n2, dtype=np.int64)
        fix2[prevK], fix2[prevK + 1] = K, K + 1
        cls2, trace, moves = exchange(l2, r2, c2, n2, K, mov2, f2, fix2, None, None,
                                      max_passes, stop_share)
        prev = cls2[prev]
        prevK = K
        codes[:, li] = prev[:V]
        info.append({"level": K, "items": int(mov2.sum()), "passes": len(moves),
                     "moves": moves.tolist(), "objective": trace.tolist()})
        if log:
            log(f"exchange level {K}: passes {len(moves)} moves {moves.tolist()}")
    return codes, info
