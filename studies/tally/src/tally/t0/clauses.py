"""Clause banks as literal lists, evaluated on the one-hot T0 inputs with compiled loops.

A bank is given by its include matrix (clauses x 2F, from tmu's get_literals): column f < F
includes feature f, column F + f includes its negation. Inputs are the active feature indices
of each position (CONTEXT of them). A clause outputs 1 when every included feature is active
and no negated feature is active. A clause with no literal outputs 0, as in tmu's predict path
(checked against tmu in tests/test_clauses.py).
"""
import numba as nb
import numpy as np


class Bank:
    def __init__(self, include, F):
        include = np.asarray(include)
        self.C = include.shape[0]
        self.F = F
        pos = include[:, :F] > 0
        neg = include[:, F:] > 0
        self.npos = pos.sum(1).astype(np.int32)
        self.nlit = (self.npos + neg.sum(1)).astype(np.int32)
        self.fpos_ptr, self.fpos_cl = _by_feature(pos)
        self.fneg_ptr, self.fneg_cl = _by_feature(neg)
        # clauses with no positive literal fire unless a negated feature is active
        self.free = np.flatnonzero((self.npos == 0) & (self.nlit > 0)).astype(np.int32)

    @staticmethod
    def random_like(trained, rng):
        """Untrained control: every clause gets as many positive and negated literals as the
        trained clause in the same place, drawn uniformly without replacement."""
        F = trained.F
        npos = trained.npos
        nneg = trained.nlit - trained.npos
        include = np.zeros((trained.C, 2 * F), dtype=np.uint8)
        for j in range(trained.C):
            feats = rng.choice(F, npos[j] + nneg[j], replace=False)
            include[j, feats[: npos[j]]] = 1
            include[j, F + feats[npos[j]:]] = 1
        return Bank(include, F)

    def fired(self, active):
        """Dense uint8 (n, C) outputs for active feature indices (n, CONTEXT)."""
        return _fired(active, self.C, self.npos, self.fpos_ptr, self.fpos_cl, self.fneg_ptr, self.fneg_cl, self.free)

    def counts(self, active, y, K):
        """N[j, k]: positions where clause j outputs 1 and the target class is k; also n_k."""
        return _counts(active, y.astype(np.int32), K, self.C, self.npos, self.fpos_ptr, self.fpos_cl,
                       self.fneg_ptr, self.fneg_cl, self.free)


def _by_feature(mask):
    C, F = mask.shape
    cl, f = np.nonzero(mask.T)  # sorted by feature
    ptr = np.zeros(F + 1, dtype=np.int64)
    np.add.at(ptr, cl + 1, 1)
    return np.cumsum(ptr), f.astype(np.int32)


@nb.njit(cache=True)
def _row(active_row, out, hits, npos, fpos_ptr, fpos_cl, fneg_ptr, fneg_cl, free, killed):
    # out[j] = 1 for fired clauses; hits and killed are scratch arrays left zeroed afterwards
    for a in active_row:
        for q in range(fneg_ptr[a], fneg_ptr[a + 1]):
            killed[fneg_cl[q]] = 1
    for a in active_row:
        for q in range(fpos_ptr[a], fpos_ptr[a + 1]):
            j = fpos_cl[q]
            hits[j] += 1
            if hits[j] == npos[j] and killed[j] == 0:
                out[j] = 1
    for j in free:
        if killed[j] == 0:
            out[j] = 1
    for a in active_row:
        for q in range(fpos_ptr[a], fpos_ptr[a + 1]):
            hits[fpos_cl[q]] = 0
        for q in range(fneg_ptr[a], fneg_ptr[a + 1]):
            killed[fneg_cl[q]] = 0


@nb.njit(cache=True)
def _fired(active, C, npos, fpos_ptr, fpos_cl, fneg_ptr, fneg_cl, free):
    n = active.shape[0]
    out = np.zeros((n, C), dtype=np.uint8)
    hits = np.zeros(C, dtype=np.int32)
    killed = np.zeros(C, dtype=np.uint8)
    for i in range(n):
        _row(active[i], out[i], hits, npos, fpos_ptr, fpos_cl, fneg_ptr, fneg_cl, free, killed)
    return out


@nb.njit(cache=True)
def _counts(active, y, K, C, npos, fpos_ptr, fpos_cl, fneg_ptr, fneg_cl, free):
    N = np.zeros((C, K), dtype=np.int64)
    nk = np.zeros(K, dtype=np.int64)
    row = np.zeros(C, dtype=np.uint8)
    hits = np.zeros(C, dtype=np.int32)
    killed = np.zeros(C, dtype=np.uint8)
    for i in range(active.shape[0]):
        _row(active[i], row, hits, npos, fpos_ptr, fpos_cl, fneg_ptr, fneg_cl, free, killed)
        k = y[i]
        nk[k] += 1
        for j in range(C):
            if row[j]:
                N[j, k] += 1
                row[j] = 0
    return N, nk
