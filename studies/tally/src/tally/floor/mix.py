"""Linear interpolation of members, weights fitted by EM on the calibration split (section 4).

P(w) = sum_m lambda_m[b] P_m(w), lambda >= 0 summing to 1 per bucket b. EM (deleted
interpolation): each step sets lambda to the average posterior responsibility of each member for
the true word. The log-likelihood is concave in lambda, so EM goes to the global optimum from
any start inside the simplex.

Bucketed version: b = longest match length x count bucket of the matched context (from the
Kneser-Ney tables). Sparse buckets are pulled to the global weights by tau pseudo-tokens
(MAP-EM with a Dirichlet prior centred on the global weights).
"""
import numpy as np

COUNT_EDGES = (1, 2, 4, 8, 16, 64, 256)


def em(P, iters=1000, tol=1e-10, init=None):
    """P: (n, M) probabilities of the true word under each member. Returns (lambda, loglik)."""
    n, M = P.shape
    lam = np.full(M, 1.0 / M) if init is None else np.asarray(init, dtype=np.float64)
    trace = []
    for _ in range(iters):
        mix = P @ lam
        trace.append(np.log(mix).sum())
        lam_new = (P * lam / mix[:, None]).mean(0)
        if len(trace) > 1 and trace[-1] - trace[-2] < tol * n:
            lam = lam_new
            break
        lam = lam_new
    return lam, np.array(trace)


def em_bucketed(P, b, n_buckets, prior, tau=20.0, iters=1000, tol=1e-10):
    """Per-bucket weights (n_buckets, M), pulled to prior by tau pseudo-tokens."""
    n, M = P.shape
    lam = np.tile(np.asarray(prior, dtype=np.float64), (n_buckets, 1))
    nb = np.bincount(b, minlength=n_buckets).astype(np.float64)
    trace = []
    for _ in range(iters):
        mix = (P * lam[b]).sum(1)
        trace.append(np.log(mix).sum() + tau * (np.log(lam) * prior).sum())
        r = P * lam[b] / mix[:, None]
        R = np.zeros((n_buckets, M))
        np.add.at(R, b, r)
        lam = (R + tau * prior) / (nb + tau)[:, None]
        if len(trace) > 1 and trace[-1] - trace[-2] < tol * n:
            break
    return lam, np.array(trace)


def buckets(L, cnt, edges=COUNT_EDGES):
    """Bucket id = longest match length L x count bucket of the matched context's raw count."""
    cb = np.searchsorted(np.asarray(edges), cnt, side="right")   # 0 for cnt < 1
    return np.asarray(L) * (len(edges) + 1) + cb


def n_buckets(order, edges=COUNT_EDGES):
    return order * (len(edges) + 1)


def logprob(P, lam, b=None):
    """Natural log of the mixed true-word probability."""
    if b is None:
        return np.log(P @ lam)
    return np.log((P * lam[b]).sum(1))
