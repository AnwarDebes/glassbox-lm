"""Read-outs over K = 1,001 classes and the full distribution over V = 10,000 words.

Learned: logits = class sums of the machine / tau.
Counted: vote of clause j for class k = log((N[j,k] + alpha) / (n_j + alpha K)), N from train,
logits = sum of the votes of the clauses that output 1, / tau.
tau (and alpha) are fitted on the calibration split by minimum log-loss. Scoring is in float64.
Word probability: P(w) = P(class of w) for the 1,000 words, P(other) u(w) for the rest.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

CHUNK = 20000
LN2 = np.log(2.0)


def log_softmax(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def scores(fired_fn, M, n):
    """fired_fn(lo, hi) -> uint8 (hi-lo, C); returns float64 (n, K) = fired @ M."""
    out = np.empty((n, M.shape[1]), dtype=np.float64)
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        f = sparse.csr_matrix(fired_fn(lo, hi), dtype=np.float64)
        r = f @ M
        out[lo:hi] = r.toarray() if sparse.issparse(r) else r
    return out


def counted_matrix(N, alpha):
    K = N.shape[1]
    nj = N.sum(1, keepdims=True)
    return np.log((N + alpha) / (nj + alpha * K))


def class_nll(S, y, tau):
    """Mean negative log-likelihood (nats) of classes y under softmax(S / tau)."""
    lp = log_softmax(S / tau)
    return -lp[np.arange(len(y)), y].mean()


def fit_tau(S, y, lo=-6.0, hi=12.0):
    """Bounded Brent search for log(tau) on the calibration split (scipy, xatol 1e-4)."""
    r = minimize_scalar(lambda t: class_nll(S, y, np.exp(t)), bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-4))
    return float(np.exp(r.x)), float(r.fun)


def word_bits(S, tau, words, cls, u, K_TOP):
    """Bits per position for the true words, full distribution over V."""
    lp = log_softmax(S / tau)
    c = cls[words]
    out = -lp[np.arange(len(words)), c] / LN2
    oth = c == K_TOP
    out[oth] += -np.log2(u[words[oth]])
    return out


def check_distribution(S, tau, cls, u, K_TOP, rng, n=1000):
    """Design contract: on n random positions the full vector over V sums to 1 within 1e-6,
    its minimum is above 0, and the fast path (word_bits) equals the vector entry."""
    rows = rng.choice(S.shape[0], min(n, S.shape[0]), replace=False)
    lp = log_softmax(S[rows] / tau)
    P = np.exp(lp)
    top = cls < K_TOP
    full = np.where(top[None, :], P[:, np.minimum(cls, K_TOP)], P[:, [K_TOP]] * u[None, :])
    sums = full.sum(1)
    words = np.random.default_rng(0).integers(0, len(cls), len(rows))
    fast = word_bits(S[rows], tau, words, cls, u, K_TOP)
    slow = -np.log2(full[np.arange(len(rows)), words])
    ok = bool(np.all(np.abs(sums - 1) < 1e-6) and full.min() > 0 and np.allclose(fast, slow, rtol=0, atol=1e-9))
    return ok, float(np.abs(sums - 1).max()), float(full.min())
