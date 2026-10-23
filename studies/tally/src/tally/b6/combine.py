"""Team scores, composite rules and the team error of rule T.

Composite of Granmo 2023 (Eq. for alpha_t): each member's class sums are divided by
alpha_t = max - min of that member's class sums over a whole input set, then added. The paper
and the CAIR scripts take the set being scored (the test file); here alpha_t is taken on the
calib split and frozen, so no dev-test number depends on the dev-test inputs. tmu's own
TMComposite divides per sample (np.ptp over the classes of each row); that variant is reported
as a secondary row, and the neural gate uses it because the library does.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def alpha(S):
    a = float(np.max(S) - np.min(S))
    return a if a > 0 else 1.0


def paper(scores, alphas):
    return sum(s / a for s, a in zip(scores, alphas))


def per_sample(S):
    return S / np.maximum(np.ptp(S, axis=1), 1e-8)[:, None]


def library(scores, weights=None):
    """tmu TMComposite: per-sample normalised scores, times an optional gate weight per member."""
    out = 0.0
    for i, s in enumerate(scores):
        w = 1.0 if weights is None else weights[:, i:i + 1]
        out = out + w * per_sample(s.astype(np.float64))
    return out


def confidence(scores):
    """MixCTME confidence gating: member weight = its largest class sum on the sample."""
    return sum(s.max(axis=1, keepdims=True) * s for s in scores)


def accuracy_weighted(scores, accs):
    """WCoCo-WTM: weights = held-out accuracy ratios, on per-sample normalised scores."""
    w = np.asarray(accs, dtype=np.float64)
    w = w / w.sum()
    return sum(wi * per_sample(s.astype(np.float64)) for wi, s in zip(w, scores))


def acc(S, y):
    return float((np.asarray(S).argmax(axis=1) == y).mean())


def _log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def fit_tau(S, y):
    """Temperature of softmax(S / tau) that minimises the log loss of the true class."""
    if np.ptp(S) == 0:
        return 1.0
    r = np.arange(len(y))

    def nll(lt):
        return -_log_softmax(S / np.exp(lt))[r, y].mean()
    res = minimize_scalar(nll, bounds=(-12.0, 6.0), method="bounded")
    return float(np.exp(res.x))


def team_error(S, y, kind="soft", tau=None):
    """Error of a (partial) team on each example, in [0, 1].

    soft: 1 - softmax(S / tau)[true class], tau fitted by log loss on the same (out-of-fold)
          scores; with no information (all scores equal) this is 1 - 1/K.
    01:   1 if the team's top class is wrong (ties count as wrong), else 0.
    """
    S = np.asarray(S, dtype=np.float64)
    r = np.arange(len(y))
    if kind == "01":
        top = S.max(axis=1)
        return ((S[r, y] < top) | ((S == top[:, None]).sum(axis=1) > 1)).astype(np.float64), None
    tau = fit_tau(S, y) if tau is None else tau
    return 1.0 - np.exp(_log_softmax(S / tau)[r, y]), tau
