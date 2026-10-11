"""The probes' L2 logistic regression with its loss and gradient evaluated on the GPU.

Only the arithmetic moves. The optimiser is the one sklearn 1.9.1 uses for
`LogisticRegression(solver="lbfgs", C, max_iter, tol)`: `scipy.optimize.minimize(method="L-BFGS-B")` with
options maxiter = max_iter, maxls = 50, gtol = tol, ftol = 64 * machine epsilon, started from w0 = 0 in
sklearn's parameter layout (binary: one vector of n_features + 1; K > 2 classes: a (K, n_features + 1)
array raveled in Fortran order). The objective is sklearn's `LinearModelLoss`:

    mean_i loss_i + 0.5 * l2 * ||coef||^2,   l2 = 1 / (C * n_samples),   intercept not penalised,

with `HalfBinomialLoss` (binary: softplus(z) - y z) or `HalfMultinomialLoss` (logsumexp(z) - z_y), all in
float64. Predictions follow sklearn: binary classes_[decision > 0], otherwise classes_[argmax]. The only
source of difference from the CPU fit is floating-point summation order on the GPU; tests/test_lr_gpu.py
checks coefficients and predictions against sklearn, and every switch of a registered run to this backend
needs the equivalence run recorded in its amendment.
"""

from __future__ import annotations

import numpy as np

_FTOL = 64 * np.finfo(float).eps


def _softplus(cp, z):
    return cp.maximum(z, 0.0) + cp.log1p(cp.exp(-cp.abs(z)))


class GpuLogisticRegression:
    """Drop-in for the subset of sklearn's LogisticRegression the probes use (fit, predict)."""

    def __init__(self, C: float, max_iter: int = 1000, tol: float = 1e-4) -> None:
        self.C, self.max_iter, self.tol = float(C), int(max_iter), float(tol)

    @staticmethod
    def _to_gpu(X):
        import cupy as cp
        import cupyx.scipy.sparse as cps
        import scipy.sparse as sp

        if sp.issparse(X):
            return cps.csr_matrix(X.tocsr().astype(np.float64)), True
        return cp.asarray(np.ascontiguousarray(X, dtype=np.float64)), False

    def fit(self, X, y) -> "GpuLogisticRegression":
        import cupy as cp
        from scipy import optimize

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        k = self.classes_.shape[0]
        if k < 2:
            raise ValueError("needs at least 2 classes")
        n, d = X.shape
        Xg, sparse = self._to_gpu(X)
        XgT = Xg.T.tocsr() if sparse else Xg.T
        l2 = 1.0 / (self.C * n)
        binary = k == 2
        if binary:
            t = cp.asarray((y == self.classes_[1]).astype(np.float64))
            w0 = np.zeros(d + 1, np.float64)

            def func(w):
                wg = cp.asarray(w)
                coef, b = wg[:d], wg[d]
                z = Xg @ coef + b
                loss = float(cp.sum(_softplus(cp, z) - t * z)) / n + 0.5 * l2 * float(coef @ coef)
                g = (1.0 / (1.0 + cp.exp(-z)) - t) / n
                grad = cp.empty(d + 1, np.float64)
                grad[:d] = XgT @ g + l2 * coef
                grad[d] = cp.sum(g)
                return loss, cp.asnumpy(grad)
        else:
            yi = np.searchsorted(self.classes_, y)
            onehot = cp.zeros((n, k), np.float64)
            onehot[cp.arange(n), cp.asarray(yi)] = 1.0
            w0 = np.zeros((k, d + 1), np.float64, order="F").ravel(order="F")

            def func(w):
                W = cp.asarray(w).reshape((k, d + 1), order="F")
                coef, b = W[:, :d], W[:, d]
                z = Xg @ coef.T + b[None, :]
                zmax = cp.max(z, axis=1, keepdims=True)
                e = cp.exp(z - zmax)
                s = cp.sum(e, axis=1, keepdims=True)
                lse = cp.log(s) + zmax
                loss = float(cp.sum(lse[:, 0] - cp.sum(z * onehot, axis=1))) / n + 0.5 * l2 * float(cp.sum(coef * coef))
                g = (e / s - onehot) / n
                grad = cp.empty((k, d + 1), np.float64, order="F")
                grad[:, :d] = (XgT @ g).T + l2 * coef
                grad[:, d] = cp.sum(g, axis=0)
                return loss, cp.asnumpy(grad).ravel(order="F")

        res = optimize.minimize(func, w0, method="L-BFGS-B", jac=True,
                                options={"maxiter": self.max_iter, "maxls": 50, "gtol": self.tol, "ftol": _FTOL})
        self.n_iter_ = np.array([int(res.nit)])
        w = np.asarray(res.x, np.float64)
        if binary:
            self.coef_, self.intercept_ = w[:d][None, :], w[d:]
        else:
            W = w.reshape((k, d + 1), order="F")
            self.coef_, self.intercept_ = W[:, :d].copy(), W[:, d].copy()
        return self

    def decision_function(self, X) -> np.ndarray:
        import cupy as cp

        Xg, _ = self._to_gpu(X)
        z = Xg @ cp.asarray(self.coef_.T) + cp.asarray(self.intercept_)[None, :]
        z = cp.asnumpy(z)
        return z[:, 0] if z.shape[1] == 1 else z

    def predict(self, X) -> np.ndarray:
        z = self.decision_function(X)
        if z.ndim == 1:
            return self.classes_[(z > 0).astype(int)]
        return self.classes_[np.argmax(z, axis=1)]
