"""Thermometer code of the embedding after a linear map fitted on the training queries and their
labels (exploration e24, after R4/R4b). A Tsetlin machine builds class regions from AND-ed
thresholds on single coordinates, so the coordinate system matters to it, while logistic
regression is rotation-invariant. Two maps:

- ``lda``: the discriminant axes of shrinkage LDA (at most C - 1 of them): within-class whitening,
  then the eigenvectors of the between-class scatter of the class means.
- ``white``: whitening by the within-class covariance, all D axes.

Shrinkage: Sw + shrink * trace(Sw) / D * I. Bit names: ``lda[a] >= t``.
"""

from __future__ import annotations

import numpy as np

from dl.boolean import Thermo


class Rotated:
    def __init__(self, kind: str = "lda", n_bins: int = 8, shrink: float = 0.001, max_axes: int = 0):
        if kind not in ("lda", "white"):
            raise ValueError(kind)
        self.kind, self.n_bins, self.shrink, self.max_axes = kind, n_bins, shrink, max_axes

    def fit(self, X: np.ndarray, y: np.ndarray, texts=None, class_names=None) -> "Rotated":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.mu = X.mean(0)
        classes = np.unique(y)
        means = np.stack([X[y == c].mean(0) for c in classes])
        R = X - means[np.searchsorted(classes, y)]
        D = X.shape[1]
        Sw = R.T @ R / len(X)
        Sw += self.shrink * np.trace(Sw) / D * np.eye(D)
        ev, U = np.linalg.eigh(Sw)
        Wh = U @ np.diag(ev ** -0.5) @ U.T  # Sw^{-1/2}
        if self.kind == "white":
            self.W = Wh
        else:
            counts = np.array([(y == c).sum() for c in classes], dtype=np.float64)
            M = (means - self.mu) @ Wh
            Sb = (M * counts[:, None]).T @ M / len(X)
            eb, V = np.linalg.eigh(Sb)
            k = min(len(classes) - 1, D) if self.max_axes <= 0 else min(self.max_axes, len(classes) - 1, D)
            V = V[:, np.argsort(-eb)[:k]]
            self.W = Wh @ V
        Z = ((X - self.mu) @ self.W).astype(np.float32)
        self.therm = Thermo(self.n_bins).fit(Z)
        self.names = [f"{self.kind}[{a}] >= {t:.3f}" for a, th in enumerate(self.therm.th) for t in th]
        return self

    def transform(self, X: np.ndarray, texts=None) -> np.ndarray:
        Z = ((np.asarray(X, dtype=np.float64) - self.mu) @ self.W).astype(np.float32)
        return self.therm.transform(Z)
