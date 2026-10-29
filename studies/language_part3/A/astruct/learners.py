"""LR1, LR2 and boosted trees on the arm's rows (PLAN_part3.md section 3, arms table).

- LR1: sklearn LogisticRegression (L2, lbfgs, max_iter 1000, tol 1e-4) on the plain bits (sparse CSR);
  LR2: the same on the plain bits plus every (position bit x candidate type or identity bit) product
  (Ratnaparkhi-style conjoined maximum entropy). Grid: {LR1, LR2} x C {0.03, 0.3, 3, 30}.
- Boosted trees: sklearn HistGradientBoostingClassifier, max_leaf_nodes {31, 127} x learning_rate {0.05, 0.2} x
  l2_regularization {0, 1}; max_iter 1000; early stopping (n_iter_no_change 20, validation loss) on the rows of the
  training dialogues with sha256("es:" + dialogue_id) mod 10 == 0, which are not used for fitting; random_state 0.
Scores: the predicted probability of "true" (the read-out maps it by isotonic regression).
"""

from __future__ import annotations

import numpy as np

LR_C = (0.03, 0.3, 3.0, 30.0)
HGB_GRID = [(m, lr, l2) for m in (31, 127) for lr in (0.05, 0.2) for l2 in (0.0, 1.0)]
LR_MAX_ITER = 1000
HGB_MAX_ITER = 1000


def fit_lr(X, y: np.ndarray, C: float, max_iter: int = LR_MAX_ITER):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-4)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, np.asarray(y, np.int64))
    m.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
    return m


def fit_hgb(X: np.ndarray, y: np.ndarray, es_mask: np.ndarray, leaves: int, lr: float, l2: float,
            max_iter: int = HGB_MAX_ITER):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_leaf_nodes=leaves, learning_rate=lr, l2_regularization=l2,
                                       max_iter=max_iter, early_stopping=True, n_iter_no_change=20,
                                       scoring="loss", random_state=0)
    es = np.asarray(es_mask, bool)
    y = np.asarray(y, np.int64)
    m.fit(X[~es], y[~es], X_val=X[es], y_val=y[es])
    return m


def proba(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1]
