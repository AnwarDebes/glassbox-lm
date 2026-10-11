"""Few-label transfer protocol on frozen representations (exploratory, docs/12 F23).

Label samples: for a task, a budget n and a seed s, the labelled sample is n pool documents drawn
uniformly without replacement with numpy default_rng(SeedSequence([20260924, 623, task_code, n, s]));
the budget "full" is the whole pool. Only the labelled sample is ever used for choices (regularisation
strength, TM configuration): stratified k-fold cross-validation inside the sample.

Classifiers
  lr  L2-regularised logistic regression (sklearn, lbfgs, multinomial), C from C_GRID by k-fold CV
      accuracy on the labelled sample (ties to the smaller C), refit on the whole sample.
  tm  tmu TMClassifier (CPU, weighted clauses) on binary features, configuration from a small fixed grid
      by k-fold CV accuracy on the labelled sample (ties to the first grid entry), refit on the sample.

Intervals: per cell, mean over seeds with a Student-t 95 percent interval (df = n_seeds - 1); paired
differences between feature sets use the per-seed differences (same label sample, same folds).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def label_sample(pool_size: int, budget: int | str, task_code: int, seed: int) -> np.ndarray:
    """Indices into the pool (ascending)."""
    if budget == "full" or (isinstance(budget, int) and budget >= pool_size):
        return np.arange(pool_size, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, 623, int(task_code), int(budget), int(seed)]))
    return np.sort(rng.choice(pool_size, size=int(budget), replace=False)).astype(np.int64)


def cv_splits(y: np.ndarray, seed: int, max_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds when every class present has at least 2 members (k = min(max_folds, smallest
    class count), at least 2), else plain shuffled k-fold with k = min(max_folds, n)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= 2:
        k = int(max(2, min(max_folds, counts.min())))
        spl = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    else:
        k = int(max(2, min(max_folds, y.shape[0])))
        spl = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    return [(tr, va) for tr, va in spl.split(np.zeros(y.shape[0]), y)]


def used_columns(X) -> np.ndarray:
    """Columns with at least one stored non-zero in a scipy sparse matrix (ascending)."""
    Xc = X.tocsr()
    Xc.eliminate_zeros()
    return np.unique(Xc.indices).astype(np.int64)


def _fit_lr(X, y, C: float, dense: bool, max_iter: int, backend: str = "cpu"):
    """Sparse input: the fit uses only the columns that are non-zero in some training row. Under L2
    regularisation a column that is zero in every training row has gradient w/C, so its optimal weight is
    exactly 0 and its lbfgs iterate stays 0: the restricted fit is the same model at a cost set by the used
    columns instead of all columns (tests/test_ssl_features.py checks identical predictions).
    backend 'gpu': the same objective and the same scipy L-BFGS-B call with the loss and gradient on the
    GPU (tmlm.ssl.lr_gpu; tests/test_lr_gpu.py)."""
    from sklearn.linear_model import LogisticRegression

    if np.unique(y).shape[0] < 2:
        return ("const", int(y[0]))
    cols = None
    if dense:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
    else:
        mu = sd = None
        cols = used_columns(X)
        if cols.size == 0:
            cols = np.zeros(1, np.int64)
        Xs = X.tocsr()[:, cols]
    if backend == "gpu":
        from tmlm.ssl.lr_gpu import GpuLogisticRegression

        m = GpuLogisticRegression(C=C, max_iter=max_iter, tol=1e-4)
    elif backend == "cpu":
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-4)
    else:
        raise ValueError(f"unknown LR backend {backend!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xs, y)
    return ("lr", m, mu, sd, cols)


def _predict_lr(model, X, dense: bool) -> np.ndarray:
    if model[0] == "const":
        return np.full(X.shape[0], model[1])
    _, m, mu, sd, cols = model
    Xs = (X - mu) / sd if dense else X.tocsr()[:, cols]
    return m.predict(Xs)


@dataclass
class FitResult:
    accuracy: float
    choice: object
    cv: dict
    fit_s: float
    pred: np.ndarray


def lr_cell(Xl, yl, Xe, ye, seed: int, dense: bool = False, max_iter: int = 1000,
            c_grid: Sequence[float] = C_GRID, max_folds: int = 5, backend: str = "cpu") -> FitResult:
    """L2 logistic regression with C chosen by CV on the labelled sample; accuracy on (Xe, ye)."""
    t0 = time.time()
    folds = cv_splits(yl, seed, max_folds)
    cv = {}
    for C in c_grid:
        accs = []
        for tr, va in folds:
            m = _fit_lr(Xl[tr], yl[tr], C, dense, max_iter, backend)
            accs.append(float(np.mean(_predict_lr(m, Xl[va], dense) == yl[va])))
        cv[float(C)] = float(np.mean(accs))
    best = max(cv.values())
    C = min(c for c, a in cv.items() if a == best)
    m = _fit_lr(Xl, yl, C, dense, max_iter, backend)
    pred = _predict_lr(m, Xe, dense)
    return FitResult(float(np.mean(pred == ye)), C, {"folds": len(folds), "acc": cv}, time.time() - t0, pred)


# ------------------------------------------------------------------ Tsetlin machine (tmu, CPU)
def tm_train_predict(Xl: np.ndarray, yl: np.ndarray, Xe: np.ndarray, cfg: dict, seed: int, n_classes: int) -> np.ndarray:
    """Train a tmu TMClassifier for cfg['epochs'] epochs on binary uint32 features; predict Xe.
    Missing classes: tmu sizes the model from max(y) + 1, so a sample lacking the top class still works."""
    from tmu.models.classification.vanilla_classifier import TMClassifier

    if np.unique(yl).shape[0] < 2:
        return np.full(Xe.shape[0], int(yl[0]))
    np.random.seed(seed)
    tm = TMClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]), s=float(cfg["s"]), platform="CPU",
                      weighted_clauses=True, seed=int(seed))
    Xl = np.ascontiguousarray(Xl, dtype=np.uint32)
    yl = np.ascontiguousarray(yl, dtype=np.uint32)
    for _ in range(int(cfg["epochs"])):
        tm.fit(Xl, yl)
    return np.asarray(tm.predict(np.ascontiguousarray(Xe, dtype=np.uint32))).astype(np.int64)


def tm_cell(Xl, yl, Xe, ye, seed: int, grid: Sequence[dict], n_classes: int, max_folds: int = 3) -> FitResult:
    t0 = time.time()
    folds = cv_splits(yl, seed, max_folds)
    cv = {}
    for gi, cfg in enumerate(grid):
        accs = []
        for fi, (tr, va) in enumerate(folds):
            p = tm_train_predict(Xl[tr], yl[tr], Xl[va], cfg, seed * 1000 + fi, n_classes)
            accs.append(float(np.mean(p == yl[va])))
        cv[gi] = float(np.mean(accs))
    best = max(cv.values())
    gi = min(g for g, a in cv.items() if a == best)
    pred = tm_train_predict(Xl, yl, Xe, grid[gi], seed, n_classes)
    return FitResult(float(np.mean(pred == ye)), dict(grid[gi]), {"folds": len(folds), "acc": cv}, time.time() - t0, pred)


# ------------------------------------------------------------------ intervals
def t_interval(values: Sequence[float]) -> dict:
    v = np.asarray(values, np.float64)
    n = v.shape[0]
    mean = float(v.mean()) if n else float("nan")
    if n < 2:
        return {"mean": mean, "lo": None, "hi": None, "n": int(n), "sd": None}
    sd = float(v.std(ddof=1))
    h = T_975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return {"mean": mean, "lo": mean - h, "hi": mean + h, "n": int(n), "sd": sd}


def verdict(diff_ci: Callable[[str, str, int, str, str], dict | None], classifiers: Sequence[str],
            tasks=("sst2", "agnews"), budgets=(256, 1024), margin: float = 0.02,
            all_cells: Sequence[tuple[str, int, str]] = ()) -> dict:
    """EARLY VERDICT.

    diff_ci(task, budget, classifier, a, b) -> t-interval of accuracy(a) - accuracy(b) over seeds (paired),
    a, b in {'trained', 'bag', 'young'}; 'bag' is the stronger bag-of-tokens variant of that cell.
    GO: under at least one classifier, for BOTH tasks and BOTH budgets, trained - bag >= margin and
        trained - young >= margin (means), with both intervals excluding 0 (lo > 0).
    NO-GO: trained - young has lo > 0 in NO cell of all_cells (task, budget, classifier).
    GREY: otherwise."""
    per_clf = {}
    for clf in classifiers:
        ok = True
        detail = {}
        for t in tasks:
            for b in budgets:
                for ref in ("bag", "young"):
                    ci = diff_ci(t, b, clf, "trained", ref)
                    good = bool(ci is not None and ci["lo"] is not None and ci["mean"] >= margin and ci["lo"] > 0)
                    detail[f"{t}.{b}.trained_minus_{ref}"] = {"ci": ci, "meets": good}
                    ok = ok and good
        per_clf[clf] = {"go": ok, "cells": detail}
    beats_young_anywhere = []
    for (t, b, clf) in all_cells:
        ci = diff_ci(t, b, clf, "trained", "young")
        if ci is not None and ci["lo"] is not None and ci["lo"] > 0:
            beats_young_anywhere.append(f"{t}.{b}.{clf}")
    if any(v["go"] for v in per_clf.values()):
        v = "GO"
    elif not beats_young_anywhere:
        v = "NO-GO"
    else:
        v = "GREY"
    return {"verdict": v, "per_classifier": per_clf, "trained_beats_young_cells": beats_young_anywhere}
