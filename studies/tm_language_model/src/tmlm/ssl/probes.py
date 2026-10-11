"""Probes and Q_text of docs/13 amendment A7 (family H17).

Label draws (A7 "5 stratified label draws"): for a task, a draw index d and a budget of n labels per class,
the labelled sample holds min(n, pool size of class c) examples of every class c: the first ones of a fixed
per-class order of the pool, the order being numpy default_rng(SeedSequence([20260924, 717, task code, d,
c])).permutation of that class's pool rows. The samples of one draw are therefore NESTED across budgets
(8 in 32 in 128 in 512), and the union of all samples of a task is at most draws x 512 per class (small
enough to featurise exactly). Budget "full" is the whole pool (identical for every draw). TREC's smallest
class has fewer than 128 pool examples: its sample is then the whole class (sizes are logged per cell).

Primary probe: L2 logistic regression, C from C_GRID by stratified 4-fold cross-validation inside the
labelled sample only (tmlm.ssl.transfer.lr_cell with max_folds 4; folds shuffled with the draw's seed),
refit on the whole sample, accuracy on the dev carve.
Secondary probe: tmu TMCoalescedClassifier (CPU) on binarised features, the configuration chosen from a
small fixed grid by the same 4-fold cross-validation inside the labelled sample (ties to the first entry).

Q_text: the mean accuracy over AG News, SST-2 and TREC-6 (all three required) for one feature set, probe,
budget and draw; reported as the mean over draws with a Student-t interval over draws, and compared with
paired differences (same draw, same folds) and a paired bootstrap over draws (and seeds when there are
several; `paired_bootstrap`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import transfer as TR

FOLDS = 4
Q_TASKS = ("agnews", "sst2", "trec")
C_GRID = TR.C_GRID
TM_GRID = ({"clauses": 400, "T": 300, "s": 2.0}, {"clauses": 400, "T": 300, "s": 6.0})
TM_EPOCHS = 10


# ------------------------------------------------------------------ label draws
def class_orders(y: np.ndarray, task_code: int, draw: int) -> dict[int, np.ndarray]:
    """class -> pool rows of that class in the draw's fixed random order."""
    y = np.asarray(y, np.int64)
    out = {}
    for c in np.unique(y).tolist():
        rows = np.flatnonzero(y == c)
        rng = np.random.default_rng(np.random.SeedSequence([20260924, 717, int(task_code), int(draw), int(c)]))
        out[int(c)] = rows[rng.permutation(rows.shape[0])]
    return out


def label_sample(y: np.ndarray, per_class: int | str, task_code: int, draw: int) -> np.ndarray:
    """Ascending pool rows of the labelled sample (see the module docstring)."""
    y = np.asarray(y, np.int64)
    if per_class == "full":
        return np.arange(y.shape[0], dtype=np.int64)
    n = int(per_class)
    parts = [o[:n] for o in class_orders(y, task_code, draw).values()]
    return np.sort(np.concatenate(parts)).astype(np.int64)


def sample_union(y: np.ndarray, budgets: Sequence, draws: Sequence[int], task_code: int) -> np.ndarray:
    """Ascending pool rows used by any (budget, draw)."""
    rows = [label_sample(y, b, task_code, d) for b in budgets for d in draws]
    return np.unique(np.concatenate(rows)).astype(np.int64) if rows else np.zeros(0, np.int64)


def fold_seed(task_code: int, budget, draw: int) -> int:
    b = 0 if budget == "full" else int(budget)
    return int(20260924 + 1000 * int(task_code) + 17 * b + int(draw)) % (2 ** 31)


# ------------------------------------------------------------------ probes
@dataclass
class Cell:
    accuracy: float
    choice: object
    cv: dict
    fit_s: float
    n_labelled: int
    n_eval: int


def lr_probe(Xl, yl, Xe, ye, seed: int, dense: bool = False, c_grid: Sequence[float] = C_GRID,
             folds: int = FOLDS, max_iter: int = 1000, backend: str = "cpu") -> Cell:
    """L2 logistic regression, C by stratified `folds`-fold CV inside the labelled sample only."""
    r = TR.lr_cell(Xl, np.asarray(yl), Xe, np.asarray(ye), seed, dense=dense, max_iter=max_iter,
                   c_grid=c_grid, max_folds=folds, backend=backend)
    return Cell(r.accuracy, r.choice, r.cv, r.fit_s, int(np.asarray(yl).shape[0]), int(np.asarray(ye).shape[0]))


def tm_fit_predict(Xl: np.ndarray, yl: np.ndarray, Xe: np.ndarray, cfg: dict, seed: int, epochs: int) -> np.ndarray:
    """tmu TMCoalescedClassifier (CPU) on binary uint32 features; seeded (np.random and the tmu seed)."""
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

    yl = np.asarray(yl, np.int64)
    if np.unique(yl).shape[0] < 2:
        return np.full(Xe.shape[0], int(yl[0]))
    np.random.seed(int(seed) % (2 ** 31))
    tm = TMCoalescedClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]), s=float(cfg["s"]),
                               platform="CPU", seed=int(seed) % (2 ** 31))
    Xl = np.ascontiguousarray(Xl, dtype=np.uint32)
    ylu = np.ascontiguousarray(yl, dtype=np.uint32)
    for _ in range(int(epochs)):
        tm.fit(Xl, ylu)
    return np.asarray(tm.predict(np.ascontiguousarray(Xe, dtype=np.uint32))).astype(np.int64)


def tm_probe(Xl: np.ndarray, yl, Xe: np.ndarray, ye, seed: int, grid: Sequence[dict] = TM_GRID,
             epochs: int = TM_EPOCHS, folds: int = FOLDS) -> Cell:
    """Coalesced TM probe: configuration from `grid` by `folds`-fold CV inside the labelled sample."""
    t0 = time.time()
    yl, ye = np.asarray(yl, np.int64), np.asarray(ye, np.int64)
    splits = TR.cv_splits(yl, seed, folds)
    cv = {}
    for gi, cfg in enumerate(grid):
        accs = []
        for fi, (tr, va) in enumerate(splits):
            p = tm_fit_predict(Xl[tr], yl[tr], Xl[va], cfg, seed * 1000 + fi, epochs)
            accs.append(float(np.mean(p == yl[va])))
        cv[gi] = float(np.mean(accs))
    best = max(cv.values())
    gi = min(g for g, a in cv.items() if a == best)
    pred = tm_fit_predict(Xl, yl, Xe, grid[gi], seed, epochs)
    return Cell(float(np.mean(pred == ye)), dict(grid[gi]), {"folds": len(splits), "acc": cv}, time.time() - t0,
                int(yl.shape[0]), int(ye.shape[0]))


# ------------------------------------------------------------------ Q_text and comparisons
def q_text(acc_by_task: dict[str, float], tasks: Sequence[str] = Q_TASKS) -> float:
    """Mean accuracy over the three tasks; raises if one is missing."""
    missing = [t for t in tasks if t not in acc_by_task]
    if missing:
        raise KeyError(f"Q_text needs every task, missing {missing}")
    return float(np.mean([acc_by_task[t] for t in tasks]))


def t_interval(values: Sequence[float]) -> dict:
    return TR.t_interval(values)


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 20260924) -> dict:
    """Paired bootstrap of the mean of per-unit differences (units = label draws, or (seed, draw) pairs):
    resample units with replacement; 95 percent percentile interval. `diffs` 1-D."""
    d = np.asarray(diffs, np.float64).reshape(-1)
    n = d.shape[0]
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 717, n]))
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)),
            "n": int(n), "n_boot": int(n_boot), "excludes_0": bool(np.quantile(means, 0.025) > 0 or np.quantile(means, 0.975) < 0)}
