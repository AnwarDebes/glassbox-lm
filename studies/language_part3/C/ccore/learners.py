"""LR and LightGBM class models of C-core (proposal C section 5; PLAN_part3.md section 4, K0b; equal
selection budgets, review 8: 8 trials each, all scored on valid by the system's bpw).

LR: sklearn multinomial logistic regression (lbfgs, L2, intercept) on the one-hot tree nodes of the
  60 bits (layout.lr_matrix: 1,700 features), C in {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}, fitted in
  ascending C with warm start, max_iter 200, tol 1e-4, n_jobs 1.
LightGBM 4.7.0: multiclass (65) on the 60 bits, num_leaves {15, 63} x learning_rate {0.05, 0.2} x
  min_data_in_leaf {20, 200}, at most 400 rounds with early stopping on calib (multi_logloss, patience
  20), num_threads 2, deterministic, force_col_wise, seed 0; predictions at the best iteration.
Shuffled-label runs: the selected setting of each, trained on a seeded 10% of the train blocks (seed 0)
  with the class targets permuted across those positions (seed 1); LR fitted from scratch."""
import numpy as np

from . import layout

LR_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LR_MAX_ITER = 200
LR_TOL = 1e-4
LGBM_GRID = tuple({"num_leaves": nl, "learning_rate": lr, "min_data_in_leaf": md}
                  for nl in (15, 63) for lr in (0.05, 0.2) for md in (20, 200))
LGBM_ROUNDS = 400
LGBM_PATIENCE = 20
LGBM_BASE = {"objective": "multiclass", "num_class": layout.N_CLASSES, "metric": "multi_logloss",
             "num_threads": 2, "deterministic": True, "force_col_wise": True, "seed": 0, "verbosity": -1,
             "max_bin": 255}
SUBSET_SHARE = 0.10
SUBSET_SEED = 0
PERM_SEED = 1


def subset_blocks(blocks, share=SUBSET_SHARE, seed=SUBSET_SEED):
    ub = np.unique(blocks)
    k = int(round(share * len(ub)))
    chosen = np.random.default_rng(seed).choice(ub, size=k, replace=False)
    return np.isin(blocks, chosen)


def permuted(y, seed=PERM_SEED):
    return np.asarray(y)[np.random.default_rng(seed).permutation(len(y))]


def lr_path(Xtr, ytr, evals, grid=LR_GRID, max_iter=LR_MAX_ITER, log=print):
    """Yield (C, {split: P}, info) for ascending C, warm-started along the path."""
    import time
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=grid[0], solver="lbfgs", max_iter=max_iter, tol=LR_TOL, warm_start=True)  # n_jobs: no effect since sklearn 1.8 (no process pool)
    for C in sorted(grid):
        m.set_params(C=C)
        t = time.time()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(Xtr, ytr)
        assert list(m.classes_) == list(range(layout.N_CLASSES)), "a class is missing from train"
        P = {s: m.predict_proba(X) for s, X in evals.items()}
        info = {"C": C, "n_iter": int(np.max(m.n_iter_)), "seconds": round(time.time() - t, 1),
                "converged": not any(issubclass(w.category, ConvergenceWarning) for w in wl)}
        log(f"LR C={C}: {info}")
        yield C, P, {"coef": m.coef_.copy(), "intercept": m.intercept_.copy()}, info


def full_proba(classes, P):
    """predict_proba columns mapped onto all 65 classes (0 for a class absent from the training rows)."""
    out = np.zeros((P.shape[0], layout.N_CLASSES))
    out[:, np.asarray(classes, dtype=np.int64)] = P
    return out


def lr_fit(Xtr, ytr, C, evals, max_iter=LR_MAX_ITER):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=LR_TOL).fit(Xtr, ytr)
    P = {s: full_proba(m.classes_, m.predict_proba(X)) for s, X in evals.items()}
    return P, {"n_iter": int(np.max(m.n_iter_)), "classes_in_train": int(len(m.classes_))}


def lgbm_fit(params, Xtr, ytr, Xcal, ycal, evals, rounds=LGBM_ROUNDS, patience=LGBM_PATIENCE):
    import lightgbm as lgb
    p = dict(LGBM_BASE, **params)
    dtr = lgb.Dataset(np.asarray(Xtr, dtype=np.float32), label=ytr, params={"max_bin": 255}, free_raw_data=False)
    dcal = lgb.Dataset(np.asarray(Xcal, dtype=np.float32), label=ycal, reference=dtr, free_raw_data=False)
    hist = {}
    bst = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dcal], valid_names=["calib"],
                    callbacks=[lgb.early_stopping(patience, verbose=False), lgb.record_evaluation(hist)])
    it = bst.best_iteration if bst.best_iteration > 0 else bst.current_iteration()
    P = {s: bst.predict(np.asarray(X, dtype=np.float32), num_iteration=it) for s, X in evals.items()}
    curve = hist["calib"]["multi_logloss"]
    return P, bst, {"best_iteration": int(it), "rounds_run": int(len(curve)),
                    "calib_multi_logloss": float(curve[it - 1])}
