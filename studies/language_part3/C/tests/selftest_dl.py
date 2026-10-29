"""Self-tests of C-core in the learner environment (.venv-dl: numpy, scipy, sklearn, LightGBM).

    scripts/run_stage.sh selftest_dl

Synthetic data only. Prints PASS or FAIL per test and exits 1 on any failure. Under 2 minutes."""
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import gates, hashing, layout, learners, paths, systems  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def synthetic(n, rng, signal=True):
    """Random 60-bit inputs from a tiny vocabulary with codes; class = a function of the t-1 code."""
    V, eos = 200, 0
    p6 = rng.integers(0, 64, size=V)
    p8 = rng.integers(0, 256, size=V)
    x = []
    while len(x) < n:
        x += list(rng.integers(1, V, size=rng.integers(1, 12))) + [eos]
    x = np.array(x[:n - 1] + [eos])
    X = layout.features(x, eos, p6, p8)
    E6, B8, pad = layout.decode(X)
    y = np.where(pad[:, 0], 64, (E6[:, 0] + (B8[:, 0] >> 6)) % 64) if signal else rng.integers(0, 65, size=n)
    noise = rng.random(n) < 0.3
    y = np.where(noise, rng.integers(0, 65, size=n), y)
    y[:65] = np.arange(65)                     # every class present
    return X, y.astype(np.int64)


@test
def lr_features_layout():
    rng = np.random.default_rng(0)
    X, _ = synthetic(3000, rng)
    M = layout.lr_matrix(X)
    assert M.shape == (3000, 1700)
    E6, B8, pad = layout.decode(X)
    nnz = np.diff(M.indptr)
    want = (7 * (~pad)).sum(1) + pad.sum(1)
    assert np.array_equal(nnz, want), "wrong number of one-hot features per row"
    assert (nnz[~pad.any(1)] == 28).all()
    r = int(np.flatnonzero(~pad.any(1))[0])
    cols = set(M.indices[M.indptr[r]:M.indptr[r + 1]].tolist())
    for j in range(4):
        base = j * layout.LR_PER_POS
        assert base + (E6[r, j] >> 4) in cols                 # E depth 2
        assert base + 4 + 16 + E6[r, j] in cols               # E depth 6
        assert base + 84 + 4 + 16 + 64 + B8[r, j] in cols     # B depth 8
    assert M.data.min() == 1.0 and M.indices.max() < 1700


@test
def lr_path_and_fit():
    rng = np.random.default_rng(1)
    X, y = synthetic(4000, rng)
    Xc, yc = synthetic(1000, rng)
    M, Mc = layout.lr_matrix(X), layout.lr_matrix(Xc)
    seen = []
    for C, P, coef, info in learners.lr_path(M, y, {"calib": Mc}, grid=(0.01, 1.0), max_iter=50, log=lambda *a: None):
        assert P["calib"].shape == (1000, 65) and np.allclose(P["calib"].sum(1), 1)
        seen.append((C, -np.mean(np.log(P["calib"][np.arange(1000), yc]))))
    assert [c for c, _ in seen] == [0.01, 1.0]
    sub = np.arange(4000) < 300                # a small subset can miss classes: columns are filled
    P, info = learners.lr_fit(M[np.flatnonzero(sub)], y[sub] % 50, 1.0, {"calib": Mc}, max_iter=50)
    assert P["calib"].shape == (1000, 65) and np.allclose(P["calib"].sum(1), 1)
    assert (P["calib"][:, 50:] == 0).all() and info["classes_in_train"] == 50


@test
def lightgbm_fit_early_stopping():
    rng = np.random.default_rng(2)
    X, y = synthetic(3000, rng)
    Xc, yc = synthetic(800, rng)
    P, bst, info = learners.lgbm_fit({"num_leaves": 15, "learning_rate": 0.2, "min_data_in_leaf": 20}, X, y, Xc, yc,
                                     {"calib": Xc}, rounds=60, patience=5)
    assert P["calib"].shape == (800, 65) and np.allclose(P["calib"].sum(1), 1) and P["calib"].min() > 0
    assert 1 <= info["best_iteration"] <= info["rounds_run"] <= 60
    assert info["rounds_run"] == 60 or info["rounds_run"] == info["best_iteration"] + 5, info
    assert bst.params["num_threads"] == 2 and bst.params["deterministic"] is True
    P2, _, info2 = learners.lgbm_fit({"num_leaves": 15, "learning_rate": 0.2, "min_data_in_leaf": 20}, X, y, Xc, yc,
                                     {"calib": Xc}, rounds=60, patience=5)
    assert np.array_equal(P["calib"], P2["calib"]), "LightGBM is not deterministic"
    assert len(learners.LGBM_GRID) == 8 and len(learners.LR_GRID) == 8


@test
def shuffled_subset_rules():
    blocks = np.repeat(np.arange(379), 50)
    sub = learners.subset_blocks(blocks)
    assert len(np.unique(blocks[sub])) == 38 and np.array_equal(sub, learners.subset_blocks(blocks))
    y = np.arange(1000)
    p = learners.permuted(y)
    assert sorted(p.tolist()) == y.tolist() and not np.array_equal(p, y)
    assert np.array_equal(p, learners.permuted(y))


@test
def systems_work_here():
    rng = np.random.default_rng(3)
    V, K, n = 150, 65, 400
    cls = np.concatenate([np.arange(K), rng.integers(0, K, size=V - K)])
    F = rng.dirichlet(np.ones(V), size=n)
    w = np.array([rng.choice(V, p=F[i]) for i in range(n)])
    Fc = F @ np.eye(K)[cls]
    e = rng.random(V) + 0.1
    for c in range(K):
        e[cls == c] /= e[cls == c].sum()
    b = rng.integers(0, 40, size=n)
    sp = systems.Split(w, cls[w], F[np.arange(n), w], Fc, b, b // 8, np.arange(n) // 40, e)
    PX = rng.dirichlet(np.ones(K), size=n)
    _, lamb, _ = systems.fit_system(PX, sp)
    pos = np.arange(0, n, 4)
    assert systems.contract(PX, sp, lamb, pos, F[pos], cls, e)["pass"]
    assert gates.k0b(6.9, 6.905, 7.1)["pass"]


@test
def no_neural_network_in_the_learners():
    import lightgbm  # noqa: F401
    import sklearn.linear_model  # noqa: F401

    from ccore import runlog  # noqa: F401
    assert env.neural_modules_loaded() == [], env.neural_modules_loaded()


@test
def no_writes_outside_C():
    cmd = "for d in " + " ".join(f'"{d}"' for d in paths.LISTED_DIRS) + \
        '; do echo "== $d"; ls -la --time-style=full-iso "$d" 2>&1; done'
    before = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    rng = np.random.default_rng(4)
    X, y = synthetic(1500, rng)
    learners.lgbm_fit({"num_leaves": 15, "learning_rate": 0.2, "min_data_in_leaf": 20}, X, y, X[:300], y[:300],
                      {"x": X[:10]}, rounds=5, patience=2)
    list(learners.lr_path(layout.lr_matrix(X), y, {"x": layout.lr_matrix(X[:10])}, grid=(1.0,), max_iter=5,
                          log=lambda *a: None))
    hashing.snapshot()
    after = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    assert before == after, "a listed folder changed"


def main():
    t0 = time.time()
    fails = 0
    for fn in RESULTS:
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t:.1f} s)", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__} ({time.time() - t:.1f} s)\n{traceback.format_exc()}", flush=True)
    print(f"selftest_dl: {len(RESULTS) - fails} of {len(RESULTS)} passed in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
