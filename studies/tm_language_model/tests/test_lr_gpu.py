"""tmlm.ssl.lr_gpu against sklearn's LogisticRegression(lbfgs): same objective, same optimiser, GPU arithmetic."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

cp = pytest.importorskip("cupy")
try:
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    HAS_GPU = False
pytestmark = pytest.mark.skipif(not HAS_GPU, reason="no CUDA device visible to CuPy")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from tmlm.ssl import transfer as TR  # noqa: E402
from tmlm.ssl.lr_gpu import GpuLogisticRegression  # noqa: E402


def _data(n, d, k, density, seed, dense=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = sp.random(n, d, density=density, random_state=seed, format="csr", dtype=np.float64)
    X.data = np.log1p(np.ceil(X.data * 5))
    signal = rng.normal(size=(k, d)) * (rng.random(d) < 0.05)
    X = X + sp.csr_matrix((np.eye(k)[y] @ signal) * (X.toarray() > 0))
    X = sp.csr_matrix(X)
    if dense:
        return X.toarray(), y
    return X, y


def _both(X, y, C, max_iter=1000):
    ref = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref.fit(X, y)
    gpu = GpuLogisticRegression(C=C, max_iter=max_iter, tol=1e-4).fit(X, y)
    return ref, gpu


@pytest.mark.parametrize("k", [2, 4, 6])
@pytest.mark.parametrize("C", [0.01, 1.0, 100.0])
def test_matches_sklearn_sparse(k, C):
    X, y = _data(120, 400, k, 0.05, seed=10 * k + int(C * 10) % 7)
    ref, gpu = _both(X, y, C)
    np.testing.assert_array_equal(gpu.classes_, ref.classes_)
    scale = max(1.0, float(np.abs(ref.coef_).max()))
    assert np.abs(gpu.coef_ - ref.coef_).max() <= 1e-6 * scale
    assert np.abs(gpu.intercept_ - ref.intercept_).max() <= 1e-6 * scale
    Xe, _ = _data(300, 400, k, 0.05, seed=99)
    np.testing.assert_array_equal(gpu.predict(Xe), ref.predict(Xe))


def test_matches_sklearn_dense_and_iteration_cap():
    X, y = _data(80, 50, 3, 0.3, seed=3, dense=True)
    for max_iter in (5, 1000):
        ref, gpu = _both(X, y, 10.0, max_iter=max_iter)
        assert int(gpu.n_iter_[0]) == int(np.max(ref.n_iter_))
        assert np.abs(gpu.coef_ - ref.coef_).max() <= 1e-6 * max(1.0, float(np.abs(ref.coef_).max()))
        np.testing.assert_array_equal(gpu.predict(X), ref.predict(X))


def test_lr_cell_backend_equivalent():
    X, y = _data(160, 300, 4, 0.06, seed=7)
    Xe, ye = _data(200, 300, 4, 0.06, seed=8)
    a = TR.lr_cell(X, y, Xe, ye, seed=5, max_folds=4, backend="cpu")
    b = TR.lr_cell(X, y, Xe, ye, seed=5, max_folds=4, backend="gpu")
    assert a.choice == b.choice
    assert a.cv == b.cv
    np.testing.assert_array_equal(a.pred, b.pred)


def test_unknown_backend_rejected():
    X, y = _data(40, 30, 2, 0.2, seed=1)
    with pytest.raises(ValueError):
        TR._fit_lr(X, y, 1.0, False, 100, backend="tpu")
