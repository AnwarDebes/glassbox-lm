"""Load the prepared B6 views (written by tally.b6.prepare) and their labels."""
import json
import os

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(STUDY, "data", "b6")
SPLITS = ("train", "calib", "valid", "devtest")
AG_VIEWS = ("word1", "word2", "char4")


def path(ds, name):
    return os.path.join(ROOT, ds, name)


def manifest(ds):
    with open(path(ds, "manifest.json")) as f:
        return json.load(f)


def labels(ds, split):
    return np.load(path(ds, f"y_{split}.npy")).astype(np.uint32)


def view(ds, name, split):
    """uint8 array (Fashion-MNIST) or uint8 CSR matrix (AG News); 'concat' joins the AG views."""
    if ds == "agnews":
        import scipy.sparse as sps
        if name == "concat":
            return sps.hstack([view(ds, v, split) for v in AG_VIEWS], format="csr")
        return sps.load_npz(path(ds, f"{name}_{split}.npz")).tocsr()
    return np.load(path(ds, f"{name}_{split}.npy"))


def dense(X, rows=None):
    """Dense uint32 copy of the selected rows (tmu wants uint32)."""
    if rows is not None:
        X = X[rows]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.ascontiguousarray(X, dtype=np.uint32)


def files(ds, names):
    """Paths of the prepared files a run reads (for the data hashes of the run record)."""
    out = [path(ds, "manifest.json")]
    for n in names:
        out.append(path(ds, n))
    return out


def subset_rows(y, n, seed):
    """Stratified subset of n rows (smoke runs)."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, n // len(classes))
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in classes]))
