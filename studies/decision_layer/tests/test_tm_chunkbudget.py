"""The chunked update with the exact budget (src/dl/tm_chunkbudget.py) keeps every clause within the
budget and learns a small synthetic task about as well as the unchunked update."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dl.tm as tm_mod  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_chunkbudget import enable  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def task(n=3000, f=40, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.uniform(size=(n, f)) < 0.3).astype(np.uint8)
    y = (X[:, 0] & X[:, 1]) + 2 * (X[:, 2] & ~X[:, 3] & 1)
    y = np.minimum(y, 2)
    return X, y


def run(chunked: bool, budget: int = 6):
    X, y = task()
    cfg = TMConfig(n_clauses=60, T=60, s=4.0, epochs=6, max_literals=budget, budget_select="rank", seed=3, batch_size=64)
    old = tm_mod.CHUNK_ELEMS
    try:
        if chunked:
            tm_mod.CHUNK_ELEMS = 2 * X.shape[1] * 7  # 7 clauses per chunk, so the chunked path runs
        tm = make_tm(3, X.shape[1], cfg, init="trim")
        if chunked:
            enable(tm)
        tm.fit(X[:2400], y[:2400], X[2400:], y[2400:])
        acc = float((tm.predict(torch.as_tensor(X[2400:], dtype=torch.uint8, device="cuda")).cpu().numpy() == y[2400:]).mean())
        return acc, tm.stats()["max_literals"]
    finally:
        tm_mod.CHUNK_ELEMS = old


def test_chunked_budget():
    acc_c, max_c = run(True)
    acc_u, max_u = run(False)
    assert max_c <= 6 and max_u <= 6, (max_c, max_u)
    assert acc_c > 0.8 and abs(acc_c - acc_u) < 0.08, (acc_c, acc_u)
    print("chunked", acc_c, max_c, "unchunked", acc_u, max_u)


if __name__ == "__main__":
    test_chunked_budget()
    print("ok")
