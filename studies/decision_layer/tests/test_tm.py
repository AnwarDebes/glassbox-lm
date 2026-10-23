"""fp16 and fp32 matrix products give identical training (same seed, same draws)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dl.tm import CoTM, TMConfig  # noqa: E402


def test_half_equals_float():
    rng = np.random.default_rng(0)
    X = (rng.random((3000, 700)) < 0.3).astype(np.uint8)
    y = (X[:, :5].sum(1) + X[:, 5:9].sum(1)) % 7
    outs = []
    for half in (True, False):
        for maxlit in (0, 12):
            cfg = TMConfig(n_clauses=300, T=300, s=5.0, epochs=3, batch_size=64, half=half, seed=3, max_literals=maxlit)
            tm = CoTM(7, 700, cfg)
            tm.fit(X, y)
            outs.append((half, maxlit, tm.ta.cpu().numpy(), tm.w.cpu().numpy()))
    for maxlit in (0, 12):
        a = [o for o in outs if o[1] == maxlit]
        assert np.array_equal(a[0][2], a[1][2]) and np.array_equal(a[0][3], a[1][3]), maxlit
    # learns an easy rule: class k when marker bit k is on and bit 100+k is off
    y2 = rng.integers(0, 7, 3000)
    X2 = (rng.random((3000, 700)) < 0.2).astype(np.uint8)
    X2[:, :7] = 0
    X2[np.arange(3000), y2] = 1
    X2[np.arange(3000), 100 + y2] = 0
    cfg = TMConfig(n_clauses=300, T=300, s=5.0, epochs=10, batch_size=64, seed=3)
    tm = CoTM(7, 700, cfg).fit(X2[:2500], y2[:2500])
    acc = float((tm.predict(torch.as_tensor(X2[2500:], device="cuda")).cpu().numpy() == y2[2500:]).mean())
    print("acc", acc)
    assert acc > 0.95


if __name__ == "__main__":
    test_half_equals_float()
    print("PASS")
