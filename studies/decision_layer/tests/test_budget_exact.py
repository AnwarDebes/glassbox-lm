"""The exact literal budget keeps exactly k crossings with budget_select="rank", also when random keys
tie; the original "threshold" rule keeps k + 1 when two keys tie at the k-th place.

    .venv-dl/bin/python tests/test_budget_exact.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dl.tm import CoTM, TMConfig, select_k  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1. a forced tie at the k-th place
keys = torch.tensor([[0.9, 0.5, 0.5, 0.1, -1.0]], device=dev)
cr = keys >= 0
k = torch.tensor([2], device=dev)
assert int(select_k(keys, cr, k, "threshold").sum()) == 3, "the old rule keeps k + 1 on a tie"
assert int(select_k(keys, cr, k, "rank").sum()) == 2
# 2. without ties both rules agree
g = torch.Generator(device=dev)
g.manual_seed(0)
for _ in range(50):
    cr = torch.rand((64, 300), generator=g, device=dev) < 0.1
    keys = torch.rand(cr.shape, generator=g, device=dev).masked_fill(~cr, -1.0)
    k = torch.randint(0, 20, (64,), generator=g, device=dev)
    a, b = select_k(keys, cr, k, "threshold"), select_k(keys, cr, k, "rank")
    ties = any(len(torch.unique(r[r >= 0])) < int((r >= 0).sum()) for r in keys)
    if not ties:
        assert torch.equal(a, b)
    assert bool((b.sum(1) == torch.minimum(k, cr.sum(1))).all())
# 3. a short training run: no clause over the budget, ever, with the trim start and "rank"
rng = np.random.default_rng(0)
X = (rng.random((600, 120)) < 0.05).astype(np.uint8)
y = rng.integers(0, 5, 600)
for L in (4, 8):
    cfg = TMConfig(n_clauses=300, T=300, epochs=3, max_literals=L, budget_select="rank", seed=1, eval_every=1)
    tm = make_tm(5, 120, cfg, "trim", device=dev)
    for e in range(3):
        for i in range(0, 600, 64):
            tm._step(torch.as_tensor(X[i:i + 64], device=dev), torch.as_tensor(y[i:i + 64], device=dev))
            assert int((tm.ta > cfg.n_states).sum(1).max()) <= L
print("ok: exact selection with ties; equal to the old rule without ties; budget never exceeded")
