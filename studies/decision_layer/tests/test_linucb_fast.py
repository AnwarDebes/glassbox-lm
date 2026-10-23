"""LinUCBFast (Woodbury updates) equals LinUCB (fresh inverses) on a random stream."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dl.bandit import LinUCB, LinUCBFast  # noqa: E402


def test_same_choices_and_inverses():
    g = torch.Generator().manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K, d = 6, 40
    slow, fast = LinUCB(K, d, alpha=0.5, device=dev), LinUCBFast(K, d, alpha=0.5, device=dev)
    for _ in range(60):
        X = (torch.rand((32, d), generator=g) < 0.3).float().to(dev)
        y = torch.randint(0, K, (32,), generator=g).to(dev)
        a1, a2 = slow.act(X), fast.act(X)
        assert torch.equal(a1, a2)
        r = (a1 == y).float()
        slow.learn(X, a1, r)
        fast.learn(X, a2, r)
    err = (torch.linalg.inv(slow.A) - fast.Ainv64).abs().max().item()
    assert err < 1e-8, err
    assert torch.equal(slow.greedy(X), fast.greedy(X))


if __name__ == "__main__":
    test_same_choices_and_inverses()
    print("ok")
