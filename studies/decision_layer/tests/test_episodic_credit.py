"""The Poisson-binomial credit of scripts/abcd_episodic.py against brute-force enumeration."""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abcd_episodic import Calibrator, mem_bits, pb_dist  # noqa: E402
from dl.abcd import memory_from_rows  # noqa: E402


def brute(q, m):
    K = len(q)
    dist = np.zeros(K + 1)
    post = np.zeros((K + 1, K))
    for bits in itertools.product([0, 1], repeat=K):
        if any(b and not mm for b, mm in zip(bits, m)):
            continue
        p = np.prod([(qi if b else 1 - qi) if mm else 1.0 for qi, b, mm in zip(q, bits, m)])
        g = sum(bits)
        dist[g] += p
        post[g] += p * np.array(bits)
    return dist, post / np.maximum(dist[:, None], 1e-300)


def test_pb_and_posterior():
    rng = np.random.default_rng(0)
    for _ in range(20):
        K = int(rng.integers(1, 7))
        q = rng.uniform(0.05, 0.95, size=K)
        m = rng.uniform(size=K) < 0.8
        m[0] = True
        d, post = brute(q, m)
        got = pb_dist(q[None], m[None])[0]
        assert np.allclose(got, d, atol=1e-12)
        cal = Calibrator()
        # make the calibrator return q exactly: z = logit(q), a = 0, b = 1
        z = np.log(q / (1 - q))[None]
        for G in range(int(m.sum()) + 1):
            p = cal.posterior(z, m[None], np.array([G]))[0]
            exp = np.where(m, post[G], 0.0)
            assert np.allclose(p, exp, atol=1e-9), (q, m, G, p, exp)


def test_mem_bits_match_memory_from_rows():
    rng = np.random.default_rng(1)
    B = 30
    buttons = [f"b{i}" for i in range(B)]
    for _ in range(50):
        n = int(rng.integers(0, 10))
        seq = rng.integers(0, B, size=n).tolist()
        done = torch.zeros((1, B), dtype=torch.uint8)
        last = torch.zeros((1, B), dtype=torch.uint8)
        cnt = torch.zeros(1, dtype=torch.long)
        for a in seq:
            done[0, a] = 1
            last[0] = 0
            last[0, a] = 1
            cnt[0] += 1
        ref = memory_from_rows([{"prev": [buttons[a] for a in seq]}], buttons)
        assert np.array_equal(mem_bits(done, last, cnt).numpy(), ref)


if __name__ == "__main__":
    test_mem_bits_match_memory_from_rows()
    test_pb_and_posterior()
    print("ok")
