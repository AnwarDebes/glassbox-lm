"""Lane GX (exclusive GPU): experiments/cvtm_transformer_tuned.py, the H8.5 cap and rerun rules, the H9.7
learning-rate rule, the per-token scoring and the AMP training loop (CPU torch; skipped without torch).

    .venv-dl/bin/python -m pytest tests/test_cvtm_gx_transformer.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))


@pytest.fixture(scope="module")
def tt():
    pytest.importorskip("torch")
    import cvtm_transformer_tuned as tt

    return tt


def test_cap_and_rerun_rules(tt):
    assert tt.REGISTERED_POWER_W == 259.2
    assert tt.matched_cap_s(2592.0) == pytest.approx(10.0)
    assert tt.matched_cap_s(1000.0, 100.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        tt.matched_cap_s(0.0)
    assert not tt.needs_rerun(1100.0, 1000.0) and not tt.needs_rerun(900.0, 1000.0)
    assert tt.needs_rerun(1100.1, 1000.0) and tt.needs_rerun(899.9, 1000.0)


def test_lr_rule_lowest_then_smaller(tt):
    assert tt.select_lr({3e-4: 1.30, 1e-3: 1.25, 2e-3: 1.27}) == 1e-3
    assert tt.select_lr({3e-4: 1.25, 1e-3: 1.25, 2e-3: 1.26}) == 3e-4
    with pytest.raises(ValueError):
        tt.select_lr({})


def _tiny_stream(vocab: int = 50, n_docs: int = 9, seed: int = 3):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(5, 60, size=n_docs)
    tok = rng.integers(0, vocab, size=int(lengths.sum())).astype(np.uint16)
    ds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return tok, ds


def test_per_token_bits_reproduce_bits_per_byte(tt):
    import torch

    ts = tt.ts
    vocab, seq_len = 50, 16
    tok, ds = _tiny_stream(vocab)
    packed, doc = ts.pack_stream(tok, ds, vocab)
    torch.manual_seed(0)
    model = ts.SmallTransformer(vocab, 2, 32, 2, seq_len)
    dev = torch.device("cpu")
    n_bytes = 1234
    ref = ts.bits_per_byte(model, packed, doc, n_bytes, seq_len=seq_len, batch_seqs=4, device=dev)
    bits, tot, n_scored, n_double = tt.per_token_bits(model, packed, doc, seq_len=seq_len, batch_seqs=4, device=dev)
    assert n_double == 0 and n_scored == tok.shape[0] == bits.shape[0] and np.all(np.isfinite(bits))
    assert tot / math.log(2.0) / n_bytes == ref["bits_per_byte"]
    assert float(bits.sum()) / n_bytes == pytest.approx(ref["bits_per_byte"], rel=1e-6)


def test_ignore_index_loss_equals_the_kept_positions_loss():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(1)
    V = 11
    x = torch.randint(0, V + 1, (3, 9), generator=g)          # id V plays BOS
    logits = torch.randn(3, 8, V, generator=g)
    keep = x[:, 1:] < V
    a = F.cross_entropy(logits[keep], x[:, 1:][keep])
    target = x[:, 1:].masked_fill(x[:, 1:] >= V, -100)
    b = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=-100)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-7)


def test_train_amp_on_cpu_runs_and_counts_targets(tt):
    import torch

    ts = tt.ts
    vocab, seq_len = 50, 16
    tok, ds = _tiny_stream(vocab, n_docs=40)
    packed, doc = ts.pack_stream(tok, ds, vocab)
    torch.manual_seed(0)
    model = ts.SmallTransformer(vocab, 1, 16, 2, seq_len)
    t0 = time.time()
    st = tt.train_amp(model, packed, doc, seq_len=seq_len, batch_seqs=4, lr=1e-3, warmup=2, time_cap_s=0.3, seed=100,
                      device=torch.device("cpu"), log=lambda m: None, amp=True)
    assert time.time() - t0 < 30
    assert st["steps"] >= 1 and st["final_grad_scale"] is None          # AMP is CUDA-only; disabled on the CPU
    # every step predicts the real targets of 4 windows of seq_len (BOS targets excluded)
    assert 0 < st["tokens_predicted"] <= st["steps"] * 4 * (seq_len - 1)
    assert st["loss_curve"][0][0] == 1
