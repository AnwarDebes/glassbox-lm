"""Temperature scaling of score vectors and calibration metrics."""

from __future__ import annotations

import numpy as np
import torch


def fit_temperature(S: torch.Tensor, y, lo: float = -6.0, hi: float = 12.0, iters: int = 80) -> float:
    """tau minimising the NLL of softmax(S / tau); golden-section search on log tau."""
    y = torch.as_tensor(y, device=S.device)
    S = S.double()

    def nll(lt):
        return float(torch.nn.functional.cross_entropy(S / np.exp(lt), y))

    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2))


def ece(probs: torch.Tensor, y, n_bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    y = torch.as_tensor(y, device=probs.device)
    conf, pred = probs.max(1)
    correct = (pred == y).double()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    e = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += float(m.sum()) / n * abs(float(correct[m].mean()) - float(conf[m].double().mean()))
    return e


def nll(probs: torch.Tensor, y) -> float:
    y = torch.as_tensor(y, device=probs.device)
    return float(-torch.log(probs[torch.arange(len(y), device=probs.device), y].clamp(min=1e-12)).mean())
