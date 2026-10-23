"""Multinomial logistic regression on the GPU (full-batch L-BFGS), sklearn-compatible C.

Objective (same scaling as scikit-learn's LogisticRegression with penalty "l2"):
    sum_i CE(x_i, y_i) + ||W||^2 / (2 C), bias not penalised,
minimised as its mean over i (the same minimiser).
"""

from __future__ import annotations

import numpy as np
import torch


def fit_logreg(X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, n_classes: int, C: float,
               max_iter: int = 500, device: str = "cuda", tol: float = 1e-7):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n, d = X.shape
    W = torch.zeros((d, n_classes), device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * n)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def logits(X, W, b, device="cuda"):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    return X @ W + b


def accuracy(scores: torch.Tensor, y) -> float:
    y = torch.as_tensor(y, device=scores.device)
    return float((scores.argmax(1) == y).float().mean())


def auroc(is_pos: np.ndarray, score: np.ndarray) -> float:
    """AUROC with ties counted half (Mann-Whitney)."""
    from scipy.stats import rankdata

    is_pos = np.asarray(is_pos).astype(bool)
    r = rankdata(score)
    n1 = is_pos.sum()
    n0 = len(is_pos) - n1
    return float((r[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_precision(is_pos: np.ndarray, score: np.ndarray, min_precision: float = 0.95) -> float:
    """Largest recall of the positives (high score = positive) at a threshold whose precision
    is at least ``min_precision``; 0.0 if no threshold reaches it."""
    is_pos = np.asarray(is_pos).astype(bool)
    score = np.asarray(score)
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(is_pos[order])
    k = np.arange(1, len(order) + 1)
    prec = tp / k
    s_sorted = score[order]
    ends = np.r_[s_sorted[1:] != s_sorted[:-1], True]  # a threshold can only fall between distinct scores
    ok = (prec >= min_precision) & ends
    return float(tp[ok].max() / is_pos.sum()) if ok.any() else 0.0
