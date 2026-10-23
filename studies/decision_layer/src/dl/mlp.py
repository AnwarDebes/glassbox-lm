"""A small neural decision head: one hidden layer (ReLU, dropout), Adam, best-validation epoch.

The "NN agent" baseline for the decision layer: the same inputs as the Tsetlin Machine (bits
as 0/1 floats) or the raw embeddings. Deterministic given the seed (cuBLAS aside).
"""

from __future__ import annotations

import numpy as np
import torch


def fit_mlp(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, n_classes: int, hidden: int = 512,
            dropout: float = 0.2, lr: float = 1e-3, wd: float = 1e-4, epochs: int = 60, batch: int = 256,
            patience: int = 10, seed: int = 0, device: str = "cuda") -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    Xvt = torch.as_tensor(Xv, dtype=torch.float32, device=device)
    yvt = torch.as_tensor(yv, dtype=torch.long, device=device)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, n_classes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(Xt.shape[0], generator=g, device=device)
        for a in range(0, Xt.shape[0], batch):
            idx = perm[a : a + batch]
            loss = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xvt).argmax(1) == yvt).float().mean())
        hist.append(acc)
        if acc > best:
            best, best_ep, bad = acc, ep + 1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_val": best, "best_epoch": best_ep, "epochs_run": len(hist)}


@torch.no_grad()
def mlp_scores(net: torch.nn.Module, X: np.ndarray, device: str = "cuda") -> torch.Tensor:
    return net(torch.as_tensor(X, dtype=torch.float32, device=device))
