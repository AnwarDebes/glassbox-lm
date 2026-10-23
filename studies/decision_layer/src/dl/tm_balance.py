"""Class-balanced training for the coalesced TM. A new module, so `tm.py` and every frozen run stay as they are.

Diagnosis d1 (2026-10-17, STAR dev, results/explore/star_gap.jsonl): with 187 actions the CoTM is
7 to 9 points behind LR on the less common actions and level on the common ones, and the
positive weight mass of a class grows with its train count. ``fit_balanced`` changes only which
examples an epoch feeds: n draws with replacement, an example of class c with probability
proportional to n_c ** -rho (rho 0: uniform draws; 0.5: square-root balancing; 1: every class
equally often). The update rule, the best-dev snapshot and eval_every are those of CoTM.fit.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from dl.tm import CoTM


def fit_balanced(tm: CoTM, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, rho: float = 0.5) -> CoTM:
    cfg = tm.cfg
    t0 = time.time()
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8, device=tm.dev)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=tm.dev)
    Xv = torch.as_tensor(np.ascontiguousarray(X_val), dtype=torch.uint8, device=tm.dev) if X_val is not None else None
    yv = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=tm.dev) if y_val is not None else None
    n = Xt.shape[0]
    cnt = torch.bincount(yt, minlength=tm.C).double()
    wex = cnt[yt].pow(-rho)
    wex = (wex / wex.sum()).float()
    best, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        draw = torch.multinomial(wex, n, replacement=True, generator=tm.gen)
        for start in range(0, n, cfg.batch_size):
            bidx = draw[start : start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
        if Xv is not None and (epoch % cfg.eval_every == cfg.eval_every - 1 or epoch == cfg.epochs - 1):
            acc = float((tm.predict(Xv) == yv).float().mean())
            tm.history.append({"epoch": epoch + 1, "val_acc": acc, "t": round(time.time() - t0, 1)})
            if acc > best:
                best = acc
                best_state = (tm.ta.clone(), tm.w.clone(), epoch + 1)
    if best_state is not None:
        tm.ta, tm.w, tm.best_epoch = best_state
        tm.best_val = best
    torch.cuda.synchronize()
    tm.train_seconds = time.time() - t0
    return tm
