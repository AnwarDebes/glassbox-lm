"""The TM run loop of A-Struct: dl CoTM (glassbox-lm/studies/decision_layer/src/dl/tm.py, read-only), made with
dl.tm_init.make_tm(init "half"), trained by G1's loop (glassbox-lm/studies/text_understanding/linec/g1_tm.py): every epoch a random
permutation from the machine's own generator, minibatches of 128 through CoTM._step, no best-state restore.
fit() is not used: it calls torch.cuda.synchronize(), so this loop also runs on the CPU (K3c).
Score of a row = class sum of "true" (class 1) minus class sum of "not true" (class 0).
"""

from __future__ import annotations

import time

import numpy as np

from .paths import add_study_paths

add_study_paths(dl=True)

TM_FIXED = {"n_clauses": 4000, "epochs": 10, "batch_size": 128, "neg_sampling": "hardest", "negation": True,
            "max_literals": 0, "budget_mode": "exact", "budget_select": "threshold", "max_negated": 0,
            "clause_drop": 0.0, "half": True, "eval_every": 1}


def train_tm(X: np.ndarray, y: np.ndarray, T: int, s: float, seed: int, device: str = "cpu", log=None,
             epoch_hook=None, **over):
    import torch
    from dl.tm import TMConfig
    from dl.tm_init import make_tm
    cfg = TMConfig(**{**TM_FIXED, **over, "T": int(T), "s": float(s), "seed": int(seed)})
    tm = make_tm(2, X.shape[1], cfg, "half", device=device)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8, device=tm.dev)
    yt = torch.as_tensor(np.asarray(y, np.int64), dtype=torch.long, device=tm.dev)
    n = Xt.shape[0]
    times = []
    for epoch in range(cfg.epochs):
        te = time.time()
        perm = torch.randperm(n, generator=tm.gen, device=tm.dev)
        for start in range(0, n, cfg.batch_size):
            bidx = perm[start:start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
        if tm.dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(round(time.time() - te, 2))
        if log:
            log(f"epoch {epoch + 1}/{cfg.epochs}: {times[-1]} s; {tm.stats()}")
        if epoch_hook is not None:
            epoch_hook(epoch + 1, tm)
    tm.epoch_seconds = times
    return tm


def tm_scores(tm, X: np.ndarray, bs: int = 65536) -> np.ndarray:
    import torch
    out = []
    for i in range(0, X.shape[0], bs):
        S = tm.class_sums(torch.as_tensor(np.ascontiguousarray(X[i:i + bs]), dtype=torch.uint8, device=tm.dev))
        out.append((S[:, 1] - S[:, 0]).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def clause_report(tm) -> dict:
    A = (tm.ta > tm.cfg.n_states).cpu().numpy()
    lens = A.sum(1)
    hist, edges = np.histogram(lens, bins=[0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 10**6])
    return {**tm.stats(), "length_hist": {"edges": edges.tolist(), "counts": hist.tolist()}}
