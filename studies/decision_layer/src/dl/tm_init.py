"""Start-state variants for the coalesced TM. A new module, so `tm.py` and every frozen run stay as they are.

- ``half``: the start of `tm.py` itself (every automaton at the boundary, about half of all
  literals included). This is what every run before 2026-10-16 09:00 UTC used.
- ``trim``: the same start, then each clause keeps a random subset of at most ``max_literals``
  of its included literals. With the exact budget, no clause can then ever hold more than the
  budget: the budget holds from the first step, not only for literals added later. R2 missed its
  length rule by one clause of 33 literals that was left over from the half start.
- ``empty``: every automaton on the exclude side of the boundary, as tmu starts.
"""

from __future__ import annotations

import numpy as np
import torch

from dl.tm import CoTM, TMConfig

INITS = ("half", "trim", "empty")


def make_tm(n_classes: int, n_features: int, cfg: TMConfig, init: str = "half", device: str = "cuda") -> CoTM:
    tm = CoTM(n_classes, n_features, cfg, device=device)
    if init == "half":
        return tm
    n = cfg.n_states
    ta = tm.ta.cpu().numpy()
    if init == "empty":
        ta[:] = n
    elif init == "trim":
        if cfg.max_literals <= 0:
            raise ValueError("init 'trim' needs a literal budget (max_literals > 0)")
        rng = np.random.default_rng(cfg.seed + 7919)  # own stream: the weight draws of CoTM stay the same
        for j in range(ta.shape[0]):
            inc = np.flatnonzero(ta[j] > n)
            if len(inc) > cfg.max_literals:
                ta[j, rng.choice(inc, len(inc) - cfg.max_literals, replace=False)] = n
    else:
        raise ValueError(f"unknown init {init!r}; one of {INITS}")
    tm.ta = torch.as_tensor(ta, dtype=torch.int32, device=tm.dev)
    return tm
