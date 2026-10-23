"""The exact literal budget in the clause-chunked automaton update. A new module, so `tm.py` and every frozen run stay as they are.

`tm.py` updates the automata in clause chunks when clauses x literals exceed ``CHUNK_ELEMS`` (so
that jobs sharing the GPU fit), and that path raises NotImplementedError for a literal budget. The
budget is a rule per clause, so it applies chunk by chunk unchanged: ``enable(tm)`` binds a chunked
update that runs, inside each chunk, the same exact-budget block as the unchunked update (at most
``max_literals - included`` literals may cross into "include" per step, a random subset when more
would). Only ``budget_mode="exact"`` and no negated cap. The random stream differs from the
unchunked code (as for every chunked run). Added 2026-10-17 after R18 (e50: 16,000 clauses on STAR).
"""

from __future__ import annotations

import types

import torch

import dl.tm as tm_mod
from dl.tm import CoTM, select_k


def _automata_chunked_budget(self: CoTM, L, g1, g1nf, g2) -> None:
    cfg = self.cfg
    if cfg.max_literals > 0 and cfg.budget_mode != "exact":
        raise NotImplementedError("only the exact budget in the chunked update")
    if cfg.max_negated > 0:
        raise NotImplementedError("no negated cap in the chunked update")
    n_states = cfg.n_states
    N = cfg.n_clauses
    dt = self.mm_dtype
    notL = 1 - L
    cs = max(1, tm_mod.CHUNK_ELEMS // self.m)
    for a in range(0, N, cs):
        b = min(N, a + cs)
        ta = self.ta[a:b]
        g1c, g1nfc, g2c = g1[:, a:b], g1nf[:, a:b], g2[:, a:b]
        cnt_inc = (g1c.to(dt).T @ L).float()
        cnt_dec = (g1c.to(dt).T @ notL).float() + g1nfc.float().sum(0).unsqueeze(1)
        cnt_t2 = (g2c.to(dt).T @ notL).float()
        excluded = ta <= n_states
        inc = torch.binomial(cnt_inc, torch.full_like(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
        del cnt_inc
        dec = torch.binomial(cnt_dec, torch.full_like(cnt_dec, 1.0 / cfg.s), generator=self.gen)
        del cnt_dec
        new = ta + (inc - dec + cnt_t2 * excluded.float()).to(torch.int32)
        del inc, dec, cnt_t2
        if cfg.max_literals > 0:
            n_incl = (~excluded).sum(1)
            crossing = excluded & (new > n_states)
            allowed = (cfg.max_literals - n_incl).clamp(min=0)
            over = crossing.sum(1) > allowed
            if bool(over.any()):
                rows = over.nonzero().squeeze(1)
                cr = crossing[rows]
                keys = torch.rand(cr.shape, generator=self.gen, device=self.dev).masked_fill(~cr, -1.0)
                keep = select_k(keys, cr, allowed[rows], cfg.budget_select)
                sub = new[rows]
                sub[cr & ~keep] = n_states
                new[rows] = sub
        self.ta[a:b] = new.clamp_(1, 2 * n_states)


def enable(tm: CoTM) -> CoTM:
    """Make this machine's chunked update honour the exact literal budget."""
    tm._automata_chunked = types.MethodType(_automata_chunked_budget, tm)
    return tm
