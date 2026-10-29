"""Prototype of exact (bitwise-identical) speedups for the decision_layer CoTM (src/dl/tm.py), as a subclass in this folder.

Nothing of the decision_layer is edited; this file only reads dl.tm. Every random draw of CoTM.feedback / _automata_chunked is kept:
the same generator, the same calls (torch.rand, torch.randint, torch.binomial, torch.rand for the budget keys) in the
same order, with the same shapes, dtypes and memory layout (contiguous) and the same input values. Only the
deterministic arithmetic around the draws changes, and it stays exact (integer counts held in float16 / float32 below
2^11 / 2^24). Switches (all OFF by default, pace 0: a bare FastCoTM(...) runs the rewritten feedback with every
switch off, i.e. the original's arithmetic and draws; the candidate settings are named in configs.py of this folder):

- ``xonly``   (T1): the automaton counts from the plain bits only. With negation, L = [X, 1 - X] and 1 - L = [1 - X, X],
               so g1^T L = [A, r1 - A], g1^T (1 - L) = [r1 - A, A] and g2^T (1 - L) = [r2 - B, B] with A = g1^T X,
               B = g2^T X [N, f] and r1, r2 the clauses' event counts. Two products of width f replace three of width 2f
               (one third of the FLOPs). The counts are the same integers.
- ``probs``   (T2): the binomial probability tensors are made once per shape and reused (same float32 values) instead of
               torch.full_like each step. ``probs_expand``: a stride-0 expanded scalar instead of a full tensor.
- ``elem``    (T3): fewer elementwise passes: the exclude mask is the complement of the include mask the step already
               computed; the Type II term is a masked select instead of a float multiply; inc - dec + t2 in place; the
               chunked update writes the automata in place (as the original's ``self.ta[a:b] = new`` does).
- ``evalonce``(E1): predict / class_sums compute the include mask and the empty-clause flags once per call, not once
               per 4096-row block.
Only the default supervised path is rewritten (no literal budget, no negated cap, no clause drop); any other setting
falls back to the original method, so the prototype can never silently differ there.
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import sys

import torch

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import dl.tm as T  # noqa: E402
from dl.tm import CoTM  # noqa: E402


class FastCoTM(CoTM):
    def __init__(self, *a, xonly=False, probs=False, probs_expand=False, elem=False, evalonce=False, pace=0, **k):
        super().__init__(*a, **k)
        self.opt = dict(xonly=xonly, probs=probs, probs_expand=probs_expand, elem=elem, evalonce=evalonce, pace=pace)
        self._pcache = {}
        self._events = []

    def _step(self, Xb, yb) -> None:
        """R1 (pace > 0, CUDA only): after each step, record a blocking-sync CUDA event and wait for the one recorded
        ``pace`` steps earlier, so the host thread sleeps instead of spinning while the GPU works (measured: the
        unpaced job's main thread spins at 1.00 core, 0 voluntary switches in 10 s). Only host-side timing changes;
        no tensor, draw or order of kernels does. A no-op on the CPU."""
        super()._step(Xb, yb)
        if self.opt["pace"] and self.dev.type == "cuda":
            ev = torch.cuda.Event(blocking=True)
            ev.record()
            self._events.append(ev)
            if len(self._events) > self.opt["pace"]:
                self._events.pop(0).synchronize()

    # ----- helpers --------------------------------------------------------------------------------------------
    def _plain(self) -> bool:
        c = self.cfg
        return c.max_literals <= 0 and c.max_negated <= 0 and c.clause_drop <= 0

    def _prob(self, like: torch.Tensor, p: float) -> torch.Tensor:
        if not self.opt["probs"]:
            return torch.full_like(like, p)
        if self.opt["probs_expand"]:
            key = ("x", p, like.dtype, like.device)
            if key not in self._pcache:
                self._pcache[key] = torch.full((), p, dtype=like.dtype, device=like.device)
            return self._pcache[key].expand(like.shape)
        key = (tuple(like.shape), p, like.dtype, like.device)
        t = self._pcache.get(key)
        if t is None:
            t = self._pcache[key] = torch.full_like(like, p)
        return t

    def _counts(self, Xd: torch.Tensor, L: torch.Tensor, g1, g1nf, g2):
        """(cnt_inc, cnt_dec, cnt_t2) for a clause block: float32 [n, m], contiguous, the original's integers."""
        dt = self.mm_dtype
        f = self.f
        if not self.opt["xonly"]:
            notL = 1 - L
            g1f = g1.to(dt)
            cnt_inc = (g1f.T @ L).float()
            cnt_dec = (g1f.T @ notL).float() + g1nf.float().sum(0).unsqueeze(1)
            cnt_t2 = (g2.to(dt).T @ notL).float()
            return cnt_inc, cnt_dec, cnt_t2
        n = g1.shape[1]
        A = g1.to(dt).T @ Xd  # [n, f]: selected Type I, fired, plain literal 1
        B = g2.to(dt).T @ Xd  # [n, f]: Type II, fired, plain literal 1
        r1 = g1.sum(0, dtype=torch.float32).unsqueeze(1)  # [n, 1]
        r2 = g2.sum(0, dtype=torch.float32).unsqueeze(1)
        nf = g1nf.sum(0, dtype=torch.float32).unsqueeze(1)
        if self.cfg.negation:
            # every half written once, straight from A / B (fp16 on the GPU) into the float32 count tensors
            cnt_inc = torch.empty((n, 2 * f), dtype=torch.float32, device=self.dev)
            cnt_dec = torch.empty_like(cnt_inc)
            cnt_t2 = torch.empty_like(cnt_inc)
            cnt_inc[:, :f] = A  # g1^T X
            torch.sub(r1, A, out=cnt_inc[:, f:])  # g1^T (1 - X) = r1 - A
            torch.sub(r1 + nf, A, out=cnt_dec[:, :f])  # g1^T (1 - X) + nf (same integers as (r1 - A) + nf)
            torch.add(A, nf, out=cnt_dec[:, f:])  # g1^T X + nf
            torch.sub(r2, B, out=cnt_t2[:, :f])  # g2^T (1 - X)
            cnt_t2[:, f:] = B  # g2^T X
            return cnt_inc, cnt_dec, cnt_t2
        Af = A.float()
        cnt_inc = Af
        cnt_dec = (r1 - Af) + nf
        cnt_t2 = r2 - B.float()
        return cnt_inc, cnt_dec, cnt_t2

    # ----- training -----------------------------------------------------------------------------------------------
    def feedback(self, Xb, pos, pos_mask, neg, neg_mask, p_pos=None, p_neg=None) -> None:
        if not self._plain():
            return super().feedback(Xb, pos, pos_mask, neg, neg_mask, p_pos, p_neg)
        cfg = self.cfg
        T_, n_states = cfg.T, cfg.n_states
        b = Xb.shape[0]
        N = cfg.n_clauses
        dt = self.mm_dtype
        Xd = Xb.to(dt)
        L = torch.cat([Xd, 1 - Xd], dim=1) if cfg.negation else Xd
        notL = 1 - L
        incl_b = self.ta > n_states  # bool [N, m]
        incl = incl_b.to(dt)
        co = (notL @ incl.T) == 0  # [b, N], train rule: empty clause fires
        S = co.float() @ self.w.float()  # [b, C]
        yb = pos
        if neg is None:
            yn = self._neg_classes(S, yb)
        elif bool((neg < 0).any()):
            yn = torch.where(neg >= 0, neg, self._neg_classes(S, yb))
        else:
            yn = neg
        ar = torch.arange(b, device=self.dev)
        Sp = S[ar, yb].clamp(-T_, T_)
        Sn = S[ar, yn].clamp(-T_, T_)
        p_pos = (T_ - Sp) / (2.0 * T_) if p_pos is None else p_pos
        p_neg = (T_ + Sn) / (2.0 * T_) if p_neg is None else p_neg
        if pos_mask is not None:
            p_pos = torch.where(pos_mask, p_pos, torch.zeros_like(p_pos))
        if neg_mask is not None:
            p_neg = torch.where(neg_mask, p_neg, torch.zeros_like(p_neg))
        Wt = self.w[:, yb].T
        Wn = self.w[:, yn].T
        sel_pos = torch.rand((b, N), generator=self.gen, device=self.dev) < p_pos.unsqueeze(1)
        sel_neg = torch.rand((b, N), generator=self.gen, device=self.dev) < p_neg.unsqueeze(1)
        t1 = (sel_pos & (Wt >= 0)) | (sel_neg & (Wn < 0))
        g1 = t1 & co
        g1nf = t1 & ~co
        g2 = ((sel_pos & (Wt < 0)) | (sel_neg & (Wn > 0))) & co
        inc_t = (sel_pos & co).float()
        inc_n = (sel_neg & co).float()
        dW = torch.zeros((N, self.C), device=self.dev)
        dW.index_add_(1, yb, inc_t.T)
        dW.index_add_(1, yn, -inc_n.T)
        self.w += dW.to(torch.int32)
        if N * self.m > T.CHUNK_ELEMS:
            self._chunked(Xd, L, g1, g1nf, g2, incl_b)
            return
        cnt_inc, cnt_dec, cnt_t2 = self._counts(Xd, L, g1, g1nf, g2)
        inc = torch.binomial(cnt_inc, self._prob(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
        dec = torch.binomial(cnt_dec, self._prob(cnt_dec, 1.0 / cfg.s), generator=self.gen)
        if self.opt["elem"]:
            inc.sub_(dec)
            inc.add_(cnt_t2.masked_fill_(incl_b, 0.0))  # Type II only where excluded
            new = self.ta + inc.to(torch.int32)
        else:
            excluded = self.ta <= n_states
            new = self.ta + (inc - dec + cnt_t2 * excluded.float()).to(torch.int32)
        self.ta = new.clamp_(1, 2 * n_states)

    def _chunked(self, Xd, L, g1, g1nf, g2, incl_b) -> None:
        cfg = self.cfg
        n_states = cfg.n_states
        N = cfg.n_clauses
        cs = max(1, T.CHUNK_ELEMS // self.m)
        for a in range(0, N, cs):
            b = min(N, a + cs)
            ta = self.ta[a:b]
            cnt_inc, cnt_dec, cnt_t2 = self._counts(Xd, L, g1[:, a:b], g1nf[:, a:b], g2[:, a:b])
            inc = torch.binomial(cnt_inc, self._prob(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
            del cnt_inc
            dec = torch.binomial(cnt_dec, self._prob(cnt_dec, 1.0 / cfg.s), generator=self.gen)
            del cnt_dec
            if self.opt["elem"]:
                inc.sub_(dec)
                inc.add_(cnt_t2.masked_fill_(incl_b[a:b], 0.0))
                ta.add_(inc.to(torch.int32)).clamp_(1, 2 * n_states)  # a view of self.ta: in place
            else:
                excluded = ta <= n_states
                new = ta + (inc - dec + cnt_t2 * excluded.float()).to(torch.int32)
                self.ta[a:b] = new.clamp_(1, 2 * n_states)

    # ----- evaluation -------------------------------------------------------------------------------------------
    def class_sums(self, X: torch.Tensor, bs: int = 4096) -> torch.Tensor:
        if not self.opt["evalonce"]:
            return super().class_sums(X, bs)
        incl = self._incl()
        has = incl.sum(1) > 0
        wf = self.w.float()
        out = []
        for i in range(0, X.shape[0], bs):
            L = self._literals(X[i : i + bs])
            co = (((1 - L) @ incl.T) == 0) & has.unsqueeze(0)
            out.append(co.float() @ wf)
        return torch.cat(out)
