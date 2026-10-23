"""Coalesced Tsetlin Machine on the GPU (torch), batch-accumulated feedback.

Starts from the torch backend of Clause-Driven-LLM (``cdllm/tm/torch_backend.py``), which
reproduced the upstream lexical Gate 0 (test 0.8293 against 0.8278), and keeps its update
rule as the default:

- clause outputs and class sums at the minibatch-start state (an empty clause fires in
  training and is silent at inference);
- target class: each clause is selected with p = (T - clip(S_y)) / 2T; a selected clause with
  weight >= 0 gets Type I feedback (fired: include literals that are 1 with p = (s-1)/s,
  exclude literals that are 0 with p = 1/s; not fired: exclude every literal with p = 1/s), a
  selected clause with weight < 0 gets Type II feedback when it fired (include each excluded
  literal that is 0); the weight moves +1 when the clause fired;
- negative class (hardest competitor by default): p = (T + clip(S_neg)) / 2T, roles of the
  polarities swapped, weight -1 when the clause fired;
- the per-literal Bernoulli draws of a minibatch are summed into Binomial draws.

Additions (all off by default, so the defaults equal the upstream rule):
- ``negation=False``: only plain literals exist (clauses cannot say NOT x);
- ``max_literals``: a clause that already includes that many literals cannot include more;
- ``neg_sampling``: "hardest", "random", or "mix:q" (hardest with probability q);
- ``clause_drop``: each clause is left out of a minibatch with this probability;
- ``half``: the 0/1 matrix products run in float16 (exact: every product is a count of
  non-negative integers no larger than the number of literals, and only "== 0" or counts
  up to the batch size are read; see tests/test_tm.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


# Above this many (clause, literal) pairs the automaton update runs in clause chunks, so that the
# float32 count tensors fit in GPU memory when several jobs share the card. Same update rule and
# the same distribution of every draw; the random stream differs from the unchunked code, so a
# machine trained above the threshold is not bit-identical to one trained without chunks.
CHUNK_ELEMS = 50_000_000


@dataclass
class TMConfig:
    n_clauses: int = 4000
    T: int = 4000
    s: float = 8.0
    n_states: int = 256
    epochs: int = 40
    batch_size: int = 128
    neg_sampling: str = "hardest"
    negation: bool = True
    max_literals: int = 0  # 0 = no budget
    budget_mode: str = "exact"  # "exact": crossings into include capped per step; "tmu": as tmu's
    # max_included_literals, a clause with more than the budget included gets Type Ib (erosion only)
    # instead of Type Ia when it fired (Type II unchanged)
    max_negated: int = 0  # > 0: a clause may hold at most this many negated literals (exact, like the exact budget)
    budget_select: str = "threshold"  # how the exact budget and the negated cap pick the k crossings they keep:
    # "threshold" (every run up to 2026-10-16 11:30 UTC): keys >= the k-th largest key, which keeps k + 1 when
    # two float32 keys tie at the k-th place (rare; it let a few clauses end one literal over the budget);
    # "rank": exactly the k crossings with the largest keys, ties broken by position.
    clause_drop: float = 0.0
    half: bool = True
    seed: int = 0
    eval_every: int = 2


def select_k(keys: torch.Tensor, cr: torch.Tensor, k: torch.Tensor, mode: str = "threshold") -> torch.Tensor:
    """Which of the crossing literals ``cr`` [r, m] each row keeps, at most ``k`` [r] of them, by
    random ``keys`` (-1 where not crossing). ``threshold`` is the original rule; ``rank`` is exact."""
    if mode == "threshold":
        srt = keys.sort(dim=1, descending=True).values
        thr = srt.gather(1, (k - 1).clamp(min=0).unsqueeze(1))
        return (keys >= thr) & (k > 0).unsqueeze(1) & cr
    if mode == "rank":
        order = keys.argsort(dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(keys.shape[1], device=keys.device).expand_as(order).contiguous())
        return (rank < k.unsqueeze(1)) & cr
    raise ValueError(mode)


class CoTM:
    def __init__(self, n_classes: int, n_features: int, cfg: TMConfig, device: str = "cuda"):
        self.cfg = cfg
        self.C = int(n_classes)
        self.f = int(n_features)
        self.m = 2 * self.f if cfg.negation else self.f
        self.dev = torch.device(device)
        self.mm_dtype = torch.float16 if (cfg.half and self.dev.type == "cuda") else torch.float32
        rng = np.random.default_rng(cfg.seed)
        # Same initialisation as the upstream numba and torch code: automata at the decision
        # boundary (half include, half exclude), weights +1 or -1.
        ta = cfg.n_states + rng.integers(0, 2, size=(cfg.n_clauses, self.m))
        w = 2 * rng.integers(0, 2, size=(cfg.n_clauses, self.C)) - 1
        self.ta = torch.as_tensor(ta, dtype=torch.int32, device=self.dev)
        self.w = torch.as_tensor(w, dtype=torch.int32, device=self.dev)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(cfg.seed))
        self.history: list[dict] = []
        self.best_val = float("nan")
        self.train_seconds = 0.0

    # ----- literals and clause outputs -------------------------------------------------
    def _literals(self, Xb: torch.Tensor) -> torch.Tensor:
        Xf = Xb.to(self.mm_dtype)
        if self.cfg.negation:
            return torch.cat([Xf, 1 - Xf], dim=1)
        return Xf

    def _incl(self) -> torch.Tensor:
        return (self.ta > self.cfg.n_states).to(self.mm_dtype)

    def clause_outputs(self, X: torch.Tensor, train_rule: bool = False, bs: int = 4096) -> torch.Tensor:
        """Clause outputs [n, N] as bool. ``X`` is a uint8 tensor on the device."""
        incl = self._incl()
        has = incl.sum(1) > 0
        out = []
        for i in range(0, X.shape[0], bs):
            L = self._literals(X[i : i + bs])
            notfire = (1 - L) @ incl.T
            co = notfire == 0
            if not train_rule:
                co = co & has.unsqueeze(0)
            out.append(co)
        return torch.cat(out)

    def class_sums(self, X: torch.Tensor, bs: int = 4096) -> torch.Tensor:
        wf = self.w.float()
        out = []
        for i in range(0, X.shape[0], bs):
            co = self.clause_outputs(X[i : i + bs], bs=bs)
            out.append(co.float() @ wf)
        return torch.cat(out)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.class_sums(X).argmax(1)

    # ----- training ---------------------------------------------------------------------
    def _neg_classes(self, S: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        b = yb.shape[0]
        mode = self.cfg.neg_sampling
        rand = torch.randint(0, self.C - 1, (b,), generator=self.gen, device=self.dev)
        rand = rand + (rand >= yb).long()  # uniform over the other classes
        if mode == "random":
            return rand
        masked = S.clone()
        masked[torch.arange(b, device=self.dev), yb] = -1e30
        hard = masked.argmax(1)
        if mode == "hardest":
            return hard
        if mode.startswith("mix:"):
            q = float(mode.split(":")[1])
            pick = torch.rand(b, generator=self.gen, device=self.dev) < q
            return torch.where(pick, hard, rand)
        raise ValueError(mode)

    def _step(self, Xb: torch.Tensor, yb: torch.Tensor) -> None:
        """Supervised step: positive feedback for the label, negative for one other class."""
        self.feedback(Xb, yb, None, None, None)

    def feedback(self, Xb: torch.Tensor, pos: torch.Tensor, pos_mask, neg, neg_mask, p_pos=None, p_neg=None) -> None:
        """General feedback on a minibatch.

        ``pos`` [b]: class that gets target-class feedback where ``pos_mask`` is True (all if None).
        ``neg`` [b]: class that gets negative-class feedback where ``neg_mask`` is True (all if None).
        ``neg`` None: the negative class is drawn by ``neg_sampling`` among the classes other than
        ``pos`` (the supervised rule). Used by the bandit learner, where a reward of 0 only says
        that the chosen action was wrong (negative feedback for it, no target class).
        """
        cfg = self.cfg
        T, n_states = cfg.T, cfg.n_states
        b = Xb.shape[0]
        N = cfg.n_clauses
        L = self._literals(Xb)  # [b, m]
        incl = self._incl()
        co = ((1 - L) @ incl.T) == 0  # [b, N], train rule: empty clause fires
        if cfg.clause_drop > 0:
            keep = torch.rand(N, generator=self.gen, device=self.dev) >= cfg.clause_drop
            co = co & keep.unsqueeze(0)
        else:
            keep = None
        S = co.float() @ self.w.float()  # [b, C]
        yb = pos
        if neg is None:
            yn = self._neg_classes(S, yb)
        elif bool((neg < 0).any()):  # -1 = draw by the negative-sampling rule
            yn = torch.where(neg >= 0, neg, self._neg_classes(S, yb))
        else:
            yn = neg
        ar = torch.arange(b, device=self.dev)
        Sp = S[ar, yb].clamp(-T, T)
        Sn = S[ar, yn].clamp(-T, T)
        # explicit probabilities (regression / TD use) replace the margin rule
        p_pos = (T - Sp) / (2.0 * T) if p_pos is None else p_pos
        p_neg = (T + Sn) / (2.0 * T) if p_neg is None else p_neg
        if pos_mask is not None:
            p_pos = torch.where(pos_mask, p_pos, torch.zeros_like(p_pos))
        if neg_mask is not None:
            p_neg = torch.where(neg_mask, p_neg, torch.zeros_like(p_neg))
        Wt = self.w[:, yb].T  # [b, N]
        Wn = self.w[:, yn].T
        sel_pos = torch.rand((b, N), generator=self.gen, device=self.dev) < p_pos.unsqueeze(1)
        sel_neg = torch.rand((b, N), generator=self.gen, device=self.dev) < p_neg.unsqueeze(1)
        if keep is not None:
            sel_pos &= keep.unsqueeze(0)
            sel_neg &= keep.unsqueeze(0)
        t1 = (sel_pos & (Wt >= 0)) | (sel_neg & (Wn < 0))
        g1 = t1 & co
        g1nf = t1 & ~co
        g2 = ((sel_pos & (Wt < 0)) | (sel_neg & (Wn > 0))) & co

        # weights
        inc_t = (sel_pos & co).float()
        inc_n = (sel_neg & co).float()
        dW = torch.zeros((N, self.C), device=self.dev)
        dW.index_add_(1, yb, inc_t.T)
        dW.index_add_(1, yn, -inc_n.T)
        self.w += dW.to(torch.int32)

        # automata
        if N * self.m > CHUNK_ELEMS:
            self._automata_chunked(L, g1, g1nf, g2)
            return
        dt = self.mm_dtype
        g1f = g1.to(dt)
        notL = 1 - L
        cnt_inc = (g1f.T @ L).float()  # Type I, fired, literal 1
        cnt_dec = (g1f.T @ notL).float() + g1nf.float().sum(0).unsqueeze(1)
        cnt_t2 = (g2.to(dt).T @ notL).float()
        excluded = self.ta <= n_states
        if cfg.max_literals > 0 and cfg.budget_mode == "tmu":
            over = ((~excluded).sum(1) > cfg.max_literals).unsqueeze(1)  # tmu: Type Ia only if included <= budget
            n_t1 = (g1.float().sum(0) + g1nf.float().sum(0)).unsqueeze(1)  # every Type I event of the clause
            cnt_inc = torch.where(over, torch.zeros_like(cnt_inc), cnt_inc)
            cnt_dec = torch.where(over, n_t1.expand_as(cnt_dec), cnt_dec)
        inc = torch.binomial(cnt_inc, torch.full_like(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
        dec = torch.binomial(cnt_dec, torch.full_like(cnt_dec, 1.0 / cfg.s), generator=self.gen)
        t2 = cnt_t2 * excluded.float()
        new = self.ta + (inc - dec + t2).to(torch.int32)
        if cfg.max_literals > 0 and cfg.budget_mode == "exact":
            # exact budget: a clause may gain at most (budget - included) literals in this step;
            # when more literals would cross into "include", a random subset of that size does.
            n_incl = (~excluded).sum(1)
            crossing = excluded & (new > n_states)
            n_cross = crossing.sum(1)
            allowed = (cfg.max_literals - n_incl).clamp(min=0)
            over = n_cross > allowed
            if bool(over.any()):
                rows = over.nonzero().squeeze(1)
                cr = crossing[rows]
                keys = torch.rand(cr.shape, generator=self.gen, device=self.dev).masked_fill(~cr, -1.0)
                keep = select_k(keys, cr, allowed[rows], cfg.budget_select)
                block = cr & ~keep
                sub = new[rows]
                sub[block] = n_states
                new[rows] = sub
        if cfg.max_negated > 0 and cfg.negation:
            # cap on negated literals: crossings into include among the negated half are limited to
            # (cap - negated already included); a random subset of the crossing negated literals is kept
            f = self.f
            exc_neg = excluded[:, f:]
            n_neg = (~exc_neg).sum(1)
            cross = exc_neg & (new[:, f:] > n_states)
            allowed = (cfg.max_negated - n_neg).clamp(min=0)
            over = cross.sum(1) > allowed
            if bool(over.any()):
                rows = over.nonzero().squeeze(1)
                cr = cross[rows]
                keys = torch.rand(cr.shape, generator=self.gen, device=self.dev).masked_fill(~cr, -1.0)
                keep = select_k(keys, cr, allowed[rows], cfg.budget_select)
                block = cr & ~keep
                sub = new[rows, f:]
                sub[block] = n_states
                new[rows, f:] = sub
        if cfg.max_literals > 0 and cfg.budget_mode == "block_full":
            # the first budget code of this study (used by the exploratory run e2_lex_ml16): a clause
            # that already includes the budget gains no literal in this step; one below it may
            # cross freely (so it can overshoot within the step)
            n_incl = (~excluded).sum(1)
            full = (n_incl >= cfg.max_literals).unsqueeze(1)
            crossing = excluded & (new > n_states)
            new = torch.where(full & crossing, torch.full_like(new, n_states), new)
        self.ta = new.clamp_(1, 2 * n_states)

    def _automata_chunked(self, L, g1, g1nf, g2) -> None:
        cfg = self.cfg
        n_states = cfg.n_states
        N = cfg.n_clauses
        dt = self.mm_dtype
        notL = 1 - L
        cs = max(1, CHUNK_ELEMS // self.m)
        for a in range(0, N, cs):
            b = min(N, a + cs)
            ta = self.ta[a:b]
            g1c, g1nfc, g2c = g1[:, a:b], g1nf[:, a:b], g2[:, a:b]
            cnt_inc = (g1c.to(dt).T @ L).float()
            cnt_dec = (g1c.to(dt).T @ notL).float() + g1nfc.float().sum(0).unsqueeze(1)
            cnt_t2 = (g2c.to(dt).T @ notL).float()
            excluded = ta <= n_states
            if cfg.max_literals > 0 and cfg.budget_mode == "tmu":
                over = ((~excluded).sum(1) > cfg.max_literals).unsqueeze(1)
                n_t1 = (g1c.float().sum(0) + g1nfc.float().sum(0)).unsqueeze(1)
                cnt_inc = torch.where(over, torch.zeros_like(cnt_inc), cnt_inc)
                cnt_dec = torch.where(over, n_t1.expand_as(cnt_dec), cnt_dec)
            inc = torch.binomial(cnt_inc, torch.full_like(cnt_inc, (cfg.s - 1.0) / cfg.s), generator=self.gen)
            del cnt_inc
            dec = torch.binomial(cnt_dec, torch.full_like(cnt_dec, 1.0 / cfg.s), generator=self.gen)
            del cnt_dec
            new = ta + (inc - dec + cnt_t2 * excluded.float()).to(torch.int32)
            del inc, dec, cnt_t2
            if cfg.max_literals > 0 and cfg.budget_mode in ("exact", "block_full"):
                raise NotImplementedError("literal budgets are not supported in the chunked update")
            self.ta[a:b] = new.clamp_(1, 2 * n_states)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, log=None) -> "CoTM":
        cfg = self.cfg
        t0 = time.time()
        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.uint8, device=self.dev)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.dev)
        Xv = torch.as_tensor(np.ascontiguousarray(X_val), dtype=torch.uint8, device=self.dev) if X_val is not None else None
        yv = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=self.dev) if y_val is not None else None
        n = Xt.shape[0]
        best = -1.0
        best_state = None
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n, generator=self.gen, device=self.dev)
            for start in range(0, n, cfg.batch_size):
                bidx = perm[start : start + cfg.batch_size]
                self._step(Xt[bidx], yt[bidx])
            if Xv is not None and (epoch % cfg.eval_every == cfg.eval_every - 1 or epoch == cfg.epochs - 1):
                acc = float((self.predict(Xv) == yv).float().mean())
                self.history.append({"epoch": epoch + 1, "val_acc": acc, "t": round(time.time() - t0, 1)})
                if log:
                    log(f"epoch {epoch + 1} val {acc:.4f} ({time.time() - t0:.0f}s)")
                if acc > best:
                    best = acc
                    best_state = (self.ta.clone(), self.w.clone(), epoch + 1)
        if best_state is not None:
            self.ta, self.w, self.best_epoch = best_state
            self.best_val = best
        torch.cuda.synchronize()
        self.train_seconds = time.time() - t0
        return self

    # ----- inspection -------------------------------------------------------------------
    def stats(self) -> dict:
        A = (self.ta > self.cfg.n_states)
        lens = A.sum(1).cpu().numpy()
        out = {
            "n_clauses": int(A.shape[0]),
            "empty": int((lens == 0).sum()),
            "median_literals": float(np.median(lens)),
            "mean_literals": float(lens.mean()),
            "max_literals": int(lens.max()),
            "p95_literals": float(np.percentile(lens, 95)),
        }
        if self.cfg.negation:
            pos = A[:, : self.f].sum(1).cpu().numpy()
            out["no_positive_literal"] = int(((pos == 0) & (lens > 0)).sum())
            out["mean_positive_literals"] = float(pos.mean())
        return out

    def state(self) -> dict:
        return {
            "ta": self.ta.cpu().numpy().astype(np.int16),
            "w": self.w.cpu().numpy(),
            "cfg": json.dumps(asdict(self.cfg)),
            "C": self.C,
            "f": self.f,
        }
