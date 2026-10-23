"""Contextual-bandit learners for routing from reward only (the agent loop observe -> act ->
reward -> learn), and the full-information references.

Protocol shared by every learner: the stream is ``passes`` random permutations of the training
queries; in rounds of ``batch`` queries the learner picks one action per query, sees only the
reward of that action (1 if it is the correct route, else 0), then updates. Logged: cumulative
reward (online accuracy, including exploration) and the greedy accuracy on a held-out split after
every pass.

Learners:
- ``TMBandit``: Coalesced TM, one class per action, action = argmax of the clause votes with
  epsilon-greedy exploration. Reward 1: target-class feedback for the action and negative
  feedback for the hardest other action (the reward pins down the label, so this is the
  supervised update). Reward 0: negative feedback for the chosen action only.
- ``LinUCB``: disjoint LinUCB (Li et al. 2010) on the raw embedding, ridge per action.
- ``EpsLogistic``: one logistic reward model per action (online, Adam), epsilon-greedy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .tm import CoTM, TMConfig


class TMBandit:
    name = "tm"

    def __init__(self, n_actions: int, n_bits: int, cfg: TMConfig, eps: float, eps_decay: float = 0.0, device="cuda",
                 topk: int = 0):
        self.tm = CoTM(n_actions, n_bits, cfg, device=device)
        self.K = n_actions
        self.topk = topk  # > 0: exploration picks uniformly among the top-k actions instead of all
        self.boltzmann = 0.0  # > 0: Boltzmann exploration with this temperature (exploratory only)
        self.eps0, self.decay = eps, eps_decay
        self.t = 0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(cfg.seed + 7919)
        self.dev = device

    def eps(self) -> float:
        return self.eps0 / (1.0 + self.decay * self.t) if self.decay > 0 else self.eps0

    def greedy(self, X: torch.Tensor) -> torch.Tensor:
        return self.tm.class_sums(X).argmax(1)

    def act(self, X: torch.Tensor) -> torch.Tensor:
        S = self.tm.class_sums(X)
        # ties (e.g. all-zero sums at the start) broken at random
        noise = torch.rand(S.shape, generator=self.gen, device=self.dev) * 1e-3
        if self.boltzmann > 0:
            # exploratory option (not used by R5): sample the action from softmax(S / (tau * std of S))
            z = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6) / self.boltzmann
            return torch.multinomial(torch.softmax(z, 1), 1, generator=self.gen).squeeze(1)
        a = (S + noise).argmax(1)
        explore = torch.rand(X.shape[0], generator=self.gen, device=self.dev) < self.eps()
        rnd = _explore_actions(S + noise, self.K, self.topk, self.gen)
        return torch.where(explore, rnd, a)

    def learn(self, X: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> None:
        hit = r > 0.5
        neg = torch.where(hit, torch.full_like(a, -1), a)
        self.tm.feedback(X, a, hit, neg, None)
        self.t += X.shape[0]


def _explore_actions(S: torch.Tensor, K: int, topk: int, gen: torch.Generator) -> torch.Tensor:
    """Uniform over all K actions (topk = 0) or over the top-k scored actions."""
    b = S.shape[0]
    if topk <= 0 or topk >= K:
        return torch.randint(0, K, (b,), generator=gen, device=S.device)
    top = S.topk(topk, dim=1).indices
    j = torch.randint(0, topk, (b,), generator=gen, device=S.device)
    return top[torch.arange(b, device=S.device), j]


class LinUCB:
    name = "linucb"

    def __init__(self, n_actions: int, dim: int, alpha: float, lam: float = 1.0, device="cuda"):
        self.K, self.d, self.alpha = n_actions, dim, alpha
        self.A = (torch.eye(dim, device=device, dtype=torch.float64) * lam).repeat(n_actions, 1, 1)
        self.Ainv = torch.eye(dim, device=device).repeat(n_actions, 1, 1) / lam
        self.bvec = torch.zeros((n_actions, dim), device=device)
        self.dev = device

    def _theta(self):
        return torch.einsum("kij,kj->ki", self.Ainv, self.bvec)

    def greedy(self, X):
        return (X @ self._theta().T).argmax(1)

    def act(self, X):
        th = self._theta()
        mean = X @ th.T  # [b, K]
        # x^T Ainv_k x for every k
        var = torch.einsum("bi,kij,bj->bk", X, self.Ainv, X)
        return (mean + self.alpha * var.clamp(min=0).sqrt()).argmax(1)

    def learn(self, X, a, r):
        # A_k += sum of x x^T over the round's queries routed to k; the inverses of the touched
        # arms are recomputed exactly (batched, float64). Same estimator as sequential
        # Sherman-Morrison updates, without a Python loop over queries.
        Xd = X.double()
        self.A.index_add_(0, a, Xd.unsqueeze(2) * Xd.unsqueeze(1))
        self.bvec.index_add_(0, a, r.float().unsqueeze(1) * X)
        touched = torch.unique(a)
        self.Ainv[touched] = torch.linalg.inv(self.A[touched]).float()


class EpsLogistic:
    name = "eps_logistic"

    def __init__(self, n_actions: int, dim: int, eps: float, lr: float = 0.05, seed: int = 0, device="cuda", topk: int = 0):
        self.topk = topk
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = torch.zeros((dim, n_actions), device=device, requires_grad=True)
        self.b = torch.full((n_actions,), -math.log(n_actions), device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.W, self.b], lr=lr)
        self.K, self.eps = n_actions, eps
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 31)
        self.dev = device

    def greedy(self, X):
        with torch.no_grad():
            return (X @ self.W + self.b).argmax(1)

    def act(self, X):
        with torch.no_grad():
            S = X @ self.W + self.b
        a = S.argmax(1)
        explore = torch.rand(X.shape[0], generator=self.gen, device=self.dev) < self.eps
        rnd = _explore_actions(S, self.K, self.topk, self.gen)
        return torch.where(explore, rnd, a)

    def learn(self, X, a, r):
        self.opt.zero_grad()
        z = (X @ self.W + self.b).gather(1, a.view(-1, 1)).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, r.float())
        loss.backward()
        self.opt.step()


def run_stream(learner, Xs, ys, Xeval, yeval, passes: int, batch: int, seed: int, X_for_eval=None, log=None,
               replay: int = 0):
    """Feed ``passes`` permutations of (Xs, ys) in rounds of ``batch``. Returns a record.

    ``replay`` > 0: after each round, ``replay`` extra updates on minibatches drawn uniformly from
    every (query, action, reward) logged so far (the same logged interactions, no new labels).
    """
    dev = Xs.device
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    n = Xs.shape[0]
    total, rounds = 0.0, 0
    curve = []
    log_i, log_a, log_r = [], [], []
    for p in range(passes):
        perm = torch.randperm(n, generator=g, device=dev)
        rew_pass = 0.0
        for st in range(0, n, batch):
            idx = perm[st : st + batch]
            X, y = Xs[idx], ys[idx]
            a = learner.act(X)
            r = (a == y).float()
            learner.learn(X, a, r)
            rew_pass += float(r.sum())
            rounds += len(idx)
            if replay > 0:
                log_i.append(idx); log_a.append(a); log_r.append(r)
                Li, La, Lr = torch.cat(log_i), torch.cat(log_a), torch.cat(log_r)
                log_i, log_a, log_r = [Li], [La], [Lr]
                for _ in range(replay):
                    j = torch.randint(0, Li.shape[0], (batch,), generator=g, device=dev)
                    learner.learn(Xs[Li[j]], La[j], Lr[j])
        total += rew_pass
        acc = float((learner.greedy(Xeval) == yeval).float().mean())
        curve.append({"pass": p + 1, "online_acc_pass": rew_pass / n, "cum_online_acc": total / rounds, "eval_acc": acc})
        if log:
            log(curve[-1])
    return {"curve": curve, "cum_online_acc": total / rounds, "final_eval_acc": curve[-1]["eval_acc"], "replay": replay,
            "best_eval_acc": max(c["eval_acc"] for c in curve)}


class LinUCBFast(LinUCB):
    """LinUCB with the same estimator as ``LinUCB``, but each round's inverse is updated by the
    Woodbury identity (A + X^T X)^-1 = A^-1 - A^-1 X^T (I + X A^-1 X^T)^-1 X A^-1, per touched arm,
    in float64, instead of inverting A from scratch: O(d^2 n) instead of O(d^3) per arm and round.
    Added 2026-10-17 for R10 (1,706 inputs); tests/test_linucb_fast.py checks it against LinUCB."""

    name = "linucb"

    def __init__(self, n_actions: int, dim: int, alpha: float, lam: float = 1.0, device="cuda"):
        super().__init__(n_actions, dim, alpha, lam, device)
        self.Ainv64 = torch.eye(dim, device=device, dtype=torch.float64).repeat(n_actions, 1, 1) / lam

    def learn(self, X, a, r):
        Xd = X.double()
        self.A.index_add_(0, a, Xd.unsqueeze(2) * Xd.unsqueeze(1))
        self.bvec.index_add_(0, a, r.float().unsqueeze(1) * X)
        for k in torch.unique(a).tolist():
            Xk = Xd[a == k]  # [n, d]
            Ai = self.Ainv64[k]
            AXt = Ai @ Xk.T  # [d, n]
            M = torch.eye(Xk.shape[0], device=X.device, dtype=torch.float64) + Xk @ AXt
            self.Ainv64[k] = Ai - AXt @ torch.linalg.solve(M, AXt.T)
            self.Ainv[k] = self.Ainv64[k].float()


class NeuralBandit:
    """A neural reward model for routing from reward only (the "NN agent" baseline of R14).

    One hidden layer (ReLU) maps the bits to one logit per action, read as the predicted chance
    that the action gets reward 1; after each round one AdamW step on the binary cross-entropy of
    the chosen actions' rewards (the same one update per round as the TM bandit). Exploration:
    Boltzmann on the standardised logits (as the TM bandit) or epsilon-greedy. Added 2026-10-17.
    """

    name = "neural"

    def __init__(self, n_actions: int, dim: int, hidden: int = 512, lr: float = 1e-3, wd: float = 1e-4,
                 boltzmann: float = 0.5, eps: float = 0.0, seed: int = 0, device="cuda"):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, n_actions)).to(device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        self.K, self.boltzmann, self.eps = n_actions, boltzmann, eps
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 31)
        self.dev = device

    def greedy(self, X):
        with torch.no_grad():
            return self.net(X.float()).argmax(1)

    def act(self, X):
        with torch.no_grad():
            S = self.net(X.float())
        if self.boltzmann > 0:
            z = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6) / self.boltzmann
            return torch.multinomial(torch.softmax(z, 1), 1, generator=self.gen).squeeze(1)
        a = S.argmax(1)
        explore = torch.rand(X.shape[0], generator=self.gen, device=self.dev) < self.eps
        return torch.where(explore, _explore_actions(S, self.K, 0, self.gen), a)

    def learn(self, X, a, r):
        z = self.net(X.float()).gather(1, a.view(-1, 1)).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, r.float())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
