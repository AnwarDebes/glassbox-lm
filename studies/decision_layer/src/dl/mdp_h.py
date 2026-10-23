"""Clarify-or-route over H views (R8): the query is revealed in H growing prefixes.

Episode: one query. At step t (0 .. H-1) the agent sees a fixed router's class sums on the first
ceil((t+1) * len / H) words. Actions: ROUTE (commit to the router's top intent: +1 if right, -1 if
wrong or if the query is out of scope; ends), ASK (-c now, the next prefix is revealed; not
allowed at the last step), ESCALATE (hand over to a human: 0; ends). The value of asking at step 0
shows up to H-1 steps later, so a learner needs credit assignment over the whole episode.

Observation bits, each with a readable name:
- the step as a thermometer ("step >= 1", ...);
- thermometers of the router's top class sum and margin on the current prefix;
- memory literals (dropped in the no-memory ablation): the same thermometers on the previous
  prefix, "the top intent is the same as on the previous prefix", and "the top intent is the same
  on the last three prefixes".
"""

from __future__ import annotations

import torch

ROUTE, ASK, ESCALATE = 0, 1, 2
ACTIONS = ["ROUTE", "ASK", "ESCALATE"]


def prefix_text(t: str, k: int, H: int) -> str:
    """The first ceil(k * len / H) words (at least one) of ``t``, for k = 1 .. H."""
    w = t.split()
    return " ".join(w[: max(1, (k * len(w) + H - 1) // H)])


class EpisodesH:
    """Router outputs on the H prefixes of every query of one split.

    ``top`` [n, H]: router's top intent; ``ok`` [n, H]: it is right (False for out of scope);
    ``conf`` [n, H, 2]: (top class sum, margin); ``agree1``/``agree2`` [n, H]: the top intent equals
    the one of the previous prefix / of the two previous prefixes (False where there is none).
    """

    def __init__(self, S_views: list[torch.Tensor], y, is_oos):
        dev = S_views[0].device
        self.H = len(S_views)
        self.n = S_views[0].shape[0]
        self.is_oos = torch.as_tensor(is_oos, dtype=torch.bool, device=dev)
        yt = torch.as_tensor(y, device=dev)
        tops, confs, oks = [], [], []
        for S in S_views:
            t2 = S.float().topk(2, dim=1)
            top = t2.indices[:, 0]
            tops.append(top)
            confs.append(torch.stack([t2.values[:, 0], t2.values[:, 0] - t2.values[:, 1]], 1))
            oks.append((top == yt) & ~self.is_oos)
        self.top = torch.stack(tops, 1)
        self.conf = torch.stack(confs, 1).float()
        self.ok = torch.stack(oks, 1)
        a1 = torch.zeros_like(self.ok)
        a1[:, 1:] = self.top[:, 1:] == self.top[:, :-1]
        a2 = torch.zeros_like(self.ok)
        a2[:, 2:] = a1[:, 2:] & a1[:, 1:-1]
        self.agree1, self.agree2 = a1, a2


class MetaCoderH:
    """Named bits of the observation; thermometer thresholds at quantiles of the training episodes
    (pooled over all prefixes, per feature)."""

    def __init__(self, H: int, bins: int = 16, memory: bool = True):
        self.H, self.bins, self.memory = H, bins, memory

    def fit(self, ep: EpisodesH) -> "MetaCoderH":
        qs = torch.linspace(0, 1, self.bins + 1, device=ep.conf.device)[1:-1]
        self.th = [torch.quantile(ep.conf[:, :, k].reshape(-1), qs).unique() for k in range(2)]
        nm = ["top", "margin"]
        self.names = [f"step >= {k}" for k in range(1, self.H)]
        for view in (("now", "before") if self.memory else ("now",)):
            for k in range(2):
                self.names += [f"{nm[k]}_{view} >= {float(t):.0f}" for t in self.th[k]]
        if self.memory:
            self.names += ["same top intent as before", "same top intent on the last 3 prefixes"]
        return self

    def encode(self, ep: EpisodesH, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        parts = [step.unsqueeze(1) >= torch.arange(1, self.H, device=idx.device)[None, :]]
        cur = ep.conf[idx, step]
        for k in range(2):
            parts.append(cur[:, k : k + 1] >= self.th[k][None, :])
        if self.memory:
            has_prev = (step >= 1).unsqueeze(1)
            prev = ep.conf[idx, (step - 1).clamp(min=0)]
            for k in range(2):
                parts.append((prev[:, k : k + 1] >= self.th[k][None, :]) & has_prev)
            parts.append(ep.agree1[idx, step].unsqueeze(1))
            parts.append(ep.agree2[idx, step].unsqueeze(1))
        return torch.cat(parts, 1).to(torch.uint8)


def cont_features_h(ep: EpisodesH, idx, step, mu, sd, memory: bool = True) -> torch.Tensor:
    """Continuous twin of the bits for linear Q: one-hot step, standardized (top, margin) of the
    current and previous prefix with squares, the two agreement flags, and step interactions."""
    H = ep.H
    oh = torch.nn.functional.one_hot(step, H).float()
    cur = (ep.conf[idx, step] - mu) / sd
    f = [oh, cur, cur.pow(2), (oh[:, :, None] * cur[:, None, :]).reshape(len(idx), -1)]
    if memory:
        has = (step >= 1).float().unsqueeze(1)
        prev = (ep.conf[idx, (step - 1).clamp(min=0)] - mu) / sd * has
        f += [prev, prev.pow(2), ep.agree1[idx, step].float().unsqueeze(1), ep.agree2[idx, step].float().unsqueeze(1)]
    return torch.cat(f, 1)


def step_rewards_h(ep: EpisodesH, idx, step, a, R):
    """Reward and done flag. ASK at the last step counts as ESCALATE (policies mask it anyway)."""
    last = step == ep.H - 1
    ok = ep.ok[idx, step]
    r = torch.where(ok, torch.full_like(step, 0, dtype=torch.float) + R.correct, torch.full_like(step, 0, dtype=torch.float) + R.wrong)
    r = torch.where(a == ESCALATE, torch.full_like(r, R.escalate), r)
    ask = (a == ASK) & ~last
    r = torch.where(ask, torch.full_like(r, R.ask), r)
    r = torch.where((a == ASK) & last, torch.full_like(r, R.escalate), r)
    return r, ~ask


def masked_greedy_h(Q: torch.Tensor, step: torch.Tensor, H: int) -> torch.Tensor:
    Q = Q.clone()
    Q[step == H - 1, ASK] = -1e9
    return Q.argmax(1)


def returns_h(policy, ep: EpisodesH, R) -> torch.Tensor:
    """Per-episode return of a greedy rollout, with no statistics (fast, for tuning the rules).
    Same steps as ``evaluate_h``."""
    dev = ep.conf.device
    idx = torch.arange(ep.n, device=dev)
    ret = torch.zeros(ep.n, device=dev)
    alive = torch.ones(ep.n, dtype=torch.bool, device=dev)
    for t in range(ep.H):
        i = idx[alive]
        if len(i) == 0:
            break
        st = torch.full_like(i, t)
        a = policy(i, st)
        if t == ep.H - 1:
            a = torch.where(a == ASK, torch.full_like(a, ESCALATE), a)
        r, done = step_rewards_h(ep, i, st, a, R)
        ret[i] += r
        alive[i[done]] = False
    return ret


def evaluate_h(policy, ep: EpisodesH, R) -> dict:
    """Greedy rollout of every episode to the end; mean return and where the decisions fell."""
    dev = ep.conf.device
    H = ep.H
    idx = torch.arange(ep.n, device=dev)
    ret = torch.zeros(ep.n, device=dev)
    alive = torch.ones(ep.n, dtype=torch.bool, device=dev)
    asks = torch.zeros(ep.n, device=dev)
    final = torch.full((ep.n,), -1, dtype=torch.long, device=dev)
    end_step = torch.full((ep.n,), -1, dtype=torch.long, device=dev)
    for t in range(H):
        i = idx[alive]
        if len(i) == 0:
            break
        st = torch.full_like(i, t)
        a = policy(i, st)
        if t == H - 1:
            a = torch.where(a == ASK, torch.full_like(a, ESCALATE), a)
        r, done = step_rewards_h(ep, i, st, a, R)
        ret[i] += r
        asks[i] += (~done).float()
        fin = i[done]
        final[fin] = a[done]
        end_step[fin] = t
        alive[fin] = False
    oos = ep.is_oos
    out = {"mean_return": float(ret.mean()), "mean_return_inscope": float(ret[~oos].mean()),
           "mean_asks": float(asks.mean()), "share_route": float((final == ROUTE).float().mean()),
           "share_escalate": float((final == ESCALATE).float().mean()),
           "end_step_shares": [float((end_step == t).float().mean()) for t in range(H)],
           "routed_accuracy": float(ep.ok[idx, end_step.clamp(min=0)][final == ROUTE].float().mean()) if (final == ROUTE).any() else None}
    if oos.any():
        out["mean_return_oos"] = float(ret[oos].mean())
        out["oos_escalated"] = float((final[oos] == ESCALATE).float().mean())
    out["_returns"] = ret
    return out
