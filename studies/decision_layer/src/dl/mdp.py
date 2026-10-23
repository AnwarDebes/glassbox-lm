"""Clarify-or-route: a two-step decision problem with delayed reward, built on CLINC150.

Episode: one user query. At step 0 the agent sees only the first half of the query (the first
ceil(len/2) words); a fixed router (a supervised TM trained on full training queries) scores it.
Actions: ROUTE (commit to the router's top intent: +1 if it is the true intent, -1 if not or if
the query is out of scope), ASK (a clarifying question: -c_ask now, the full query is revealed and
the episode moves to step 1), ESCALATE (hand over to a human: 0). At step 1 only ROUTE and
ESCALATE are allowed. So the value of ASK is only visible one step later: a learner that looks at
the immediate reward alone never asks.

Meta-observation for the decision layer (bits): step flag, thermometers of the router's top
class sum and margin on the current view, and, at step 1, the same thermometers of the step-0
view (memory literals). Names are kept for the readable policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ROUTE, ASK, ESCALATE = 0, 1, 2
ACTIONS = ["ROUTE", "ASK", "ESCALATE"]


@dataclass
class Rewards:
    correct: float = 1.0
    wrong: float = -1.0
    escalate: float = 0.0
    ask: float = -0.1


def half_text(t: str) -> str:
    w = t.split()
    return " ".join(w[: max(1, (len(w) + 1) // 2)])


class Episodes:
    """Router outputs for the half and the full view of every query of one split.

    ``top``/``ok``: router's top intent on each view and whether it is right (False for OOS).
    ``conf``: [n, 2 views, 2] = (top class sum, margin), in router units.
    """

    def __init__(self, S_half: torch.Tensor, S_full: torch.Tensor, y: np.ndarray, is_oos: np.ndarray):
        self.n = len(y)
        self.is_oos = torch.as_tensor(is_oos, dtype=torch.bool, device=S_half.device)
        yt = torch.as_tensor(y, device=S_half.device)
        feats, oks = [], []
        for S in (S_half, S_full):
            top2 = S.topk(2, dim=1)
            top = top2.indices[:, 0]
            ok = (top == yt) & ~self.is_oos
            feats.append(torch.stack([top2.values[:, 0], top2.values[:, 0] - top2.values[:, 1]], 1))
            oks.append(ok)
        self.conf = torch.stack(feats, 1).float()  # [n, 2, 2]
        self.ok = torch.stack(oks, 1)  # [n, 2]


class MetaCoder:
    """Thermometer bits of (top sum, margin) with thresholds at quantiles of the training episodes."""

    def __init__(self, bins: int = 16, memory: bool = True):
        self.bins = bins
        self.memory = memory  # False: the step-0 view is not shown at step 1 (ablation)

    def fit(self, ep: Episodes) -> "MetaCoder":
        qs = torch.linspace(0, 1, self.bins + 1, device=ep.conf.device)[1:-1]
        # pooled over both views, per feature
        self.th = [torch.quantile(ep.conf[:, :, k].reshape(-1), qs).unique() for k in range(2)]
        nm = ["top", "margin"]
        self.names = ["step1"]
        for view in ("now", "before_ask"):
            for k in range(2):
                self.names += [f"{nm[k]}_{view} >= {float(t):.0f}" for t in self.th[k]]
        return self

    def encode(self, ep: Episodes, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Bits for episodes ``idx`` at step ``step`` (0 or 1 per row)."""
        b = idx.shape[0]
        cur = ep.conf[idx, step]  # [b, 2]
        prev = ep.conf[idx, 0]
        s1 = (step == 1)
        parts = [s1.unsqueeze(1)]
        for k in range(2):
            parts.append(cur[:, k : k + 1] >= self.th[k][None, :])
        for k in range(2):
            parts.append((prev[:, k : k + 1] >= self.th[k][None, :]) & s1.unsqueeze(1) & self.memory)
        return torch.cat(parts, 1).to(torch.uint8)


def step_rewards(ep: Episodes, idx: torch.Tensor, step: torch.Tensor, a: torch.Tensor, R: Rewards):
    """Reward and done flag of action ``a`` at ``step``; ASK at step 1 is treated as ESCALATE."""
    ok = ep.ok[idx, step]
    r = torch.where(ok, torch.full_like(ok, R.correct, dtype=torch.float), torch.full_like(ok, R.wrong, dtype=torch.float))
    r = torch.where(a == ESCALATE, torch.full_like(r, R.escalate), r)
    ask = (a == ASK) & (step == 0)
    r = torch.where(ask, torch.full_like(r, R.ask), r)
    late_ask = (a == ASK) & (step == 1)
    r = torch.where(late_ask, torch.full_like(r, R.escalate), r)
    done = ~ask
    return r, done


def evaluate(policy, ep: Episodes, R: Rewards) -> dict:
    """Greedy two-step rollout over every episode; returns mean return and action shares."""
    dev = ep.conf.device
    idx = torch.arange(ep.n, device=dev)
    step = torch.zeros(ep.n, dtype=torch.long, device=dev)
    a0 = policy(idx, step)
    r0, done0 = step_rewards(ep, idx, step, a0, R)
    ret = r0.clone()
    asked = ~done0
    if asked.any():
        i1 = idx[asked]
        s1 = torch.ones_like(i1)
        a1 = policy(i1, s1)
        r1, _ = step_rewards(ep, i1, s1, a1, R)
        ret[asked] += r1
        a1_full = torch.full_like(a0, -1)
        a1_full[asked] = a1
    else:
        a1_full = torch.full_like(a0, -1)
    out = {
        "mean_return": float(ret.mean()),
        "share_route0": float((a0 == ROUTE).float().mean()),
        "share_ask": float((a0 == ASK).float().mean()),
        "share_escalate0": float((a0 == ESCALATE).float().mean()),
        "share_route_after_ask": float(((a1_full == ROUTE) & asked).float().sum() / max(1, int(asked.sum()))),
    }
    oos = ep.is_oos
    out["mean_return_inscope"] = float(ret[~oos].mean())
    if oos.any():
        out["mean_return_oos"] = float(ret[oos].mean())
    return out
