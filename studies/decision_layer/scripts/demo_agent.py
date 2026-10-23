#!/usr/bin/env python
"""Glass-box agent loop on CLINC150: the learned TM policy decides ROUTE / ASK / ESCALATE, and every
decision is printed with the clause that carried it.

    demo_agent.py --policy results/explore/r6_tm_q_seed1_e12_tmq_k0005.npz --router results/explore/r6_router_seed1.pt --split val --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.data import load  # noqa: E402
from dl.explain import clause_literals, simplify  # noqa: E402
from dl.mdp import ACTIONS, ASK, Episodes, MetaCoder, half_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = load("clinc150")
    sc = torch.load(args.router)
    lab = args.split
    y = ds.labels[lab]
    S_half = torch.cat([sc[lab]["half"], sc[f"oos_{lab}"]["half"]])
    S_full = torch.cat([sc[lab]["full"], sc[f"oos_{lab}"]["full"]])
    texts = ds.texts[lab] + ds.texts[f"oos_{lab}"]
    yy = np.r_[y, -np.ones(len(ds.texts[f"oos_{lab}"]), dtype=np.int64)]
    is_oos = yy < 0
    ep = Episodes(S_half, S_full, yy, is_oos)
    tr = Episodes(torch.cat([sc["train"]["half"], sc["oos_train"]["half"]]), torch.cat([sc["train"]["full"], sc["oos_train"]["full"]]),
                  np.r_[ds.labels["train"], -np.ones(100, dtype=np.int64)], np.r_[np.zeros(15000, bool), np.ones(100, bool)])
    m = np.load(args.policy)
    names = [str(x) for x in m["names"]]
    coder = MetaCoder(16).fit(tr)
    assert coder.names == names, "policy was trained with another coder"
    ta, w = m["ta"].astype(np.int32), m["w"]
    cfg = json.loads(str(m["cfg"]))
    T = 1000.0
    f = len(names)
    incl = ta > cfg["n_states"]

    def decide(i, step):
        bits = coder.encode(ep, torch.tensor([i], device="cuda"), torch.tensor([step], device="cuda")).cpu().numpy()[0]
        lit = np.concatenate([bits, 1 - bits])
        fired = ((incl & (lit[None, :] == 0)).sum(1) == 0) & (incl.sum(1) > 0)
        q = (fired[:, None] * w).sum(0) / T
        if step == 1:
            q[ASK] = -1e9
        a = int(np.argmax(q))
        j = int(np.argmax(fired * w[:, a]))
        lits = simplify(clause_literals(ta[j], cfg["n_states"], f, names))
        if step == 0:  # memory literals are all 0 at step 0, so they hold trivially: leave them out
            lits = [l for l in lits if "before_ask" not in l and l != "NOT step1"]
        rule = " AND ".join(lits) if lits else "(no condition beyond the step)"
        return a, q, f"+{int(w[j, a])} (strongest firing clause of the {int(fired.sum())} that fired): IF {rule}"

    rng = np.random.default_rng(args.seed)
    pick = list(rng.choice(np.flatnonzero(~is_oos), args.n - 2, replace=False)) + list(rng.choice(np.flatnonzero(is_oos), 2, replace=False))
    for i in pick:
        t = texts[i]
        top0 = int(S_half[i].argmax())
        print(f'\nUSER (first half): "{half_text(t)}"   [true intent: {"out of scope" if is_oos[i] else ds.intents[yy[i]]}]')
        print(f"  router guess: {ds.intents[top0]}  (top {float(ep.conf[i, 0, 0]):.0f}, margin {float(ep.conf[i, 0, 1]):.0f})")
        a, q, why = decide(i, 0)
        print(f"  POLICY: {ACTIONS[a]}   Q = " + ", ".join(f"{ACTIONS[k]} {q[k]:+.2f}" for k in range(3)))
        print(f"    because {why}")
        if a == ASK:
            top1 = int(S_full[i].argmax())
            print(f'  USER (full query): "{t}"')
            print(f"  router guess: {ds.intents[top1]}  (top {float(ep.conf[i, 1, 0]):.0f}, margin {float(ep.conf[i, 1, 1]):.0f})")
            a1, q1, why1 = decide(i, 1)
            print(f"  POLICY: {ACTIONS[a1]}   Q = " + ", ".join(f"{ACTIONS[k]} {q1[k]:+.2f}" for k in (0, 2)))
            print(f"    because {why1}")
            ok = bool(ep.ok[i, 1]) if a1 == 0 else None
        else:
            ok = bool(ep.ok[i, 0]) if a == 0 else None
        print("  outcome:", {True: "routed correctly", False: "routed WRONG", None: "escalated to a human"}[ok])


if __name__ == "__main__":
    main()
