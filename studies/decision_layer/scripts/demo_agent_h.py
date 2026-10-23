#!/usr/bin/env python
"""Glass-box agent loop over 4 prefixes (R8): the learned TM policy decides ROUTE / ASK / ESCALATE at
every step, and each decision is printed with the clause that carried it.

    demo_agent_h.py --policy results/explore/r8_tm_q_seed1_e22_tmq.npz --split val --n 8 [--seed 0]
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
sys.path.insert(0, str(ROOT / "scripts"))

from dl.data import load  # noqa: E402
from dl.explain import clause_literals, simplify  # noqa: E402
from dl.mdp_h import ACTIONS, ASK, ROUTE, MetaCoderH, prefix_text  # noqa: E402
from r8_run import H, build, router_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-oos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--router-seed", type=int, default=1)
    args = ap.parse_args()
    ds = load("clinc150")
    sc = router_scores(args.router_seed)
    ep_tr, ep = build(sc, "train", ds), build(sc, args.split, ds)
    m = np.load(args.policy)
    names = [str(x) for x in m["names"]]
    memory = "same top intent as before" in names
    coder = MetaCoderH(H, 16, memory=memory).fit(ep_tr)
    assert coder.names == names, "policy was trained with another coder"
    ta, w = m["ta"].astype(np.int32), m["w"]
    cfg = json.loads(str(m["cfg"]))
    T = float(cfg["T"])
    incl = ta > cfg["n_states"]
    f = len(names)
    texts = ds.texts[args.split] + ds.texts[f"oos_{args.split}"]
    y = np.r_[ds.labels[args.split], -np.ones(len(ds.texts[f"oos_{args.split}"]), dtype=np.int64)]

    def decide(i, t):
        bits = coder.encode(ep, torch.tensor([i], device="cuda"), torch.tensor([t], device="cuda")).cpu().numpy()[0]
        lit = np.concatenate([bits, 1 - bits])
        fired = ((incl & (lit[None, :] == 0)).sum(1) == 0) & (incl.sum(1) > 0)
        q = (fired[:, None] * w).sum(0) / T
        if t == H - 1:
            q[ASK] = -1e9
        a = int(np.argmax(q))
        j = int(np.argmax(fired * w[:, a]))
        lits = simplify(clause_literals(ta[j], cfg["n_states"], f, names))
        if t == 0:  # memory literals are all 0 at step 0, so their negations hold trivially there
            lits = [l for l in lits if not ("before" in l or "same top" in l or l.startswith("NOT step"))]
        rule = " AND ".join(lits) if lits else "(no condition)"
        return a, q, f"+{int(w[j, a])} (strongest of the {int(fired.sum())} clauses that fired): IF {rule}"

    rng = np.random.default_rng(args.seed)
    is_oos = ep.is_oos.cpu().numpy()
    pick = list(rng.choice(np.flatnonzero(~is_oos), args.n - args.n_oos, replace=False)) + \
        list(rng.choice(np.flatnonzero(is_oos), args.n_oos, replace=False))
    for i in pick:
        print(f'\nQUERY {i} [true intent: {"out of scope" if is_oos[i] else ds.intents[y[i]]}]')
        total = 0.0
        for t in range(H):
            top = int(ep.top[i, t])
            print(f'  step {t}: user so far: "{prefix_text(texts[i], t + 1, H)}"')
            print(f"    router guess: {ds.intents[top]} (top {float(ep.conf[i, t, 0]):.0f}, margin {float(ep.conf[i, t, 1]):.0f})"
                  + (f", same as before: {bool(ep.agree1[i, t])}" if t else ""))
            a, q, why = decide(i, t)
            qs = ", ".join(f"{ACTIONS[k]} {q[k]:+.2f}" for k in range(3) if not (t == H - 1 and k == ASK))
            print(f"    POLICY: {ACTIONS[a]}   Q = {qs}")
            print(f"      because {why}")
            if a == ASK:
                total -= 0.1
                continue
            if a == ROUTE:
                ok = bool(ep.ok[i, t])
                total += 1.0 if ok else -1.0
                print(f"    outcome: routed {'correctly' if ok else 'WRONG'}; return {total:+.1f}")
            else:
                print(f"    outcome: escalated to a human; return {total:+.1f}")
            break


if __name__ == "__main__":
    main()
