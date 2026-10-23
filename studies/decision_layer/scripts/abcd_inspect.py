#!/usr/bin/env python
"""Inspect a saved ABCD exploration TM (dev only): accuracy by button and by action index, the most
frequent confusions, and the strongest clauses for chosen buttons.

    abcd_inspect.py results/explore/abcd_models/<tag>_seed1.npz --parts mem,intent,step,words [--buttons a,b]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402
from dl.explain import decision_receipt, top_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--lex-k", type=int, default=1000)
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--buttons", default="")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--receipts", type=int, default=0)
    args = ap.parse_args()
    m = np.load(args.model, allow_pickle=False)
    ta, w = m["ta"], m["w"]
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    E = np.load(ROOT / "data" / "emb" / f"abcd__{args.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    edv = {f: E[f"dev__{f}"] for f in ("cust", "recent", "full")}
    view = ABCDView(parts=tuple(args.parts.split(",")), lex_k=args.lex_k, seed=0).fit(tr, etr, blist)
    X = view.transform(dv, edv)
    assert list(m["names"]) == view.names
    f = X.shape[1]
    lit = np.concatenate([X, 1 - X], 1)
    incl = ta > 256
    fired = ((incl[None, :, :] & (lit[:, None, :] == 0)).sum(2) == 0) if X.shape[0] * ta.shape[0] < 2e7 else None
    if fired is None:
        It = torch.as_tensor(incl, dtype=torch.float16, device="cuda")
        L0 = torch.as_tensor(1 - lit, dtype=torch.float16, device="cuda")
        fired = ((L0 @ It.T) == 0).cpu().numpy()
    fired &= incl.sum(1)[None, :] > 0
    S = fired.astype(np.int64) @ w
    pred = S.argmax(1)
    y = np.array([r["y"] for r in dv])
    print(f"dev acc {np.mean(pred == y):.4f}")
    ks = np.array([r["k"] for r in dv])
    for k in range(0, 6):
        sel = ks == k if k < 5 else ks >= 5
        print(f"  action index {k if k < 5 else '5+'}: n={sel.sum()} acc={np.mean(pred[sel] == y[sel]):.4f}")
    print("by button (n, acc):")
    for b in range(len(blist)):
        sel = y == b
        if sel.sum():
            print(f"  {blist[b]:22s} {sel.sum():5d} {np.mean(pred[sel] == b):.3f}")
    conf = Counter((blist[a], blist[b]) for a, b in zip(y, pred) if a != b)
    print("top confusions (true -> pred):")
    for (a, b), n in conf.most_common(15):
        print(f"  {a} -> {b}: {n}")
    sel = [blist.index(b) for b in args.buttons.split(",") if b] or None
    for line in top_rules(ta, w, view.names, blist, per_class=args.per_class, classes=sel):
        print(line[:400])
    rng = np.random.default_rng(0)
    for i in rng.choice(len(dv), size=args.receipts, replace=False) if args.receipts else []:
        r = dv[i]
        print(f"\n--- dev example {i}: subflow {r['subflow']}, prev {r['prev']}, true {r['button']}\n  recent: {r['recent'][:200]}")
        print(decision_receipt(ta, w, X[i], view.names, blist)[:1200])


if __name__ == "__main__":
    main()
