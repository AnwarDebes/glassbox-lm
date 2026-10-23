#!/usr/bin/env python
"""One exploratory TM run on a (dataset, encoder, boolean view), scored on VAL only.

Writes one JSON line to results/explore/tm.jsonl: val accuracy at the best snapshot, the
epoch curve, clause statistics, and for CLINC150 the out-of-scope AUROC on val with three
scores (minus max class sum, minus margin, minus max class sum / fired clauses). Also the
logistic regression on the same bits (C from a small grid, val) as a same-features control.
No test split is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import accuracy, auroc, fit_logreg, logits  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402
from dl.tm_init import INITS, make_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--view", default="thermo:8")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=0, help="0 = same as clauses")
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--neg", default="hardest")
    ap.add_argument("--no-negation", action="store_true")
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--budget-mode", default="exact", choices=["exact", "tmu", "block_full"])
    ap.add_argument("--clause-drop", type=float, default=0.0)
    ap.add_argument("--max-negated", type=int, default=0)
    ap.add_argument("--init", default="half", choices=INITS, help="start state (src/dl/tm_init.py)")
    ap.add_argument("--budget-select", default="threshold", choices=["threshold", "rank"])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--bits-logreg", action="store_true", help="also fit logreg on the same bits")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--readout", action="store_true", help="logreg on the clause outputs (and on clauses + input bits)")
    ap.add_argument("--ensemble", type=int, default=1, help="M machines with seeds seed..seed+M-1, sums averaged")
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    ds = load(args.dataset)
    view = make_view(args.view, seed=args.seed)
    view.fit(d["train"], d["y_train"], texts=ds.texts["train"], class_names=ds.intents)
    Btr, Bva = view.transform(d["train"], ds.texts["train"]), view.transform(d["val"], ds.texts["val"])
    ytr, yva = d["y_train"], d["y_val"]
    C = len(ds.intents)
    cfg = TMConfig(n_clauses=args.clauses, T=args.T or args.clauses, s=args.s, epochs=args.epochs,
                   batch_size=args.batch, neg_sampling=args.neg, negation=not args.no_negation,
                   max_literals=args.max_literals, budget_mode=args.budget_mode, budget_select=args.budget_select, max_negated=args.max_negated, clause_drop=args.clause_drop, half=not args.fp32,
                   seed=args.seed, eval_every=args.eval_every)
    members = []
    for mi in range(args.ensemble):
        cfg_i = TMConfig(**{**asdict(cfg), "seed": args.seed + mi})
        tm_i = make_tm(C, Btr.shape[1], cfg_i, args.init)
        tm_i.fit(Btr, ytr, Bva, yva)
        members.append(tm_i)
    tm = members[0]
    rec = {
        "tag": args.tag, "dataset": args.dataset, "encoder": args.encoder, "view": args.view,
        "n_bits": int(Btr.shape[1]), "cfg": asdict(cfg), "init": args.init, "val_acc": round(tm.best_val, 4),
        "best_epoch": getattr(tm, "best_epoch", None), "train_s": round(tm.train_seconds, 1),
        "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats(),
    }
    Xva_t = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
    yva_t = torch.as_tensor(yva, device="cuda")
    if args.ensemble > 1:
        rec["member_val"] = [round(m.best_val, 4) for m in members]
        tot = 0
        for m in members:
            Sm = m.class_sums(Xva_t)
            tot = tot + Sm / Sm.std()
        rec["ensemble_val"] = round(float((tot.argmax(1) == yva_t).float().mean()), 4)
    if args.readout:
        Xtr_t = torch.as_tensor(Btr, dtype=torch.uint8, device="cuda")
        Ctr = torch.cat([m.clause_outputs(Xtr_t) for m in members], 1).float()
        Cva = torch.cat([m.clause_outputs(Xva_t) for m in members], 1).float()
        ro = {}
        for name, (A, B) in {"clauses": (Ctr, Cva),
                             "clauses+bits": (torch.cat([Ctr, Xtr_t.float()], 1), torch.cat([Cva, Xva_t.float()], 1))}.items():
            best = (-1, None)
            for Cc in [0.3, 1.0, 3.0, 10.0, 30.0]:
                W, b = fit_logreg(A, ytr, C, Cc)
                acc = accuracy(logits(B, W, b), yva)
                if acc > best[0]:
                    best = (acc, Cc)
            ro[name] = {"val": round(best[0], 4), "C": best[1]}
        rec["readout"] = ro
    if "oos_val" in d.files:
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
        Xoo = torch.as_tensor(view.transform(d["oos_val"], ds.texts["oos_val"]), dtype=torch.uint8, device="cuda")
        S_in, S_oo = tm.class_sums(Xin), tm.class_sums(Xoo)
        is_oos = np.r_[np.zeros(len(S_in)), np.ones(len(S_oo))]
        S = torch.cat([S_in, S_oo]).cpu().numpy()
        top2 = np.sort(S, 1)[:, -2:]
        rec["oos_auroc_val"] = {
            "max_sum": round(auroc(is_oos, -top2[:, 1]), 4),
            "margin": round(auroc(is_oos, -(top2[:, 1] - top2[:, 0])), 4),
        }
    if args.bits_logreg:
        best = (-1, None)
        for Cc in [1.0, 10.0, 100.0]:
            W, b = fit_logreg(Btr.astype(np.float32), ytr, C, Cc)
            acc = accuracy(logits(Bva.astype(np.float32), W, b), yva)
            if acc > best[0]:
                best = (acc, Cc)
        rec["bits_logreg_val"] = round(best[0], 4)
        rec["bits_logreg_C"] = best[1]
    rec["total_s"] = round(time.time() - t0, 1)
    out = ROOT / "results" / "explore"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tm.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if args.save:
        tag = args.tag or f"{args.dataset}_{args.encoder}_{args.seed}"
        np.savez_compressed(out / f"model_{tag}.npz", **tm.state())
    print(json.dumps({k: v for k, v in rec.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
