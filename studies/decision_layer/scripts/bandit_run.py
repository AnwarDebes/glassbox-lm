#!/usr/bin/env python
"""Routing learned from reward only (contextual bandit), one run.

    bandit_run.py --dataset clinc150 --encoder minilm --target intent --learner tm --seed 1

Targets: ``intent`` (CLINC150 150, Banking77 77, HWU64 64 actions) or ``domain`` (CLINC150 10
domains from clinc/oos-eval domains.json; HWU64 18 scenarios = first word of the label).
Learners: ``tm`` (thermometer bits, unsupervised quantiles of the training embeddings),
``linucb`` and ``eps_logistic`` (raw embedding), and the full-information references
``sup_tm`` (the same TM trained on labels) and ``sup_logreg`` (C from the val grid).
Eval split: ``val`` (exploration) or ``test`` (registered runs only). One JSON line per run in
results/<out>/bandit.jsonl.
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

from dl.bandit import EpsLogistic, LinUCB, TMBandit, run_stream  # noqa: E402
from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import accuracy, fit_logreg, logits  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def targets(ds, target: str):
    if target == "intent":
        return ds.labels, ds.intents
    if ds.name == "clinc150":
        dom = json.loads((ROOT / "data" / "raw" / "clinc150_domains.json").read_text())
        inv = {i: k for k, v in dom.items() for i in v}
        names = sorted(dom)
        m = np.array([names.index(inv[i]) for i in ds.intents])
    elif ds.name == "hwu64":
        names = sorted({i.split()[0] for i in ds.intents})
        m = np.array([names.index(i.split()[0]) for i in ds.intents])
    else:
        raise ValueError("domain target exists for clinc150 and hwu64")
    return {k: m[v] for k, v in ds.labels.items()}, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--target", default="intent")
    ap.add_argument("--learner", required=True)
    ap.add_argument("--view", default="thermo:8")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=0, help="exploration among the top-k actions (0 = all)")
    ap.add_argument("--boltzmann", type=float, default=0.0, help="exploratory: Boltzmann exploration temperature (TM only)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", default="val")
    ap.add_argument("--out", default="explore")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.eval == "test" and args.out == "explore":
        raise SystemExit("test split only in registered runs (--out confirm)")

    t0 = time.time()
    torch.manual_seed(args.seed)
    d = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    ds = load(args.dataset)
    labels, names = targets(ds, args.target)
    K = len(names)
    ytr = torch.as_tensor(labels["train"], device="cuda")
    yev = torch.as_tensor(labels[args.eval], device="cuda")
    rec = {"tag": args.tag, "dataset": args.dataset, "encoder": args.encoder, "target": args.target, "K": K,
           "learner": args.learner, "seed": args.seed, "passes": args.passes, "batch": args.batch, "eval": args.eval}

    if args.learner in ("tm", "sup_tm"):
        view = make_view(args.view, seed=args.seed)
        if args.view.startswith("proto"):
            raise SystemExit("proto views use labels; not allowed for the bandit")
        view.fit(d["train"], None, texts=ds.texts["train"])
        Btr = torch.as_tensor(view.transform(d["train"], ds.texts["train"]), dtype=torch.uint8, device="cuda")
        Bev = torch.as_tensor(view.transform(d[args.eval], ds.texts[args.eval]), dtype=torch.uint8, device="cuda")
        cfg = TMConfig(n_clauses=args.clauses, T=args.T or args.clauses, s=args.s, epochs=args.passes,
                       batch_size=args.batch, seed=args.seed, eval_every=1)
        rec.update({"view": args.view, "tm_cfg": asdict(cfg)})
        if args.learner == "tm":
            ln = TMBandit(K, Btr.shape[1], cfg, eps=args.eps, eps_decay=args.eps_decay, topk=args.topk)
            ln.boltzmann = args.boltzmann
            rec.update({"eps": args.eps, "eps_decay": args.eps_decay, "topk": args.topk, "boltzmann": args.boltzmann})
            rec.update(run_stream(ln, Btr, ytr, Bev, yev, args.passes, args.batch, args.seed, replay=args.replay))
            rec["stats"] = ln.tm.stats()
        else:
            tm = CoTM(K, Btr.shape[1], cfg)
            tm.fit(Btr.cpu().numpy(), labels["train"])  # no snapshot: same budget as the stream
            rec["final_eval_acc"] = float((tm.predict(Bev) == yev).float().mean())
            rec["stats"] = tm.stats()
    else:
        Xtr = torch.as_tensor(d["train"], device="cuda")
        Xev = torch.as_tensor(d[args.eval], device="cuda")
        if args.learner == "linucb":
            ln = LinUCB(K, Xtr.shape[1], alpha=args.alpha)
            rec["alpha"] = args.alpha
            rec.update(run_stream(ln, Xtr, ytr, Xev, yev, args.passes, args.batch, args.seed, replay=args.replay))
        elif args.learner == "eps_logistic":
            ln = EpsLogistic(K, Xtr.shape[1], eps=args.eps, lr=args.lr, seed=args.seed, topk=args.topk)
            rec.update({"eps": args.eps, "lr": args.lr, "topk": args.topk})
            rec.update(run_stream(ln, Xtr, ytr, Xev, yev, args.passes, args.batch, args.seed, replay=args.replay))
        elif args.learner == "sup_logreg":
            best = (-1.0, None)
            yva = labels["val"]
            for C in [1.0, 3.0, 10.0, 30.0, 100.0]:
                W, b = fit_logreg(d["train"], labels["train"], K, C)
                acc = accuracy(logits(d["val"], W, b), yva)
                if acc > best[0]:
                    best = (acc, (C, W, b))
            C, W, b = best[1]
            rec["C"] = C
            rec["final_eval_acc"] = accuracy(logits(d[args.eval], W, b), labels[args.eval])
        else:
            raise SystemExit(args.learner)
    rec["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bandit.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tm_cfg",)}))


if __name__ == "__main__":
    main()
