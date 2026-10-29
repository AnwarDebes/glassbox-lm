#!/usr/bin/env python
"""Exploration on ABCD next-action decisions, scored on DEV only (the test split is not read).

Arms (``--arms``):
- ``tm``        Tsetlin Machine on the readable bits (src/dl/abcd.py ABCDView, parts ``--parts``);
- ``lr_bits``   logistic regression on the same bits (C from a dev grid);
- ``mlp_bits``  one-hidden-layer network on the same bits (best dev epoch);
- ``gbdt_bits`` LightGBM on the same bits (best dev iteration);
- ``lr_raw``    logistic regression on the raw embeddings of cust, recent and full plus the memory bits;
- ``mlp_raw``   the network on the same raw input;
- ``rule``      one rule: the button whose centre is nearest to the last 3 turns (whitened);
- ``table``     the most frequent next button given (nearest subflow, last action) in train.

Scores: accuracy over action turns, and the paper's AST B-Slot accuracy (accuracy weighted by
the number of AST examples per action turn, BERT and RoBERTa example sets).
Writes one JSON line per arm to results/explore/abcd.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples, memory_from_rows  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.mlp import fit_mlp, mlp_scores  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import INITS, make_tm  # noqa: E402

GRID_C = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def scores_block(pred: np.ndarray, rows: list[dict]) -> dict:
    y = np.array([r["y"] for r in rows])
    ok = pred == y
    out = {"acc": float(ok.mean())}
    for k in ("bert", "roberta"):
        w = np.array([r[f"ast_count_{k}"] for r in rows], dtype=float)
        out[f"ast_bslot_{k}"] = float((ok * w).sum() / w.sum())
    first = np.array([r["k"] == 0 for r in rows])
    out["acc_first_action"] = float(ok[first].mean())
    out["acc_later_actions"] = float(ok[~first].mean())
    return out


def emb_split(E, split: str) -> dict:
    return {f: E[f"{split}__{f}"] for f in ("cust", "recent", "full")}


def best_lr(Xtr, ytr, Xdv, ydv, C_):
    best = None
    for c in GRID_C:
        W, b = fit_logreg(Xtr, ytr, C_, c)
        acc = float((logits(Xdv, W, b).argmax(1).cpu().numpy() == ydv).mean())
        if best is None or acc > best[0]:
            best = (acc, c, W, b)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star"], help="star: scripts/star_prepare.py examples (added 2026-10-17 13:43)")
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--lex-k", type=int, default=1000)
    ap.add_argument("--arms", default="tm,lr_bits")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--init", default="half", choices=INITS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    tr, blist = load_examples("train", args.dataset)
    dv, _ = load_examples("dev", args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    etr, edv = emb_split(E, "train"), emb_split(E, "dev")
    ytr = np.array([r["y"] for r in tr])
    ydv = np.array([r["y"] for r in dv])
    C_ = len(blist)
    parts = tuple(p for p in args.parts.split(",") if p)
    arms = args.arms.split(",")
    out_path = ROOT / "results" / "explore" / "abcd.jsonl"
    base = {"tag": args.tag, "dataset": args.dataset, "encoder": args.encoder, "parts": parts, "lex_k": args.lex_k, "seed": args.seed}

    def emit(arm: str, pred: np.ndarray, extra: dict) -> None:
        rec = {**base, "arm": arm, **scores_block(pred, dv), **extra,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    view = None
    if any(a in arms for a in ("tm", "lr_bits", "mlp_bits", "gbdt_bits", "rule", "table")):
        t0 = time.time()
        view = ABCDView(parts=parts if parts else ("mem",), lex_k=args.lex_k, seed=0).fit(tr, etr, blist)
        Xtr, Xdv = view.transform(tr, etr), view.transform(dv, edv)
        print(f"view {Xtr.shape[1]} bits ({time.time() - t0:.0f}s)", flush=True)

    if "tm" in arms:
        T = args.T or args.clauses
        cfg = TMConfig(n_clauses=args.clauses, T=T, s=args.s, epochs=args.epochs, max_literals=args.max_literals,
                       budget_select="rank", seed=args.seed)
        tm = make_tm(C_, Xtr.shape[1], cfg, init=args.init)
        tm.fit(Xtr, ytr, Xdv, ydv)
        pred = tm.predict(torch.as_tensor(Xdv, dtype=torch.uint8, device="cuda")).cpu().numpy()
        emit("tm", pred, {"cfg": asdict(cfg), "init": args.init, "best_epoch": getattr(tm, "best_epoch", None), "history": tm.history,
                          "train_s": round(tm.train_seconds, 1), "stats": tm.stats(), "n_bits": int(Xtr.shape[1])})
        if args.save:
            sd = ROOT / "results" / "explore" / "abcd_models"
            sd.mkdir(parents=True, exist_ok=True)
            st = tm.state()
            np.savez_compressed(sd / f"{args.tag or 'tm'}_seed{args.seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"],
                                names=np.array(view.names), buttons=np.array(blist))

    if "lr_bits" in arms:
        acc, c, W, b = best_lr(Xtr, ytr, Xdv, ydv, C_)
        emit("lr_bits", logits(Xdv, W, b).argmax(1).cpu().numpy(), {"C": c, "n_bits": int(Xtr.shape[1])})

    if "mlp_bits" in arms:
        net, info = fit_mlp(Xtr, ytr, Xdv, ydv, C_, seed=args.seed)
        emit("mlp_bits", mlp_scores(net, Xdv).argmax(1).cpu().numpy(), {**info, "n_bits": int(Xtr.shape[1])})

    if "gbdt_bits" in arms:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1, num_leaves=31, random_state=args.seed, verbose=-1,
                                 n_jobs=4)
        clf.fit(Xtr, ytr, eval_set=[(Xdv, ydv)], callbacks=[lgb.early_stopping(30, verbose=False)])
        emit("gbdt_bits", clf.predict(Xdv), {"best_iteration": int(clf.best_iteration_ or 0)})

    if "rule" in arms and view is not None and view.step is not None:
        pred = view.step.sims(edv["recent"]).argmax(1).cpu().numpy()  # k=1: prototype j = button j
        emit("rule_nearest_step_centre", pred, {})

    if "table" in arms and view is not None and view.intent is not None:
        sub_tr = view.intent.sims(etr["cust"]).argmax(1).cpu().numpy()
        sub_dv = view.intent.sims(edv["cust"]).argmax(1).cpu().numpy()
        t2, t1 = defaultdict(Counter), defaultdict(Counter)
        glob = Counter(ytr.tolist())
        for r, s in zip(tr, sub_tr):
            last = r["prev"][-1] if r["prev"] else "-"
            t2[(int(s), last)][r["y"]] += 1
            t1[last][r["y"]] += 1
        pred = []
        for r, s in zip(dv, sub_dv):
            last = r["prev"][-1] if r["prev"] else "-"
            c = t2.get((int(s), last)) or t1.get(last) or glob
            pred.append(c.most_common(1)[0][0])
        emit("table_subflow_last", np.array(pred), {})

    if "lr_raw" in arms or "mlp_raw" in arms:
        def raw(e, rows):
            return np.concatenate([e["full"], e["cust"], e["recent"], memory_from_rows(rows, blist).astype(np.float32)], 1)

        Rtr, Rdv = raw(etr, tr), raw(edv, dv)
        if "lr_raw" in arms:
            acc, c, W, b = best_lr(Rtr, ytr, Rdv, ydv, C_)
            emit("lr_raw", logits(Rdv, W, b).argmax(1).cpu().numpy(), {"C": c, "dim": int(Rtr.shape[1])})
        if "mlp_raw" in arms:
            net, info = fit_mlp(Rtr, ytr, Rdv, ydv, C_, seed=args.seed)
            emit("mlp_raw", mlp_scores(net, Rdv).argmax(1).cpu().numpy(), {**info, "dim": int(Rtr.shape[1])})


if __name__ == "__main__":
    main()
