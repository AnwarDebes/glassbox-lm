#!/usr/bin/env python
"""ABCD next action learned from reward only (the agent loop observe -> act -> reward -> learn).

The stream is ``passes`` random permutations of the training action turns, in rounds of
``batch``. The learner presses one button per turn and sees only whether it was the button the
human agent pressed (reward 1) or not (0), as in R5 to R5d. Inputs carry no label of any kind:
``ABCDFreeView`` (src/dl/abcd.py): memory (the buttons pressed so far in the conversation),
words of the last 3 turns, and unsupervised clusters of the last 3 turns and of the customer's
words. The intent and next-button centres of the supervised view are NOT used here (they are
built from labels).

Learners: ``tm`` (TMBandit, Boltzmann exploration on the class sums), ``linucb`` (on the same
bits, or ``--linucb-input raw``: the raw recent and customer embeddings plus the memory bits),
``sup_tm`` (the same TM trained on labels, the full-information reference).
Eval: greedy accuracy on dev (exploration) or test (registered runs, ``--out confirm/<reg>``).
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

from dl.abcd import ABCDFreeView, load_examples, memory_from_rows  # noqa: E402
from dl.bandit import LinUCB, LinUCBFast, NeuralBandit, TMBandit, run_stream  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True, choices=["tm", "linucb", "sup_tm", "neural"])
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star"], help="star: scripts/star_prepare.py examples (added 2026-10-17 14:28)")
    ap.add_argument("--parts", default="mem,words,rclust,cclust")
    ap.add_argument("--lex-k", type=int, default=1000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--boltzmann", type=float, default=0.5)
    ap.add_argument("--max-literals", type=int, default=0, help="literal budget (with --init trim and the exact rank selection); added after R10")
    ap.add_argument("--init", default="half", choices=["half", "trim", "empty"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--nn-lr", type=float, default=1e-3, help="neural bandit: AdamW step size")
    ap.add_argument("--nn-eps", type=float, default=0.0, help="neural bandit: epsilon-greedy instead of Boltzmann when > 0")
    ap.add_argument("--linucb-input", default="bits", choices=["bits", "raw"])
    ap.add_argument("--linucb-impl", default="woodbury", choices=["woodbury", "inverse"])
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--replay", type=int, default=0, help="extra TM updates per round on minibatches of the logged (context, press, reward) history (as R5)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="explore")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-model", action="store_true", help="save the trained TM (ta, w, cfg) next to the results (added 2026-10-17 11:50 for the closed loop)")
    args = ap.parse_args()
    if args.eval == "test" and not args.out.startswith("confirm"):
        raise SystemExit("test split only in registered runs (--out confirm/<registration>)")

    t0 = time.time()
    torch.manual_seed(args.seed)
    tr, blist = load_examples("train", args.dataset)
    ev, _ = load_examples(args.eval, args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent")}
    eev = {f: E[f"{args.eval}__{f}"] for f in ("cust", "recent")}
    K = len(blist)
    ytr = torch.as_tensor([r["y"] for r in tr], device="cuda")
    yev = torch.as_tensor([r["y"] for r in ev], device="cuda")
    parts = tuple(p for p in args.parts.split(",") if p)
    rec = {"tag": args.tag, "dataset": args.dataset, "learner": args.learner, "encoder": args.encoder, "parts": parts, "K": K, "seed": args.seed,
           "passes": args.passes, "batch": args.batch, "eval": args.eval}

    view = ABCDFreeView(parts=parts, lex_k=args.lex_k, k=args.k, seed=0).fit(tr, etr, blist)
    Btr = torch.as_tensor(view.transform(tr, etr), dtype=torch.uint8, device="cuda")
    Bev = torch.as_tensor(view.transform(ev, eev), dtype=torch.uint8, device="cuda")
    rec["n_bits"] = int(Btr.shape[1])
    if args.learner in ("tm", "sup_tm"):
        cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.passes, batch_size=args.batch,
                       seed=args.seed, eval_every=1, max_literals=args.max_literals,
                       budget_select="rank" if args.max_literals else "threshold")
        rec["tm_cfg"] = asdict(cfg)
        if args.learner == "tm":
            ln = TMBandit(K, Btr.shape[1], cfg, eps=0.0)
            if args.init != "half":
                from dl.tm_init import make_tm

                ln.tm = make_tm(K, Btr.shape[1], cfg, init=args.init)
            ln.boltzmann = args.boltzmann
            rec["boltzmann"] = args.boltzmann
            rec.update(run_stream(ln, Btr, ytr, Bev, yev, args.passes, args.batch, args.seed, replay=args.replay))
            rec["stats"] = ln.tm.stats()
            tm = ln.tm
        else:
            tm = CoTM(K, Btr.shape[1], cfg)
            tm.fit(Btr.cpu().numpy(), np.array([r["y"] for r in tr]))  # no snapshot: same budget as the stream
            rec["final_eval_acc"] = float((tm.predict(Bev) == yev).float().mean())
            rec["stats"] = tm.stats()
        pred = tm.predict(Bev).cpu().numpy()
        if args.save_model:
            st = tm.state()
            mdir = ROOT / "results" / args.out
            mdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mdir / f"model_{args.learner}_{args.tag}_seed{args.seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"],
                                names=np.array(view.names))
    elif args.learner == "neural":
        ln = NeuralBandit(K, Btr.shape[1], lr=args.nn_lr, boltzmann=0.0 if args.nn_eps > 0 else args.boltzmann, eps=args.nn_eps, seed=args.seed)
        rec.update({"nn_lr": args.nn_lr, "nn_eps": args.nn_eps, "boltzmann": 0.0 if args.nn_eps > 0 else args.boltzmann})
        rec.update(run_stream(ln, Btr.float(), ytr, Bev.float(), yev, args.passes, args.batch, args.seed))
        pred = ln.greedy(Bev).cpu().numpy()
    else:
        if args.linucb_input == "bits":
            Xtr, Xev = Btr.float(), Bev.float()
        else:
            def raw(e, rows):
                return np.concatenate([e["recent"], e["cust"], memory_from_rows(rows, blist).astype(np.float32)], 1)

            Xtr = torch.as_tensor(raw(etr, tr), device="cuda")
            Xev = torch.as_tensor(raw(eev, ev), device="cuda")
        # the Woodbury version (same estimator, src/dl/bandit.py, tests/test_linucb_fast.py) since 2026-10-17 00:10 UTC;
        # the exploration runs e28_linucb* used the original class
        ln = (LinUCBFast if args.linucb_impl == "woodbury" else LinUCB)(K, Xtr.shape[1], alpha=args.alpha)
        rec.update({"alpha": args.alpha, "linucb_input": args.linucb_input, "linucb_impl": args.linucb_impl})
        rec.update(run_stream(ln, Xtr, ytr, Xev, yev, args.passes, args.batch, args.seed))
        pred = ln.greedy(Xev).cpu().numpy()
    # the paper's AST B-Slot accuracy (weighted by the AST example count of each action turn)
    ok = pred == np.array([r["y"] for r in ev])
    for k in ("bert", "roberta"):
        w = np.array([r[f"ast_count_{k}"] for r in ev], dtype=float)
        rec[f"final_ast_bslot_{k}"] = float((ok * w).sum() / w.sum())
    rec["seconds"] = round(time.time() - t0, 1)
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "abcd_bandit.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if args.eval == "test":
        np.savez(out / f"preds_{args.learner}_{args.tag}_seed{args.seed}.npz", pred=pred, y=yev.cpu().numpy())
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tm_cfg", "curve")}), flush=True)


if __name__ == "__main__":
    main()
