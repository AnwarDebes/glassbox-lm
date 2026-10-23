#!/usr/bin/env python
"""Train the ABCD decision layer for the closed loop (its memory bits come from its own presses).

Teacher-forced training (R9b) sees only the human agent's logged history in its memory bits; in
the closed loop the model sees its own, sometimes wrong, history. Two remedies:

- ``dagger``: data aggregation (Ross et al. 2011, one round, cross-fitted). The training
  conversations are split into K folds; a TM trained on K-1 folds decides the held-out fold's
  action turns in closed loop, which gives each training action turn the memory bits the model
  itself would have had there. The final TM trains on the logged-memory examples plus these
  own-memory examples (labels: the human agent's buttons in both).
- ``noise``: each earlier press in the memory is replaced by a random button with probability p,
  and the final TM trains on the logged-memory examples plus these noisy copies.

Scored: accuracy with the logged history (open loop) and in closed loop, on dev (exploration) or
test (registered runs, ``--out <registration>``).
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
sys.path.insert(1, str(ROOT / "scripts"))

from abcd_closed_loop import closed_loop  # noqa: E402
from dl.abcd import ABCDView, load_examples, memory_from_rows  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dagger", "noise", "plain"])
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--fold-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("test split only in registered runs (--out <registration>)")
    t0 = time.time()
    parts = tuple(args.parts.split(","))
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    ev = dv if args.split == "dev" else load_examples("test")[0]
    E = np.load(ROOT / "data" / "emb" / f"abcd__{args.encoder}.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev", args.split)}
    view = ABCDView(parts=parts, seed=0).fit(tr, em["train"], blist)
    Xtr, Xdv = view.transform(tr, em["train"]), view.transform(dv, em["dev"])
    Xev = Xdv if args.split == "dev" else view.transform(ev, em[args.split])
    ytr = np.array([r["y"] for r in tr])
    ydv = np.array([r["y"] for r in dv])
    yev = np.array([r["y"] for r in ev])
    n_mem = 2 * len(blist) + 6
    C_ = len(blist)
    rng = np.random.default_rng(args.seed + 100)

    def cfg(epochs, seed):
        return TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=epochs, seed=seed)

    def scorer(tm):
        return lambda Z: tm.class_sums(torch.as_tensor(Z, dtype=torch.uint8, device="cuda")).cpu().numpy()

    extra_X = None
    info = {}
    if args.mode == "dagger":
        convs = np.array(sorted({r["convo_id"] for r in tr}))
        fold_of = dict(zip(rng.permutation(convs), np.arange(len(convs)) % args.folds))
        f_row = np.array([fold_of[r["convo_id"]] for r in tr])
        extra_X = Xtr.copy()
        fold_acc = []
        for f in range(args.folds):
            tm_f = make_tm(C_, Xtr.shape[1], cfg(args.fold_epochs, args.seed * 100 + f), init="half")
            tm_f.fit(Xtr[f_row != f], ytr[f_row != f], Xdv, ydv)
            held = np.flatnonzero(f_row == f)
            rows_f = [tr[i] for i in held]
            p = closed_loop(scorer(tm_f), Xtr[held], rows_f, blist, n_mem)
            # the own-memory bits of the held-out rows, as closed_loop built them
            prev_own = {}
            by_conv = {}
            for j, r in enumerate(rows_f):
                by_conv.setdefault(r["convo_id"], []).append(j)
            fake = [None] * len(rows_f)
            for idx in by_conv.values():
                idx.sort(key=lambda j: rows_f[j]["k"])
                for kk, j in enumerate(idx):
                    fake[j] = {"prev": [blist[p[i]] for i in idx[:kk]]}
            extra_X[held, :n_mem] = memory_from_rows(fake, blist)
            fold_acc.append(float((p == ytr[held]).mean()))
        info["fold_closed_acc"] = fold_acc
        info["share_memory_rows_changed"] = float((extra_X[:, :n_mem] != Xtr[:, :n_mem]).any(1).mean())
    elif args.mode == "noise":
        fake = []
        for r in tr:
            prev = [b if rng.random() >= args.noise else blist[int(rng.integers(0, C_))] for b in r["prev"]]
            fake.append({"prev": prev})
        extra_X = Xtr.copy()
        extra_X[:, :n_mem] = memory_from_rows(fake, blist)
        info["share_memory_rows_changed"] = float((extra_X[:, :n_mem] != Xtr[:, :n_mem]).any(1).mean())
    X_fit = Xtr if extra_X is None else np.concatenate([Xtr, extra_X])
    y_fit = ytr if extra_X is None else np.concatenate([ytr, ytr])
    c = cfg(args.epochs, args.seed)
    tm = make_tm(C_, X_fit.shape[1], c, init="half")
    tm.fit(X_fit, y_fit, Xdv, ydv)
    sc = scorer(tm)
    p_open = sc(Xev).argmax(1)
    p_closed = closed_loop(sc, Xev, ev, blist, n_mem)
    rec = {"tag": args.tag, "mode": args.mode, "parts": parts, "seed": args.seed, "split": args.split, "cfg": asdict(c),
           "folds": args.folds, "fold_epochs": args.fold_epochs, "noise": args.noise, "n_fit": int(len(y_fit)),
           "acc_open": float((p_open == yev).mean()), "acc_closed": float((p_closed == yev).mean()),
           "best_epoch": getattr(tm, "best_epoch", None), "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}
    if args.split == "test":
        out = ROOT / "results" / "confirm" / args.out / f"tm_{args.mode}"
        out.mkdir(parents=True, exist_ok=True)
        ks = np.array([r["k"] for r in ev])
        base = {"y": yev, "convo_id": np.array([r["convo_id"] for r in ev]), "k": ks,
                "w_bert": np.array([r["ast_count_bert"] for r in ev]), "w_roberta": np.array([r["ast_count_roberta"] for r in ev])}
        np.savez_compressed(out / f"seed{args.seed}.npz", pred=p_closed, **base)
        (out / f"seed{args.seed}.json").write_text(json.dumps(rec, indent=1))
        oo = ROOT / "results" / "confirm" / args.out / f"tm_{args.mode}_open"
        oo.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oo / f"seed{args.seed}.npz", pred=p_open, **base)
        (oo / f"seed{args.seed}.json").write_text(json.dumps(rec, indent=1))
        st = tm.state()
        np.savez_compressed(out / f"model_seed{args.seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"])
    else:
        with (ROOT / "results" / "explore" / "abcd_closed_loop.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "cfg"}), flush=True)


if __name__ == "__main__":
    main()
