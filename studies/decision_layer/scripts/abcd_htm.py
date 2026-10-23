#!/usr/bin/env python
"""Granmo's hierarchical Tsetlin machine (PyHierarchicalTsetlinMachineCUDA, commit 5b4a223, in
.venv-htm) as the ABCD decision layer, against the flat machine of the same package.

Run with .venv-htm/bin/python. Input: the readable view exported by scripts/abcd_export_bits.py
(parts mem, intent, step, words; block sizes 66 / 335 / 180 / 1,000). Each block is zero-padded to
F = 1,000 features, so the input has G = 4 groups of F features:

- ``flat``: ((AND_GROUP, F), (AND_GROUP, G)): a clause is one conjunction of literals over all
  groups (an ordinary TM clause);
- ``hier``: ((AND_GROUP, F), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, K), (AND_GROUP, G)): within
  each group a clause holds K OR-groups of A options, each option a conjunction of that group's
  literals, and the groups are AND-ed: (a1 OR a2 ...) AND (...) AND ... per block.

One binary machine per button (the package's MultiClassTsetlinMachine: one-vs-rest with balanced
negatives), negated literals on, 8 state bits. Scored on dev (exploration; best epoch reported) or
test (registered runs: the epoch is fixed in advance, no snapshot on test).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def padded(X: np.ndarray, sizes: list[int], F: int) -> np.ndarray:
    out = np.zeros((X.shape[0], F * len(sizes)), dtype=np.uint32)
    a = 0
    for g, n in enumerate(sizes):
        out[:, g * F : g * F + n] = X[:, a : a + n]
        a += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hier", choices=["hier", "flat"])
    ap.add_argument("--clauses", type=int, default=200, help="clauses per button machine")
    ap.add_argument("--T", type=float, default=200)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--A", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--F", type=int, default=1000)
    ap.add_argument("--boost", type=int, default=1)
    ap.add_argument("--state-bits", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1, help="dev accuracy every N epochs and after the last (prediction is slow: one kernel per example)")
    ap.add_argument("--no-negation", action="store_true")
    ap.add_argument("--batch", type=int, default=5000, help="examples per fit() call (as the thesis driver)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="short")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("test split only in registered runs (--out <registration>)")
    import PyHierarchicalTsetlinMachineCUDA.tm as htm

    t0 = time.time()
    tr = np.load(ROOT / "data" / "abcd" / f"bits_{args.name}_train.npz")
    dv = np.load(ROOT / "data" / "abcd" / f"bits_{args.name}_dev.npz")
    sizes = [int(x) for x in tr["sizes"]]
    assert max(sizes) <= args.F
    Xtr, ytr = padded(tr["X"], sizes, args.F), tr["y"].astype(np.uint32)
    Xdv, ydv = padded(dv["X"], sizes, args.F), dv["y"]
    G = len(sizes)
    if args.model == "hier" and args.K > 1:
        hs = ((htm.AND_GROUP, args.F), (htm.OR_ALTERNATIVES, args.A), (htm.AND_ALTERNATIVES, args.K), (htm.AND_GROUP, G))
    elif args.model == "hier":  # the Hex demo's shape: one OR over A options per block, blocks AND-ed
        hs = ((htm.AND_GROUP, args.F), (htm.OR_ALTERNATIVES, args.A), (htm.AND_GROUP, G))
    else:
        hs = ((htm.AND_GROUP, args.F), (htm.AND_GROUP, G))
    tm = htm.MultiClassTsetlinMachine(args.clauses, args.T, args.s, hierarchy_structure=hs, append_negated=not args.no_negation,
                                      number_of_state_bits=args.state_bits, boost_true_positive_feedback=args.boost, seed=args.seed)
    np.random.seed(args.seed)  # the package balances and shuffles with the global numpy RNG
    hist = []
    best = (-1.0, 0)
    for ep in range(1, args.epochs + 1):
        t1 = time.time()
        perm = np.random.permutation(len(ytr))
        for b in np.array_split(perm, max(1, len(perm) // args.batch)):
            tm.fit(Xtr[b], ytr[b])
        t_fit = time.time() - t1
        if ep % args.eval_every and ep != args.epochs:
            print(json.dumps({"epoch": ep, "fit_s": round(t_fit, 1)}), flush=True)
            continue
        # scoring uses no random state (the package's _score), so skipping it leaves training unchanged
        acc = float((tm.predict(Xdv) == ydv).mean())
        hist.append({"epoch": ep, "dev_acc": acc, "fit_s": round(t_fit, 1), "epoch_s": round(time.time() - t1, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if acc > best[0]:
            best = (acc, ep)
    rec = {"tag": args.tag, "model": args.model, "hierarchy": [[int(t), int(n)] for t, n in hs], "clauses": args.clauses, "T": args.T,
           "s": args.s, "A": args.A, "K": args.K, "epochs": args.epochs, "boost": args.boost, "seed": args.seed, "split": args.split,
           "final_dev_acc": hist[-1]["dev_acc"], "best_dev_acc": best[0], "best_epoch": best[1], "history": hist,
           "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.split == "test":
        te = np.load(ROOT / "data" / "abcd" / f"bits_{args.name}_test.npz")
        Xte = padded(te["X"], sizes, args.F)
        pred = tm.predict(Xte)
        rec["test_acc"] = float((pred == te["y"]).mean())
        out = ROOT / "results" / "confirm" / args.out / f"htm_{args.model}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"seed{args.seed}.npz", pred=pred, y=te["y"], convo_id=te["convo_id"], k=te["k"],
                            w_bert=te["w_bert"], w_roberta=te["w_bert"])
        (out / f"seed{args.seed}.json").write_text(json.dumps(rec, indent=1))
    else:
        with (ROOT / "results" / "explore" / "abcd_htm.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
