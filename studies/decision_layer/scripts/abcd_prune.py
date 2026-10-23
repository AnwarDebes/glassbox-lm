#!/usr/bin/env python
"""Shorter clauses after training: drop the exclusions that never exclude anything in training.

A negated literal NOT x in a clause can only matter on an example where the rest of the clause
holds and x = 1. Rule ``unused``: drop NOT x when x = 0 on every training example on which all
of the clause's plain literals hold. The clause then gives the same output on every training
example (it could only fire more where the plain literals hold and x = 1, which never happens
in training), so the machine's training-set votes are unchanged; on new examples it can fire more.
Rule ``rare:m``: drop NOT x when x = 1 on at most m such training examples (changes training votes
a little; reported only).

Scored: accuracy before and after, clause lengths of all clauses and of the strongest ones, on dev
(exploration) or test (registered runs, ``--out``). Works on any saved CoTM over the ABCD view.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402


def fired(ta_incl: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """[n, N] clause outputs; ta_incl [N, 2f] bool; X [n, f] uint8 (on the GPU). An empty clause is silent."""
    lit0 = torch.cat([1 - X, X], 1).to(torch.float16)  # 1 where the literal is FALSE
    viol = lit0 @ ta_incl.to(torch.float16).T
    return (viol == 0) & (ta_incl.sum(1) > 0)[None, :]


def prune(ta_incl: np.ndarray, Xtr: np.ndarray, f: int, max_hits: int = 0, chunk: int = 256) -> np.ndarray:
    inc = torch.as_tensor(ta_incl, device="cuda")
    X = torch.as_tensor(Xtr, device="cuda", dtype=torch.float16)
    out = inc.clone()
    pos = inc[:, :f]
    for a in range(0, inc.shape[0], chunk):
        P = pos[a : a + chunk].to(torch.float16)  # [c, f]
        # examples where all plain literals of the clause hold: count of (plain literal true) == number of plain literals
        hold = (X @ P.T) == P.sum(1)[None, :]  # [n, c]
        # for each feature x: on how many such examples is x = 1 (i.e. NOT x would exclude)
        hits = hold.T.to(torch.float16) @ X  # [c, f]
        neg = inc[a : a + chunk, f:]
        drop = neg & (hits <= max_hits)
        kept = neg & ~drop
        # never turn a clause into an empty one (an empty clause is silent at inference)
        empty = (pos[a : a + chunk].sum(1) == 0) & (kept.sum(1) == 0)
        kept[empty] = neg[empty]
        out[a : a + chunk, f:] = kept
    return out.cpu().numpy()


def lengths(incl: np.ndarray, w: np.ndarray, f: int, top: int = 200) -> dict:
    L = incl.sum(1)
    P = incl[:, :f].sum(1)
    order = np.argsort(-np.abs(w).max(1))[:top]
    ne = L > 0
    return {"median": float(np.median(L)), "mean": float(L.mean()), "max": int(L.max()),
            "share_nonempty_with_plain": float(((P > 0) & ne).sum() / ne.sum()),
            "top200_mean": float(L[order].mean()), "top200_neg_mean": float((L - P)[order].mean()),
            "top200_le8": float((L[order] <= 8).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="R9b")
    ap.add_argument("--name", default="tm_mem_short")
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--rule", default="unused", help="unused, or rare:m")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("test split only in registered runs (--out <registration>)")
    max_hits = 0 if args.rule == "unused" else int(args.rule.split(":")[1])
    tr, blist = load_examples("train")
    ev, _ = load_examples(args.split)
    E = np.load(ROOT / "data" / "emb" / "abcd__bge-large.npz")
    em = lambda s: {f_: E[f"{s}__{f_}"] for f_ in ("cust", "recent", "full")}  # noqa: E731
    view = ABCDView(parts=tuple(args.parts.split(",")), seed=0).fit(tr, em("train"), blist)
    src = ROOT / "results" / "confirm" / args.registration / args.name
    assert view.names == json.loads((src / "bit_names.json").read_text())
    Xtr, Xev = view.transform(tr, em("train")), view.transform(ev, em(args.split))
    y = np.array([r["y"] for r in ev])
    ytr = np.array([r["y"] for r in tr])
    f = Xtr.shape[1]
    Xtr_t = torch.as_tensor(Xtr, device="cuda")
    Xev_t = torch.as_tensor(Xev, device="cuda")
    for seed in args.seeds:
        t0 = time.time()
        m = np.load(src / f"model_seed{seed}.npz")
        incl = m["ta"] > 256
        w = m["w"].astype(np.int64)
        new = prune(incl, Xtr, f, max_hits)
        wt = torch.as_tensor(w, device="cuda", dtype=torch.float32)
        S_old = fired(torch.as_tensor(incl, device="cuda"), Xev_t).float() @ wt
        S_new = fired(torch.as_tensor(new, device="cuda"), Xev_t).float() @ wt
        Str_old = fired(torch.as_tensor(incl, device="cuda"), Xtr_t).float() @ wt
        Str_new = fired(torch.as_tensor(new, device="cuda"), Xtr_t).float() @ wt
        p_old, p_new = S_old.argmax(1).cpu().numpy(), S_new.argmax(1).cpu().numpy()
        rec = {"tag": args.tag, "registration": args.registration, "name": args.name, "seed": seed, "rule": args.rule, "split": args.split,
               "acc_before": float((p_old == y).mean()), "acc_after": float((p_new == y).mean()), "same_decision": float((p_old == p_new).mean()),
               "train_votes_identical": bool(torch.equal(Str_old, Str_new)),
               "train_acc_before": float((Str_old.argmax(1).cpu().numpy() == ytr).mean()), "train_acc_after": float((Str_new.argmax(1).cpu().numpy() == ytr).mean()),
               "literals_before": lengths(incl, w, f), "literals_after": lengths(new, w, f), "seconds": round(time.time() - t0, 1),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if args.split == "test":
            out = ROOT / "results" / "confirm" / args.out / f"{args.name}_pruned_{args.rule.replace(':', '')}"
            out.mkdir(parents=True, exist_ok=True)
            base = {"y": y, "convo_id": np.array([r["convo_id"] for r in ev]), "k": np.array([r["k"] for r in ev]),
                    "w_bert": np.array([r["ast_count_bert"] for r in ev]), "w_roberta": np.array([r["ast_count_roberta"] for r in ev])}
            np.savez_compressed(out / f"seed{seed}.npz", pred=p_new, **base)
            (out / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
            np.savez_compressed(out / f"pruned_model_seed{seed}.npz", incl=new, w=w)
        else:
            with (ROOT / "results" / "explore" / "abcd_prune.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
