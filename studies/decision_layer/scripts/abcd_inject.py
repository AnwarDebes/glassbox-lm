#!/usr/bin/env python
"""Revise a trained policy by hand: the written guidelines added to the TM as clauses.

The plan says the policy "can be revised and verified directly". Here the revision is the
company's own written workflow (data/untrusted/abcd/guidelines.json): for each subflow s and each
written step a -> b, one clause

    IF (s is the nearest subflow of the customer's words) AND (last action = a) AND NOT done(b)
    THEN vote w for b

and for each subflow's first written step b0: IF (nearest subflow = s) AND NOT (at least 1 action
so far) THEN vote w for b0. The clauses are appended to a trained machine (its own clauses are
not changed). Each added clause can be read and checked on its own: whenever its literals hold,
button b gets exactly w more votes, for every possible input.

Scored: accuracy of the machine with and without the added clauses, on dev (exploration) or test
(registered runs, ``--out``), for each w in ``--weights``.
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
sys.path.insert(1, str(ROOT / "scripts"))

from abcd_prune import fired  # noqa: E402
from abcd_workflow import FAQ_FLOWS, RENAME  # noqa: E402
from dl.abcd import ABCD_DIR, ABCDView, load_examples, ontology  # noqa: E402


def guideline_steps(buttons: list[str]) -> dict[str, list[str]]:
    """Ontology subflow name -> the written buttons in order (guideline subflows map to the
    ontology's subflows by flow name and position, which agree in both files)."""
    g = json.loads((ABCD_DIR / "guidelines.json").read_text())
    ont = ontology()["intents"]["subflows"]
    out = {}
    for flow, v in g.items():
        key = flow.lower().replace("-", "_").replace(" ", "_")
        sub_names = ont[key]
        for (sf, w), name in zip(v["subflows"].items(), sub_names):
            seq = []
            for i, a in enumerate(w["actions"]):
                b = RENAME.get(a["button"].lower().replace(" ", "-"), a["button"].lower().replace(" ", "-"))
                if flow in FAQ_FLOWS and i == 1:
                    b = f"search-{b}"
                if b in buttons:
                    seq.append(b)
            out[name] = seq
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="R9b")
    ap.add_argument("--name", default="tm_mem_short")
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--weights", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.out:
        raise SystemExit("test split only in registered runs (--out <registration>)")
    tr, blist = load_examples("train")
    ev, _ = load_examples(args.split)
    E = np.load(ROOT / "data" / "emb" / "abcd__bge-large.npz")
    em = lambda s: {f_: E[f"{s}__{f_}"] for f_ in ("cust", "recent", "full")}  # noqa: E731
    view = ABCDView(parts=tuple(args.parts.split(",")), seed=0).fit(tr, em("train"), blist)
    src = ROOT / "results" / "confirm" / args.registration / args.name
    names = json.loads((src / "bit_names.json").read_text())
    assert view.names == names
    X = view.transform(ev, em(args.split))
    y = np.array([r["y"] for r in ev])
    f = X.shape[1]
    idx = {n: i for i, n in enumerate(names)}
    steps = guideline_steps(blist)
    rules = []  # (literal indices plain, literal indices negated, button)
    for s, seq in steps.items():
        near = f"{s} is among the 1 nearest subflow centres of the customer so far"
        if near not in idx or not seq:
            continue
        rules.append(([idx[near]], [idx["at least 1 actions so far"]], seq[0]))
        for a, b in zip(seq, seq[1:]):
            if a != b:
                rules.append(([idx[near], idx[f"last action = {a}"]], [idx[f"done({b})"]], b))
    R = len(rules)
    add_incl = np.zeros((R, 2 * f), dtype=bool)
    add_cls = np.zeros(R, dtype=np.int64)
    for j, (pos, neg, b) in enumerate(rules):
        add_incl[j, pos] = True
        add_incl[j, [f + n for n in neg]] = True
        add_cls[j] = blist.index(b)
    Xt = torch.as_tensor(X, device="cuda")
    F_add = fired(torch.as_tensor(add_incl, device="cuda"), Xt).float()  # [n, R]
    for seed in args.seeds:
        m = np.load(src / f"model_seed{seed}.npz")
        incl = torch.as_tensor(m["ta"] > 256, device="cuda")
        w = torch.as_tensor(m["w"].astype(np.float32), device="cuda")
        S = fired(incl, Xt).float() @ w
        p0 = S.argmax(1).cpu().numpy()
        rec = {"tag": args.tag, "registration": args.registration, "name": args.name, "seed": seed, "split": args.split,
               "n_rules": R, "share_turns_with_a_rule_firing": float((F_add.sum(1) > 0).float().mean()),
               "acc_before": float((p0 == y).mean()), "after": {}, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for wt in args.weights:
            W_add = torch.zeros((R, len(blist)), device="cuda")
            W_add[torch.arange(R), torch.as_tensor(add_cls, device="cuda")] = float(wt)
            p1 = (S + F_add @ W_add).argmax(1).cpu().numpy()
            rec["after"][str(wt)] = {"acc": float((p1 == y).mean()), "changed": float((p1 != p0).mean()),
                                     "fixed": int(((p1 == y) & (p0 != y)).sum()), "broken": int(((p1 != y) & (p0 == y)).sum())}
            if args.split == "test":
                out = ROOT / "results" / "confirm" / args.out / f"{args.name}_rules_w{wt}"
                out.mkdir(parents=True, exist_ok=True)
                base = {"y": y, "convo_id": np.array([r["convo_id"] for r in ev]), "k": np.array([r["k"] for r in ev]),
                        "w_bert": np.array([r["ast_count_bert"] for r in ev]), "w_roberta": np.array([r["ast_count_roberta"] for r in ev])}
                np.savez_compressed(out / f"seed{seed}.npz", pred=p1, **base)
                (out / f"seed{seed}.json").write_text(json.dumps({**rec, "weight": wt}, indent=1))
        if args.split == "dev":
            with (ROOT / "results" / "explore" / "abcd_inject.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    if args.split == "test":
        (ROOT / "results" / "confirm" / args.out / "added_rules.json").write_text(json.dumps(
            [{"plain": [names[i] for i in pos], "negated": [names[i] for i in neg], "button": b} for pos, neg, b in rules], indent=1))


if __name__ == "__main__":
    main()
