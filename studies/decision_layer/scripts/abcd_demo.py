#!/usr/bin/env python
"""Print whole ABCD test conversations as the registered TM agent sees them: at every action turn
the button the TM chooses, the human agent's button, and the clauses behind the choice.

    abcd_demo.py --registration R9 --name tm_mem --seed 1 --convos 3 [--parts ...]

Run after the registered runs (it reads the saved model; it trains nothing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402
from dl.explain import clause_literals  # noqa: E402


def short(lit: str) -> str:
    neg = lit.startswith("NOT ")
    lit = lit[4:] if neg else lit
    lit = re.sub(r"sim\((.+?), (.+?)\) >= (-?[0-9.]+)", r"near(\1: \2)", lit)
    lit = re.sub(r"^(.+?) is among the 1 nearest (subflow|next-button) centres of the (.+)$", r"(nearest \2 of the \3 = \1)", lit)
    lit = re.sub(r"^(.+?) is among the 3 nearest (subflow|next-button) centres of the (.+)$", r"(\1 in top 3 \2s of the \3)", lit)
    return ("NOT " + lit) if neg else lit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="R9")
    ap.add_argument("--name", default="tm_mem")
    ap.add_argument("--parts", default="mem,intent,step,words,mem2,fstep,fwords")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--convos", type=int, default=3)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--max-lits", type=int, default=8)
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration / args.name
    m = np.load(base / f"model_seed{args.seed}.npz")
    ta, w = m["ta"], m["w"]
    names = json.loads((base / "bit_names.json").read_text())
    tr, blist = load_examples("train")
    te, _ = load_examples("test")
    E = np.load(ROOT / "data" / "emb" / "abcd__bge-large.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    ete = {f: E[f"test__{f}"] for f in ("cust", "recent", "full")}
    view = ABCDView(parts=tuple(args.parts.split(",")), seed=0).fit(tr, etr, blist)
    assert view.names == names, "view differs from the registered run"
    X = view.transform(te, ete)
    f = X.shape[1]
    incl = ta > 256
    rng = np.random.default_rng(args.seed)
    cids = sorted({r["convo_id"] for r in te})
    pick = rng.choice(cids, size=args.convos, replace=False)
    for cid in pick:
        rows = [i for i, r in enumerate(te) if r["convo_id"] == cid]
        print(f"\n=== test conversation {cid} (subflow {te[rows[0]]['subflow']}) ===")
        print(f"customer: {te[rows[-1]]['cust'][:300]}")
        for i in rows:
            r = te[i]
            lit = np.concatenate([X[i], 1 - X[i]])
            fired = ((incl & (lit[None, :] == 0)).sum(1) == 0) & (incl.sum(1) > 0)
            S = (fired[:, None] * w).sum(0)
            c = int(np.argmax(S))
            ok = "right" if blist[c] == r["button"] else f"WRONG (agent pressed {r['button']})"
            print(f"\n  [{r['k'] + 1}] last 3 turns: {r['recent'][:160]}")
            print(f"      TM presses {blist[c]} -- {ok}; vote {int(S[c])}, runner-up {blist[int(np.argsort(-S)[1])]} {int(np.sort(S)[-2])}; {int(fired.sum())} clauses fire")
            contrib = fired * w[:, c]
            for j in np.argsort(-contrib)[: args.top]:
                if contrib[j] <= 0:
                    break
                L = [short(x) for x in clause_literals(ta[j], 256, f, names)]
                pos = [x for x in L if not x.startswith("NOT ")]
                neg = [x for x in L if x.startswith("NOT ")]
                shown = (pos + neg)[: args.max_lits]
                more = len(L) - len(shown)
                print(f"      +{int(contrib[j])}: IF " + " AND ".join(shown) + (f" AND {more} more NOT-literals" if more else ""))


if __name__ == "__main__":
    main()
