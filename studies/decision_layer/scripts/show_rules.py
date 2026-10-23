#!/usr/bin/env python
"""Print the learned clauses of a saved model as readable rules.

    show_rules.py router <model.npz> --dataset clinc150 --view lex:2000 [--intents a b c] [--queries 5]
    show_rules.py r6 <model.npz> [--per-action 6]

For a router model, the view is rebuilt from the training data (deterministic) to name the bits;
``--queries k`` also prints the decision receipt of the first k test queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.explain import decision_receipt, top_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["router", "r6"])
    ap.add_argument("model")
    ap.add_argument("--dataset", default="clinc150")
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--view", default="lex:2000")
    ap.add_argument("--intents", nargs="*", default=None)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--queries", type=int, default=0)
    args = ap.parse_args()
    m = np.load(args.model, allow_pickle=False)
    ta, w = m["ta"].astype(np.int32), m["w"]
    cfg = json.loads(str(m["cfg"]))
    neg = cfg.get("negation", True)
    if args.kind == "r6":
        names = [str(x) for x in m["names"]]
        actions = ["ROUTE", "ASK", "ESCALATE"]
        for line in top_rules(ta, w, names, actions, cfg["n_states"], per_class=args.per_class, negation=neg):
            print(line)
        return
    from dl.boolean import make_view
    from dl.data import load

    d = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    ds = load(args.dataset)
    view = make_view(args.view, seed=0)
    view.fit(d["train"], d["y_train"], texts=ds.texts["train"], class_names=ds.intents)
    classes = [ds.intents.index(c) for c in args.intents] if args.intents else None
    for line in top_rules(ta, w, view.names, ds.intents, cfg["n_states"], per_class=args.per_class,
                          classes=classes, negation=neg):
        print(line)
    if args.queries:
        B = view.transform(d["test"], ds.texts["test"])
        for i in range(args.queries):
            print(f'\nQUERY: "{ds.texts["test"][i]}"  (true: {ds.intents[d["y_test"][i]]})')
            print(decision_receipt(ta, w, B[i], view.names, ds.intents, cfg["n_states"], negation=neg))


if __name__ == "__main__":
    main()
