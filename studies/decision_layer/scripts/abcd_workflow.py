#!/usr/bin/env python
"""Does the learned layer restate the written workflow? Transition rules against ABCD's guidelines.

The guidelines (data/untrusted/abcd/guidelines.json) list, per subflow, the buttons an agent
should press in order. Their consecutive pairs give the written transitions (a -> b); the
first button of each subflow gives the written first steps.

- TM: for each pair (a, b), the summed positive weight for class b of the clauses that include
  the plain literal "last action = a".
- LR: the weight of the bit "last action = a" for class b.
- Logs: how often b followed a in the training conversations.

Each source ranks all pairs; reported: precision at K = number of written transitions (the
share of the K top-ranked pairs that are written transitions), for every saved TM seed of a run,
the LR run and the logs. Nothing here is fitted on test data.

    abcd_workflow.py --registration R9 --tm tm_full --lr lr_full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCD_DIR, load_examples  # noqa: E402

RENAME = {"membership-privileges": "membership", "notify-internal-team": "notify-team", "log-out/in": "log-out-in",
          "enter-detail": "enter-details", "select-answer": "select-faq"}
FAQ_FLOWS = {"Single-Item Query", "Storewide Query"}


def written_workflows(buttons: list[str]) -> dict:
    g = json.loads((ABCD_DIR / "guidelines.json").read_text())
    out = {}
    for flow, v in g.items():
        for sf, w in v["subflows"].items():
            seq = []
            for i, a in enumerate(w["actions"]):
                b = a["button"].lower().replace(" ", "-")
                b = RENAME.get(b, b)
                if flow in FAQ_FLOWS and i == 1:
                    b = f"search-{b}"
                if b in buttons:
                    seq.append(b)
            out[f"{flow} / {sf}"] = seq
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="")
    ap.add_argument("--tm", default="")
    ap.add_argument("--lr", default="")
    ap.add_argument("--models", nargs="*", default=[], help="explicit TM model files (exploration)")
    ap.add_argument("--names", default="", help="bit names json (exploration)")
    args = ap.parse_args()
    tr, blist = load_examples("train")
    wf = written_workflows(blist)
    G = {(a, b) for seq in wf.values() for a, b in zip(seq, seq[1:]) if a != b}
    K = len(G)
    pairs = [(a, b) for a in blist for b in blist if a != b]

    def prec(score: dict) -> float:
        top = sorted(pairs, key=lambda p: -score.get(p, 0.0))[:K]
        return sum(p in G for p in top) / K

    res = {"n_written_transitions": K, "written": {k: v for k, v in wf.items()}}
    cnt = Counter()
    for r in tr:
        if r["prev"]:
            cnt[(r["prev"][-1], r["button"])] += 1
    res["logs_precision_at_K"] = prec({p: float(c) for p, c in cnt.items()})
    base = ROOT / "results" / "confirm" / args.registration if args.registration else None
    model_files = [Path(m) for m in args.models] or (sorted((base / args.tm).glob("model_seed*.npz")) if base and args.tm else [])
    if args.names:
        names = json.loads(Path(args.names).read_text())
    elif base:
        names = json.loads((base / args.tm / "bit_names.json").read_text())
    else:  # exploration model files carry their bit names
        names = [str(x) for x in np.load(model_files[0])["names"]]
    last_idx = {b: names.index(f"last action = {b}") for b in blist}
    res["tm_precision_at_K"] = {}
    for mf in model_files:
        m = np.load(mf)
        incl = m["ta"] > 256
        w = m["w"]
        score = {}
        for a in blist:
            js = np.flatnonzero(incl[:, last_idx[a]])
            if len(js) == 0:
                continue
            pos = np.clip(w[js], 0, None).sum(0)
            for bi, b in enumerate(blist):
                if b != a:
                    score[(a, b)] = float(pos[bi])
        res["tm_precision_at_K"][mf.stem] = prec(score)
    if base and args.lr:
        for wf_ in sorted((base / args.lr).glob("lr_weights_seed*.npz")):
            W = np.load(wf_)["W"]
            score = {(a, b): float(W[last_idx[a], bi]) for a in blist for bi, b in enumerate(blist) if a != b}
            res.setdefault("lr_precision_at_K", {})[wf_.stem] = prec(score)
    print(json.dumps({k: v for k, v in res.items() if k != "written"}, indent=1))
    if base:
        (base / "workflow_precision.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
