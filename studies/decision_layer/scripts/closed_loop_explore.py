#!/usr/bin/env python
"""Closed loop on DEV for a saved exploration model (scripts/abcd_explore.py --save): the memory bits come from
the model's own earlier presses, as in scripts/abcd_closed_loop.py, whose closed_loop() this reuses.

    closed_loop_explore.py --dataset star --parts mem,cwords,xcust,xcrecent --encoder potion-32m \
        --model results/explore/abcd_models/<tag>_seed1.npz --tag e65_closed

- The view is fitted on train as in the run that made the model (lex_k 1000, seed 0). Its names must equal the
  model's saved names.
- The decision runs on the CPU (src/dl/cpu_tm_fast.py, class sums equal to CoTM's).
- Output: open and closed accuracy on dev, overall and by position, as one JSON line in
  results/explore/abcd_closed_loop.jsonl. The test split is not read.
Added 2026-10-18 for R20's dev work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abcd_closed_loop import closed_loop  # noqa: E402
from dl import cpu_tm_fast  # noqa: E402
from dl.abcd import ABCDView, load_examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star"])
    ap.add_argument("--parts", required=True)
    ap.add_argument("--encoder", default="potion-32m")
    ap.add_argument("--model", required=True, help="a model saved by abcd_explore.py --save (with its bit names)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    parts = tuple(args.parts.split(","))
    assert parts[0] == "mem", "memory bits must be the first block"
    tr, blist = load_examples("train", args.dataset)
    dv, _ = load_examples("dev", args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    edv = {f: E[f"dev__{f}"] for f in ("cust", "recent", "full")}
    view = ABCDView(parts=parts, seed=0).fit(tr, etr, blist)
    z = np.load(ROOT / args.model)
    assert [str(n) for n in z["names"]] == view.names, "view differs from the model's"
    X = view.transform(dv, edv)
    m = cpu_tm_fast.pack(z["ta"], z["w"])

    def score(Z):
        return cpu_tm_fast.class_sums(m, Z)

    t0 = time.time()
    y = np.array([r["y"] for r in dv])
    ks = np.array([r["k"] for r in dv])
    p_open = score(X).argmax(1)
    p_closed = closed_loop(score, X, dv, blist, 2 * len(blist) + 6)
    rec = {"tag": args.tag, "script": "closed_loop_explore.py", "dataset": args.dataset, "parts": parts, "encoder": args.encoder,
           "model": args.model, "kind": "tm", "split": "dev", "acc_open": float((p_open == y).mean()), "acc_closed": float((p_closed == y).mean()),
           "open_by_index": {str(k): float((p_open == y)[ks == k].mean()) for k in range(5)} | {"5+": float((p_open == y)[ks >= 5].mean())},
           "closed_by_index": {str(k): float((p_closed == y)[ks == k].mean()) for k in range(5)} | {"5+": float((p_closed == y)[ks >= 5].mean())},
           "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (ROOT / "results" / "explore" / "abcd_closed_loop.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
