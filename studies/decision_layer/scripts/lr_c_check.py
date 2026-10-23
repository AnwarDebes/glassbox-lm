#!/usr/bin/env python
"""LR below the registered C grid on a cached view's bits, scored on DEV only (a post-hoc check of a grid edge).

    lr_c_check.py --dataset star --parts mem,words,xcust,xrecent --encoder potion-32m --C 0.01 0.03 0.1 --tag e67_star_B

Fits dl.linear.fit_logreg for each C on train, as scripts/abcd_confirm.py's lr arm does, and scores dev. One JSON
line per C goes to results/explore/abcd.jsonl (arm "lr_bits"). The test split is not read. Added 2026-10-18 after
R19: internal audit 9 noted that the word-table LR runs chose C = 0.1, the lowest value of the grid, and that lower values
had not been checked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.bitcache import view_bits  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DL_GPU_FRAC", "0.15")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="star", choices=["abcd", "star"])
    ap.add_argument("--parts", default="mem,words,xcust,xrecent")
    ap.add_argument("--encoder", default="potion-32m")
    ap.add_argument("--C", type=float, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    parts = tuple(args.parts.split(","))
    Xtr, ytr, Xdv, ydv, names, blist, _tr, _dv = view_bits(args.dataset, parts, encoder=args.encoder)
    for c in args.C:
        t0 = time.time()
        W, b = fit_logreg(Xtr, ytr, len(blist), c)
        acc = float((logits(Xdv, W, b).argmax(1).cpu().numpy() == ydv).mean())
        rec = {"tag": args.tag, "script": "lr_c_check.py", "dataset": args.dataset, "encoder": args.encoder, "parts": parts,
               "arm": "lr_bits", "C": c, "acc": acc, "n_bits": int(Xtr.shape[1]), "seconds": round(time.time() - t0, 1),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with (ROOT / "results" / "explore" / "abcd.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
