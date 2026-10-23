#!/usr/bin/env python
"""Exploratory baselines on raw frozen embeddings, scored on VAL only.

Logistic regression with C chosen on val from a grid, and cosine kNN (k = 10).
For CLINC150 also the out-of-scope AUROC on val (3,000 in-scope + 100 oos_val) with the
maximum softmax probability. Writes results/explore/baselines.jsonl. No test split is read.
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

from dl.linear import accuracy, auroc, fit_logreg, logits  # noqa: E402

GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["clinc150", "banking77", "hwu64"])
    ap.add_argument("--encoders", nargs="+", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "explore"
    out.mkdir(parents=True, exist_ok=True)
    for enc in args.encoders:
        for ds in args.datasets:
            p = ROOT / "data" / "emb" / f"{ds}__{enc}.npz"
            if not p.exists():
                print("missing", p)
                continue
            d = np.load(p)
            Xtr, ytr, Xva, yva = d["train"], d["y_train"], d["val"], d["y_val"]
            C_cls = int(max(ytr.max(), yva.max()) + 1)
            rec = {"dataset": ds, "encoder": enc, "dim": int(Xtr.shape[1]), "logreg_val": {}}
            t0 = time.time()
            best = (-1, None, None)
            for C in GRID:
                W, b = fit_logreg(Xtr, ytr, C_cls, C)
                acc = accuracy(logits(Xva, W, b), yva)
                rec["logreg_val"][str(C)] = round(acc, 4)
                if acc > best[0]:
                    best = (acc, C, (W, b))
            rec["logreg_best_C"] = best[1]
            rec["logreg_best_val"] = round(best[0], 4)
            if "oos_val" in d.files:
                W, b = best[2]
                p_in = torch.softmax(logits(Xva, W, b), 1).max(1).values.cpu().numpy()
                p_oos = torch.softmax(logits(d["oos_val"], W, b), 1).max(1).values.cpu().numpy()
                is_oos = np.r_[np.zeros(len(p_in)), np.ones(len(p_oos))]
                rec["logreg_oos_auroc_val"] = round(auroc(is_oos, -np.r_[p_in, p_oos]), 4)
            # cosine kNN, k = 10, majority vote (embeddings are L2-normalised)
            A = torch.as_tensor(Xtr, device="cuda")
            Bv = torch.as_tensor(Xva, device="cuda")
            sims = Bv @ A.T
            top = sims.topk(10, dim=1).indices
            votes = torch.zeros((len(Xva), C_cls), device="cuda")
            votes.scatter_add_(1, torch.as_tensor(ytr, device="cuda")[top], torch.ones_like(top, dtype=torch.float))
            rec["knn10_val"] = round(accuracy(votes, yva), 4)
            rec["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(rec))
            with open(out / "baselines.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
