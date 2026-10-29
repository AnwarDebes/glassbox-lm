#!/usr/bin/env python
"""N7 follow-up F1 (EXPLORATORY, carved val only; rule in LOG.md before the run): calibrate on CAIR's machine type.

    n7_tmu_calib.py --dataset trec|mr --seed 1 [--view bow] [--epochs 100]

Machine: tmu 0.8.3 TMClassifier, the standard multi-class Tsetlin machine (one clause bank per class, half the
clauses positive and half negative, each voting +-1: weighted_clauses False; feature_negation True;
boost_true_positive_feedback 1; CPU, one thread), with Yadav et al. 2021's settings: clauses per class TREC 2,000 and
MR 3,000, T 80, s 9. Data: N7's explore caches (emb_n7/{ds}__explore.npz: our bag of words, TREC 6,000 words, MR
5,000; the 90/10 carved split of the published train set, numpy default_rng(2026)); the test split is never opened.
Each epoch = one tm.fit pass (tmu shuffles), then the val accuracy. Statistics: S1-val = mean val accuracy over
epochs 51-100 (Yadav et al.'s statistic, on val), best val. One row in results_n7_tmu.jsonl.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CLAUSES = {"trec": 2000, "mr": 3000}  # Yadav et al. 2021, per class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(CLAUSES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--view", default="bow")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="only print; write no row")
    args = ap.parse_args()
    from tmu.models.classification.vanilla_classifier import TMClassifier
    import tmu

    t0 = time.time()
    cache = HERE / "emb_n7" / f"{args.dataset}__explore.npz"
    z = np.load(cache)
    Xtr = np.ascontiguousarray(z[f"{args.view}_train"]).astype(np.uint32)
    Xva = np.ascontiguousarray(z[f"{args.view}_val"]).astype(np.uint32)
    ytr, yva = z["y_train"].astype(np.uint32), z["y_val"].astype(np.uint32)
    tm = TMClassifier(CLAUSES[args.dataset], 80, 9.0, platform="CPU", weighted_clauses=False, feature_negation=True,
                      boost_true_positive_feedback=1, seed=args.seed)
    curve = []
    for ep in range(args.epochs):
        tm.fit(Xtr, ytr)
        acc = float((tm.predict(Xva) == yva).mean()) * 100
        curve.append(round(acc, 3))
        print(f"epoch {ep + 1} val {acc:.2f} ({time.time() - t0:.0f}s)", flush=True)
    rec = {"followup": "N7-F1 (exploratory, carved val)", "dataset": args.dataset, "view": args.view, "seed": args.seed,
           "machine": f"tmu {getattr(tmu, '__version__', '?')} TMClassifier (standard multi-class TM, CPU)",
           "clauses_per_class": CLAUSES[args.dataset], "T": 80, "s": 9.0, "n": {"train": int(len(ytr)), "val": int(len(yva))},
           "bits": int(Xtr.shape[1]), "val_S1_mean_51_100": round(float(np.mean(curve[50:100])), 3) if len(curve) >= 100 else None,
           "val_best": max(curve), "val_best_epoch": int(np.argmax(curve)) + 1, "val_final": curve[-1], "curve": curve,
           "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()[:16],
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)
    if args.smoke:
        return
    with open(HERE / "results_n7_tmu.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
