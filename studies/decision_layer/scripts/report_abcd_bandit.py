#!/usr/bin/env python
"""R10 report: ABCD next action learned from reward only. Final greedy test accuracy per learner
(mean over seeds), online accuracy, and paired differences with a bootstrap over test
conversations (10,000 resamples, seed 0, 95%)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl.abcd import load_examples  # noqa: E402
from report_abcd import cluster_boot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", default="R10")
    ap.add_argument("--arms", nargs="+", required=True, help="learner_tag names, e.g. tm_R10_tm")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", default="report_R10.json")
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration
    te, _ = load_examples("test")
    cid = np.array([r["convo_id"] for r in te])
    w = np.array([r["ast_count_bert"] for r in te], dtype=float)
    recs = [json.loads(l) for l in (base / "abcd_bandit.jsonl").read_text().splitlines() if l.strip()]
    out = {"arms": {}, "pairs": {}}
    corr = {}
    for a in args.arms:
        files = sorted(base.glob(f"preds_{a}_seed*.npz"))
        C = np.stack([(np.load(f)["pred"] == np.load(f)["y"]) for f in files]).astype(float)
        corr[a] = C.mean(0)
        learner = next(L for L in ("sup_tm", "linucb", "neural", "tm") if a.startswith(L + "_"))
        tag = a[len(learner) + 1:]
        rs = [r for r in recs if r["learner"] == learner and r["tag"] == tag]
        out["arms"][a] = {"n_seeds": len(files), "test_acc": float(C.mean()), "test_acc_per_seed": [round(float(x), 5) for x in C.mean(1)],
                          "ast_weighted": float((C.mean(0) * w).sum() / w.sum()),
                          "online_acc_mean": float(np.mean([r["cum_online_acc"] for r in rs])) if rs and "cum_online_acc" in rs[0] else None}
    ones = np.ones(len(te))
    for p in args.pairs:
        a, b = p.split(":")
        out["pairs"][p] = cluster_boot(corr[a] - corr[b], ones, cid)
        if b.startswith("sup"):
            out["pairs"][p]["ratio"] = out["arms"][a]["test_acc"] / out["arms"][b]["test_acc"]
    (base / args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
