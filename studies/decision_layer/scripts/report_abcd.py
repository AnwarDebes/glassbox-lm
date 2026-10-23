#!/usr/bin/env python
"""Summarise registered ABCD runs: accuracy per arm and paired differences with a cluster bootstrap.

    report_abcd.py --registration R9 --arms tm_full lr_full ... --pairs tm_full:tm_nomem tm_full:lr_full ...

Per test action turn, each arm's correctness is averaged over its seeds. A difference A - B is
the mean over action turns of (A - B); its 95% interval comes from a bootstrap over
conversations (the action turns of one conversation are resampled together), 10,000 resamples,
seed 0. ``_ast`` rows weight every action turn by its number of AST examples (the paper's
B-Slot accuracy, BERT example set).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_arm(folder: Path):
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
    Z = [np.load(folder / f"seed{k}.npz") for k in seeds]
    meta = {k: Z[0][k] for k in ("y", "convo_id", "k", "w_bert", "w_roberta")}
    corr = np.stack([z["pred"] == meta["y"] for z in Z]).astype(float)
    for z in Z[1:]:
        assert np.array_equal(z["y"], meta["y"])
    return seeds, corr, meta


def cluster_boot(d: np.ndarray, w: np.ndarray, cid: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """Mean of d weighted by w, with a bootstrap over clusters cid."""
    uniq, inv = np.unique(cid, return_inverse=True)
    D = np.bincount(inv, weights=d * w, minlength=len(uniq))
    Wc = np.bincount(inv, weights=w, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    bs = D[idx].sum(1) / Wc[idx].sum(1)
    return {"diff": float(D.sum() / Wc.sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--composite", default="", help="TM arm whose two disjoint 5-machine composites (seeds 1-5, 6-10; sum of class sums) become arm <name>_composite")
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration
    arms, meta = {}, None
    out = {"arms": {}, "pairs": {}}
    for a in args.arms:
        seeds, corr, m = load_arm(base / a)
        if meta is None:
            meta = m
        assert np.array_equal(m["y"], meta["y"]) and np.array_equal(m["convo_id"], meta["convo_id"])
        arms[a] = corr.mean(0)
        w = meta["w_bert"].astype(float)
        accs = corr.mean(1)
        out["arms"][a] = {"seeds": seeds, "test_acc": float(accs.mean()), "test_acc_sd": float(accs.std(ddof=1)) if len(seeds) > 1 else 0.0,
                          "test_acc_per_seed": [round(float(x), 5) for x in accs],
                          "ast_bslot_bert": float((corr.mean(0) * w).sum() / w.sum()),
                          "acc_by_action_index": {str(k): float(corr.mean(0)[meta["k"] == k].mean()) for k in range(5)} |
                                                 {"5+": float(corr.mean(0)[meta["k"] >= 5].mean())}}
    if args.composite:
        folder = base / args.composite
        seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
        groups = [g for g in (seeds[:5], seeds[5:10]) if len(g) == 5]
        cc = []
        for g in groups:
            S = sum(np.load(folder / f"seed{k}.npz")["S"].astype(np.float64) for k in g)
            cc.append(S.argmax(1) == meta["y"])
        cc = np.stack(cc).astype(float)
        name = f"{args.composite}_composite"
        arms[name] = cc.mean(0)
        w = meta["w_bert"].astype(float)
        out["arms"][name] = {"seeds": groups, "test_acc": float(cc.mean()), "test_acc_sd": 0.0,
                             "test_acc_per_seed": [round(float(x), 5) for x in cc.mean(1)],
                             "ast_bslot_bert": float((cc.mean(0) * w).sum() / w.sum()), "acc_by_action_index": {}}
    ones = np.ones_like(meta["y"], dtype=float)
    for p in args.pairs:
        a, b = p.split(":")
        d = arms[a] - arms[b]
        out["pairs"][p] = {"acc": cluster_boot(d, ones, meta["convo_id"]),
                           "ast": cluster_boot(d, meta["w_bert"].astype(float), meta["convo_id"])}
    out["n_test_action_turns"] = int(len(meta["y"]))
    out["n_test_conversations"] = int(len(np.unique(meta["convo_id"])))
    (base / args.out).write_text(json.dumps(out, indent=1))
    for a, r in out["arms"].items():
        print(f"{a:16s} acc {r['test_acc']:.4f} (sd {r['test_acc_sd']:.4f}, {len(r['seeds'])} seeds)  AST-weighted {r['ast_bslot_bert']:.4f}")
    for p, r in out["pairs"].items():
        x, y_ = r["acc"], r["ast"]
        print(f"{p:28s} {100 * x['diff']:+.2f} [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]   AST {100 * y_['diff']:+.2f} [{100 * y_['lo']:+.2f}, {100 * y_['hi']:+.2f}]")


if __name__ == "__main__":
    main()
