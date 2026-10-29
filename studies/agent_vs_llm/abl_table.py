"""PLAN_ablation_dev.md: the cost of removing the written policy from the LLM prompt, per model and dataset.

usage: python -I abl_table.py --out runs/abl_table.json
With-policy arms: L9 = runs/llm_hybrid_dev/<ds>_alone, L35 = runs/llm_q3_L35/<ds>; without: runs/llm_abl/<arm>/<ds>.
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

SEED = 20261026
H = os.path.dirname(os.path.abspath(__file__))
WITH = {"L9": "runs/llm_hybrid_dev/{ds}_alone", "L35": "runs/llm_q3_L35/{ds}"}


def oks(path, nid, gold):
    recs = {json.loads(x)["i"]: json.loads(x) for x in open(os.path.join(H, path, "preds.jsonl")) if x.strip()}
    return {i: float(nid.get(r["pred"], -1) == gold[i]) for i, r in recs.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args(argv)
    res = {"plan": "PLAN_ablation_dev.md", "cost": {}, "cost_gap": {}}
    rng = np.random.default_rng(SEED)
    per = {}
    for ds in ("abcd", "star"):
        with gzip.open(os.path.join(F.DATA, ds, "examples_dev.json.gz"), "rt") as h:
            d = json.load(h)
        nid = {b: i for i, b in enumerate(d["buttons"])}
        gold = np.array([r["y"] for r in d["rows"]])
        for arm, wp in WITH.items():
            w = oks(wp.format(ds=ds), nid, gold)
            wo = oks(f"runs/llm_abl/{arm}/{ds}", nid, gold)
            idx = sorted(set(w) & set(wo))
            diff = np.array([w[i] - wo[i] for i in idx])
            b = diff[rng.integers(0, len(idx), size=(a.draws, len(idx)))].mean(1)
            per[(arm, ds)] = (diff, b)
            res["cost"][f"{arm}/{ds}"] = {"n": len(idx), "acc_with": float(np.mean([w[i] for i in idx])),
                                          "acc_without": float(np.mean([wo[i] for i in idx])), "cost": float(diff.mean()),
                                          "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    ok = []
    for arm in WITH:
        ca, cs = res["cost"][f"{arm}/abcd"], res["cost"][f"{arm}/star"]
        gap = per[(arm, "abcd")][1] - per[(arm, "star")][1]  # independent rows, bootstrap draws paired by index
        res["cost_gap"][arm] = {"abcd_minus_star": ca["cost"] - cs["cost"],
                                "ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]}
        ok.append(ca["cost"] >= 0.05 and ca["ci95"][0] > 0 and ca["cost"] - cs["cost"] >= 0.03)
    res["statement_supported"] = all(ok)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    for k, v in res["cost"].items():
        print(f"{k:9s} with {v['acc_with']:.3f} without {v['acc_without']:.3f} cost {v['cost']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for k, v in res["cost_gap"].items():
        print(f"{k}: ABCD cost minus STAR cost {v['abcd_minus_star']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("statement supported:", res["statement_supported"])


if __name__ == "__main__":
    main()
