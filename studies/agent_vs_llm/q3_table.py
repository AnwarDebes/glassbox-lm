"""PLAN_llm_q3_dev.md: the readable agent against every LLM arm on the same fresh dev rows (seed 2).

usage: python -I q3_table.py --dataset star --out runs/q3_star.json
Arms are read when their preds.jsonl exists: L9 (runs/llm_hybrid_dev/<ds>_alone), L9t (runs/llm_q3_L9t/<ds>),
L35 (runs/llm_q3_L35/<ds>), L35t (runs/llm_q3_L35t/<ds>). An arm with fewer rows than the 500 is compared on its rows
only and flagged as partial. Reading: the claim is open for a dataset only if the agent minus EVERY complete arm is
at least +3.0 points with the interval above 0.
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
ARMS = {"L9": "runs/llm_hybrid_dev/{ds}_alone", "L9t": "runs/llm_q3_L9t/{ds}", "L35": "runs/llm_q3_L35/{ds}",
        "L35t": "runs/llm_q3_L35t/{ds}"}
FT = {"star": "runs/ft_star_bgebase_s1/dev_preds.npz", "abcd": "runs/ft_abcd_bert_s1/dev_preds.npz",
      "multiwoz": "runs/ft_mwoz_bgebase_s1/dev_preds.npz"}
ARMS_MWOZ = {"L9": "runs/llm_mwoz_dev/L9", "L35": "runs/llm_mwoz_dev/L35"}  # PLAN_mwoz_dev.md


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ds = a.dataset
    with gzip.open(os.path.join(F.DATA, ds, "examples_dev.json.gz"), "rt") as h:
        d = json.load(h)
    nid = {b: i for i, b in enumerate(d["buttons"])}
    gold = np.array([r["y"] for r in d["rows"]])
    agent = np.load(os.path.join(H, f"runs/agent_{ds}_c4_tm_s1/tm.npz"))["top_ids"][:, 0].astype(int)
    ftp = os.path.join(H, FT[ds])
    ft = np.load(ftp)["top_ids"][:, 0].astype(int) if os.path.exists(ftp) else np.full(len(gold), -1)
    arms = ARMS_MWOZ if ds == "multiwoz" else ARMS
    base = [json.loads(x)["i"] for x in open(os.path.join(H, arms["L9"].format(ds=ds), "preds.jsonl"))]
    rows500 = sorted(base)
    res = {"plan": "PLAN_llm_q3_dev.md", "dataset": ds, "n_rows": len(rows500), "arms": {}}
    rng = np.random.default_rng(SEED)
    complete_ok = []
    for arm, path in arms.items():
        p = os.path.join(H, path.format(ds=ds), "preds.jsonl")
        if not os.path.exists(p):
            continue
        recs = {json.loads(x)["i"]: json.loads(x) for x in open(p) if x.strip()}
        idx = np.array([i for i in rows500 if i in recs])
        if len(idx) == 0:
            continue
        pred = np.array([nid.get(recs[i]["pred"], -1) for i in idx])
        a_ok, l_ok = (agent[idx] == gold[idx]).astype(float), (pred == gold[idx]).astype(float)
        diff = a_ok - l_ok
        b = diff[rng.integers(0, len(idx), size=(a.draws, len(idx)))].mean(1)
        r = {"n": int(len(idx)), "partial": bool(len(idx) < len(rows500)), "llm_acc": float(l_ok.mean()),
             "agent_acc": float(a_ok.mean()), "ft_acc": float((ft[idx] == gold[idx]).mean()),
             "agent_minus_llm": float(diff.mean()),
             "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        res["arms"][arm] = r
        if not r["partial"]:
            complete_ok.append(r["agent_minus_llm"] >= 0.03 and r["ci95"][0] > 0)
    res["claim_open_on_complete_arms"] = bool(complete_ok) and all(complete_ok)
    res["complete_arms"] = [k for k, v in res["arms"].items() if not v["partial"]]
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    for k, v in res["arms"].items():
        print(f"{ds} {k:5s} n {v['n']:3d}{' partial' if v['partial'] else ''}: LLM {v['llm_acc']:.3f} agent "
              f"{v['agent_acc']:.3f} FT {v['ft_acc']:.3f} | agent - LLM {v['agent_minus_llm']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print(ds, "claim open on complete arms:", res["claim_open_on_complete_arms"], res["complete_arms"])


if __name__ == "__main__":
    main()
