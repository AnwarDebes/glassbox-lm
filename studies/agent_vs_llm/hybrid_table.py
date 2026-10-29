"""PLAN_hybrid_dev.md: Hybrid-k3 against the agent alone and the LLM alone on the same fresh dev rows, and the rule.

usage: python -I hybrid_table.py --dataset star --dir runs/llm_hybrid_dev --agent runs/agent_star_c4_tm_s1/tm.npz \
           [--ft runs/ft_star_bgebase_s1/dev_preds.npz ...] [--tfidf-c 4] --out runs/hybrid_star.json
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


def load_llm(path, nid):
    with open(path) as h:
        recs = [json.loads(x) for x in h if x.strip()]
    recs.sort(key=lambda r: r["i"])
    return (np.array([r["i"] for r in recs]), np.array([nid.get(r["pred"], -1) for r in recs]),
            sum(bool(r.get("fallback")) for r in recs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ft", action="append", default=[])
    ap.add_argument("--tfidf-c", type=float)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with gzip.open(os.path.join(F.DATA, a.dataset, "examples_dev.json.gz"), "rt") as h:
        d = json.load(h)
    nid = {b: i for i, b in enumerate(d["buttons"])}
    gold_all = np.array([r["y"] for r in d["rows"]])
    arms, fallbacks = {}, {}
    idx = None
    for arm in ("k3", "k5", "alone"):
        p = os.path.join(a.dir, f"{a.dataset}_{arm}", "preds.jsonl")
        i, pred, fb = load_llm(p, nid)
        if idx is None:
            idx = i
        assert np.array_equal(i, idx), f"{arm}: not the same rows"
        arms["hybrid_" + arm if arm != "alone" else "llm_alone"] = pred
        fallbacks[arm] = fb
    gold = gold_all[idx]
    A = np.load(a.agent)
    assert np.array_equal(A["gold"], gold_all)
    arms["agent"] = A["top_ids"][idx, 0].astype(int)
    for p in a.ft:
        P = np.load(p)
        assert np.array_equal(P["gold"], gold_all)
        arms["ft:" + os.path.basename(os.path.dirname(p))] = P["top_ids"][idx, 0].astype(int)
    if a.tfidf_c:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        xtr, ytr, _, _ = F.load(a.dataset, "train")
        xdv, _, _, _ = F.load(a.dataset, "dev")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        clf = LogisticRegression(C=a.tfidf_c, solver="saga", max_iter=200, tol=1e-4).fit(vec.fit_transform(xtr), ytr)
        arms[f"tfidf_lr_C{a.tfidf_c:g}"] = clf.predict(vec.transform([xdv[i] for i in idx]))
    ok = {k: (v == gold).astype(float) for k, v in arms.items()}
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(idx), size=(a.draws, len(idx)))

    def diff(x, y):
        dd = ok[x] - ok[y]
        b = dd[boots].mean(1)
        return {"diff": float(dd.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    res = {"plan": "PLAN_hybrid_dev.md", "dataset": a.dataset, "n": int(len(idx)), "fallbacks": fallbacks,
           "acc": {k: float(v.mean()) for k, v in ok.items()},
           "gold_in_agent_top3": float((A["top_ids"][idx, :3] == gold[:, None]).any(1).mean()),
           "gold_in_agent_top5": float((A["top_ids"][idx, :5] == gold[:, None]).any(1).mean()),
           "k3_minus_agent": diff("hybrid_k3", "agent"), "k3_minus_llm": diff("hybrid_k3", "llm_alone"),
           "k5_minus_agent": diff("hybrid_k5", "agent"), "agent_minus_llm": diff("agent", "llm_alone")}
    for k in ok:
        if k.startswith("ft:"):
            res["k3_minus_" + k] = diff("hybrid_k3", k)
    r1, r2 = res["k3_minus_agent"], res["k3_minus_llm"]
    res["rule_met_this_dataset"] = bool(r1["diff"] >= 0.02 and r1["ci95"][0] > 0 and r2["diff"] >= 0.02 and r2["ci95"][0] > 0)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
