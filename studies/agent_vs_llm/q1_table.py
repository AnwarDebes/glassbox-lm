"""Q1 of PLAN_llm_dev.md: every arm on the same dev rows the LLM answered, with paired bootstrap intervals.

usage: python -I q1_table.py --dataset abcd --llm runs/llm_q1_dev/abcd/preds.jsonl --agent runs/agent_abcd_c4_tm_s1/tm.npz \
           --ft runs/ft_abcd_bert_s1/dev_preds.npz [--ft ...] --tfidf-c 16 --out runs/q1_abcd.json
TF-IDF LR is refitted here at the given C (the best C of NOTE_tfidf_dev.md) to get its row predictions.
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd"], required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--agent")
    ap.add_argument("--ft", action="append", default=[])
    ap.add_argument("--tfidf-c", type=float)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with gzip.open(os.path.join(F.DATA, a.dataset, "examples_dev.json.gz"), "rt") as h:
        d = json.load(h)
    buttons = d["buttons"]
    gold_all = np.array([r["y"] for r in d["rows"]])
    with open(a.llm) as h:
        recs = [json.loads(x) for x in h if x.strip()]
    idx = np.array([r["i"] for r in recs])
    gold = gold_all[idx]
    name_to_id = {b: i for i, b in enumerate(buttons)}
    arms = {"llm": np.array([name_to_id.get(r["pred"], -1) for r in recs]),
            "nearest_demo": np.array([name_to_id[r["demo_labels"][-1]] for r in recs]),
            "demo_majority": np.array([name_to_id[max(set(r["demo_labels"]), key=r["demo_labels"].count)] for r in recs])}
    if a.agent:
        A = np.load(a.agent)
        assert np.array_equal(A["gold"], gold_all)
        arms["agent_tm_c4"] = A["top_ids"][idx, 0].astype(int)
    for p in a.ft:
        P = np.load(p)
        assert np.array_equal(P["gold"], gold_all)
        arms["ft:" + os.path.basename(os.path.dirname(p))] = P["top_ids"][idx, 0].astype(int)
    if a.tfidf_c:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        xtr, ytr, _, _ = F.load(a.dataset, "train")
        xdv, ydv, _, _ = F.load(a.dataset, "dev")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        clf = LogisticRegression(C=a.tfidf_c, solver="saga", max_iter=200, tol=1e-4).fit(vec.fit_transform(xtr), ytr)
        arms[f"tfidf_lr_C{a.tfidf_c:g}"] = clf.predict(vec.transform([xdv[i] for i in idx]))
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(idx), size=(a.draws, len(idx)))
    ok = {k: (v == gold) for k, v in arms.items()}
    res = {"dataset": a.dataset, "n": int(len(idx)), "llm_invalid": int((arms["llm"] < 0).sum()), "arms": {}}
    for k, v in ok.items():
        diff = v.astype(float) - ok["llm"]
        bd = diff[boots].mean(1)
        res["arms"][k] = {"acc": float(v.mean()), "minus_llm": float(diff.mean()),
                          "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    for k, v in res["arms"].items():
        print(f"{k:28s} acc {v['acc']:.4f}  minus LLM {v['minus_llm']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
