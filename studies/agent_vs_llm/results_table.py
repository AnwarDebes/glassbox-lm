"""One table of every dev arm on the fresh 500 rows of each dataset (seed 2), with paired intervals against the agent.

usage: python -I results_table.py --out runs/results_table.md
Arms are read where their predictions exist; partial arms are marked with their row count.
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
SEED = 20261026
LLM = {}
for _ds in ("star", "abcd", "multiwoz"):
    LLM[_ds] = {
        "Qwen3.5-9B": f"runs/llm_hybrid_dev/{_ds}_alone" if _ds != "multiwoz" else "runs/llm_mwoz_dev/L9",
        "Qwen3.5-9B think 4k (dropped)": f"runs/llm_q3_L9t/{_ds}",
        "Qwen3.6-35B-A3B": f"runs/llm_q3_L35/{_ds}" if _ds != "multiwoz" else "runs/llm_mwoz_dev/L35",
        "Qwen3.6-35B-A3B think 4k cap": f"runs/llm_q3_L35t/{_ds}",
        "Qwen3.6-35B-A3B think 16k cap": f"runs/llm_q3_L35t16/{_ds}",
        "Qwen3.6-35B-A3B k50": f"runs/llm_q3_L35k50/{_ds}",
        "Qwen3.6-35B-A3B k100": f"runs/llm_q3_L35k100/{_ds}",
        "Qwen3.5-9B no policy": f"runs/llm_abl/L9/{_ds}",
        "Qwen3.6-35B-A3B no policy": f"runs/llm_abl/L35/{_ds}",
        "9B + agent top-3": f"runs/llm_hybrid_dev/{_ds}_k3"}
FT = {"star": ["runs/ft_star_bgebase_s1", "runs/ft_star_roberta_s1", "runs/ft_star_bert_s1"],
      "abcd": ["runs/ft_abcd_bert_s1", "runs/ft_abcd_bgebase_s1", "runs/ft_abcd_roberta_s1"],
      "multiwoz": ["runs/ft_mwoz_bgebase_s1"]}
TFIDF = {ds: f"runs/dry/tfidf_{ds}/dev_preds.npz" for ds in ("star", "abcd", "multiwoz")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    lines = ["| Dataset | Arm | Rows | Accuracy | Agent minus arm (95% interval) |", "|---|---|---|---|---|"]
    for ds in ("star", "abcd", "multiwoz"):
        with gzip.open(os.path.join(F.DATA, ds, "examples_dev.json.gz"), "rt") as h:
            d = json.load(h)
        nid = {b: i for i, b in enumerate(d["buttons"])}
        gold = np.array([r["y"] for r in d["rows"]])
        agent = np.load(os.path.join(H, f"runs/agent_{ds}_c4_tm_s1/tm.npz"))["top_ids"][:, 0]
        rows = sorted(json.loads(x)["i"] for x in open(os.path.join(H, LLM[ds]["Qwen3.5-9B"], "preds.jsonl")))
        rows = np.array(rows)
        a_ok = (agent[rows] == gold[rows]).astype(float)
        lines.append(f"| {ds} | readable agent (no neural network) | {len(rows)} | {a_ok.mean():.3f} | |")
        arms = {}
        for p in FT[ds]:
            f = os.path.join(H, p, "dev_preds.npz")
            if os.path.exists(f):
                arms["fine-tuned " + json.load(open(os.path.join(H, p, "result.json")))["model"].split("/")[-1]] = \
                    {i: int(v) for i, v in zip(range(len(gold)), np.load(f)["top_ids"][:, 0])}
        if os.path.exists(os.path.join(H, TFIDF[ds])):
            arms["TF-IDF logistic regression (no neural network)"] = \
                {i: int(v) for i, v in zip(range(len(gold)), np.load(os.path.join(H, TFIDF[ds]))["top_ids"][:, 0])}
        for name, p in LLM[ds].items():
            f = os.path.join(H, p, "preds.jsonl")
            if os.path.exists(f):
                arms[name] = {json.loads(x)["i"]: nid.get(json.loads(x)["pred"], -1) for x in open(f) if x.strip()}
        for name, pr in arms.items():
            idx = np.array([i for i in rows if i in pr])
            if len(idx) == 0:
                continue
            ok = np.array([pr[i] == gold[i] for i in idx], dtype=float)
            ag = (agent[idx] == gold[idx]).astype(float)
            diff = ag - ok
            b = diff[rng.integers(0, len(idx), size=(10000, len(idx)))].mean(1)
            lines.append(f"| {ds} | {name} | {len(idx)}{'' if len(idx) == len(rows) else ' (partial)'} | {ok.mean():.3f} | "
                         f"{diff.mean() * 100:+.1f} ({np.percentile(b, 2.5) * 100:+.1f} to {np.percentile(b, 97.5) * 100:+.1f}) |")
    with open(a.out, "w") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
