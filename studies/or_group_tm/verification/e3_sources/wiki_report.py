"""Report for Wikipedia pair runs of this line, with the baselines and the paired article bootstrap of the
hierarchical_tm study's one final test. The baseline definitions (word overlap = jaccard, tfidf_cosine,
shared_count, ppmi_svd_cosine, the two logistic regressions), AUC, the resampling of whole articles (1,000 resamples,
numpy seed 0, a pair belongs to the article of its first paragraph) and the verdict words ("beats X" if the whole
95% interval of the mean paired difference in AUC is above 0, "below X" if it is all below 0, else "not
distinguishable") are imported or copied unchanged from that study's data_checks.py, htm_driver.py and
final_test_report.py.

  .venv/bin/python -I src/wiki_report.py --data sa --split dev --runs runs/<run> [runs/<run> ...]
        [--stock <study score dir> ...] [--reading unit|weight|logodds] [--best lr_A_B_AandB] [--out FILE]
  --split test needs --registered <existing frozen record>.

--runs: run folders of src/run_wiki.py (scores_<split>.npz in file order). --stock: score folders of the study
(scores_<split>.npz with score, y, art_a, par_a; on dev in the study's permuted row order, which is undone here).
--best: the baseline that was best on dev in the study (lr_A_B_AandB on sa and np, ppmi_svd_cosine on ns_all).
"""
import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hierarchical_tm"))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(STUDY, "src"))
import data_checks as dc
import htm_driver as hd
from pairdata import load_pairs

BASELINES = ("jaccard", "tfidf_cosine", "shared_count", "ppmi_svd_cosine", "lr_A_AND_B", "lr_A_B_AandB")
N_RESAMPLES, RESAMPLE_SEED = 1000, 0
BEST_ON_DEV = {"sa": "lr_A_B_AandB", "np": "lr_A_B_AandB", "ns_all": "ppmi_svd_cosine"}


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def lr_baseline_scores(X_tr, y_tr, X_ev, y_ev):
    """data_checks.lr_baseline, unchanged; its decision_function values are captured (as final_test_report.py does)."""
    got, report = [], dc.report

    def capture(pred, score, y):
        got.append(score)
        return report(pred, score, y)
    dc.report = capture
    try:
        dc.lr_baseline(X_tr, y_tr, X_ev, y_ev)
    finally:
        dc.report = report
    return got[0]


def resamples(art):
    uniq, inv = np.unique(art, return_inverse=True)
    rows = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(RESAMPLE_SEED)
    for _ in range(N_RESAMPLES):
        yield np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=["sa", "np", "ns_all"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--registered", default="")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--stock", nargs="*", default=[])
    ap.add_argument("--reading", choices=["unit", "weight", "logodds"], default="unit")
    ap.add_argument("--best", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.split == "test" and not (a.registered and os.path.exists(a.registered)):
        ap.error("--split test needs --registered <existing frozen record>")
    best = a.best or BEST_ON_DEV[a.data]
    tr, ev = load_pairs(a.data, "train", dc.F), load_pairs(a.data, a.split, dc.F)
    y, n = ev["y"], len(ev["y"])
    P = dc.train_paragraphs()
    idf2 = np.log(P.shape[0] / np.maximum(np.asarray(P.sum(0)).ravel(), 1)) ** 2
    W = dc.ppmi_vectors(P)
    s_ev = dc.scores(ev, idf2, W)
    for k, kind in (("lr_A_AND_B", "AND"), ("lr_A_B_AandB", "ALL")):
        s_ev[k] = lr_baseline_scores(dc.feats(tr, kind), tr["y"], dc.feats(ev, kind), ev["y"])
    items, ours, stock, info = {k: np.asarray(s_ev[k], float) for k in BASELINES}, [], [], {}
    for r in a.runs:
        z = np.load(os.path.join(r, f"scores_{a.split}.npz"))
        assert np.array_equal(z["y"], y), f"{r}: labels differ from the {a.data} {a.split} pairs"
        with open(os.path.join(r, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["data"] == a.data and cfg["split"] == a.split and cfg["n_dev"] == 0, r
        key = f"ours_seed{cfg['seed']}"
        assert key not in items, f"two runs of seed {cfg['seed']}"
        items[key] = np.asarray(z["votes"] if a.reading == "unit" else z[a.reading], float)
        ours.append(key)
        info[key] = {"run": os.path.basename(os.path.normpath(r)), "engine": cfg["orgroup_tm_sha256"]}
    for r in a.stock:
        z = np.load(os.path.join(r, f"scores_{a.split}.npz"))
        sc = np.asarray(z["score"], float)
        if a.split == "dev":  # the study's dev scores are in the row order rng(1).permutation(n)
            order = np.random.default_rng(1).permutation(n)
            assert len(sc) == n, f"{r}: a dev subset was scored"
            assert np.array_equal(z["y"], y[order]) and np.array_equal(z["art_a"], ev["art_a"][order]), r
            full = np.empty(n)
            full[order] = sc
            sc = full
        else:
            assert np.array_equal(z["y"], y) and np.array_equal(z["art_a"], ev["art_a"]), r
        key = f"stock_{len(stock) + 1}"
        items[key] = sc
        stock.append(key)
        info[key] = {"run": os.path.basename(os.path.normpath(r))}
    keys = list(items)
    B = np.empty((N_RESAMPLES, len(keys)))
    for b, i in enumerate(resamples(ev["art_a"])):
        B[b] = [hd.auc(items[k][i], y[i]) for k in keys]
    col = {k: B[:, j] for j, k in enumerate(keys)}
    auc = {k: hd.auc(items[k], y) for k in keys}
    S = np.stack([col[k] for k in ours], 1)

    def paired(ref_cols, ref_auc, name, what):
        lo, hi = ci(S.mean(1) - ref_cols)
        return {"vs": name, "what": what, "mean_diff": float(np.mean([auc[k] for k in ours]) - ref_auc), "ci95": [lo, hi],
                "verdict": f"beats {name}" if lo > 0 else f"below {name}" if hi < 0 else "not distinguishable"}
    comps = {}
    if a.data == "ns_all":
        assert not np.asarray(s_ev["shared_count"]).any(), "ns_all pairs share words"
        comps["word_overlap"] = {"vs": "jaccard", "value": "not a comparison (by construction)"}
    else:
        comps["word_overlap"] = paired(col["jaccard"], auc["jaccard"], "jaccard", "word overlap")
    comps["best_on_dev"] = paired(col[best], auc[best], best, "the baseline that was best on dev in the study")
    comps["ppmi_svd_cosine"] = paired(col["ppmi_svd_cosine"], auc["ppmi_svd_cosine"], "ppmi_svd_cosine", "count-based word vectors")
    if stock:
        ref = np.stack([col[k] for k in stock], 1).mean(1)
        comps["stock"] = paired(ref, float(np.mean([auc[k] for k in stock])), "the stock model",
                                f"mean of {len(stock)} stock seeds on the same pairs")
    out = {"data": a.data, "split": a.split, "reading": a.reading, "n_pairs": int(n), "n_articles": len(np.unique(ev["art_a"])),
           "ours": {k: {"auc": auc[k], "auc_ci95": ci(col[k]), **info[k]} for k in ours},
           "ours_mean_auc": float(np.mean([auc[k] for k in ours])),
           "ours_sd_auc": float(np.std([auc[k] for k in ours], ddof=1)) if len(ours) > 1 else None,
           "stock": {k: {"auc": auc[k], **info[k]} for k in stock},
           "stock_mean_auc": float(np.mean([auc[k] for k in stock])) if stock else None,
           "baselines": {k: {"auc": auc[k], "auc_ci95": ci(col[k])} for k in BASELINES},
           "comparisons": comps, "registered": a.registered}
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    return out


if __name__ == "__main__":
    main()
