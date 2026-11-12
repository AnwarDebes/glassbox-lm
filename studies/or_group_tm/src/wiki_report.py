"""Report for Wikipedia pair runs of this line, with the baselines and the paired article bootstrap of the
hierarchical_tm study's one final test. The baseline definitions (word overlap = jaccard, tfidf_cosine,
shared_count, ppmi_svd_cosine, the two logistic regressions), AUC, the resampling of whole articles (1,000 resamples,
numpy seed 0, a pair belongs to the article of its first paragraph) and the verdict words ("beats X" if the whole
95% interval of the mean paired difference in AUC is above 0, "below X" if it is all below 0, else "not
distinguishable") are imported or copied unchanged from that study's data_checks.py, htm_driver.py and
final_test_report.py.

  .venv/bin/python -I src/wiki_report.py --data sa --split dev --runs runs/<run> [runs/<run> ...]
        [--stock <study score dir> ...] [--reading unit|ratio|weight|logodds] [--best lr_A_B_AandB]
        [--subsets FILE [--subsets-only]] [--out FILE]
  --split test is refused unless --registered names the record of registrations/E5_wikipedia_test.md and the plan,
  the sources and the pinned files are unchanged since that record (e5_common.py); with --split test every run must be
  a registered run of that record and the stock folders must be the study's five test score folders of the data
  version, in order. These checks run before any pair is read.

--runs: run folders of src/run_wiki.py (scores_<split>.npz in file order). --stock: score folders of the study
(scores_<split>.npz with score, y, art_a, par_a; on dev in the study's permuted row order, which is undone here; y,
art_a and par_a are compared with the pairs, as the study did). --best: the baseline that was best on dev in the
study (lr_A_B_AandB on sa and np, ppmi_svd_cosine on ns_all). --reading: unit = the vote sum (the score of the
study's driver); ratio = clauses that fire divided by clauses whose groups are all true on at least one paragraph;
weight and logodds = the two weighted readings (runs with weights). --subsets: a JSON file with named article lists
per split ({split: {"subsets": {name: [article, ...]}}}); for each list the comparisons are repeated without the pairs
whose paragraph A or B comes from one of its articles (with --subsets-only, only those are computed).

Every comparison carries three intervals of the paired difference of mean AUCs:
  ci95                     the study's paired article bootstrap, unchanged: 1,000 resamples of the articles of
                           paragraph A, numpy seed 0, the seeds of each side fixed;
  ci95_seeds_and_articles  10,000 resamples (numpy seed 0) of the articles of paragraph A, in each of which the seeds
                           of this study and of the stock model are also drawn with replacement;
  ci95_two_way             1,000 resamples (numpy seed 0) of all articles of the split, each pair weighted by the
                           copies of its article (A and B from one article) or by the product of the copies of its
                           two articles; seeds fixed. It shows what the article of paragraph B adds; product weights
                           may overstate the spread. Not computed for the subsets.
The last two use pair weights: the AUC of a resample equals the AUC in which each pair counts as often as it was
drawn. Added to the study's comparisons, as a reading: the paired comparison with tf-idf cosine (not on ns_all,
where it is 0 for every pair).
"""
import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hierarchical_tm"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
sys.path.insert(1, os.path.join(STUDY, "src"))
import data_checks as dc
import htm_driver as hd
from pairdata import load_pairs

import e5_common as e5

BASELINES = ("jaccard", "tfidf_cosine", "shared_count", "ppmi_svd_cosine", "lr_A_AND_B", "lr_A_B_AandB")
N_RESAMPLES, RESAMPLE_SEED = 1000, 0              # the study's bootstrap, unchanged
SEED_RESAMPLES, SEED_RESAMPLE_SEED = 10000, 0     # articles of paragraph A and the seeds of both sides
TWO_WAY_RESAMPLES, TWO_WAY_SEED = 1000, 0         # articles of paragraphs A and B (product weights), seeds fixed
BEST_ON_DEV = {"sa": "lr_A_B_AandB", "np": "lr_A_B_AandB", "ns_all": "ppmi_svd_cosine"}
READINGS = ("unit", "ratio", "weight", "logodds")


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


class WeightedAUC:
    """AUC of one score in which pair i counts w[i] times (ties count half, as in htm_driver.auc); the order of the
    scores is computed once."""

    def __init__(self, score, y):
        o = np.argsort(score, kind="stable")
        s = np.asarray(score)[o]
        self.o, self.pos = o, np.asarray(y)[o] == 1
        self.g = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]) if len(s) else np.zeros(0, np.int64)
        self.n = int(self.g[-1]) + 1 if len(s) else 0

    def __call__(self, w):
        ws = np.asarray(w, float)[self.o]
        wp = np.bincount(self.g, np.where(self.pos, ws, 0.0), self.n)
        wn = np.bincount(self.g, np.where(self.pos, 0.0, ws), self.n)
        tp, tn = wp.sum(), wn.sum()
        if tp == 0 or tn == 0:
            return float("nan")
        return float((wp * (np.cumsum(wn) - 0.5 * wn)).sum() / (tp * tn))


def two_way_weights(counts, ia, ib, same):
    """Pair weights of one two-way resample: the copies of the article (A and B from one article) or the product of
    the copies of the two articles."""
    return np.where(same, counts[ia], counts[ia] * counts[ib]).astype(float)


def keep_mask(art_a, art_b, articles):
    """True for the pairs whose paragraphs A and B both come from articles outside the list."""
    s = np.asarray(sorted(articles), dtype=np.asarray(art_a).dtype)
    return ~(np.isin(art_a, s) | np.isin(art_b, s))


def compare(items, ours, stock, y, art_a, art_b, best, data, two_way=True):
    """(auc, bootstrap columns of the study's resamples, comparisons) for the pairs given."""
    keys = list(items)
    B = np.empty((N_RESAMPLES, len(keys)))
    for b, i in enumerate(resamples(art_a)):
        B[b] = [hd.auc(items[k][i], y[i]) for k in keys]
    col = {k: B[:, j] for j, k in enumerate(keys)}
    auc = {k: hd.auc(items[k], y) for k in keys}
    refs = ["jaccard", "tfidf_cosine", "ppmi_svd_cosine", best]
    W = {k: WeightedAUC(items[k], y) for k in dict.fromkeys(ours + stock + refs)}
    # the articles of paragraph A and the seeds of both sides
    uniq, inv = np.unique(art_a, return_inverse=True)
    rng = np.random.default_rng(SEED_RESAMPLE_SEED)
    so, st, sb = np.empty(SEED_RESAMPLES), np.empty(SEED_RESAMPLES), {k: np.empty(SEED_RESAMPLES) for k in refs}
    for b in range(SEED_RESAMPLES):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)[inv]
        a_ours = np.array([W[k](w) for k in ours])
        so[b] = a_ours[rng.integers(0, len(ours), len(ours))].mean()
        if stock:
            a_stock = np.array([W[k](w) for k in stock])
            st[b] = a_stock[rng.integers(0, len(stock), len(stock))].mean()
        for k in refs:
            sb[k][b] = W[k](w)
    # the articles of paragraphs A and B, seeds fixed (a reading; its failure leaves the other intervals)
    tw, two_way_error = None, None
    if two_way:
        try:
            arts = np.unique(np.concatenate([art_a, art_b]))
            ia, ib, same = np.searchsorted(arts, art_a), np.searchsorted(arts, art_b), np.asarray(art_a) == np.asarray(art_b)
            rng2 = np.random.default_rng(TWO_WAY_SEED)
            tw = {"ours": np.empty(TWO_WAY_RESAMPLES), "stock": np.empty(TWO_WAY_RESAMPLES), **{k: np.empty(TWO_WAY_RESAMPLES) for k in refs}}
            for b in range(TWO_WAY_RESAMPLES):
                c = np.bincount(rng2.integers(0, len(arts), len(arts)), minlength=len(arts))
                w = two_way_weights(c, ia, ib, same)
                tw["ours"][b] = np.mean([W[k](w) for k in ours])
                if stock:
                    tw["stock"][b] = np.mean([W[k](w) for k in stock])
                for k in refs:
                    tw[k][b] = W[k](w)
        except Exception as ex:  # noqa: BLE001 - the two-way interval is a reading
            tw, two_way_error = None, f"{type(ex).__name__}: {ex}"
    S = np.stack([col[k] for k in ours], 1).mean(1)

    def paired(ref_col, ref_auc, ref_seeds, ref_key, name, what):
        lo, hi = ci(S - ref_col)
        out = {"vs": name, "what": what, "mean_diff": float(np.mean([auc[k] for k in ours]) - ref_auc), "ci95": [lo, hi],
               "verdict": f"beats {name}" if lo > 0 else f"below {name}" if hi < 0 else "not distinguishable",
               "ci95_seeds_and_articles": ci(so - ref_seeds)}
        if tw is not None:
            out["ci95_two_way"] = ci(tw["ours"] - tw[ref_key])
        elif two_way_error:
            out["ci95_two_way_failed"] = two_way_error
        return out
    comps = {}
    if data == "ns_all":
        comps["word_overlap"] = {"vs": "jaccard", "value": "not a comparison (by construction)"}
        comps["tfidf_cosine"] = {"vs": "tfidf_cosine", "value": "not a comparison (by construction)"}
    else:
        comps["word_overlap"] = paired(col["jaccard"], auc["jaccard"], sb["jaccard"], "jaccard", "jaccard", "word overlap")
        comps["tfidf_cosine"] = paired(col["tfidf_cosine"], auc["tfidf_cosine"], sb["tfidf_cosine"], "tfidf_cosine",
                                       "tfidf_cosine", "tf-idf cosine")
    comps["best_on_dev"] = paired(col[best], auc[best], sb[best], best, best, "the baseline that was best on dev in the study")
    comps["ppmi_svd_cosine"] = paired(col["ppmi_svd_cosine"], auc["ppmi_svd_cosine"], sb["ppmi_svd_cosine"], "ppmi_svd_cosine",
                                      "ppmi_svd_cosine", "count-based word vectors")
    if stock:
        comps["stock"] = paired(np.stack([col[k] for k in stock], 1).mean(1), float(np.mean([auc[k] for k in stock])), st,
                                "stock", "the stock model", f"mean of {len(stock)} stock seeds on the same pairs")
    return auc, col, comps


def test_problems(registered):
    """Problems that refuse the test pairs: no record, the dry record, or a record that no longer matches the files."""
    if not registered:
        return ["--split test needs --registered <the record of registrations/E5_wikipedia_test.md>"]
    if e5.is_dry_record(registered):
        return ["the dry record does not open the test pairs"]
    return e5.preflight(registered, False)[1]


def registered_run_problems(cfg, data, run):
    """Problems of a run folder given for --split test: it must be a registered run of E5's record."""
    problems = []
    rec = cfg.get("registered") or ""
    if os.path.abspath(os.path.join(e5.ROOT, rec)) != os.path.abspath(e5.record_path(False)):
        problems.append(f"{os.path.basename(run)}: not a run of the record of {e5.PLAN[False]}")
    if cfg.get("seed") not in e5.SEEDS[False] or cfg.get("name") != e5.run_name(data, cfg.get("seed"), False) or cfg.get("data") != data:
        problems.append(f"{os.path.basename(run)}: not a registered run of {data}")
    return problems


def stock_problems(stock, data):
    """Problems of the stock folders given for --split test: the study's five test score folders, in order."""
    if [os.path.basename(os.path.normpath(r)) for r in stock] != list(e5.STOCK_TEST[data]):
        return [f"the stock folders are not the study's five test score folders of {data}, in order"]
    return []


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=["sa", "np", "ns_all"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--registered", default="")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--stock", nargs="*", default=[])
    ap.add_argument("--reading", choices=READINGS, default="unit")
    ap.add_argument("--best", default="")
    ap.add_argument("--subsets", default="")
    ap.add_argument("--subsets-only", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.subsets_only and not a.subsets:
        ap.error("--subsets-only needs --subsets FILE")
    if a.split == "test":
        problems = test_problems(a.registered)
        if not problems:
            for r in a.runs:
                with open(os.path.join(r, "config.json")) as f:
                    problems += registered_run_problems(json.load(f), a.data, r)
            problems += stock_problems(a.stock, a.data)
        if problems:
            ap.error("test pairs refused before any pair was read: " + "; ".join(problems))
    return a


def main(argv=None):
    a = get_args(argv)
    best = a.best or BEST_ON_DEV[a.data]
    tr, ev = load_pairs(a.data, "train", dc.F), load_pairs(a.data, a.split, dc.F)
    y, n = ev["y"], len(ev["y"])
    P = dc.train_paragraphs()
    idf2 = np.log(P.shape[0] / np.maximum(np.asarray(P.sum(0)).ravel(), 1)) ** 2
    W = dc.ppmi_vectors(P)
    s_ev = dc.scores(ev, idf2, W)
    for k, kind in (("lr_A_AND_B", "AND"), ("lr_A_B_AandB", "ALL")):
        s_ev[k] = lr_baseline_scores(dc.feats(tr, kind), tr["y"], dc.feats(ev, kind), ev["y"])
    if a.data == "ns_all":
        assert not np.asarray(s_ev["shared_count"]).any(), "ns_all pairs share words"
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
            full = np.empty(n)
            full[order] = sc
            sc = full
        else:
            order = np.arange(n)
        for k in ("y", "art_a", "par_a"):
            assert np.array_equal(z[k], ev[k][order]), f"{r}: {k} differs from the {a.data} {a.split} pairs"
        key = f"stock_{len(stock) + 1}"
        items[key] = sc
        stock.append(key)
        info[key] = {"run": os.path.basename(os.path.normpath(r))}
    out = {"data": a.data, "split": a.split, "reading": a.reading, "n_pairs": int(n), "n_articles": len(np.unique(ev["art_a"])),
           "n_resamples": N_RESAMPLES, "resample_seed": RESAMPLE_SEED, "seed_resamples": SEED_RESAMPLES,
           "seed_resample_seed": SEED_RESAMPLE_SEED, "two_way_resamples": TWO_WAY_RESAMPLES, "two_way_seed": TWO_WAY_SEED,
           "registered": os.path.relpath(os.path.abspath(a.registered), e5.ROOT) if a.registered else ""}
    if not a.subsets_only:
        auc, col, comps = compare(items, ours, stock, y, ev["art_a"], ev["art_b"], best, a.data)
        out.update({"ours": {k: {"auc": auc[k], "auc_ci95": ci(col[k]), **info[k]} for k in ours},
                    "ours_mean_auc": float(np.mean([auc[k] for k in ours])),
                    "ours_sd_auc": float(np.std([auc[k] for k in ours], ddof=1)) if len(ours) > 1 else None,
                    "stock": {k: {"auc": auc[k], **info[k]} for k in stock},
                    "stock_mean_auc": float(np.mean([auc[k] for k in stock])) if stock else None,
                    "baselines": {k: {"auc": auc[k], "auc_ci95": ci(col[k])} for k in BASELINES},
                    "comparisons": comps})
    if a.subsets:
        with open(a.subsets) as f:
            lists = json.load(f)[a.split]["subsets"]
        out["subsets"] = {}
        for name, arts in lists.items():
            m = keep_mask(ev["art_a"], ev["art_b"], arts)
            auc_s, _, comps_s = compare({k: v[m] for k, v in items.items()}, ours, stock, y[m], ev["art_a"][m],
                                        ev["art_b"][m], best, a.data, two_way=False)
            out["subsets"][name] = {"articles_listed": len(arts), "n_pairs": int(m.sum()), "n_pairs_dropped": int((~m).sum()),
                                    "ours_mean_auc": float(np.mean([auc_s[k] for k in ours])),
                                    "stock_mean_auc": float(np.mean([auc_s[k] for k in stock])) if stock else None,
                                    "baselines": {k: auc_s[k] for k in BASELINES}, "comparisons": comps_s}
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    return out


if __name__ == "__main__":
    main()
