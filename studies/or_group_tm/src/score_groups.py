"""Word-group scores for Wikipedia pair runs of this study, with the measures and reference data of the
hierarchical_tm study (its wiki_groups.py and npmi_groups.py, imported unchanged):

  NPMI     mean normalised pointwise mutual information of a group's word pairs over the 2,847 reference articles
           that are in no split (how often the words of a group appear together in new articles)
  WordNet  share of a group's word pairs that are synonyms or hypernym-close in WordNet 3.0 (that study's rule)

Each set of groups is compared with frequency-matched random groups (1,000 draws per group, group bootstrap and the
calibrated test of npmi_groups.compare), and the groups of this study's runs are compared with the groups of the
stock model's runs (npmi_groups.compare_two) when --stock is given.

A group of this study is the member set of one OR-group of one clause (tied runs: one set per clause and group
index); groups of fewer than 2 words are left out, as in that study. --unique also drops repeated groups.

Usage (from the study folder):
  .venv/bin/python -I src/score_groups.py --data sa --runs runs/<run> [...] [--stock <study run dir> ...] [--out FILE]
"""
import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(os.path.dirname(SRC), "..", "hierarchical_tm"))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(STUDY, "src"))
import npmi_groups as ng
import wiki_groups as wg
from pairdata import load_vocab

SKEWED = ("began", "big", "since", "called")  # the four words that study leaves out of its synonym verdict


def our_groups(run, unique):
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    st = np.load(os.path.join(run, "state.npz"))
    mem = st["state"] >= 0
    C, G, _ = mem.shape
    n_groups = G // 2 if cfg.get("tied") else G
    out, seen = [], set()
    for c in range(C):
        if st["pol"][c] <= 0:
            continue
        for g in range(n_groups):
            words = np.flatnonzero(mem[c, g]).tolist()
            if len(words) < 2:
                continue
            key = tuple(words)
            if unique and key in seen:
                continue
            seen.add(key)
            out.append({"words": words})
    return out, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=["sa", "np", "ns_all"], required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--stock", nargs="*", default=[])
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    vocab = load_vocab(1024)
    X, n_art = wg.reference(vocab, "article")
    M, df = ng.npmi_matrix(X)
    bins = ng.freq_bins(df)
    W, covered, _ = wg.wordnet(vocab)
    skew = frozenset(vocab.index(w) for w in SKEWED)
    ours, stock = [], []
    for r in a.runs:
        g, cfg = our_groups(r, a.unique)
        assert cfg["data"] == a.data and cfg["F"] == 1024, r
        ours += g
    for r in a.stock:
        stock += wg.load_run(r, "model", a.data)["groups"]

    def both(groups):
        res = {}
        for name, mat, cov, skip in (("npmi", M, None, frozenset()), ("wordnet", W, covered, frozenset()),
                                     ("wordnet_without_skewed_words", W, covered, skew)):
            b = bins.copy()
            b[sorted(skip)] = -2
            picked = wg.pick(groups, cov, skip)
            c = ng.compare(picked, mat, b, a.draws, 1000, 0, 2000)
            res[name] = {"n_groups": c["n_groups"], "mean_size": c.get("mean_size"), "score_groups": c.get("npmi_groups"),
                         "score_random": c.get("npmi_random"), "diff": c.get("diff"), "ci95": c.get("ci95_group_bootstrap"),
                         "beats_random": c.get("beats_random"), "beats_random_calibrated": c.get("beats_random_calibrated")}
        return res
    out = {"data": a.data, "reference_articles": int(n_art), "unique": a.unique,
           "ours": {"runs": [os.path.basename(os.path.normpath(r)) for r in a.runs], "n_groups_2plus": len(ours),
                    "mean_words": float(np.mean([len(g["words"]) for g in ours])) if ours else None, **both(ours)}}
    if stock:
        out["stock"] = {"runs": [os.path.basename(os.path.normpath(r)) for r in a.stock], "n_counted_groups": len(stock),
                        "mean_words": float(np.mean([len(g["words"]) for g in stock])), **both(stock)}
        for name, mat, cov in (("npmi", M, None), ("wordnet", W, covered)):
            c = ng.compare_two(wg.pick(ours, cov), wg.pick(stock, cov), mat, 1000, 0, 2000)
            out[f"ours_minus_stock_{name}"] = {k: c[k] for k in ("n_a", "n_b", "mean_a", "mean_b", "diff", "ci95_group_bootstrap",
                                                                 "a_beats_b", "a_beats_b_calibrated")}
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    return out


if __name__ == "__main__":
    main()
