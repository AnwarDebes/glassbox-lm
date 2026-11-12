"""Duplicate articles of the Wikipedia data of the hierarchical_tm study, counted from the word ids of the paragraphs
alone (its data/paragraphs_F3072.npz and data/splits.json; no pair, no label, no model and no score is read). Writes
the article lists of the E5 reading "without the pairs that touch a duplicate" (registrations/E5_duplicate_articles.json).

A paragraph is its set of word ids among the 3,072 most frequent words, as the study stored it. A paragraph of a DEV or
test article is duplicated when its set is not empty and equals the set of some paragraph of a training article. An
article "has a duplicated paragraph" when at least one of its paragraphs is duplicated; it is a "whole copy" when it has
at least one paragraph and its paragraphs, in order, equal those of some training article. Equal word sets at 3,072
words are an upper bound for paragraphs that are equal word for word (the text itself is not in the data).

Usage (from the study folder):
  .venv/bin/python -I src/e5_duplicates.py --out registrations/E5_duplicate_articles.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.path.normpath(os.path.join(ROOT, "..", "hierarchical_tm", "data"))
SUBSETS = ("whole_copy_articles", "articles_with_a_duplicated_paragraph")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def article_paragraphs(paragraphs_file):
    """A function art -> tuple of the word-id sets (as bytes) of the article's paragraphs, in order."""
    z = np.load(paragraphs_file)
    ids, off, npar = z["ids"], z["off"], z["npar"]
    start = np.r_[0, np.cumsum(npar.astype(np.int64))]

    def paras(art):
        return tuple(ids[off[p]:off[p + 1]].tobytes() for p in range(int(start[art]), int(start[art + 1])))
    return paras


def count(arts, paras, ref_paragraphs, ref_articles):
    out = {"articles": len(arts), "paragraphs": 0, "duplicated_paragraphs": 0}
    dup, whole = [], []
    for a in arts:
        ps = paras(a)
        out["paragraphs"] += len(ps)
        d = sum(1 for p in ps if p and p in ref_paragraphs)
        out["duplicated_paragraphs"] += d
        if d:
            dup.append(int(a))
        if ps and ps in ref_articles:
            whole.append(int(a))
    out["articles_with_a_duplicated_paragraph"] = len(dup)
    out["whole_copy_articles"] = len(whole)
    return out, {"whole_copy_articles": sorted(whole), "articles_with_a_duplicated_paragraph": sorted(dup)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    pf, sf = os.path.join(DATA, "paragraphs_F3072.npz"), os.path.join(DATA, "splits.json")
    with open(sf) as f:
        splits = json.load(f)
    paras = article_paragraphs(pf)
    ref = {}
    for name in ("train", "dev"):
        ps = [paras(x) for x in splits[name]]
        ref[name] = ({p for art in ps for p in art if p}, {art for art in ps if art})
    out = {"made_by": "src/e5_duplicates.py", "inputs_sha256": {"paragraphs_F3072.npz": sha(pf), "splits.json": sha(sf)},
           "definition": __doc__.split("\n\n")[1].replace("\n", " ")}
    for split in ("dev", "test"):
        counts, lists = count(splits[split], paras, *ref["train"])
        out[split] = {"counts_against_training_articles": counts, "subsets": lists}
    out["test_against_dev_articles"] = count(splits["test"], paras, *ref["dev"])[0]
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("dev", "test")} | {
        s: out[s]["counts_against_training_articles"] for s in ("dev", "test")}, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
