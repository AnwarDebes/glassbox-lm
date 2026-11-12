"""Facts about the slide-10 data of the hierarchical_tm study that the registration E3 quotes, computed from the
arrays alone. No trained model is involved: the only rule applied to a sentence is a hand-written 'template clause'
built from the brackets in slide10_meta.json (one group of single words per bracket, plus one group with the fixed
word that no negated twin holds).

  .venv/bin/python -I src/slide10_facts.py [--out logs/slide10_facts.json]

Prints one JSON object: sentences per kind and split, the 216 view, the make-up of text/train.npz and text/dev.npz,
overlaps between the splits, how near the EVAL family sentences are to training sentences, where the template
clause fires, the ceiling that the data put on the bracket rule of Check 2b for a clause of single-word groups, and
how many deletion texts are training negatives once the filler word is dropped.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
sys.path.insert(1, os.path.normpath(os.path.join(os.path.dirname(SRC), "..", "hierarchical_tm", "src")))
import slide10

import probe_slide10 as ps

KINDS = ("slide10", "negated", "nearmiss", "broken", "other")
SPLITS = (("train", 0), ("dev", 1), ("eval", 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    d = np.load(os.path.join(ps.D, "pairs_slide10.npz"))
    with open(os.path.join(ps.D, "slide10_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ps.D, "vocab_slide10.json")) as f:
        vocab = json.load(f)
    idx = {w: i for i, w in enumerate(vocab)}
    X, fam, split, in216 = d["Xp"].astype(bool), d["fam"], d["split"], d["in216"].astype(bool)
    kind = np.array([f["kind"] for f in meta["families"]])[fam]
    fam0 = meta["families"][0]
    fixed = set(fam0["fixed"])
    out = {"sentences": len(X), "words": len(vocab), "families": len(meta["families"]),
           "distinct_bags_of_words": len({r.tobytes() for r in X}),
           "sentences_per_kind": {k: {n: int(((kind == k) & (split == s)).sum()) for n, s in SPLITS} for k in KINDS},
           "family_sentences_in_the_216_view": {n: int(((kind == "slide10") & (split == s) & in216).sum()) for n, s in SPLITS}}
    for name in ("train", "dev"):
        t = np.load(os.path.join(ps.D, "text", f"{name}.npz"))
        rows = t["row"]
        out[f"text_{name}_npz"] = {
            "rows": len(rows), "label_1_rows": int(t["y"].sum()), "splits_of_the_rows": sorted(set(split[rows].tolist())),
            "rows_per_kind": {k: int((kind[rows] == k).sum()) for k in KINDS},
            "distinct_sentences_per_kind": {k: len(np.unique(rows[kind[rows] == k])) for k in KINDS}}
    bags = {n: {r.tobytes() for r in X[split == s]} for n, s in SPLITS}
    out["equal_bags_between_splits"] = {"train_dev": len(bags["train"] & bags["dev"]), "train_eval": len(bags["train"] & bags["eval"]),
                                        "dev_eval": len(bags["dev"] & bags["eval"])}

    # the alternatives in use, per family sentence
    combos = {}
    for n, s in SPLITS:
        _, toks, brackets, _, _, _, _ = ps.family(s)
        combos[n] = [tuple(" ".join(t[i:i + m]) for i, m in ps.spans(t, brackets)) for t in toks]
    out["alternatives_per_bracket"] = [len(b) for b in fam0["brackets"]]
    out["times_each_alternative_is_in_a_training_family_sentence"] = [
        [sum(c[k] == alt for c in combos["train"]) for alt in alts] for k, alts in enumerate(fam0["brackets"])]
    out["combinations_shared_between_splits"] = len(set(combos["train"]) & set(combos["dev"])) + len(
        set(combos["train"]) & set(combos["eval"])) + len(set(combos["dev"]) & set(combos["eval"]))
    for n in ("dev", "eval"):
        dist = [min(sum(x != y for x, y in zip(c, t)) for t in combos["train"]) for c in combos[n]]
        out[f"{n}_family_sentences_by_brackets_changed_against_the_nearest_training_sentence"] = {
            str(k): int(v) for k, v in zip(*np.unique(dist, return_counts=True))}
    fam_words = X[kind == "slide10"].any(0)
    task = X[np.isin(kind, KINDS[:4])].any(0)
    out["words_of_the_family"] = int(fam_words.sum())
    out["words_of_the_direct_label_task"] = int(task.sum())
    out["eval_family_words_missing_from_training_family_sentences"] = int(
        (X[(kind == "slide10") & (split == 2)].any(0) & ~X[(kind == "slide10") & (split == 0)].any(0)).sum())

    # the hand-written template clause
    groups = [sorted(set(" ".join(alts).split()) - fixed) for alts in fam0["brackets"]]
    lack = [w for w in sorted(fixed) if not X[kind == "negated"][:, idx[w]].any()]

    def clause(gs):
        M = np.zeros((len(gs), len(vocab)), bool)
        for j, g in enumerate(gs):
            M[j, [idx[w] for w in g]] = True
        return M

    def fires(M, T):
        ok = np.ones(len(T), bool)
        for g in M:
            ok &= T[:, g].any(1)
        return ok
    full = groups + [lack]
    f = fires(clause(full), X)
    out["template_clause"] = {
        "words_per_group": [len(g) for g in full],
        "fires_on": {n: {k: [int(f[(kind == k) & (split == s)].sum()), int(((kind == k) & (split == s)).sum())] for k in KINDS}
                     for n, s in SPLITS}}

    # ceiling of the bracket rule for a clause with one group of single words per bracket: the best Jaccard a group
    # can reach for bracket k while the clause still covers at least 0.90 of the TRAIN family sentences and fires on
    # at most 0.05 of each kind of TRAIN negative. First pass: the other groups hold their full brackets. A bracket
    # whose full group is not allowed (it lets too many negatives through) keeps its largest allowed subset, and the
    # brackets for which nothing was allowed in the first pass are tried again with those subsets in place.
    tr = split == 0

    def allowed(gs):
        fk = fires(clause(gs), X[tr])
        return fk[kind[tr] == "slide10"].mean() >= 0.9 and max(fk[kind[tr] == n].mean() for n in slide10.NEG_KINDS) <= 0.05

    def best_subsets(base, k):
        """{view: (Jaccard, subset)} over the allowed subsets of bracket k, the other groups as in base."""
        best = {"288": (0.0, None), "216": (0.0, None)}
        for m in range(len(groups[k]), 0, -1):
            for sub in itertools.combinations(groups[k], m):
                if not allowed(base[:k] + [list(sub)] + base[k + 1:]):
                    continue
                for view, key in (("288", "brackets"), ("216", "brackets_216")):
                    true = set(" ".join(fam0[key][k]).split()) - fixed
                    j = len(set(sub) & true) / len(set(sub) | true)
                    if j > best[view][0]:
                        best[view] = (j, list(sub))
        return best
    first = [best_subsets(full, k) for k in range(len(groups))]
    base = list(full)
    for k, b in enumerate(first):
        if b["288"][1] is not None and len(b["288"][1]) < len(groups[k]):
            base[k] = b["288"][1]
    final = [b if b["288"][1] is not None else best_subsets(base, k) for k, b in enumerate(first)]
    ceil = {view: [round(b[view][0], 3) for b in final] for view in ("288", "216")}
    out["template_clause_allowed_as_it_is"] = bool(allowed(full))
    out["brackets_whose_full_group_is_not_allowed"] = [k + 1 for k in range(len(groups)) if base[k] != full[k]]
    out["ceiling_of_the_bracket_jaccard_per_bracket"] = ceil
    out["brackets_that_can_be_recovered"] = {v: int(sum(j >= 0.8 for j in c)) for v, c in ceil.items()}

    for n, s in (("dev", 1), ("eval", 2)):
        Xd, _, which = ps.deletion_texts(s)
        seen = ps.is_training_negative(Xd)
        out[f"deletion_texts_{n}"] = {"n": len(Xd), "training_negatives_once_the_filler_is_dropped": int(seen.sum()),
                                      "per_bracket": [int(seen[which == k].sum()) for k in range(6)]}
        X1, which1 = ps.one_word_texts(s)
        out[f"one_word_texts_{n}"] = {"n": len(X1), "per_bracket": {str(int(k) + 1): int((which1 == k).sum()) for k in np.unique(which1)}}
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "x") as fh:
            fh.write(text + "\n")
    return out


if __name__ == "__main__":
    main()
