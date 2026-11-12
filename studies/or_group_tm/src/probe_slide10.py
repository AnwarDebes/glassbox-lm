"""Deletion and one-word probes for a slide-10 run of the text version (run_slide10.py --version text).

Texts, all built from the family sentences of the chosen split and from nothing else:
  deletion texts   each family sentence with the words of one bracket's alternative deleted and nothing added
                   (6 texts per sentence). The construction is that of the deletion probe of the hierarchical_tm
                   study (its src/recovered/e3fb_deletion_probe.py), rebuilt here with a split argument; for the
                   EVAL sentences the texts are compared with that study's own function and must be equal.
  one-word texts   each family sentence with ONE word of a multi-word alternative removed (brackets whose
                   alternative in use has at least two words; only cases where the bag of words changes).
Scored twice, with the model's own semantics (a group is true iff one member word is present, a clause fires iff
every group is true): on the best rule (the clause that check2b chose on the TRAIN sentences) and on the whole model
(at least one clause fires), which is the unit the study's probe scored. Reported per bracket and pooled, with
the study's two-part reading: template-like = accepts at least 90% of the intact sentences and at most 5% of the
deletion texts; blacklist-like = accepts more than 50% of the deletion texts.

For this model a deletion text holds a subset of the words of the matching broken sentences (a broken sentence is a
deletion text plus one filler word), so the best-rule shares follow from the firing rates that check2b reports. The
shares are therefore also given for the deletion texts that are NOT a training negative once the filler word is
dropped.

--split eval is refused unless the run is a registered run of registrations/E3_slide10_text.md (its config names
the record, the record and the pinned files are unchanged, and the run was scored on the EVAL sentences).

Usage (from the study folder): .venv/bin/python -I src/probe_slide10.py runs/<run> [--split dev|eval]
Writes probe_<split>.json into the run folder.
"""
import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
STUDY = os.path.normpath(os.path.join(ROOT, "..", "hierarchical_tm"))
D = os.path.join(STUDY, "data", "slide10")
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
sys.path.insert(1, os.path.join(STUDY, "src"))
import slide10

import e3_common as e3

SPLITS = {"dev": 1, "eval": 2}


def family(split):
    """The family sentences of one split: (rows, token lists, bracket alternatives, bag-of-words function, Xp, kinds)."""
    d = np.load(os.path.join(D, "pairs_slide10.npz"))
    with open(os.path.join(D, "slide10_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(D, "vocab_slide10.json")) as f:
        vocab = json.load(f)
    idx = {w: i for i, w in enumerate(vocab)}
    kinds = np.array([f["kind"] for f in meta["families"]])[d["fam"]]
    rows = np.flatnonzero((kinds == "slide10") & (d["split"] == split))

    def bow(toks):
        return np.isin(np.arange(len(vocab)), [idx[w] for w in toks])
    toks = [slide10.words(meta["sentences"][r]) for r in rows]
    for r, t in zip(rows, toks):
        assert (bow(t) == d["Xp"][r].astype(bool)).all(), f"row {r}: text does not rebuild its bag of words"
    return rows, toks, meta["families"][0]["brackets"], bow, d["Xp"], kinds, vocab


def spans(toks, brackets):
    """(start, length) of the alternative in use, one per bracket."""
    out = []
    for k, alts in enumerate(brackets):
        hits = []
        for alt in alts:
            a = alt.split()
            hits += [(i, len(a)) for i in range(len(toks) - len(a) + 1) if toks[i:i + len(a)] == a]
        assert len(hits) == 1, (k, hits)
        out.append(hits[0])
    return out


def deletion_texts(split):
    """(deletion texts [n * 6, F] bool, intact [n, F] bool, bracket index per deletion text)."""
    _, toks, brackets, bow, _, _, _ = family(split)
    X, intact, which = [], [], []
    for t in toks:
        intact.append(bow(t))
        for k, (i, n) in enumerate(spans(t, brackets)):
            X.append(bow(t[:i] + t[i + n:]))
            which.append(k)
    return np.array(X), np.array(intact), np.array(which)


def one_word_texts(split):
    """(texts [m, F] bool, bracket index per text): one word of a multi-word alternative removed."""
    _, toks, brackets, bow, _, _, _ = family(split)
    X, which = [], []
    for t in toks:
        b0 = bow(t)
        for k, (i, n) in enumerate(spans(t, brackets)):
            if n < 2:
                continue
            for j in range(i, i + n):
                b = bow(t[:j] + t[j + 1:])
                if (b != b0).any():  # the removed word does not occur elsewhere in the sentence
                    X.append(b)
                    which.append(k)
    return np.array(X), np.array(which)


def is_training_negative(X):
    """bool per text: its bag of words equals a broken sentence of text/train.npz with the filler words dropped."""
    _, _, _, _, Xp, kinds, vocab = family(1)
    t = np.load(os.path.join(D, "text", "train.npz"))
    rows = np.unique(t["row"][kinds[t["row"]] == "broken"])
    bags = Xp[rows].astype(bool)
    bags[:, [vocab.index(w) for w in slide10.FILLERS]] = False
    seen = {b.tobytes() for b in bags}
    return np.array([x.tobytes() in seen for x in X.astype(bool)])


def reading(intact_share, deletion_share):
    if deletion_share > 0.5:
        return "blacklist-like (accepts > 50% of deletions)"
    if deletion_share <= 0.05:
        return "template-like" if intact_share >= 0.9 else "reject-all (degenerate)"
    return "neither (between)"


def eval_problems(run_dir, cfg):
    """Why this run may not be probed on the EVAL sentences (empty list: it may)."""
    problems = []
    rel = cfg.get("registered", "")
    record = os.path.join(ROOT, rel) if rel else ""
    if cfg.get("split") != "eval":
        problems.append("the run was not scored on the EVAL sentences")
    if not rel or e3.is_dry_record(record):
        problems.append("the run names no record of registrations/E3_slide10_text.md")
    else:
        problems += e3.preflight(record, False)[1]
    folder = os.path.abspath(run_dir)
    name = f"{e3.PREFIX[False]}{cfg.get('seed')}"
    if (cfg.get("name") != name or not os.path.basename(folder).endswith("_" + name)
            or os.path.dirname(folder) != os.path.join(ROOT, e3.RUNS_ROOT[False])):
        problems.append("not the folder of a registered run")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    a = ap.parse_args(argv)
    with open(os.path.join(a.run_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["version"] == "text", "the probe is defined for the text version"
    if a.split == "eval":
        problems = eval_problems(a.run_dir, cfg)
        if problems:
            ap.error("the EVAL sentences are refused: " + "; ".join(problems))
    with open(os.path.join(a.run_dir, "metrics.json")) as f:
        chk = json.load(f)["final"]["check2b"]
    st = np.load(os.path.join(a.run_dir, "state.npz"))
    mem = st["state"] >= 0
    pos = np.flatnonzero(st["pol"] > 0)
    best = int(pos[chk["best_rule"] // 2])  # check2b numbers the positive rules 0, 2, 4, ...
    X, intact, which = deletion_texts(SPLITS[a.split])
    if a.split == "eval":  # the study's own function builds the EVAL texts; they must be the same
        sys.path.insert(2, os.path.join(STUDY, "src", "recovered"))
        import e3fb_deletion_probe as ref
        Xr, ir, info, _ = ref.deletion_texts()
        assert np.array_equal(Xr, X) and np.array_equal(ir, intact) and [k for _, k in info] == which.tolist(), \
            "the deletion texts differ from those of the study's probe"
    X1, which1 = one_word_texts(SPLITS[a.split])
    new = ~is_training_negative(X)

    def accept(texts):
        """[n, clauses]: every group of the clause has a present member."""
        out = np.ones((len(texts), mem.shape[0]), bool)
        for c in range(mem.shape[0]):
            for g in range(mem.shape[1]):
                out[:, c] &= (texts & mem[c, g]).any(1)
        return out
    acc_i, acc_d, acc_1 = accept(intact), accept(X), accept(X1)
    out = {"run": os.path.basename(os.path.normpath(a.run_dir)), "split": a.split, "best_clause": best,
           "n_intact": len(intact), "n_deletion_texts": len(X), "n_deletion_texts_not_training_negatives": int(new.sum()),
           "n_one_word_texts": len(X1),
           "n_one_word_texts_per_bracket": {str(int(k) + 1): int((which1 == k).sum()) for k in np.unique(which1)},
           "template_like_clauses": int(((acc_i[:, pos].mean(0) >= 0.9) & (acc_d[:, pos].mean(0) <= 0.05)).sum()),
           "n_clauses": len(pos)}
    for name, ai, ad, a1 in (("best_rule", acc_i[:, best], acc_d[:, best], acc_1[:, best]),
                             ("model", acc_i[:, pos].any(1), acc_d[:, pos].any(1), acc_1[:, pos].any(1))):
        out[f"{name}_accepts_intact"] = float(ai.mean())
        out[f"{name}_accepts_deletions"] = float(ad.mean())
        out[f"{name}_accepts_deletions_per_bracket"] = [float(ad[which == k].mean()) for k in range(6)]
        out[f"{name}_accepts_deletions_not_training_negatives"] = float(ad[new].mean()) if new.any() else None
        out[f"{name}_reading"] = reading(float(ai.mean()), float(ad.mean()))
        out[f"{name}_accepts_one_word"] = float(a1.mean())
        out[f"{name}_accepts_one_word_per_bracket"] = {str(int(k) + 1): float(a1[which1 == k].mean()) for k in np.unique(which1)}
    with open(os.path.join(a.run_dir, f"probe_{a.split}.json"), "x" if a.split == "eval" else "w") as f:
        json.dump(out, f, indent=1)
    print("PROBE", json.dumps(out))
    return out


if __name__ == "__main__":
    main()
