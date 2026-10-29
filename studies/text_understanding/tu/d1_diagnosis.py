#!/usr/bin/env python
"""Text understanding, step 1 (DESCRIPTIVE diagnosis, intent VAL only; plan PLAN_text_understanding.md, approved by
the decided 21:38Z; rules in LOG.md before the run). CPU only, one thread. No test split is read: the MiniLM
cache's "test" arrays are never loaded.

Probes: W2's logistic-regression probe (dl.linear.fit_logreg on the CPU) on three bit sets of CLINC150, Banking77 and
HWU64: the C4 table ppw_S1000's existsT (W1-S caches) and potion's existsT (test 2's caches), each with the C that
w2/results_w2_lr.jsonl chose; MiniLM: dl.boolean.Thermo(8) fitted on the training embeddings of the decision-layer
study's cache data/emb/{ds}__minilm.npz (all-MiniLM-L6-v2), C in 1 / 10 / 100, the best on val (a diagnostic choice).
Groups (val examples): potion-only = potion right, ppw wrong; ppw-only; MiniLM-only-vs-potion; MiniLM-only-vs-ppw.
Automatic types per example (glove_feats.glove_tokens; fixed before the run):
- N negation: a token in NEGATORS;
- R rare: a token seen fewer than 5 times in that dataset's training texts, or with no C4 row (ppw_S1000);
- T typo: a token with no C4 row within edit distance 2 of a C4 word (Norvig's edits; letters a-z);
- P paraphrase: none of N / R / T, and the content-word Jaccard to the nearest training example of the true class is
  below 0.5 (content words: tokens outside the 150 most frequent C4 words);
- X other: none of the above.
Hand-check file: 100 potion-only and 100 MiniLM-only-vs-potion examples, a stratified draw per dataset in proportion
to the group sizes (numpy default_rng(0)), with the true and the losing probe's labels; the main type is written by
hand afterwards (N negation, O order, A antonym, P paraphrase, R rare, T typo, B brand or entity, X other), judged by
an AI model, not a person. Outputs tu/d1_counts.json, tu/d1_diagnosis.txt, tu/handcheck_d1_200.tsv (refuses to overwrite).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from glove_feats import glove_tokens  # noqa: E402
from dl.boolean import Thermo  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402

SETS = ("clinc150", "banking77", "hwu64")
DIRS = {"potion": EMB / "emb_potion_test2", "ppw_S1000": EMB / "w1" / "feats" / "ppw_S1000_d500_e0.5"}
MINILM = Path(f"{_REPO_ROOT}/studies/decision_layer/data/emb")
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
VOCAB = EMB / "w1" / "corpus" / "c4_words_vocab.json"
NEGATORS = {"not", "no", "never", "n't", "dont", "cant", "cannot", "without", "nothing", "none", "nobody", "nowhere",
            "neither", "nor", "wont", "isnt", "arent", "wasnt", "doesnt", "didnt", "havent", "hasnt", "shouldnt", "wouldnt",
            "couldnt", "aint", "stop"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def edits1(w):
    s = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in s if b} | {a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1}
            | {a + c + b[1:] for a, b in s if b for c in LETTERS} | {a + c + b for a, b in s for c in LETTERS})


def near(w, known):
    e1 = edits1(w)
    if e1 & known:
        return True
    return any(e2 in known for e in e1 for e2 in edits1(e))


def probe(Btr, ytr, Bva, yva, grid):
    n_classes = int(max(ytr.max(), yva.max())) + 1
    best = None
    for C in grid:
        W, b = fit_logreg(Btr.astype(np.float32), ytr, n_classes, C, device="cpu")
        p = logits(Bva.astype(np.float32), W, b, device="cpu").argmax(1).cpu().numpy()
        acc = float((p == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, C, p)
    return best


def table_bits(name, ds):
    F, meta = X.features(ds, 512, DIRS[name], "cpu")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    v = X.View("existsT", meta).fit(F)
    return v.transform(F, "train"), v.transform(F, "val"), F["y_train"], F["y_val"]


def main() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    outs = [HERE / "d1_counts.json", HERE / "d1_diagnosis.txt", HERE / "handcheck_d1_200.tsv"]
    assert not any(p.exists() for p in outs), "outputs exist; not overwritten"
    lr_rows = {}
    for x in (EMB / "w2" / "results_w2_lr.jsonl").read_text().splitlines():
        r = json.loads(x)
        lr_rows[(r["name"], r["dataset"])] = r
    z = np.load(TABLE)
    live = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
    known = {str(w) for w, l in zip(z["words"], live) if l}
    voc = json.loads(VOCAB.read_text())
    stop = set(voc["words"][:150])
    counts, lines, pool = {}, [], []
    for ds in SETS:
        data = load(ds)
        tr_texts, va_texts = data.texts["train"], data.texts["val"]
        pred, acc = {}, {}
        y_ref = None
        for name in DIRS:
            Btr, Bva, ytr, yva = table_bits(name, ds)
            acc[name], _, pred[name] = probe(Btr, ytr, Bva, yva, [lr_rows[(name, ds)]["C"]])
            y_ref = yva if y_ref is None else y_ref
            assert np.array_equal(yva, y_ref)
        m = np.load(MINILM / f"{ds}__minilm.npz")
        mtr, mva = m["train"], m["val"]  # the "test" arrays are never loaded
        assert np.array_equal(m["y_val"], y_ref) and np.array_equal(np.asarray(data.labels["val"]), y_ref)
        assert np.array_equal(m["y_train"], np.asarray(data.labels["train"]))
        th = Thermo(8).fit(mtr)
        acc["minilm"], c_m, pred["minilm"] = probe(th.transform(mtr), m["y_train"], th.transform(mva), y_ref, [1.0, 10.0, 100.0])
        right = {k: pred[k] == y_ref for k in pred}
        groups = {"potion_only": right["potion"] & ~right["ppw_S1000"], "ppw_only": right["ppw_S1000"] & ~right["potion"],
                  "minilm_only_vs_potion": right["minilm"] & ~right["potion"],
                  "potion_only_vs_minilm": right["potion"] & ~right["minilm"],
                  "minilm_only_vs_ppw": right["minilm"] & ~right["ppw_S1000"]}
        tr_tok = [glove_tokens(t) for t in tr_texts]
        freq = {}
        for tk in tr_tok:
            for w in tk:
                freq[w] = freq.get(w, 0) + 1
        ytr_all = np.asarray(data.labels["train"])
        by_class = {}
        for t, c in zip(tr_tok, ytr_all):
            by_class.setdefault(int(c), []).append({w for w in t if w not in stop})
        va_tok = [glove_tokens(t) for t in va_texts]
        types = []
        typo_cache = {}
        for i, tk in enumerate(va_tok):
            ty = set()
            if any(w in NEGATORS for w in tk):
                ty.add("N")
            if any(freq.get(w, 0) < 5 or w not in known for w in tk):
                ty.add("R")
            for w in tk:
                if w not in known and w.isalpha():
                    if w not in typo_cache:
                        typo_cache[w] = near(w, known)
                    if typo_cache[w]:
                        ty.add("T")
            if not ty:
                cw = {w for w in tk if w not in stop}
                best = max((len(cw & s) / max(len(cw | s), 1) for s in by_class.get(int(y_ref[i]), [])), default=0.0)
                if best < 0.5:
                    ty.add("P")
            types.append(ty or {"X"})
        c = {"n_val": int(len(y_ref)), "acc": acc, "minilm_C": c_m}
        for g, mask in groups.items():
            idx = np.flatnonzero(mask)
            tc = {t: int(sum(1 for i in idx if t in types[i])) for t in ("N", "R", "T", "P", "X")}
            c[g] = {"n": int(len(idx)), "types": tc}
        c["all"] = {t: int(sum(1 for ty in types if t in ty)) for t in ("N", "R", "T", "P", "X")}
        counts[ds] = c
        lines.append(f"== {ds}: val {len(y_ref)}; LR ppw_S1000 {acc['ppw_S1000']:.4f}, potion {acc['potion']:.4f}, MiniLM bits {acc['minilm']:.4f} (C {c_m})")
        for g in groups:
            e = c[g]
            lines.append(f"  {g:24s} n {e['n']:4d}  " + "  ".join(f"{t} {e['types'][t]}" for t in ("N", "R", "T", "P", "X")))
        lines.append("  all val                     " + "  ".join(f"{t} {c['all'][t]}" for t in ("N", "R", "T", "P", "X")))
        names = data.intents
        for g, loser in (("potion_only", "ppw_S1000"), ("minilm_only_vs_potion", "potion")):
            for i in np.flatnonzero(groups[g]):
                pool.append((ds, g, va_texts[i], names[y_ref[i]], names[pred[loser][i]], "".join(sorted(types[i]))))
    # the stratified hand-check draw
    rng = np.random.default_rng(0)
    rows = []
    for g in ("potion_only", "minilm_only_vs_potion"):
        cand = [p for p in pool if p[1] == g]
        tot = len(cand)
        for ds in SETS:
            cd = [p for p in cand if p[0] == ds]
            k = round(100 * len(cd) / tot) if tot else 0
            pick = rng.choice(len(cd), min(k, len(cd)), replace=False) if cd else []
            rows += [cd[j] for j in sorted(pick)]
    with open(HERE / "handcheck_d1_200.tsv", "w") as f:
        f.write("id\tdataset\tgroup\ttext\ttrue\tloser_said\tauto_types\tmain_type_by_hand\n")
        for n, r in enumerate(rows):
            f.write(f"{n}\t" + "\t".join(str(x).replace("\t", " ") for x in r) + "\t\n")
    (HERE / "d1_counts.json").write_text(json.dumps(counts, indent=1))
    lines.append(f"hand-check file: {len(rows)} rows (potion-only and MiniLM-only-vs-potion, stratified); {time.time() - t0:.0f} s")
    (HERE / "d1_diagnosis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
