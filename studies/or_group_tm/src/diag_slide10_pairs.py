"""Diagnostic for the pair version of slide 10: for every clause of a run, the family whose TRAIN sentences both
halves accept most often, and for the slide-10 family the acceptance of each half on each kind of sentence.
Usage (from the study folder): .venv/bin/python -I src/diag_slide10_pairs.py runs/<run> [n_clauses_to_print]"""
import json
import os
import sys

import numpy as np

S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hierarchical_tm", "data", "slide10")


def main():
    d = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    z = np.load(f"{S}/pairs_slide10.npz")
    with open(f"{S}/slide10_meta.json") as f:
        meta = json.load(f)
    with open(f"{S}/vocab_slide10.json") as f:
        vocab = json.load(f)
    X, fam, split = z["Xp"].astype(bool), z["fam"], z["split"]
    kind = np.array([f["kind"] for f in meta["families"]])[fam]
    st = np.load(f"{d}/state.npz")["state"]
    C, G, _ = st.shape
    K = G // 2
    mem = st >= 0
    acc = np.zeros((len(X), C, 2), bool)
    for c in range(C):
        for s in range(2):
            ok = np.ones(len(X), bool)
            for k in range(K):
                ok &= (X & mem[c, s * K + k]).any(1)
            acc[:, c, s] = ok
    tr = split == 0
    nf = len(meta["families"])
    best_fam = np.zeros(C, int)
    best_rate = np.zeros(C)
    for c in range(C):
        both = acc[:, c].all(1)
        rate = np.array([both[tr & (fam == i)].mean() for i in range(nf)])
        best_fam[c], best_rate[c] = int(rate.argmax()), rate.max()
    names = [f"{f['kind']}{i}" for i, f in enumerate(meta["families"])]
    per = {names[i]: int(((best_fam == i) & (best_rate >= 0.9)).sum()) for i in range(nf)}
    print("clauses that accept >= 90% of a family's train sentences on both halves, by family:", per)
    print("clauses with an empty group:", int(((st >= 0).sum(-1) == 0).any(1).sum()), "of", C)
    # the clauses closest to the slide-10 family
    s10 = tr & (kind == "slide10")
    cover = acc[s10].all(2).mean(0)
    order = np.argsort(-cover)[:top]
    for c in order:
        print(f"clause {c}: both halves accept {cover[c]:.3f} of the slide-10 train sentences; per half and kind (A / B):",
              {k: (round(float(acc[tr & (kind == k), c, 0].mean()), 3), round(float(acc[tr & (kind == k), c, 1].mean()), 3))
               for k in ("slide10", "negated", "nearmiss", "broken", "other")})
        for g in range(G):
            words = np.flatnonzero(mem[c, g])
            print("   ", "AB"[g // K] + str(g % K + 1), "[" + " | ".join(f"{vocab[i]}:{st[c, g, i]}" for i in words) + "]")


if __name__ == "__main__":
    main()
