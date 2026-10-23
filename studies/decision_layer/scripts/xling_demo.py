#!/usr/bin/env python
"""Post hoc demo for R12: Norwegian test utterances routed by the English-trained readable TM.

R12's runs saved predictions, not models. This script retrains seed 1 with the same view and
setting as `scripts/xling_run.py`, checks that it reproduces the registered class sums of seed 1
on the nb-NO test split exactly, and prints decision receipts (the strongest firing clauses) for a
few Norwegian utterances next to their English originals. Nothing here is a registered result.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.explain import clause_literals  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def short(lit: str, intents: list[str]) -> str:
    lit = re.sub(r"sim_w\(q, proto\d+\[(.+?)\]\) >= (-?[0-9.]+)", r"near(\1) >= \2", lit)
    return re.sub(r"one of the (\d+) nearest prototypes is (.+)", r"\2 among the \1 nearest", lit)


def main() -> None:
    enc, seed = "bge-m3", 1
    en, nb = load("massive_en"), load("massive_nb")
    de = np.load(ROOT / "data" / "emb" / f"massive_en__{enc}.npz")
    dn = np.load(ROOT / "data" / "emb" / f"massive_nb__{enc}.npz")
    view = make_view("proto:k=1:whiten=0.001", seed=0)
    view.fit(de["train"], en.labels["train"], texts=en.texts["train"], class_names=en.intents)
    Xtr, Xva = view.transform(de["train"]), view.transform(de["val"])
    Xnb = view.transform(dn["test"])
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, seed=seed)
    tm = make_tm(len(en.intents), Xtr.shape[1], cfg, init="half")
    tm.fit(Xtr, en.labels["train"], Xva, en.labels["val"])
    S = tm.class_sums(torch.as_tensor(Xnb, dtype=torch.uint8, device="cuda")).cpu().numpy()
    reg = np.load(ROOT / "results" / "confirm" / "R12" / enc / f"preds_seed{seed}.npz")
    same = bool(np.array_equal(S.astype(np.int32), reg["S_nb"]))
    print(f"retrained seed {seed} reproduces the registered nb-NO class sums exactly: {same}")
    ta, w = tm.ta.cpu().numpy(), tm.w.cpu().numpy()
    incl = ta > 256
    names = [short(n, en.intents) for n in view.names]
    f = len(names)
    rng = np.random.default_rng(0)
    y = nb.labels["test"]
    out = [f"retrained seed {seed} reproduces the registered nb-NO class sums exactly: {same}"]
    for i in rng.choice(len(y), size=8, replace=False):
        lit = np.concatenate([Xnb[i], 1 - Xnb[i]])
        fired = ((incl & (lit[None, :] == 0)).sum(1) == 0) & (incl.sum(1) > 0)
        Sv = (fired[:, None] * w).sum(0)
        c = int(np.argmax(Sv))
        contrib = fired * w[:, c]
        j = int(np.argmax(contrib))
        L = clause_literals(ta[j], 256, f, names)
        pos = [x for x in L if not x.startswith("NOT ")]
        neg = [x for x in L if x.startswith("NOT ")]
        mark = "right" if c == y[i] else f"wrong (true: {en.intents[y[i]]})"
        block = [f"\nNorwegian: {nb.texts['test'][i]}", f"English original: {en.texts['test'][i]}",
                 f"TM routes to {en.intents[c]}: {mark}; strongest clause +{int(contrib[j])}: IF " + " AND ".join((pos + neg)[:6])
                 + (f" AND {len(L) - 6} more" if len(L) > 6 else "")]
        print("\n".join(block))
        out += block
    p = ROOT.parents[2] / "research" / "decision_layer" / "r12_norwegian_receipts_post_hoc.txt"
    p.write_text("\n".join(out) + "\n")
    print("written", p)


if __name__ == "__main__":
    main()
