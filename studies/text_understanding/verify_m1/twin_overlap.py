#!/usr/bin/env python
"""The leakage check's minor 1, checked independently (descriptive; CPU; reads the M1 a3B_twin caches, the saved test
predictions and the test labels; scores no model): test rows whose existsT bits (the thermometer fit on the originals,
as m1_tm.py) equal an EDA-twin row's bits but no original's, and H3b without those test rows.

    python -I -B verify_m1/twin_overlap.py
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import json
import sys

import numpy as np

sys.path.insert(0, f"{_REPO_ROOT}/studies/text_understanding")
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from dl.data import load  # noqa: E402

M1 = f"{_REPO_ROOT}/studies/text_understanding/tu/m1"
lr = list(map(json.loads, open(f"{M1}/results_m1_lr.jsonl")))
tm = [r for r in map(json.loads, open(f"{M1}/results_m1_tm.jsonl")) if r.get("tag") == "m1"]
out = {}
for d in ("clinc150", "banking77", "hwu64"):
    z = np.load(f"{M1}/feats/a3B_twin/{d}__p32tok_K512.npz")
    meta = json.load(open(f"{M1}/feats/a3B_twin/{d}__p32tok_K512.meta.json"))
    F = {k: z[k] for k in z.files}
    for sp in ("train", "test"):
        F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    n = int(F["n_orig"])
    view = X.View("existsT", meta).fit({k: (v[:n] if k.startswith("train__") else v) for k, v in F.items()})
    Btr, Bte = view.transform(F, "train"), view.transform(F, "test")
    orig = {r.tobytes() for r in np.packbits(Btr[:n], axis=1)}
    twin = {r.tobytes() for r in np.packbits(Btr[n:], axis=1)}
    pt = np.packbits(Bte, axis=1)
    hit = np.array([(r.tobytes() in twin) and (r.tobytes() not in orig) for r in pt])
    y = load(d).labels["test"]
    keep = ~hit
    lrp = {a: np.load(next(r for r in lr if r["dataset"] == d and r["name"] == a)["preds"]) for a in ("a3B_3c", "a3B_twin")}
    h3b_all = 100 * ((lrp["a3B_3c"] == y).mean() - (lrp["a3B_twin"] == y).mean())
    h3b_wo = 100 * ((lrp["a3B_3c"][keep] == y[keep]).mean() - (lrp["a3B_twin"][keep] == y[keep]).mean())
    tw_on_hit = float((lrp["a3B_twin"][hit] == y[hit]).mean()) if hit.any() else None
    out[d] = {"n_test": int(len(y)), "twin_only_bit_matches": int(hit.sum()), "H3b_lr_all": round(h3b_all, 4),
              "H3b_lr_without": round(h3b_wo, 4), "twin_lr_acc_on_matched": tw_on_hit}
print(json.dumps(out))
