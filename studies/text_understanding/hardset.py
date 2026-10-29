#!/usr/bin/env python
"""Fixed before the existential-bits runs (VAL only): the "composition losses" of each dataset.

Logistic regression on the raw potion-32m average and on the raw MiniLM embedding (C grid of errors.py,
best on val). lost = val queries MiniLM gets right and the static average gets wrong; won = the other way.
Writes hardset.json here. Reads only the train and val arrays of the MiniLM cache, never test.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import sys

import numpy as np

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.linear import fit_logreg, logits  # noqa: E402

EMB = f"{_REPO_ROOT}/studies/text_understanding/emb_potion_test2"
MINI = f"{_REPO_ROOT}/studies/decision_layer/data/emb"


def pred(Xtr, ytr, Xva, yva, C):
    best = None
    for c in (10.0, 30.0, 100.0):
        W, b = fit_logreg(Xtr.astype(np.float32), ytr, C, c)
        p = logits(Xva.astype(np.float32), W, b).argmax(1).cpu().numpy()
        acc = float((p == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, c, p)
    return best


out = {}
for name in ("clinc150", "banking77", "hwu64"):
    s = np.load(f"{EMB}/{name}__potion-32m.npz")
    m = np.load(f"{MINI}/{name}__minilm.npz")
    ytr, yva = s["y_train"], s["y_val"]
    assert (m["y_train"] == ytr).all() and (m["y_val"] == yva).all()
    C = int(ytr.max()) + 1
    a_s, c_s, p_s = pred(s["train"], ytr, s["val"], yva, C)
    a_m, c_m, p_m = pred(m["train"], ytr, m["val"], yva, C)
    lost = [int(i) for i in np.flatnonzero((p_m == yva) & (p_s != yva))]
    won = [int(i) for i in np.flatnonzero((p_m != yva) & (p_s == yva))]
    out[name] = {"lr_static": round(a_s, 4), "C_static": c_s, "lr_minilm": round(a_m, 4), "C_minilm": c_m,
                 "n_val": int(len(yva)), "lost": lost, "won": won}
    print(name, out[name]["lr_static"], out[name]["lr_minilm"], len(lost), len(won))
with open(f"{_REPO_ROOT}/studies/text_understanding/hardset.json", "w") as f:
    json.dump(out, f)
