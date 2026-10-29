#!/usr/bin/env python
"""Post hoc (VAL only): where does the static embedding lose against MiniLM? Logistic regression on each raw
embedding (C from the static_tm.py grid, best on val), CLINC150 val; lists queries that MiniLM gets right and
potion-32m gets wrong, and counts how many of them contain a negation or depend on word order (by eye)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json, sys
import numpy as np
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import load
from dl.linear import fit_logreg, logits
emb = f"{_REPO_ROOT}/studies/text_understanding/emb_potion_test2"
ds = load("clinc150"); C = len(ds.intents)
s = np.load(f"{emb}/clinc150__potion-32m.npz")
m = np.load(f"{_REPO_ROOT}/studies/decision_layer/data/emb/clinc150__minilm.npz")
def pred(z, grid):
    best = None
    for c in grid:
        W, b = fit_logreg(z["train"], z["y_train"], C, c)
        p = logits(z["val"], W, b).argmax(1).cpu().numpy()
        acc = (p == z["y_val"]).mean()
        if best is None or acc > best[0]: best = (acc, c, p)
    return best
a_s, c_s, p_s = pred(s, [10.0, 30.0, 100.0])
a_m, c_m, p_m = pred({"train": m["train"], "y_train": m["y_train"], "val": m["val"], "y_val": m["y_val"]}, [10.0, 30.0, 100.0])
y = s["y_val"]; texts = ds.texts["val"]
lost = [i for i in range(len(y)) if p_m[i] == y[i] and p_s[i] != y[i]]
won = [i for i in range(len(y)) if p_m[i] != y[i] and p_s[i] == y[i]]
neg = {"not", "no", "don't", "dont", "doesn't", "isn't", "can't", "cannot", "never", "without", "won't", "didn't"}
has_neg = lambda t: any(w.strip("?.,!").lower() in neg for w in t.split())
print(json.dumps({"lr_val_static": round(float(a_s), 4), "C_static": c_s, "lr_val_minilm": round(float(a_m), 4), "C_minilm": c_m,
                  "static_wrong_minilm_right": len(lost), "minilm_wrong_static_right": len(won),
                  "neg_share_lost": round(np.mean([has_neg(texts[i]) for i in lost]), 3) if lost else None,
                  "neg_share_all_val": round(np.mean([has_neg(t) for t in texts]), 3)}))
rng = np.random.default_rng(0)
for i in rng.choice(lost, size=min(25, len(lost)), replace=False):
    print(f"{texts[i]!r}  true={ds.intents[y[i]]}  static={ds.intents[p_s[i]]}")
