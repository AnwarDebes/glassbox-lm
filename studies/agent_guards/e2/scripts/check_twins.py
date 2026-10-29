"""Exactness of exported D5 twins: refit LR and tree with the recorded configuration on the same rows (deterministic:
same data, random_state 0) and compare sklearn's P(ALLOW) with the pure-Python decision (e2guard.models) read from the
exported JSON. Decision-layer venv: .venv-dl/bin/python -I -B scripts/check_twins.py --models DIR --rows R [R ...] --qbits Q"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(E2 / "scripts"))
import train_loso as TL

from e2guard.bits import QueryBits
from e2guard.models import LRTwin, TreeTwin

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--qbits", required=True)
a = ap.parse_args()
rows = TL.load_rows(a.rows)
out = {}
for held in TL.SUITES:
    qb = QueryBits(Path(a.qbits) / f"fold_{held}.json")
    by_suite = {x: [r for r in rows if r["suite"] == x] for x in TL.SUITES}  # train_loso's exact row order
    fit = [r for x in TL.SUITES if x != held for r in by_suite[x] if not r["_veto"]]
    X, y, names = TL.matrix(fit, qb)
    lrj, trj = LRTwin.load(Path(a.models) / f"d5_lr_{held}.json"), TreeTwin.load(Path(a.models) / f"d5_tree_{held}.json")
    lr = LogisticRegression(C=lrj.d["config"]["C"], max_iter=3000, class_weight="balanced").fit(X, y)
    t = DecisionTreeClassifier(max_depth=trj.d["config"]["depth"], class_weight="balanced", random_state=0).fit(X, y)
    pick = np.random.default_rng(0).choice(len(X), size=min(500, len(X)), replace=False)
    pl = lr.predict_proba(X[pick])[:, list(lr.classes_).index(1)]
    pt = t.predict_proba(X[pick])[:, list(t.classes_).index(1)]
    dl_ = [abs(lrj.p_allow(X[i].tolist()) - pl[k]) for k, i in enumerate(pick)]
    out[held] = {"rows": len(pick),
                 "lr_equal_1e-9": int(sum(d < 1e-9 for d in dl_)), "lr_max_abs_diff": float(max(dl_)),
                 "lr_same_decision": int(sum((lrj.p_allow(X[i].tolist()) >= lrj.threshold) == (pl[k] >= lrj.threshold)
                                             for k, i in enumerate(pick))),
                 "tree_equal": int(sum(abs(trj.p_allow(X[i].tolist()) - pt[k]) < 1e-9 for k, i in enumerate(pick)))}
    print(held, out[held], flush=True)
print(json.dumps(out))
