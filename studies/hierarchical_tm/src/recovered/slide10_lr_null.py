"""Null distribution of the one-sentence check on slide-10 pair data: repeat 'fit a logistic regression on side A of
20,000 fresh train pairs, test on 20,000 fresh dev pairs' R times. If one sentence says nothing about the label,
z = (acc - 0.5) / sqrt(0.25 / n) is N(0, 1): mean 0, sd 1, about 5% beyond 1.96. Usage: python slide10_lr_null.py DATA_DIR [R]"""
import json, os, sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slide10 as s10
d = np.load(os.path.join(sys.argv[1], "pairs_slide10.npz")); X, fam, split = d["Xp"], d["fam"], d["split"]
groups = lambda sp: [r for f in range(fam.max() + 1) if len(r := np.flatnonzero((fam == f) & (split == sp))) >= 2]
def one(rep):
    rng = np.random.default_rng([11, rep])
    Pf, Yf = s10.sample_pairs(groups(0), 20000, rng)
    Pe, Ye = s10.sample_pairs(groups(1), 20000, rng)
    acc = (LogisticRegression(max_iter=2000).fit(X[Pf[:, 0]], Yf).predict(X[Pe[:, 0]]) == Ye).mean()
    return (acc - 0.5) / np.sqrt(0.25 / len(Ye))
R = int(sys.argv[2]) if len(sys.argv) > 2 else 160
z = np.array(Parallel(n_jobs=32)(delayed(one)(i) for i in range(R)))
m = {"repeats": R, "z_mean": float(z.mean()), "z_mean_se": float(z.std(ddof=1) / np.sqrt(R)), "z_sd": float(z.std(ddof=1)),
     "share_abs_z_above_1_96": float((np.abs(z) > 1.96).mean())}
print("NULL", json.dumps(m))
if os.environ.get("TMLM_RUN_DIR"):
    json.dump({k: {"value": v, "label": "measured"} for k, v in m.items()}, open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1)
