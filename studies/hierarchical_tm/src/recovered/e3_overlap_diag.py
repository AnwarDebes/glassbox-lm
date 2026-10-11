"""Post-hoc diagnostic for E3 (pair version), no training: word-overlap (Jaccard) AUC on the slide-10 pair data, overall
and by pair type, to tell a data problem from a learning failure. The same numbers were first
computed read-only; reproduced here as a logged run before they are quoted.
Pair types use family kinds from slide10_meta.json: 'variant' = slide10, negated, nearmiss or broken (families 0-12),
'other' = the other-topic families (13-20). AUC = htm_driver.auc (average ranks for ties)."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import htm_driver as hd
out_dir = os.environ.get("TMLM_RUN_DIR", ".")
shutil.copy(os.path.abspath(__file__), out_dir)
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "slide10")
d, meta = np.load(f"{D}/pairs_slide10.npz"), json.load(open(f"{D}/slide10_meta.json"))
X, fam = d["Xp"].astype(bool), d["fam"]
kinds = np.array([f["kind"] for f in meta["families"]])
variant = np.isin(kinds, ["slide10", "negated", "nearmiss", "broken"])
res = {}
for sp, P, Y in (("train", d["P_train"], d["Y_train"]), ("held", d["P_held"], d["Y_held"])):
    a, b, y = X[P[:, 0]], X[P[:, 1]], Y.astype(int)
    jac = (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
    fa, fb = fam[P[:, 0]], fam[P[:, 1]]
    res[f"{sp}_label_equals_same_family_share"] = float(((fa == fb).astype(int) == y).mean())
    sel = {"all": np.ones(len(y), bool), "both_variant": variant[fa] & variant[fb], "both_other": ~variant[fa] & ~variant[fb],
           "involves_slide10": (fa == 0) | (fb == 0), "mixed_variant_other": variant[fa] != variant[fb]}
    for k, m in sel.items():
        res[f"{sp}_jaccard_auc_{k}"] = hd.auc(jac[m], y[m]) if 0 < y[m].sum() < m.sum() else None
        res[f"{sp}_n_{k}"] = int(m.sum())
        res[f"{sp}_pos_{k}"] = int(y[m].sum())
for k, v in res.items():
    print(f"{k}: {v}")
print("DIAG", json.dumps(res))
json.dump({k: {"value": v, "label": "measured"} for k, v in res.items()}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
