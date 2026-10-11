"""Check 1b for data/ns_all (the all-pairs synonym data), reusing the functions in data_checks.py.

data_checks.py runs the synonym-data checks (overlap 0, length-only classifiers) only for the folder named "ns",
and also asserts equal shared-word histograms, which only the k-matched designs have. This wrapper runs the
generic checks (z-test, one-paragraph LR, partner-shuffle, baselines) plus overlap and length-only, and only
reports the shared-word histogram before removal.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_checks as dc  # noqa: E402
from pairdata import DATA, ids_to_csr, load_pairs, load_vocab  # noqa: E402

V = "ns_all"


def main():
    t0, run_dir = time.time(), os.environ.get("TMLM_RUN_DIR")
    P = dc.train_paragraphs()
    df = np.asarray(P.sum(0)).ravel()
    idf2 = np.log(P.shape[0] / np.maximum(df, 1)) ** 2
    W = dc.ppmi_vectors(P)
    out = dc.check_version(V, idf2, W, load_vocab(dc.F), np.random.default_rng([0, 9]))
    ov = {}
    for split in ("train", "dev", "test"):
        for FF in (dc.F, 3072):
            p = load_pairs(V, split, FF)
            ov[f"{split}_F{FF}_pairs_with_overlap"] = int((p["A"].multiply(p["B"]).sum(1) > 0).sum())
    out["overlap"] = ov
    tr, dev = load_pairs(V, "train", dc.F), load_pairs(V, "dev", dc.F)
    Ltr, Ldev = dc.length_feats(tr), dc.length_feats(dev)
    sc = dc.StandardScaler().fit(Ltr)
    m = dc.LogisticRegression().fit(sc.transform(Ltr), tr["y"])
    out["length_lr"] = dc.report(m.predict(sc.transform(Ldev)), m.decision_function(sc.transform(Ldev)), dev["y"])
    g = dc.HistGradientBoostingClassifier(random_state=0).fit(Ltr, tr["y"])
    out["length_gbt"] = dc.report(g.predict(Ldev), g.predict_proba(Ldev)[:, 1], dev["y"])
    z = np.load(os.path.join(DATA, "paragraphs_F3072.npz"))
    M, st = ids_to_csr(z["ids"], z["off"], dc.F), np.concatenate([[0], np.cumsum(z["npar"])])
    k = np.asarray(M[st[dev["art_a"]] + dev["par_a"]].multiply(M[st[dev["art_b"]] + dev["par_b"]]).sum(1)).ravel()
    out["k_hist_before_removal_dev"] = {c: np.bincount(np.minimum(k[dev["y"] == y], 10), minlength=11).tolist()
                                       for c, y in (("pos", 1), ("neg", 0))}
    gate = [out["ztest"]["pass"], out["one_paragraph_lr"]["ci_has_0.50"], out["partner_shuffle_lr"]["ci_has_0.50"],
            sum(ov.values()) == 0, out["length_lr"]["ci_has_0.50"], out["length_gbt"]["ci_has_0.50"]]
    out["check1b_pass"] = bool(all(gate))
    md = dc.tables({V: out})
    print(json.dumps(out))
    print(md)
    print(f"wall {time.time() - t0:.0f}s")
    if run_dir:
        import shutil
        for f in ("check_ns_all.py", "data_checks.py", "pairdata.py"):
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), run_dir)
        json.dump({"results": {V: out}}, open(os.path.join(run_dir, "checks.json"), "w"), indent=1)
        open(os.path.join(run_dir, "checks.md"), "w").write(md + "\n")


if __name__ == "__main__":
    main()
