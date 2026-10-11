"""Post-hoc: is a trained clause bank useful even though the package vote is near chance?
Clause truth is recomputed in numpy from saved include matrices (verified 100% agreement with the
package under binary inference). Everything selected/fitted on TRAIN pairs, evaluated on HELD pairs.
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tm import build, clause_outputs_hier, leaf_outputs

D = os.path.dirname(os.path.abspath(__file__))


def auc(s, y):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="hier")
    ap.add_argument("--F", type=int, default=1024)
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n-fit", type=int, default=10000)
    a = ap.parse_args()
    d = dict(np.load(f"{D}/pairs_F{a.F}.npz"))
    lits = np.load(f"{D}/lits_{a.tag}.npz")["lits"]
    C = lits.shape[0]
    Xtr, Ytr = build(d, "train", a.n_fit)
    Xho, Yho = build(d, "held", 4000)

    def co(X):
        if a.model == "hier":
            return clause_outputs_hier(lits, X, a.F, a.A, a.K)
        return leaf_outputs(lits[:, 0, :], X[:, :a.F]) & leaf_outputs(lits[:, 1, :], X[:, a.F:])
    Ctr, Cho = co(Xtr).astype(np.float32), co(Xho).astype(np.float32)
    sign = np.where(np.arange(C) % 2 == 0, 1, -1)
    res = dict(tag=a.tag, clauses=C)
    v = Cho @ sign
    res["package_style_vote_auc"] = auc(v, Yho)
    res["package_style_vote_acc_thr0"] = float(((v >= 0) == Yho).mean())
    # select clauses by train precision-lift, majority vote of selected positive-polarity clauses
    f1 = Ctr[Ytr == 1].mean(0); f0 = Ctr[Ytr == 0].mean(0)
    lift = f1 - f0
    for k in (10, 50, 200):
        top = np.argsort(-lift)[:k]
        s_tr = Ctr[:, top].sum(1); s_ho = Cho[:, top].sum(1)
        thr = max(range(0, k + 1), key=lambda t: ((s_tr >= t) == Ytr).mean()) if k <= 200 else 1
        res[f"top{k}_by_train_lift_any_count_auc"] = auc(s_ho, Yho)
        res[f"top{k}_by_train_lift_acc"] = float(((s_ho >= thr) == Yho).mean())
    m = LogisticRegression(C=0.1, max_iter=3000, solver="liblinear").fit(Ctr, Ytr)
    res["logreg_on_clause_bank_held_acc"] = float((m.predict(Cho) == Yho).mean())
    res["logreg_on_clause_bank_held_auc"] = auc(m.decision_function(Cho), Yho)
    res["logreg_on_clause_bank_train_acc"] = float((m.predict(Ctr) == Ytr).mean())
    print(json.dumps(res))
    json.dump(res, open(f"{D}/posthoc_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
