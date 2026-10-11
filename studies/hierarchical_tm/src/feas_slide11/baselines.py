"""Simple baselines on the same binary bag-of-words pair features."""
import argparse, json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import sparse

D = os.path.dirname(os.path.abspath(__file__))


def feats(d, split):
    P, Y = d[f"P_{split}"], d[f"Y_{split}"]
    Xp = d["Xp"]
    return Xp[P[:, 0]].astype(np.float32), Xp[P[:, 1]].astype(np.float32), Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=int, nargs="+", default=[1024, 3072])
    ap.add_argument("--n-train", type=int, nargs="+", default=[20000, 40000])
    a = ap.parse_args()
    res = []
    for F in a.F:
        d = dict(np.load(f"{D}/pairs_F{F}.npz"))
        S = {s: feats(d, s) for s in ("train", "held", "wval")}
        for ntr in a.n_train:
            x1, x2, y = (v[:ntr] for v in S["train"])
            # (a) Jaccard threshold tuned on train
            def jac(x1, x2):
                inter = (x1 * x2).sum(1); uni = np.maximum(((x1 + x2) > 0).sum(1), 1)
                return inter / uni
            j = jac(x1, x2)
            cands = np.unique(np.quantile(j, np.linspace(0.01, 0.99, 197)))
            accs = [((j >= c) == y).mean() for c in cands]
            thr = cands[int(np.argmax(accs))]
            r = dict(model="jaccard_threshold", F=F, n_train=ntr, thr=float(thr), train_acc=float(max(accs)))
            for s in ("held", "wval"):
                h1, h2, hy = S[s]
                r[f"{s}_acc"] = float(((jac(h1, h2) >= thr) == hy).mean())
            # raw overlap count threshold
            o = (x1 * x2).sum(1)
            cands = np.arange(0, 15)
            accs = [((o >= c) == y).mean() for c in cands]
            c = int(cands[int(np.argmax(accs))])
            r2 = dict(model="overlap_count_threshold", F=F, n_train=ntr, thr=c, train_acc=float(max(accs)))
            for s in ("held", "wval"):
                h1, h2, hy = S[s]
                r2[f"{s}_acc"] = float((((h1 * h2).sum(1) >= c) == hy).mean())
            res += [r, r2]
            # (b) logistic regression variants
            for name, fn in (("logreg_[x1,x2,x1&x2]", lambda u, v: np.hstack([u, v, u * v])),
                             ("logreg_[x1&x2]", lambda u, v: u * v)):
                t0 = time.time()
                best = None
                for C in (0.03, 0.1, 0.3, 1.0):
                    m = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
                    # pick C on the last 20% of train (inner split), refit not needed for a probe
                    k = int(0.8 * ntr)
                    m.fit(sparse.csr_matrix(fn(x1[:k], x2[:k])), y[:k])
                    va = (m.predict(sparse.csr_matrix(fn(x1[k:], x2[k:]))) == y[k:]).mean()
                    if best is None or va > best[0]:
                        best = (va, C)
                m = LogisticRegression(C=best[1], max_iter=2000, solver="liblinear").fit(sparse.csr_matrix(fn(x1, x2)), y)
                r = dict(model=name, F=F, n_train=ntr, C=best[1], inner_val=float(best[0]),
                         train_acc=float((m.predict(sparse.csr_matrix(fn(x1, x2))) == y).mean()), wall_s=round(time.time() - t0, 1))
                for s in ("held", "wval"):
                    h1, h2, hy = S[s]
                    r[f"{s}_acc"] = float((m.predict(sparse.csr_matrix(fn(h1, h2))) == hy).mean())
                res.append(r)
            for rr in res[-4:]:
                print(json.dumps(rr), flush=True)
    json.dump(res, open(f"{D}/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
