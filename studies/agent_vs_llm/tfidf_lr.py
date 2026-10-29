"""TF-IDF word 1-2 grams + logistic regression on the text that ft_baseline.render gives (dev only, descriptive).

usage: python -I tfidf_lr.py --dataset star --out DIR [--C 1 4 16] [--limit-train N] [--limit-dev N]
Note NOTE_tfidf_dev.md. Writes DIR/result.json with the dev accuracy for every C and the best one.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    a = ap.parse_args(argv)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    xtr, ytr, n_lab, h_tr = F.load(a.dataset, "train", a.limit_train)
    xdv, ydv, _, h_dv = F.load(a.dataset, "dev", a.limit_dev)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform(xtr)
    Xdv = vec.transform(xdv)
    rows = []
    for c in a.C:
        t1 = time.time()
        clf = LogisticRegression(C=c, solver="saga", max_iter=200, tol=1e-4)
        clf.fit(Xtr, ytr)
        acc = float((clf.predict(Xdv) == ydv).mean())
        rows.append({"C": c, "dev_acc": acc, "fit_s": round(time.time() - t1, 1), "n_iter": int(max(clf.n_iter_))})
        print(json.dumps(rows[-1]), flush=True)
    best = max(rows, key=lambda r: r["dev_acc"])
    os.makedirs(a.out, exist_ok=True)
    res = {"note": "NOTE_tfidf_dev.md", "dataset": a.dataset, "n_train": len(xtr), "n_dev": len(xdv),
           "n_features": Xtr.shape[1], "train_sha256": h_tr, "dev_sha256": h_dv, "rows": rows, "best": best,
           "seconds": round(time.time() - t0, 1), "argv": sys.argv,
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "result.json"), "w") as h:
        json.dump(res, h, indent=1)
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
