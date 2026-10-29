"""REG_agent_vs_llm, arm TFIDF (no rule): TF-IDF word 1-2 grams + logistic regression at the dev-best C of
NOTE_tfidf_dev.md, trained on train; dev and registered test predictions.

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python -I tfidf_reg.py --dataset abcd \
           --test-registration REG_agent_vs_llm.md --out runs/reg/tfidf_abcd        (test: needs the freeze record)
       ... --dry-run-dev --out runs/dry/tfidf_abcd   (the same code with dev in the place of test; test never opened)
The setting is tfidf_lr.py's (render of ft_baseline.py, TfidfVectorizer(ngram_range=(1, 2), min_df=2,
sublinear_tf=True, lowercase=True), LogisticRegression(solver="saga", max_iter=200, tol=1e-4)) at the C that was best
on dev (STAR 4, ABCD 16, MultiWOZ 1), with random_state=0 (tfidf_lr.py left it unset, so its dev number can differ a
little: the dev accuracy is written beside the earlier one, flagged if more than 0.5 points apart).
Writes OUT/dev_preds.npz and OUT/test_preds.npz (top_ids: the 5 highest classes, gold) and OUT/result.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

C_DEV = {"star": 4.0, "abcd": 16.0, "multiwoz": 1.0}
DEV_EARLIER = {"star": 0.6528662420382165, "abcd": 0.8414766558089034, "multiwoz": 0.3505911129229515}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-registration")
    g.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None, help="tests only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.test_registration:
        fr = os.path.splitext(a.test_registration)[0] + ".frozen.json"
        if not os.path.exists(fr):
            raise SystemExit(f"refused: no freeze record {fr}; the test split is read only under a frozen registration")
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    xtr, ytr, n_lab, h_tr = F.load(a.dataset, "train", a.limit_train)
    xdv, ydv, _, h_dv = F.load(a.dataset, "dev")
    if a.dry_run_dev:
        xte, yte, h_te = xdv, ydv, h_dv
    else:
        xte, yte, _, h_te = F.load(a.dataset, "test", allow_test=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform(xtr)
    clf = LogisticRegression(C=C_DEV[a.dataset], solver="saga", max_iter=200, tol=1e-4, random_state=0)
    clf.fit(Xtr, ytr)
    os.makedirs(a.out, exist_ok=True)
    accs = {}
    for name, x, y in (("dev", xdv, ydv), ("test", xte, yte)):
        p = clf.predict_proba(vec.transform(x))
        top = clf.classes_[np.argsort(-p, axis=1)[:, :5]]
        np.savez(os.path.join(a.out, f"{name}_preds.npz"), top_ids=top.astype(np.int64),
                 gold=np.asarray(y, dtype=np.int64))
        accs[name] = float((top[:, 0] == np.asarray(y)).mean())
    res = {"registration": a.test_registration, "dry_run_dev": a.dry_run_dev, "dataset": a.dataset,
           "C": C_DEV[a.dataset], "n_train": len(xtr), "n_dev": len(xdv), "n_test": len(xte),
           "n_features": Xtr.shape[1], "n_iter": int(max(clf.n_iter_)), "dev_acc": accs["dev"],
           "dev_acc_earlier": DEV_EARLIER[a.dataset], "dev_within_0.5_points": abs(accs["dev"] - DEV_EARLIER[a.dataset]) <= 0.005,
           "test_acc": accs["test"], "train_sha256": h_tr, "dev_sha256": h_dv, "test_sha256": h_te,
           "seconds": round(time.time() - t0, 1), "argv": sys.argv,
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "result.json"), "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "C", "dev_acc", "dev_acc_earlier", "dev_within_0.5_points",
                                          "test_acc", "dry_run_dev", "seconds")}))


if __name__ == "__main__":
    main()
