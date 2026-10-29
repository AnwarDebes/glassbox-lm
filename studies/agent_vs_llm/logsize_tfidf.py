"""PLAN_logsize_dev.md, arm TFIDF(N), no rule: tfidf_reg.py's setting trained on the rows of a pool of N training
dialogues; top-1 for every dev row (dev only, CPU).

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I logsize_tfidf.py --dataset abcd \
           [--sizes 64 256 1024 4096 whole] [--out-dir runs/logsize/tfidf]
Writes OUT_DIR/<ds>_<size>.json: the pool's size, rows and labels, the top-1 label id of every dev row, the accuracy
on all dev rows and on L35k50's 500 rows. At "whole" it also compares the top-1 of every dev row with
runs/dry/tfidf_<ds>/dev_preds.npz (tfidf_reg.py's dry run): equal_to_dry_run must be true. Never overwrites.
"""
import argparse
import gzip
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import ft_baseline as F  # noqa: E402
import logsize_subsets as S  # noqa: E402

C_DEV = {"star": 4.0, "abcd": 16.0, "multiwoz": 1.0}  # tfidf_reg.py's C_DEV


def train_rows(dataset, keep):
    with open(os.path.join(F.DATA, dataset, "examples_train.json.gz"), "rb") as h:
        rows = json.loads(gzip.decompress(h.read()))["rows"]
    return rows if keep is None else [r for r in rows if r["convo_id"] in keep]


def rows500(dataset):
    with open(os.path.join(H, "runs/llm_q3_L35k50", dataset, "preds.jsonl")) as h:
        return [json.loads(x)["i"] for x in h if x.strip()]


def fit_predict(xtr, ytr, xdv, C):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform(xtr)
    clf = LogisticRegression(C=C, solver="saga", max_iter=200, tol=1e-4, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(vec.transform(xdv))
    top = clf.classes_[np.argsort(-p, axis=1)[:, :5]]  # tfidf_reg.py's ranking
    return [int(v) for v in top[:, 0]], int(Xtr.shape[1]), int(max(clf.n_iter_))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(C_DEV), required=True)
    ap.add_argument("--sizes", nargs="+", default=["64", "256", "1024", "4096", "whole"])
    ap.add_argument("--subsets", default=os.path.join(H, "runs/logsize/subsets.json"))
    ap.add_argument("--out-dir", default=os.path.join(H, "runs/logsize/tfidf"))
    ap.add_argument("--rows", default=None, help="tests only: a JSON list of dev row ids in place of L35k50's rows")
    a = ap.parse_args(argv)
    with open(a.subsets) as h:
        subsets = json.load(h)
    xdv, ydv, _, _ = F.load(a.dataset, "dev")
    ids = json.load(open(a.rows)) if a.rows else rows500(a.dataset)
    os.makedirs(a.out_dir, exist_ok=True)
    for size in a.sizes:
        t0 = time.time()
        rows = train_rows(a.dataset, S.keep_ids(subsets, a.dataset, size))
        top1, n_feat, n_iter = fit_predict([F.render(r) for r in rows], [int(r["y"]) for r in rows], xdv,
                                           C_DEV[a.dataset])
        labels = sorted({int(r["y"]) for r in rows})
        res = {"plan": "PLAN_logsize_dev.md", "dataset": a.dataset, "size": size, "C": C_DEV[a.dataset],
               "n_dialogues": len({r["convo_id"] for r in rows}), "n_train_rows": len(rows), "n_features": n_feat,
               "n_iter": n_iter, "labels_in_pool": labels,
               "acc_dev_all": sum(p == y for p, y in zip(top1, ydv)) / len(ydv),
               "ids_500": ids, "acc_500": sum(top1[i] == ydv[i] for i in ids) / len(ids),
               "unseen_gold_500": sum(ydv[i] not in set(labels) for i in ids) / len(ids),
               "top1_dev": top1, "seconds": round(time.time() - t0, 1)}
        if size in ("whole", "full"):
            import numpy as np
            dry = np.load(os.path.join(H, "runs/dry", f"tfidf_{a.dataset}", "dev_preds.npz"))["top_ids"][:, 0]
            res["n_diff_to_dry_run"] = int(sum(int(d) != p for d, p in zip(dry, top1)))
            res["equal_to_dry_run"] = res["n_diff_to_dry_run"] == 0
        res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(a.out_dir, f"{a.dataset}_{size}.json"), "x") as h:
            json.dump(res, h)
        print(json.dumps({k: res[k] for k in res if k not in ("labels_in_pool", "ids_500", "top1_dev")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
