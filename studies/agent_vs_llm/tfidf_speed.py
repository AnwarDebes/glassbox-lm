"""CPU time of one decision of the TFIDF arm (one thread, batch 1); cost only, no rule, dev rows.

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I tfidf_speed.py \
           --dataset abcd --out runs/cost/tfidf_abcd_cpu.json [--n 1000]
Fits the TFIDF arm exactly as tfidf_reg.py does (render of ft_baseline.py, word 1-2 grams, saga, the dev-best C,
random_state 0), checks its dev accuracy against tfidf_reg.py's dry run (runs/dry/tfidf_<ds>/result.json), then times
vectorizer.transform plus predict for single dev rows after 20 warm-up rows. Rough timing on a shared machine.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402
from tfidf_reg import C_DEV  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    xtr, ytr, _, _ = F.load(a.dataset, "train")
    xdv, ydv, _, _ = F.load(a.dataset, "dev")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    clf = LogisticRegression(C=C_DEV[a.dataset], solver="saga", max_iter=200, tol=1e-4, random_state=0)
    clf.fit(vec.fit_transform(xtr), ytr)
    acc = float((clf.predict(vec.transform(xdv)) == np.asarray(ydv)).mean())
    want = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"runs/dry/tfidf_{a.dataset}/result.json")))["dev_acc"]
    assert abs(acc - want) < 1e-12, (acc, want)
    idx = random.Random(0).sample(range(len(xdv)), min(a.n, len(xdv)))
    times = []
    for warm in (True, False):
        for i in (idx[:20] if warm else idx):
            t0 = time.perf_counter()
            clf.predict(vec.transform([xdv[i]]))
            t1 = time.perf_counter()
            if not warm:
                times.append((t1 - t0) * 1000)
    res = {"dataset": a.dataset, "C": C_DEV[a.dataset], "dev_acc": acc, "dev_acc_equals_dry_run": True, "n": len(times),
           "n_features": len(vec.vocabulary_), "total_median_ms": float(np.median(times)),
           "total_p90_ms": float(np.percentile(times, 90)), "label": "rough: shared machine, one CPU thread, batch 1",
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
