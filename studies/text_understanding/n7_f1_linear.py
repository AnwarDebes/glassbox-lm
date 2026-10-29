#!/usr/bin/env python
"""N7-F1 diagnostic (POST HOC, declared in LOG.md before the run; carved val only; decides nothing): linear models on
the same features as F1, to see whether the gap to Yadav et al.'s bag-of-words TM (88.05 / 75.14) is in the features
and split or in the machines.

    n7_f1_linear.py

Data: N7's explore caches (emb_n7/{ds}__explore.npz, view bow: our bag of words, TREC 6,000 words, MR 5,000; the 90/10
carved split of the published train set, numpy default_rng(2026)); the test split is never opened. Models, sklearn
defaults, fixed before the run, no tuning: LogisticRegression(C=1.0, max_iter=5000) and LinearSVC(C=1.0), both
random_state 0. One row per set and model in results_n7_f1_linear.jsonl.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    for ds in ("trec", "mr"):
        cache = HERE / "emb_n7" / f"{ds}__explore.npz"
        z = np.load(cache)
        Xtr, Xva = z["bow_train"].astype(np.float32), z["bow_val"].astype(np.float32)
        ytr, yva = z["y_train"], z["y_val"]
        for name, model in (("logreg", LogisticRegression(C=1.0, max_iter=5000, random_state=0)),
                            ("linear_svm", LinearSVC(C=1.0, random_state=0))):
            t0 = time.time()
            model.fit(Xtr, ytr)
            acc = float((model.predict(Xva) == yva).mean()) * 100
            rec = {"followup": "N7-F1 diagnostic (post hoc, carved val)", "dataset": ds, "view": "bow", "model": name,
                   "sklearn": sklearn.__version__, "C": 1.0, "n": {"train": int(len(ytr)), "val": int(len(yva))},
                   "bits": int(Xtr.shape[1]), "val_acc": round(acc, 3),
                   "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()[:16],
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                   "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            with open(HERE / "results_n7_f1_linear.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
