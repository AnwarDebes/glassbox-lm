#!/usr/bin/env python
"""N8 reference (EXPLORATORY, declared in LOG.md before the run; carved val only; decides nothing): linear models on
N7's features for all five sets (bow and bow+existsT_potion), as the reference that N8's TM settings are read against.

    n8_linear.py

Data: N7's explore caches (emb_n7/{ds}__explore.npz, views bow and bow+existsT_potion; the 90/10 carved split of the published train set, numpy default_rng(2026)); the test split is never opened. Models, sklearn
defaults, fixed before the run, no tuning: LogisticRegression(C=1.0, max_iter=5000) and LinearSVC(C=1.0), both
random_state 0. One row per set, view and model in results_n8_linear.jsonl.
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
    for ds, view in [(d, v) for d in ("trec", "mr", "sst2", "r52", "r8") for v in ("bow", "bow+existsT_potion")]:
        cache = HERE / "emb_n7" / f"{ds}__explore.npz"
        z = np.load(cache)
        Xtr = np.concatenate([z[f"{p}_train"] for p in view.split("+")], 1).astype(np.float32)
        Xva = np.concatenate([z[f"{p}_val"] for p in view.split("+")], 1).astype(np.float32)
        ytr, yva = z["y_train"], z["y_val"]
        for name, model in (("logreg", LogisticRegression(C=1.0, max_iter=5000, random_state=0)),
                            ("linear_svm", LinearSVC(C=1.0, random_state=0))):
            t0 = time.time()
            model.fit(Xtr, ytr)
            acc = float((model.predict(Xva) == yva).mean()) * 100
            rec = {"study": "N8 reference (exploratory, carved val)", "dataset": ds, "view": view, "model": name,
                   "sklearn": sklearn.__version__, "C": 1.0, "n": {"train": int(len(ytr)), "val": int(len(yva))},
                   "bits": int(Xtr.shape[1]), "val_acc": round(acc, 3),
                   "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()[:16],
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                   "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            with open(HERE / "results_n8_linear.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
