#!/usr/bin/env python
"""M1, the registered test (tu/PLAN_M1.md "Arms"): the LR probe of one arm and set, C picked on val, with ONE read of
the test split at that C.

    python -I -B m1_lr.py --name a3B --dataset clinc150 --emb-dir tu/m1/feats/a3B

tu/c3x_lr.py (e18dbfd965df158d) with these changes and nothing else:
(1) after C is picked on val (the same fits in the same order as exists_tm.best_logreg: the first C with the strictly
    highest val accuracy; every C's val accuracy kept as "acc_by_C"), the fit at that C predicts the test split once;
    its accuracy is recorded and the predictions go to <out-dir>/preds/lr_<name>_<set>.npy;
(2) the cache must contain test rows and name its table (tu/m1_feats.py);
(3) rows go to <out-dir>/results_m1_lr.jsonl (default tu/m1) and record "study": "M1"; the hash keys name the files
    themselves ("m1_lr.py", "exists_tm.py", "cache"; the record fix for c3_lr.py's "w2_lr.py" key);
(4) --device exists for the CPU tests only; --load (the tests and the chain's code check only) takes a stand-in
    dataset whose "test" is a copy of val (m1_standin.py).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X
from m1_feats import load_ds
from m1_tm import sha16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "m1")
    ap.add_argument("--device", default="cuda", help="cpu only for the tests")
    ap.add_argument("--load", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing"
    assert json.loads(cache.with_suffix(".meta.json").read_text()).get("table"), "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, dev)
    assert "test__exists" in F and "y_test" in F, "the cache has no test rows"
    for sp in X.SPLITS + ("test",):
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ds = load_ds(args.dataset, args.load)
    ytr, yva, yte = F["y_train"], F["y_val"], F["y_test"]
    assert (yva == ds.labels["val"]).all() and (yte == ds.labels["test"]).all()
    C = len(ds.intents)
    n_orig = int(F["n_orig"]) if "n_orig" in F else len(F["y_train"])  # the originals come first in "train"
    pdir = args.out_dir / "preds"
    ppath = pdir / f"lr_{args.name}_{args.dataset}.npy"
    assert not ppath.exists(), f"{ppath} exists: one test read per arm"
    view = X.View("existsT", meta).fit({k: (v[:n_orig] if k.startswith("train__") else v) for k, v in F.items()})
    Btr, Bva, Bte = view.transform(F, "train"), view.transform(F, "val"), view.transform(F, "test")
    best, acc_by_C, fits = (-1.0, None), {}, {}
    for Cc in [1.0, 10.0, 100.0]:  # exists_tm.best_logreg's loop, with each C's accuracy kept
        W, b = X.fit_logreg(Btr.astype(np.float32), ytr, C, Cc, device=dev)
        a_c = float((X.logits(Bva.astype(np.float32), W, b, device=dev).argmax(1).cpu().numpy() == yva).mean())
        acc_by_C[str(Cc)] = round(a_c, 4)
        fits[Cc] = (W, b)
        if a_c > best[0]:
            best = (a_c, Cc)
    acc, c = round(best[0], 4), best[1]
    W, b = fits[c]
    pt = X.logits(Bte.astype(np.float32), W, b, device=dev).argmax(1).cpu().numpy()  # the one test read
    pdir.mkdir(parents=True, exist_ok=True)
    with open(ppath, "xb") as f:  # exclusive: one test read per arm
        np.save(f, pt.astype(np.int16))
    rec = {"study": "M1", "arm": meta.get("arm", "orig"), "n_orig": n_orig, "n_train": len(ytr),
           "probe": "logistic regression (C 1/10/100, best on val)", "name": args.name, "dataset": args.dataset,
           "table": meta.get("table"), "table_sha256": meta.get("table_sha256"), "kmeans_run_id": meta.get("kmeans_run_id"),
           "view": "existsT", "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "lr_val": acc, "C": c,
           "acc_by_C": acc_by_C, "lr_test": round(float((pt == yte).mean()), 4), "n_test": len(pt), "preds": str(ppath),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
           "sha256": {"m1_lr.py": sha16(Path(__file__).resolve()), "exists_tm.py": sha16(EMB / "exists_tm.py"), "cache": sha16(cache)}}
    with open(args.out_dir / "results_m1_lr.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
