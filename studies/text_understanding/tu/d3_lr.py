#!/usr/bin/env python
"""Step 3d (VAL only; tu/PLAN_3d.md "Readers"): the LR probe of one arm and set, C picked on val.

    python -I -B d3_lr.py --name J --dataset clinc150 --emb-dir tu/d3/feats/J

tu/m1_lr.py (d2a41815a2ccc568) with these changes and nothing else:
(1) no test split: the cache (tu/d3_feats.py) has none and nothing is predicted beyond val (the same fits in the same
    order as exists_tm.best_logreg: the first C with the strictly highest val accuracy; every C's val accuracy kept as
    "acc_by_C");
(2) rows go to <out-dir>/results_d3_lr.jsonl (default tu/d3) and record "study": "3d" and the sha256 of this file,
    exists_tm.py and the cache;
(3) <out-dir> is created when it is missing (the parent created it only through its test-predictions folder);
(4) each row also records "n_val" and "lr_correct", the count of correct val predictions at the picked C, which must
    equal its val accuracy times n_val (else the run stops); those val predictions go to
    <out-dir>/preds_val/lr_<name>_<set>.npy, written exclusively (one file per arm and set).
--device exists for the CPU tests only; --load takes a stand-in dataset for the tests (m1_standin.py).
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
    ap.add_argument("--out-dir", type=Path, default=HERE / "d3")
    ap.add_argument("--device", default="cuda", help="cpu only for the tests")
    ap.add_argument("--load", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    args = ap.parse_args()
    t0 = time.time()
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ppath = args.out_dir / "preds_val" / f"lr_{args.name}_{args.dataset}.npy"
    assert not ppath.exists(), f"{ppath} exists: one fit per arm and set"
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing"
    assert json.loads(cache.with_suffix(".meta.json").read_text()).get("table"), "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, dev)
    assert "test__exists" not in F and "y_test" not in F, "a 3d cache has no test rows"
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ds = load_ds(args.dataset, args.load)
    ytr, yva = F["y_train"], F["y_val"]
    assert (yva == ds.labels["val"]).all()
    C = len(ds.intents)
    n_orig = int(F["n_orig"]) if "n_orig" in F else len(F["y_train"])  # the originals come first in "train"
    view = X.View("existsT", meta).fit({k: (v[:n_orig] if k.startswith("train__") else v) for k, v in F.items()})
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
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
    pv = X.logits(Bva.astype(np.float32), W, b, device=dev).argmax(1).cpu().numpy()
    n_val, lr_correct = len(yva), int((pv == yva).sum())
    assert lr_correct == round(best[0] * n_val), f"the picked C gives {lr_correct} of {n_val}, val accuracy {best[0]}"
    ppath.parent.mkdir(parents=True, exist_ok=True)
    with open(ppath, "xb") as f:  # exclusive: one file per arm and set
        np.save(f, pv.astype(np.int16))
    rec = {"study": "3d", "arm": meta.get("arm", "orig"), "n_orig": n_orig, "n_train": len(ytr),
           "probe": "logistic regression (C 1/10/100, best on val)", "name": args.name, "dataset": args.dataset,
           "table": meta.get("table"), "table_sha256": meta.get("table_sha256"), "kmeans_run_id": meta.get("kmeans_run_id"),
           "view": "existsT", "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "lr_val": acc, "C": c,
           "acc_by_C": acc_by_C, "n_val": n_val, "lr_correct": lr_correct, "preds_val": str(ppath),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
           "sha256": {"d3_lr.py": sha16(Path(__file__).resolve()), "exists_tm.py": sha16(EMB / "exists_tm.py"), "cache": sha16(cache)}}
    with open(args.out_dir / "results_d3_lr.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
