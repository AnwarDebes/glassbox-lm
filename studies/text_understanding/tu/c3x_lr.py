#!/usr/bin/env python
"""Step 3c follow-ups (a), (b) (VAL only): tu/c3_lr.py (db03297a64b0aa6a) with one change and nothing else: rows go to
tu/c3x/results_c3x_lr.jsonl and record "study": "3c-x"; and (the 11:39Z change) the val accuracy at each C
of the grid is recorded as "acc_by_C". The best is chosen exactly as exists_tm.best_logreg chooses it (the same fits, in
the same order; the first C with the strictly highest accuracy), so "lr_val" and "C" equal c3_lr.py's. Docstring of
c3_lr.py:

Text understanding, step 3c (VAL only; tu/PLAN_3c.md "Features and readers"): tu/a3_lr.py (4fd99da0cd209e3c) with these
changes and nothing else: (1) the existsT thermometer is fit on the first n_orig training rows only (as
c3_exists_tm.py); (2) rows go to tu/c3/results_c3_lr.jsonl and record "study": "3c", the arm and n_orig / n_train.
With no extra rows it equals a3_lr.py. Docstring of a3_lr.py:

Text understanding, step 3a (VAL only; tu/PLAN_3a.md "Probes"): w3/w3_lr.py (1e11c93439fa69e4) with only these
changes: rows go to tu/a3/results_a3_lr.jsonl and record "study": "3a" (this copy lives in tu/, so HERE is tu and EMB
is still glassbox-lm/studies/text_understanding). Docstring of w3_lr.py:

W3 step 1 (VAL only): w2/w2_lr.py (f424ba4c054f4a7c) with one change and nothing else: rows go to
w3/results_w3_lr.jsonl (this copy lives in w3/, so HERE is w3 and EMB is still glassbox-lm/studies/text_understanding). Docstring of w2_lr.py:

W2 H-B part 2 (VAL only; rules in LOG.md before any run): the logistic-regression probe on the same existsT bits
the TM reads.

    python -I w2_lr.py --name J --dataset clinc150 --emb-dir w2/feats/J_d500_e0.5

The bits: the feature cache of w1/wordtab_feats.py (or test 2's potion cache, --name potion), view 'existsT' of the
TM harness (exists_tm.View: Thermo(8) fitted on the training sentences' per-concept max similarities). The model:
exists_tm.best_logreg = dl.linear's multinomial logistic regression on the GPU (full-batch L-BFGS), C in 1 / 10 / 100,
the best on val (the probe of posthoc_lr.py). Rows to w2/results_w2_lr.jsonl.
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

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/w2
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
sys.path.insert(0, str(EMB / "w1"))
import exists_tm as X  # noqa: E402
import mo_tm as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    torch.cuda.set_per_process_memory_fraction(0.1)
    cache = args.emb_dir / f"{args.dataset}__p32tok_K512.npz"
    assert cache.exists(), f"{cache} missing"
    meta = json.loads(cache.with_suffix(".meta.json").read_text())
    assert meta.get("table") or args.name == "potion", "the cache meta names no table"
    F, meta = X.features(args.dataset, 512, args.emb_dir, "cuda")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    ytr, yva = F["y_train"], F["y_val"]
    C = int(max(ytr.max(), yva.max())) + 1
    n_orig = int(F["n_orig"]) if "n_orig" in F else len(F["y_train"])  # 3c: the originals come first in "train"
    view = X.View("existsT", meta).fit({k: (v[:n_orig] if k.startswith("train__") else v) for k, v in F.items()})
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    best, acc_by_C = (-1.0, None), {}
    for Cc in [1.0, 10.0, 100.0]:  # exists_tm.best_logreg's loop, with each C's accuracy kept
        W, b = X.fit_logreg(Btr.astype(np.float32), ytr, C, Cc)
        a_c = float((X.logits(Bva.astype(np.float32), W, b).argmax(1).cpu().numpy() == yva).mean())
        acc_by_C[str(Cc)] = round(a_c, 4)
        if a_c > best[0]:
            best = (a_c, Cc)
    acc, c = round(best[0], 4), best[1]
    rec = {"study": "3c-x", "arm": meta.get("arm", "orig"), "n_orig": n_orig, "n_train": int(len(ytr)), "probe": "logistic regression (C 1/10/100, best on val)",
           "name": args.name, "dataset": args.dataset, "table": meta.get("table", "potion-base-32M"),
           "table_sha256": meta.get("table_sha256"), "cache_sha256": M.sha16(cache), "view": "existsT",
           "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "lr_val": acc, "C": c, "acc_by_C": acc_by_C,
           "oov_token_share": meta.get("oov_token_share"), "utc": M.utc(), "seconds": round(time.time() - t0, 1),
           "sha256": {"w2_lr.py": M.sha16(Path(__file__).resolve()), "exists_tm.py": M.sha16(EMB / "exists_tm.py")}}
    with open(HERE / "c3x" / "results_c3x_lr.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
