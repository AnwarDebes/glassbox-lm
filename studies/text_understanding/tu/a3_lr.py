#!/usr/bin/env python
"""Text understanding, step 3a (VAL only; tu/PLAN_3a.md "Probes"): w3/w3_lr.py (1e11c93439fa69e4) with only these
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
    view = X.View("existsT", meta).fit(F)
    Btr, Bva = view.transform(F, "train"), view.transform(F, "val")
    acc, c = X.best_logreg(Btr, ytr, Bva, yva, C, [1.0, 10.0, 100.0])
    rec = {"study": "3a", "probe": "logistic regression (C 1/10/100, best on val)",
           "name": args.name, "dataset": args.dataset, "table": meta.get("table", "potion-base-32M"),
           "table_sha256": meta.get("table_sha256"), "cache_sha256": M.sha16(cache), "view": "existsT",
           "n_bits": int(Btr.shape[1]), "on_share": round(float(Btr.mean()), 3), "lr_val": acc, "C": c,
           "oov_token_share": meta.get("oov_token_share"), "utc": M.utc(), "seconds": round(time.time() - t0, 1),
           "sha256": {"w2_lr.py": M.sha16(Path(__file__).resolve()), "exists_tm.py": M.sha16(EMB / "exists_tm.py")}}
    with open(HERE / "a3" / "results_a3_lr.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
