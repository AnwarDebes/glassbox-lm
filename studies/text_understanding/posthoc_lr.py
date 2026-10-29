#!/usr/bin/env python
"""POST HOC (chosen after the seed-1 rows of test 2, VAL only): which binarization of the existential
similarities keeps their information? Seed-1 rows: logistic regression on the RAW [mean, exists] features
gains +1.8 (CLINC150) and +2.4 (HWU64) points over the raw mean, but LR on the bits only +0.2 to +0.4, so the
bits (member-percentile thresholds) seem to lose it. Here: LR (C grid 1/10/100, best on val) on bits from
- member: the thresholds used in the runs (exists_tm.py)
- thermo: per-concept quantiles of the max similarity over training sentences (as thermo:8)
for exists alone and with the mean thermometer. Uses the cached features of exists_tm.py. Writes
posthoc_lr.jsonl here.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exists_tm as X  # noqa: E402
from dl.boolean import Thermo  # noqa: E402

EMB = HERE / "emb_potion_test2"
for name in ("clinc150", "banking77", "hwu64"):
    F, meta = X.features(name, 512, EMB, "cuda")
    ytr, yva = F["y_train"], F["y_val"]
    C = int(ytr.max()) + 1
    tm_mean = Thermo(8).fit(F["train__mean"])
    tm_ex = Thermo(8).fit(F["train__exists"])
    ex_m = X.ExistsBits(meta["thresholds"], meta["concepts"])
    bits = {
        "mean_thermo": (tm_mean.transform(F["train__mean"]), tm_mean.transform(F["val__mean"])),
        "exists_member": (ex_m.transform(F["train__exists"]), ex_m.transform(F["val__exists"])),
        "exists_thermo": (tm_ex.transform(F["train__exists"]), tm_ex.transform(F["val__exists"])),
    }
    bits["mean+exists_member"] = tuple(np.concatenate([bits["mean_thermo"][i], bits["exists_member"][i]], 1) for i in (0, 1))
    bits["mean+exists_thermo"] = tuple(np.concatenate([bits["mean_thermo"][i], bits["exists_thermo"][i]], 1) for i in (0, 1))
    for k, (btr, bva) in bits.items():
        acc, c = X.best_logreg(btr, ytr, bva, yva, C, [1.0, 10.0, 100.0])
        rec = {"dataset": name, "bits": k, "n_bits": int(btr.shape[1]), "on_share": round(float(btr.mean()), 3),
               "lr_val": acc, "C": c}
        print(json.dumps(rec), flush=True)
        with open(HERE / "posthoc_lr.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
