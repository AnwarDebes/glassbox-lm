#!/usr/bin/env python
"""Independent recompute of the registered M2 verdict (tu/PLAN_M2.md, frozen 2026-10-25T15:03:02Z), written from the plan
text without m2_verdict.py or m2_data.py. Part of the one use of MTOP English en/test (the plan's report step allows this
recompute of the same 12 prediction files; no model is scored again).

- Labels: the official zip (sha256 checked), member mtop/en/test.txt, 8 tab-separated columns; the classes are the sorted
  distinct intents (column 2) of mtop/en/train.txt; a test intent not among them is label -1 (always an error).
- Accuracy in percent per prediction file; TM = the mean over seeds 7, 8, 9; LR = one fit.
- H1: a3B - C4 at least +0.5 points with the TM AND with LR. H2: a3B - potion at least -0.5 points with the TM AND with LR.
  Both on the point estimates; STRONG when the lower ends of the item-level intervals also clear the bars.
- Intervals: item-level paired bootstrap, 10,000 resamples, numpy default_rng([20261025, 2]) (not the run's seed), one
  index draw per resample shared by every comparison; the TM side averages each item's correctness over the three seeds.

Usage: <python with numpy> -I recompute_m2.py OUT.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import hashlib
import json
import sys
import zipfile

import numpy as np

E = f"{_REPO_ROOT}/studies/text_understanding"
ZIP = E + "/data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip"
ZIP_SHA256 = "0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b"
PRED = E + "/tu/m2/preds/"
ARMS = ("C4", "a3B", "potion")
SEEDS = (7, 8, 9)


def rows(z, member):
    out = [ln.split("\t") for ln in z.read(member).decode("utf-8").split("\n") if ln.strip()]
    assert all(len(r) == 8 for r in out), member
    return out


def labels():
    h = hashlib.sha256(open(ZIP, "rb").read()).hexdigest()
    assert h == ZIP_SHA256, h
    z = zipfile.ZipFile(ZIP)
    classes = sorted({r[1] for r in rows(z, "mtop/en/train.txt")})
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx.get(r[1], -1) for r in rows(z, "mtop/en/test.txt")], dtype=np.int64), len(classes)


def main(out_path):
    y, n_classes = labels()
    corr = {}
    for arm in ARMS:
        corr[("tm", arm)] = np.stack([(np.load(PRED + f"m2_{arm}_mtop_en_r1_s{s}.npy") == y) for s in SEEDS]).astype(float)
        corr[("lr", arm)] = (np.load(PRED + f"lr_{arm}_mtop_en.npy") == y).astype(float)[None, :]
    acc = {f"{p}_{a}": [round(100 * float(c.mean()), 4) for c in v] if p == "tm" else round(100 * float(v.mean()), 4)
           for (p, a), v in corr.items() for c in [v]}
    acc = {}
    for (p, a), v in corr.items():
        acc[f"{p}_{a}"] = {"per_seed": [100 * float(r.mean()) for r in v], "mean": 100 * float(v.mean())}
    item = {k: v.mean(axis=0) for k, v in corr.items()}  # per-item correctness, averaged over seeds for the TM
    comps = {"H1_tm": (("tm", "a3B"), ("tm", "C4")), "H1_lr": (("lr", "a3B"), ("lr", "C4")),
             "H2_tm": (("tm", "a3B"), ("tm", "potion")), "H2_lr": (("lr", "a3B"), ("lr", "potion")),
             "tm_minus_lr_C4": (("tm", "C4"), ("lr", "C4")), "tm_minus_lr_a3B": (("tm", "a3B"), ("lr", "a3B")),
             "tm_minus_lr_potion": (("tm", "potion"), ("lr", "potion"))}
    n = y.shape[0]
    rng = np.random.default_rng([20261025, 2])
    diffs = {k: item[a] - item[b] for k, (a, b) in comps.items()}
    boot = {k: np.empty(10000) for k in comps}
    for i in range(10000):
        ix = rng.integers(0, n, size=n)
        for k, d in diffs.items():
            boot[k][i] = d[ix].mean()
    res = {k: {"point": 100 * float(d.mean()), "ci95": [100 * float(np.percentile(boot[k], 2.5)),
                                                        100 * float(np.percentile(boot[k], 97.5))]}
           for k, d in diffs.items()}
    h1 = res["H1_tm"]["point"] >= 0.5 and res["H1_lr"]["point"] >= 0.5
    h2 = res["H2_tm"]["point"] >= -0.5 and res["H2_lr"]["point"] >= -0.5
    h1s = h1 and res["H1_tm"]["ci95"][0] >= 0.5 and res["H1_lr"]["ci95"][0] >= 0.5
    h2s = h2 and res["H2_tm"]["ci95"][0] >= -0.5 and res["H2_lr"]["ci95"][0] >= -0.5
    out = {"n_test": int(n), "train_intents": n_classes, "unseen_test_items": int((y < 0).sum()), "accuracy": acc,
           "comparisons": res, "H1_met": h1, "H1_strong": h1s, "H2_met": h2, "H2_strong": h2s,
           "bootstrap": "item-level paired, 10,000 resamples, default_rng([20261025, 2])"}
    v = json.load(open(E + "/tu/m2/verdict_m2.json"))
    out["recorded_verdict_json_keys"] = sorted(v)[:40]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("n_test", "train_intents", "unseen_test_items", "H1_met", "H1_strong", "H2_met",
                                          "H2_strong")}, indent=1))
    for k, r in res.items():
        print(f"{k}: {r['point']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    for k, a in acc.items():
        print(k, [round(x, 2) for x in a["per_seed"]], round(a["mean"], 2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
