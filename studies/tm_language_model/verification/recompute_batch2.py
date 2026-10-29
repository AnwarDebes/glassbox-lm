#!/usr/bin/env python
"""Independent recompute of the tm_language_model results of GPU batch 2 of the finish pass (2026-10-25), written from
the registered rows of docs/13, not from the drivers:

- H9.6 (docs/13 line 243): x = ln(p / b), b = (1 - p) Q / (V - 1), over the 50 most probable KN-5 classes of each of
  the 200,000 positions; bins of x of width 0.25; weighted R^2 of the bin means of S/T against the bin means of
  tanh(x/2), weights = pairs per bin; Spearman(S, ln p) with average ranks for ties, pooled over the pairs with
  p > 20 b. Thresholds: R^2 >= 0.8 AND Spearman <= 0.2.
- H8.8 (c): median of the per-repeat training tokens per second against 1,000,000.
- H8.9: bytes fraction and loss (variant minus unpruned bits per byte) of each pruned variant against 0.30 and 0.02.

Usage: <python with numpy> -I recompute_batch2.py OUT.json   (reads the run folders named below; numpy only)
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import json
import sys

import numpy as np

S_DIR = f"{_REPO_ROOT}/studies/tm_language_model/results/"
H96 = S_DIR + "20261025T134132Z_cvtm_H9_6_vote_equilibrium"
H88C = S_DIR + "20261025T131820Z_cvtm_a3_H8_8c"
H89 = S_DIR + "20261025T133240Z_cvtm_a3_H8_9"


def average_ranks(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    # boundaries of runs of equal values
    starts = np.r_[0, np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_a.size]
    avg = (starts + ends - 1) / 2.0 + 1.0  # 1-based average rank of each run
    ranks_sorted = np.repeat(avg, ends - starts)
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def h96():
    z = np.load(H96 + "/h96_pairs.npz", allow_pickle=False)
    p = z["kn_p"].astype(np.float64).ravel()
    S = z["S"].astype(np.float64).ravel()
    T, V, Q = float(z["T"]), float(z["V"]), float(z["Q"])
    b = (1.0 - p) * Q / (V - 1.0)
    x = np.log(p / b)
    y = S / T
    f = np.tanh(x / 2.0)
    bins = np.floor(x / 0.25).astype(np.int64)
    u, inv, n = np.unique(bins, return_inverse=True, return_counts=True)
    ybar = np.bincount(inv, weights=y) / n
    fbar = np.bincount(inv, weights=f) / n
    w = n.astype(np.float64)
    ymean = (w * ybar).sum() / w.sum()
    r2 = 1.0 - (w * (ybar - fbar) ** 2).sum() / (w * (ybar - ymean) ** 2).sum()
    keep = p > 20.0 * b
    rho = spearman(S[keep], np.log(p[keep]))
    return {"pairs": int(p.size), "bins": int(u.size), "weighted_r2": float(r2), "pairs_p_gt_20b": int(keep.sum()),
            "spearman_S_lnp": rho, "met": bool(r2 >= 0.8 and rho <= 0.2), "T": T, "V": V, "Q": Q}


def h88c():
    m = json.load(open(H88C + "/metrics.json"))
    per = m["throughput.tokens_per_s_median"]["per_repeat"]
    med = float(np.median(per))
    return {"per_repeat": per, "median": med, "met": bool(med >= 1e6),
            "every_repeat_reproduces_frozen_state_hash": m["check.every_repeat_reproduces_frozen_state_hash"]["value"]}


def h89():
    m = json.load(open(H89 + "/metrics.json"))
    v = m["verdict.H8.9"]["number"]
    base_bpb, base_bytes = v["unpruned_bits_per_byte"], v["unpruned_suffix_nbytes"]
    out = {}
    for name in ("prune_c1", "prune_c2"):
        var = v["variants"][name]
        frac = var["suffix_nbytes"] / base_bytes
        loss = var["bits_per_byte"] - base_bpb
        out[name] = {"bytes_fraction": frac, "loss_bits_per_byte": loss, "qualifies": bool(frac <= 0.30 and loss <= 0.02)}
    out["met"] = any(x["qualifies"] for x in out.values())
    return out


def main(path):
    res = {"H9.6": h96(), "H8.8c": h88c(), "H8.9": h89()}
    rec = {}
    m96 = json.load(open(H96 + "/metrics.json"))
    rec["H9.6"] = {"weighted_r2": m96["h96.weighted_r2"]["value"], "spearman": m96["h96.spearman_S_lnp_p_gt_20b"]["value"],
                   "pairs_p_gt_20b": m96["h96.spearman_S_lnp_p_gt_20b"]["n_pairs"], "verdict": m96["h96.verdict"]["value"]}
    res["recorded"] = rec
    res["agree"] = {
        "H9.6_r2": abs(res["H9.6"]["weighted_r2"] - rec["H9.6"]["weighted_r2"]) < 1e-9,
        "H9.6_spearman": abs(res["H9.6"]["spearman_S_lnp"] - rec["H9.6"]["spearman"]) < 1e-9,
        "H9.6_pairs": res["H9.6"]["pairs_p_gt_20b"] == rec["H9.6"]["pairs_p_gt_20b"],
        "H9.6_verdict": (res["H9.6"]["met"] is False) == (rec["H9.6"]["verdict"] == "NOT MET"),
    }
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if all(res["agree"].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
