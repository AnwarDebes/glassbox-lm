#!/usr/bin/env python
"""Independent recompute of R24 (registrations/R24_multiwoz_random_negatives.md, "Pass rules (test)").

Written from the registration text, not from scripts/report_abcd.py or report_twolevel.py:
- per test turn, each arm's correctness is averaged over its seeds; pair difference = mean over all test turns;
- 95% interval: bootstrap over test dialogues (convo_id), 10,000 resamples, numpy default_rng(0), percentiles 2.5 / 97.5
  (the RNG scheme may differ from report_abcd.py, so interval ends can differ slightly; point estimates must match);
- two-level interval (no rule): each resample draws each arm's seeds with replacement, then the dialogues;
- R24.1: diff >= -1.0 point (strong: lower end >= -1.0); R24.2: lower end > 0.
Also checks: same test turns (y, convo_id, k) in every arm and seed; per-seed test_acc recomputes; seed-file times
after the freeze; cfg equal within each arm.

Usage: python -I recompute_r24.py R24MWOZ_DIR FREEZE_ISO   (prints JSON)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TM = ("tm_mem_c4_nrand", "tm_mem_ref_nrand", "tm_mem_bge_nrand")
LR = ("lr_mem_c4", "lr_mem_ref", "lr_mem_bge")
PAIRS = (("tm_mem_c4_nrand", "tm_mem_bge_nrand"), ("tm_mem_ref_nrand", "tm_mem_bge_nrand"),
         ("tm_mem_c4_nrand", "lr_mem_c4"), ("tm_mem_ref_nrand", "lr_mem_ref"), ("tm_mem_bge_nrand", "lr_mem_bge"))
B = 10_000


def load(d: Path, arm: str):
    seeds = sorted(int(p.stem[4:]) for p in (d / arm).glob("seed*.json"))
    corr, meta, ref = [], [], None
    for s in seeds:
        z = np.load(d / arm / f"seed{s}.npz", allow_pickle=False)
        key = (z["y"], z["convo_id"], z["k"])
        if ref is None:
            ref = key
        elif not all(np.array_equal(a, b) for a, b in zip(ref, key)):
            raise SystemExit(f"{arm} seed {s}: test turns differ")
        corr.append((z["pred"] == z["y"]).astype(np.float64))
        meta.append(json.loads((d / arm / f"seed{s}.json").read_text()))
    return seeds, np.stack(corr), meta, ref


def main() -> None:
    d, freeze = Path(sys.argv[1]), datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00"))
    arms, out, problems, ref0 = {}, {"arms": {}, "pairs": {}}, [], None
    for arm in TM + LR:
        seeds, corr, meta, ref = load(d, arm)
        if ref0 is None:
            ref0 = ref
        elif not all(np.array_equal(a, b) for a, b in zip(ref0, ref)):
            problems.append(f"{arm}: test turns differ from {TM[0]}")
        arms[arm] = corr
        per_seed = corr.mean(axis=1)
        for m, a in zip(meta, per_seed):
            if abs(m["test_acc"] - a) > 1e-12:
                problems.append(f"{arm} seed {m['seed']}: json test_acc {m['test_acc']} != {a}")
            t = datetime.fromisoformat(str(m["time"]).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t <= freeze:
                problems.append(f"{arm} seed {m['seed']}: time {m['time']} not after the freeze")
        cfgs = {json.dumps({k: v for k, v in m.get("cfg", {}).items() if k != "seed"}, sort_keys=True) for m in meta}
        if len(cfgs) != 1:
            problems.append(f"{arm}: cfg differs between seeds")
        c = meta[0].get("cfg") or {k: v for k, v in meta[0].items() if k in ("C", "C_grid", "lr_C", "best_C")}
        out["arms"][arm] = {"seeds": seeds, "test_acc_mean": float(corr.mean(axis=0).mean()),
                            "per_seed": [round(float(a), 5) for a in per_seed],
                            "cfg": {k: c.get(k) for k in ("n_clauses", "T", "s", "epochs", "neg_sampling", "C")
                                    if k in c},
                            "times": [m["time"] for m in meta], "dev_acc": [m.get("dev_acc") for m in meta]}
    cid = ref0[1]
    uniq, inv = np.unique(cid, return_inverse=True)
    n_d = len(uniq)
    cnt = np.bincount(inv, minlength=n_d).astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_d, size=(B, n_d))
    w = np.zeros((B, n_d))
    for b in range(B):
        w[b] = np.bincount(idx[b], minlength=n_d)
    den = w @ cnt
    for a, b_ in PAIRS:
        diff = arms[a].mean(axis=0) - arms[b_].mean(axis=0)
        sums = np.bincount(inv, weights=diff, minlength=n_d)
        boot = (w @ sums) / den
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # two-level: seeds with replacement per arm, then dialogues (same dialogue weights w)
        rng2 = np.random.default_rng(0)
        sa = np.stack([np.bincount(inv, weights=r, minlength=n_d) for r in arms[a]])  # seeds x dialogues
        sb = np.stack([np.bincount(inv, weights=r, minlength=n_d) for r in arms[b_]])
        tl = np.empty(B)
        for i in range(B):
            ca = np.bincount(rng2.integers(0, len(sa), len(sa)), minlength=len(sa)) / len(sa)
            cb = np.bincount(rng2.integers(0, len(sb), len(sb)), minlength=len(sb)) / len(sb)
            dsum = ca @ sa - cb @ sb
            tl[i] = (w[i] @ dsum) / den[i]
        tlo, thi = np.percentile(tl, [2.5, 97.5])
        res = {"diff_pts": round(100 * float(diff.mean()), 4), "lo_pts": round(100 * lo, 4),
               "hi_pts": round(100 * hi, 4), "twolevel_lo_pts": round(100 * tlo, 4),
               "twolevel_hi_pts": round(100 * thi, 4)}
        if b_ == "tm_mem_bge_nrand":
            res.update({"R24.1": bool(100 * diff.mean() >= -1.0 - 1e-9), "R24.1_strong": bool(100 * lo >= -1.0 - 1e-9),
                        "R24.2": bool(lo > 0), "minus2_reading": bool(100 * diff.mean() >= -2.0 - 1e-9)})
        out["pairs"][f"{a}:{b_}"] = res
    out["n_turns"], out["n_dialogues"], out["problems"] = int(len(cid)), int(n_d), problems
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
