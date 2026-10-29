#!/usr/bin/env python
"""An independent recompute of step 3c's registered bar (tu/PLAN_3c.md, "Bar (fixed now)").

Written from the plan's text only (not from c3_verdict.py or verify_3c/recompute_3c.py):
- PASS: 3c-F minus the reference is at least +0.5 with BOTH probes on at least 2 of the 3 sets;
  TM = the mean of the paired gains over seeds 4-6; LR = val; unrounded gains compared with 0.5 - 1e-9.
- Reference ("Features and readers"): if the rebuilt original features are not bit-for-bit equal to the cached
  reference features, the reference TM and LR rows are run again on the rebuilt features; those rows are the reference
  (TM rows tagged 3c-ref, arm orig; LR row name ref_rebuilt).

Usage: python -I recompute_3c.py C3_DIR   (prints JSON; exit 0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (4, 5, 6)
BAR = 0.5 - 1e-9


def main() -> None:
    c3 = Path(sys.argv[1])
    tm = [json.loads(line) for line in (c3 / "results_c3_val.jsonl").read_text().splitlines() if line.strip()]
    lr = [json.loads(line) for line in (c3 / "results_c3_lr.jsonl").read_text().splitlines() if line.strip()]
    out = {"sets": {}, "problems": []}
    met = 0
    for ds in SETS:
        f_rows = {r["seed"]: r for r in tm if r["dataset"] == ds and r["arm"] == "F" and r["tag"] == "3c-tm"}
        ref_rows = {r["seed"]: r for r in tm if r["dataset"] == ds and r["arm"] == "orig" and r["tag"] == "3c-ref"}
        for name, rows in (("F", f_rows), ("ref", ref_rows)):
            if sorted(rows) != list(SEEDS):
                out["problems"].append(f"{ds} {name}: seeds {sorted(rows)} != {list(SEEDS)}")
        dup = [r for r in tm if r["dataset"] == ds and r["tag"] in ("3c-tm", "3c-ref")]
        if len(dup) != 6:
            out["problems"].append(f"{ds}: {len(dup)} TM rows tagged 3c-tm / 3c-ref (expected 6)")
        paired = [100.0 * (f_rows[s]["val_acc"] - ref_rows[s]["val_acc"]) for s in SEEDS if s in f_rows and s in ref_rows]
        tm_gain = sum(paired) / len(paired) if paired else float("nan")
        lr_f = [r for r in lr if r["dataset"] == ds and r["arm"] == "F" and r["name"] == "F"]
        lr_ref = [r for r in lr if r["dataset"] == ds and r["name"] == "ref_rebuilt"]
        if len(lr_f) != 1 or len(lr_ref) != 1:
            out["problems"].append(f"{ds}: LR rows F {len(lr_f)}, ref_rebuilt {len(lr_ref)} (expected 1 each)")
        lr_gain = 100.0 * (lr_f[0]["lr_val"] - lr_ref[0]["lr_val"]) if lr_f and lr_ref else float("nan")
        # settings parity between the arms
        cfgs = {json.dumps({k: v for k, v in r["cfg"].items() if k != "seed"}, sort_keys=True)
                for r in list(f_rows.values()) + list(ref_rows.values())}
        if len(cfgs) != 1:
            out["problems"].append(f"{ds}: TM cfg differs between F and ref rows")
        s_vals = {r["s"] for r in list(f_rows.values()) + list(ref_rows.values())}
        meets = tm_gain >= BAR and lr_gain >= BAR
        met += int(meets)
        out["sets"][ds] = {
            "tm_F": [f_rows[s]["val_acc"] for s in SEEDS if s in f_rows],
            "tm_ref": [ref_rows[s]["val_acc"] for s in SEEDS if s in ref_rows],
            "tm_paired_gains": [round(x, 4) for x in paired],
            "tm_gain": round(tm_gain, 4),
            "lr_F": lr_f[0]["lr_val"] if lr_f else None, "lr_F_C": lr_f[0]["C"] if lr_f else None,
            "lr_ref_rebuilt": lr_ref[0]["lr_val"] if lr_ref else None, "lr_ref_C": lr_ref[0]["C"] if lr_ref else None,
            "lr_gain": round(lr_gain, 4), "s": sorted(s_vals), "meets": meets,
            "n_train_F": sorted({r["n_train"] for r in f_rows.values()}),
            "n_train_ref": sorted({r["n_train"] for r in ref_rows.values()}),
        }
    out["sets_meeting"] = met
    out["verdict"] = "PASS" if met >= 2 else "FAIL"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
