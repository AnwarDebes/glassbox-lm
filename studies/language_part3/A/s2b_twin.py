#!/usr/bin/env python
"""S2b-twin (CPU): the count twin on all fit rows (cross-fitted), 8 configurations chosen on tune-cal; B = floor +
the chosen twin's mix (the best counting system); K9 (shuffled labels) and K2 (sums) for the twin.

    chain/part3_cpu_job.sh A tm s2b_twin.py [--s1 DIR] [--s2a DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.gates import k9
from astruct.readout import Arm, PosTable, contract
from astruct.rows import shuffled_labels
from astruct.store import load_rows
from astruct.twins import TWIN_KEYS, TWIN_W, CountTwin, twin_tables

SHUFFLE_SEED = 1


def full_vectors(s1: Path, n_cal: int) -> tuple[dict, dict]:
    out = []
    for nm in ("cal", "hold"):
        z = np.load(s1 / f"summ_{nm}.npz")
        out.append({int(k): z["full_vec"][i] for i, k in enumerate(z["full_pos"])})
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default=str(paths.OUT / "s1"))
    ap.add_argument("--s2a", default=str(paths.OUT / "s2a"))
    ap.add_argument("--out", default=str(paths.OUT / "s2b_twin"))
    args = ap.parse_args()
    env = paths.check_env()
    out = paths.OutDir(Path(args.out))
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    s2a = Path(args.s2a)
    twin = load_rows(s2a / "twin_rows.npz", bits=False)
    cal = load_rows(s2a / "cal_rows.npz", bits=False)
    hold = load_rows(s2a / "hold_rows.npz", bits=False)
    full_c, full_h = full_vectors(Path(args.s1), cal["n_pos"])
    ptc, pth = PosTable(cal, cal["p_true"]), PosTable(hold, hold["p_true"])
    floor_cal, floor_hold = -np.log2(cal["p_true"]), -np.log2(hold["p_true"])
    log("S2b-twin start; twin rows", len(twin["y"]), "positions", twin["n_pos"], "| tune-cal rows", len(cal["y"]))
    tables = twin_tables(twin, twin["y"], TWIN_KEYS)
    res, bits_c, bits_h, arms = {}, {}, {}, {}
    for key in TWIN_KEYS:
        for w in TWIN_W:
            name = f"{key}_w{int(w)}"
            tw = CountTwin(key, w, tables)
            sc, sh = tw.predict(cal), tw.predict(hold)
            arm = Arm(name).fit(sc, cal, ptc)
            hb = arm.bits(sh, pth)
            k2c = contract(arm.iso(sc), cal, full_c, ptc, arm.b, cal["pos_x"], arm.cal_bits)
            k2h = contract(arm.iso(sh), hold, full_h, pth, arm.b, hold["pos_x"], hb)
            if not (k2c["pass"] and k2h["pass"]):
                raise SystemExit(f"K2 failed for {name}: {k2c} {k2h}")
            res[name] = {"b": arm.b, "tune_cal_bits": float(arm.cal_bits.mean()), "tune_hold_bits": float(hb.mean()),
                         "cells": tw.cells(), "K2": {"cal": k2c, "hold": k2h}}
            bits_c[name], bits_h[name], arms[name] = arm.cal_bits, hb, (tw, arm)
            log(name, json.dumps({k: v for k, v in res[name].items() if k != "K2"}))
    pick = min(res, key=lambda n: (res[n]["tune_cal_bits"], list(res).index(n)))
    B_cal, B_hold = bits_c[pick], bits_h[pick]
    log(f"B = floor + {pick}: tune-cal {B_cal.mean():.5f}, tune-hold {B_hold.mean():.5f} (floor tune-hold {floor_hold.mean():.5f})")
    # reported, no rule: a constant scorer (no information) through the same read-out; b > 0 would mean the floor's
    # share inside C is not calibrated, so flattening it alone gains bits
    const = Arm("constant").fit(np.zeros(len(cal["y"])), cal, ptc)
    const_hold = const.bits(np.zeros(len(hold["y"])), pth)
    const_res = {"b": const.b, "tune_cal_bits": float(const.cal_bits.mean()), "tune_hold_bits": float(const_hold.mean())}
    log("constant scorer (reported)", json.dumps(const_res))
    # K9: the chosen configuration counted on shuffled labels; read-out on the true tune-cal labels
    tw_p, _ = arms[pick]
    ys = shuffled_labels(twin, SHUFFLE_SEED)
    tws = CountTwin(tw_p.level, tw_p.w, twin_tables(twin, ys, [tw_p.level]))
    sc, sh = tws.predict(cal), tws.predict(hold)
    arm_s = Arm(pick + "_shuffled").fit(sc, cal, ptc)
    hb_s = arm_s.bits(sh, pth)
    k9r = k9(arm_s.b, float((hb_s - B_hold).mean()))
    log("K9 count twin (shuffled labels)", json.dumps(k9r))
    np.savez_compressed(out / "bits.npz", **{f"cal_{k}": v for k, v in bits_c.items()}, **{f"hold_{k}": v for k, v in bits_h.items()},
                        B_cal=B_cal, B_hold=B_hold, floor_cal=floor_cal, floor_hold=floor_hold,
                        shuffled_cal=arm_s.cal_bits, shuffled_hold=hb_s)
    with open(out / "B.pkl", "wb") as f:
        pickle.dump({"pick": pick, "twin": arms[pick][0], "arm": arms[pick][1]}, f)
    summary = {"pick": pick, "configs": res, "B": {"tune_cal_bits": float(B_cal.mean()), "tune_hold_bits": float(B_hold.mean())},
               "floor": {"tune_cal_bits": float(floor_cal.mean()), "tune_hold_bits": float(floor_hold.mean())},
               "K9": k9r, "constant_scorer": const_res, "shuffle_seed": SHUFFLE_SEED, "seconds": round(time.time() - t0, 1), "env": env,
               "code_sha256": paths.code_hashes(), "s2a_done_sha256": paths.sha256_file(s2a / "done.json")}
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    out.done({"utc_end": paths.utc(), "pick": pick, "K9": k9r["pass"]})
    if not k9r["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
