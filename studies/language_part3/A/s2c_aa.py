#!/usr/bin/env python
"""S2c (CPU): K8, the A/A run (amendment 1, 2026-10-21, before any run). B is rebuilt twice, each time with 5% of
the fit dialogues dropped (tally aa.subsample_train) and its own seeded assignment of the 5 dialogue folds
(aa.folds): seed 2 and seed 3. Each rebuild is the S1 floor recipe (its configuration and sub-picks; EM weights
refitted on tune-cal) counted on its remaining dialogues, cross-fitted twin rows, the chosen twin configuration
recounted, isotonic and b refitted on tune-cal. The registered K8 is the equal-size A/A: on tune-hold, rebuilt(2)
minus rebuilt(3) must lie within +-0.0036 bits per token with its 95% dialogue-bootstrap interval covering 0; else
STOP (exit code 3). Both rebuilds count on 95% of the fit dialogues, so their difference holds only what an honest
repeat varies (which dialogues are counted, which fold each is in).

    chain/part3_cpu_job.sh A tm s2c_aa.py [--s0 DIR] [--s1 DIR] [--twin DIR] [--out DIR]

Reported, no rule (written after the K8 decision): B minus rebuilt(2) and B minus rebuilt(3). These also hold the
effect of counting on 5% less data (about 0.004-0.015 bits per token expected), which is why they are not the rule
(R1 as frozen at 05:13:40Z used B minus rebuilt(2)).

Runs right after S2b-twin and before every learner (S2b-arm), so K8 is decided before any learner's tune-hold number
exists; it refuses to start once a learner folder exists next to its output.
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
from astruct.build import AA_SEED, aa_fit, crossfit, folds_of, make_floor, rows_for
from astruct.data import CAL, FIT, HOLD, Corpus
from astruct.gates import k8, paired_bootstrap
from astruct.positions import Positions
from astruct.readout import Arm, PosTable
from astruct.twins import CountTwin, twin_tables

AA_SEED_2 = 3  # the second rebuild of the equal-size A/A


def rebuild(c: Corpus, s1: dict, tw0, P_cal: Positions, seed: int, log) -> dict:
    """B rebuilt on the fit dialogues kept by the seeded 5% drop, with the seeded fold assignment."""
    fit, cal, hold = c.replies(FIT), c.replies(CAL), c.replies(HOLD)
    fit2, dropped = aa_fit(c, fit, seed)
    fold2 = folds_of(c, fit2, seed)
    log(f"rebuild seed {seed}: {len(dropped)} fit dialogues dropped, {len(fit2)} of {len(fit)} fit replies kept")
    cf = crossfit(c, fit2, fold2, s1, P_cal, np.zeros(len(fit2), bool), bits=False, log=log, leak_test=False)
    fs = make_floor(c, fit2, s1, P_cal)
    cal_rows, _ = rows_for(c, fs, None, cal, s1["floor"], bits=False)
    hold_rows, _ = rows_for(c, fs, None, hold, s1["floor"], bits=False)
    tw = CountTwin(tw0.level, tw0.w, twin_tables(cf["twin"], cf["twin"]["y"], [tw0.level]))
    arm = Arm(f"B_rebuilt_seed{seed}").fit(tw.predict(cal_rows), cal_rows, PosTable(cal_rows, cal_rows["p_true"]))
    hb = arm.bits(tw.predict(hold_rows), PosTable(hold_rows, hold_rows["p_true"]))
    return {"hold": hb, "cal": arm.cal_bits, "b": arm.b, "dropped": dropped, "groups": hold_rows["pos_dlg"],
            "floor_hold": -np.log2(hold_rows["p_true"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=str(paths.OUT / "s0"))
    ap.add_argument("--s1", default=str(paths.OUT / "s1"))
    ap.add_argument("--twin", default=str(paths.OUT / "s2b_twin"))
    ap.add_argument("--out", default=str(paths.OUT / "s2c_aa"))
    args = ap.parse_args()
    env = paths.check_env()
    # K8 comes before any learner's tune-hold number (README job order): refuse once a learner folder exists
    learners = sorted(p.name for p in Path(args.out).parent.glob("s2b_*") if p.name.startswith(("s2b_full_", "s2b_nf_")))
    if learners:
        raise SystemExit(f"learner outputs exist ({learners}): K8 must be decided before any learner runs")
    out = paths.OutDir(Path(args.out))
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    c = Corpus.load(Path(args.s0))
    s1 = json.loads((Path(args.s1) / "results.json").read_text())
    with open(Path(args.twin) / "B.pkl", "rb") as f:
        Bp = pickle.load(f)
    B_hold = np.load(Path(args.twin) / "bits.npz")["B_hold"]
    P_cal = Positions(c, c.replies(CAL))
    r2 = rebuild(c, s1, Bp["twin"], P_cal, AA_SEED, log)
    r3 = rebuild(c, s1, Bp["twin"], P_cal, AA_SEED_2, log)
    for r in (r2, r3):
        if len(r["hold"]) != len(B_hold) or not np.array_equal(r["groups"], r2["groups"]):
            raise SystemExit("tune-hold positions differ between B and a rebuilt B")
    g = r2["groups"]
    boot = paired_bootstrap(r2["hold"] - r3["hold"], g)
    res = {"K8": k8(boot), "K8_rule": "rebuilt(seed 2) minus rebuilt(seed 3), equal size (amendment 1)",
           "bootstrap": boot, "seeds": [AA_SEED, AA_SEED_2],
           "B_tune_hold": float(B_hold.mean()), "B_rebuilt2_tune_hold": float(r2["hold"].mean()),
           "B_rebuilt3_tune_hold": float(r3["hold"].mean()), "rebuilt2_b": r2["b"], "rebuilt3_b": r3["b"],
           "dropped_dialogues": {"seed2": len(r2["dropped"]), "seed3": len(r3["dropped"]),
                                 "both": int(len(np.intersect1d(r2["dropped"], r3["dropped"])))},
           "floor_rebuilt2_tune_hold": float(r2["floor_hold"].mean()),
           "floor_rebuilt3_tune_hold": float(r3["floor_hold"].mean()), "env": env, "code_sha256": paths.code_hashes()}
    log("K8 (equal-size A/A: rebuilt(2) minus rebuilt(3))", json.dumps(res["K8"]), "| rebuilt(2)",
        round(res["B_rebuilt2_tune_hold"], 5), "rebuilt(3)", round(res["B_rebuilt3_tune_hold"], 5))
    (out / "k8.json").write_text(json.dumps(res, indent=1))  # the registered decision, written before the reported numbers
    dB2 = paired_bootstrap(B_hold - r2["hold"], g)
    dB3 = paired_bootstrap(B_hold - r3["hold"], g)
    res["reported_no_rule"] = {
        "B_minus_rebuilt2": dB2, "B_minus_rebuilt3": dB3,
        "R1_v1_rule_on_B_minus_rebuilt2_would_pass": bool(k8(dB2)["pass"]),
        "note": "reported, no rule: B counts on all fit dialogues, each rebuild on 95% of them, so B minus a rebuild "
                "holds noise plus the effect of counting on 5% less data; rebuilt(2) minus rebuilt(3) (the rule) "
                "holds the noise only"}
    log("reported (no rule): B - rebuilt(2)", json.dumps({k: round(dB2[k], 5) for k in ("mean", "lo", "hi")}),
        "| B - rebuilt(3)", json.dumps({k: round(dB3[k], 5) for k in ("mean", "lo", "hi")}))
    res["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(out / "bits.npz", B_rebuilt_hold=r2["hold"], B_rebuilt_cal=r2["cal"], B_rebuilt3_hold=r3["hold"],
                        B_rebuilt3_cal=r3["cal"])
    (out / "results.json").write_text(json.dumps(res, indent=1))
    out.done({"utc_end": paths.utc(), "K8": res["K8"]["pass"]})
    if not res["K8"]["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
