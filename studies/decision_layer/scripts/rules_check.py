#!/usr/bin/env python
"""Apply the frozen pass rules of R2/R2b and R4/R4b (with R1) to the reports.

    rules_check.py R2       (needs results/confirm/R2/A_default, B_exact32, Bp_tmu32, C_lr_bits)
    rules_check.py R4 [R4b]
    rules_check.py R7       (R7.1 against LR on the readable bits, R7.2 composite against LR on the raw embedding)
    rules_check.py R2c      (needs results/confirm/R2c/<ds>_A_default, <ds>_B_trim32, <ds>_C_lr_bits for banking77, hwu64)
    rules_check.py R7b | R2d | R7c   (R2d and R7c: the re-runs with the exact budget selection)

Writes results/confirm/<reg>/rules_<reg>.json and prints a short verdict.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-dl" / "bin" / "python")
CELLS = ["clinc150_minilm", "banking77_minilm", "hwu64_minilm", "clinc150_bge-base", "banking77_bge-base", "hwu64_bge-base"]


def report(reg, tm, tm2=None, lr=None, composite=False):
    cmd = [PY, str(ROOT / "scripts" / "report_router.py"), "--registration", reg, "--tm", tm]
    if tm2:
        cmd += ["--tm2", tm2]
    if lr:
        cmd += ["--lr", lr]
    if composite:
        cmd += ["--composite"]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads((ROOT / "results" / "confirm" / reg / f"report_{tm}{('_vs_' + tm2) if tm2 else ''}.json").read_text())


def r2() -> dict:
    out = {}
    for name, arm in (("R2", "B_exact32"), ("R2b", "Bp_tmu32")):
        rep = report("R2", "A_default", arm, "C_lr_bits")
        L = 32
        stats = rep["tm2"]["stats_first_member"]
        d_ba = rep["tm2_minus_tm"]
        d_bc = rep["tm2_minus_lr"]
        rule1 = d_ba["diff"] >= 0.03 and d_ba["lo"] > 0
        if name == "R2":
            rule2 = all(s["median_literals"] <= L and s["max_literals"] <= L for s in stats)
        else:
            rule2 = all(s["median_literals"] <= L and s["p95_literals"] <= 2 * L for s in stats)
        out[name] = {"A_acc": rep["tm"]["test_acc_mean"], "B_acc": rep["tm2"]["test_acc_mean"], "C_acc": rep["lr"]["test_acc"],
                     "B_minus_A": d_ba, "B_minus_C": d_bc, f"{name}.1": rule1, f"{name}.2": rule2, "met": rule1 and rule2,
                     f"{name}.3_noninferior_to_C": d_bc["diff"] >= -0.005, f"{name}.3_beats_C": d_bc["lo"] > 0,
                     "B_clause_stats": stats, "A_clause_stats": rep["tm"]["stats_first_member"]}
    return out


def r2c(reg: str = "R2c", b_arm: str = "B_trim32") -> dict:
    """R2c (and its re-run R2d, B arm with the exact selection): in both datasets B (budget 32, trim
    start) minus A (default) >= +3.0 points with the interval above 0 (R2c.1), and no clause of any
    B seed over 32 literals (R2c.2)."""
    out = {"datasets": {}}
    ok1 = ok2 = True
    for ds in ("banking77", "hwu64"):
        rep = report(reg, f"{ds}_A_default", f"{ds}_{b_arm}", f"{ds}_C_lr_bits")
        stats = rep["tm2"]["stats_first_member"]
        d_ba, d_bc = rep["tm2_minus_tm"], rep["tm2_minus_lr"]
        r1 = d_ba["diff"] >= 0.03 and d_ba["lo"] > 0
        r2 = len(stats) == 10 and all(s["max_literals"] <= 32 for s in stats)
        ok1, ok2 = ok1 and r1, ok2 and r2
        out["datasets"][ds] = {"A_acc": rep["tm"]["test_acc_mean"], "B_acc": rep["tm2"]["test_acc_mean"], "C_acc": rep["lr"]["test_acc"],
                               "B_minus_A": d_ba, "B_minus_C": d_bc, "R2c.1": r1, "R2c.2": r2,
                               "R2c.3_noninferior_to_C": d_bc["diff"] >= -0.005, "R2c.3_beats_C": d_bc["lo"] > 0,
                               "B_max_literals": max(s["max_literals"] for s in stats),
                               "B_share_no_plain_literal": float(np.mean([s["no_positive_literal"] / max(1, s["n_clauses"] - s["empty"]) for s in stats])),
                               "A_share_no_plain_literal": float(np.mean([s["no_positive_literal"] / max(1, s["n_clauses"] - s["empty"]) for s in rep["tm"]["stats_first_member"]]))}
    out["R2c.1"], out["R2c.2"] = ok1, ok2
    out["R2c_met"] = ok1 and ok2
    return out


def r2e() -> dict:
    """R2e: R2 (CLINC150 word router) again with arm B = budget 32, trim start, exact selection.
    R2.1: B minus A >= +3.0 points with the interval above 0; R2.2: median and maximum literals per
    clause at most 32 in every B seed. A and C are R2's files."""
    rep = report("R2e", "A_default", "B_trim32_rank", "C_lr_bits")
    stats = rep["tm2"]["stats_first_member"]
    d_ba, d_bc = rep["tm2_minus_tm"], rep["tm2_minus_lr"]
    r1 = d_ba["diff"] >= 0.03 and d_ba["lo"] > 0
    r2 = len(stats) == 10 and all(s["median_literals"] <= 32 and s["max_literals"] <= 32 for s in stats)
    return {"A_acc": rep["tm"]["test_acc_mean"], "B_acc": rep["tm2"]["test_acc_mean"], "C_acc": rep["lr"]["test_acc"],
            "B_minus_A": d_ba, "B_minus_C": d_bc, "R2.1": r1, "R2.2": r2, "R2e_met": r1 and r2,
            "R2.3_noninferior_to_C": d_bc["diff"] >= -0.005, "R2.3_beats_C": d_bc["lo"] > 0,
            "B_max_literals": max(s["max_literals"] for s in stats),
            "B_share_no_plain_literal": float(np.mean([s["no_positive_literal"] / max(1, s["n_clauses"] - s["empty"]) for s in stats]))}


def r7b(reg: str = "R7b") -> dict:
    """R7b (and its re-run R7c with the exact selection): as R7 (R7b.1, R7b.2) plus no clause over 16
    literals (R7b.3) and, on CLINC150, a mean out-of-scope AUROC of at least 0.936 (R7b.4, a separate
    verdict)."""
    out = {"datasets": {}}
    n1 = n2 = 0
    ok3 = True
    for ds in ("clinc150", "banking77", "hwu64"):
        if not (ROOT / "results" / "confirm" / reg / ds).exists():
            continue
        a = report(reg, ds, lr=f"{ds}_lrbits", composite=True)
        (ROOT / "results" / "confirm" / reg / f"report_{ds}_vs_lrbits.json").write_text(json.dumps(a, indent=1))
        b = report(reg, ds, composite=True)
        stats = a["tm"]["stats_first_member"]
        c = {"tm_single": a["tm"]["test_acc_mean"], "n_seeds": len(a["tm"]["seeds"]), "lr_readable": a["lr"]["test_acc"], "lr_raw": b["lr"]["test_acc"],
             "single_minus_lr_readable": a["tm_minus_lr"], "single_minus_lr_raw": b["tm_minus_lr"], "ece_tm_mean": a["tm"]["ece_test_mean"],
             "max_literals": max(s["max_literals"] for s in stats)}
        if "composite" in b:
            c["composite"] = b["composite"]["test_acc_mean"]
            c["composite_minus_lr_raw"] = b["composite_minus_lr"]
            n2 += c["composite_minus_lr_raw"]["diff"] >= -0.010
        n1 += c["single_minus_lr_readable"]["diff"] >= -0.005
        ok3 = ok3 and len(stats) == 10 and c["max_literals"] <= 16
        if "oos_auroc_maxsum_mean" in b["tm"]:
            c["oos_tm_maxsum_mean"] = b["tm"]["oos_auroc_maxsum_mean"]
            c["oos_lr_raw_msp"] = b["lr"]["oos_auroc_msp"]
            if "composite" in b and "oos_auroc_maxsum" in b["composite"]:
                c["oos_composite_maxsum"] = b["composite"]["oos_auroc_maxsum"]
        out["datasets"][ds] = c
    out["R7b.1_cells_ok"], out["R7b.2_cells_ok"] = int(n1), int(n2)
    out["R7b.1"], out["R7b.2"], out["R7b.3"] = n1 >= 2, n2 >= 2, ok3
    out["R7b_met"] = out["R7b.1"] and out["R7b.2"] and out["R7b.3"]
    oos = out["datasets"].get("clinc150", {}).get("oos_tm_maxsum_mean")
    out["R7b.4"] = oos is not None and oos >= 0.936
    return out


def r4(reg: str) -> dict:
    out = {"cells": {}}
    n41 = n42 = 0
    r11 = True
    for cell in CELLS:
        if not (ROOT / "results" / "confirm" / reg / cell).exists():
            continue
        rep = report(reg, cell, composite=True)
        c = {"tm_single": rep["tm"]["test_acc_mean"], "lr": rep["lr"]["test_acc"], "single_minus_lr": rep["tm_minus_lr"],
             "n_seeds": len(rep["tm"]["seeds"]), "ece_tm_mean": rep["tm"]["ece_test_mean"], "ece_lr": rep["lr"]["ece_test"]}
        if "composite" in rep:
            c["composite"] = rep["composite"]["test_acc_mean"]
            c["composite_minus_lr"] = rep["composite_minus_lr"]
            n41 += c["composite_minus_lr"]["diff"] >= -0.005
            c["composite_ahead_R4.3"] = c["composite_minus_lr"]["lo"] > 0
        n42 += c["single_minus_lr"]["diff"] >= -0.005
        c["single_ahead_R4.3"] = c["single_minus_lr"]["lo"] > 0
        r11 = r11 and c["ece_tm_mean"] <= 0.05
        if "oos_auroc_maxsum_per_seed" in rep["tm"]:
            per = rep["tm"]["oos_auroc_maxsum_per_seed"]
            c["oos_tm_maxsum_mean"] = rep["tm"]["oos_auroc_maxsum_mean"]
            c["oos_tm_margin_mean"] = rep["tm"]["oos_auroc_margin_mean"]
            c["oos_lr_msp"] = rep["lr"]["oos_auroc_msp"]
            c["R1.2_noninferior"] = c["oos_tm_maxsum_mean"] >= c["oos_lr_msp"] - 0.005
            c["R1.2_better_all_seeds"] = all(a > c["oos_lr_msp"] for a in per)
            if "composite" in rep and "oos_auroc_maxsum" in rep["composite"]:
                c["oos_composite_maxsum"] = rep["composite"]["oos_auroc_maxsum"]
        # read-out arm (R4 only)
        recs = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / "confirm" / reg / cell).glob("seed*.json"))]
        if recs and "readout" in recs[0]:
            import numpy as np

            best_kind = max(("clauses", "clauses+bits"), key=lambda k: sum(r["readout"][k]["val_acc"] for r in recs))
            c["readout_kind"] = best_kind
            c["readout_test_mean"] = sum(r["readout"][best_kind]["test_acc"] for r in recs) / len(recs)
            folder = ROOT / "results" / "confirm" / reg / cell
            seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
            P = [np.load(folder / f"preds_seed{k}.npz") for k in seeds]
            corr = np.stack([p[f"readout_{best_kind}_pred"] == p["y"] for p in P]).astype(float)
            lrp = np.load(folder / "lr_preds.npz")
            lr_corr = (lrp["pred"] == lrp["y"]).astype(float)
            rng = np.random.default_rng(0)
            dd = corr.mean(0) - lr_corr
            bs = dd[rng.integers(0, len(dd), size=(10000, len(dd)))].mean(1)
            c["readout_minus_lr"] = {"diff": float(dd.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
            c["readout_ahead_R4.3"] = bool(np.percentile(bs, 2.5) > 0)
            c["readout_ece_mean"] = sum(r["readout"][best_kind]["calib"]["ece_test"] for r in recs) / len(recs)
            if "oos_auroc_test_msp" in recs[0]["readout"][best_kind]:
                c["readout_oos_msp_mean"] = sum(r["readout"][best_kind]["oos_auroc_test_msp"] for r in recs) / len(recs)
        out["cells"][cell] = c
    out["n_cells"] = len(out["cells"])
    out[f"{reg}.1_composite_cells_ok"] = int(n41)
    out[f"{reg}.2_single_cells_ok"] = int(n42)
    out[f"{reg}.1"] = n41 >= 5
    out[f"{reg}.2_gap_closed"] = n42 >= 5
    out["R1.1_ece_all_cells"] = r11
    return out


def r7() -> dict:
    out = {"datasets": {}}
    n1 = n2 = 0
    for ds in ("clinc150", "banking77", "hwu64"):
        if not (ROOT / "results" / "confirm" / "R7" / ds).exists():
            continue
        a = report("R7", ds, lr=f"{ds}_lrbits", composite=True)  # LR on the same readable bits
        (ROOT / "results" / "confirm" / "R7" / f"report_{ds}_vs_lrbits.json").write_text(json.dumps(a, indent=1))
        b = report("R7", ds, composite=True)  # LR on the raw embedding (the folder's own lr.json)
        c = {"tm_single": a["tm"]["test_acc_mean"], "n_seeds": len(a["tm"]["seeds"]),
             "lr_readable": a["lr"]["test_acc"], "lr_raw": b["lr"]["test_acc"],
             "single_minus_lr_readable": a["tm_minus_lr"], "single_minus_lr_raw": b["tm_minus_lr"],
             "ece_tm_mean": a["tm"]["ece_test_mean"], "stats_first_member": a["tm"]["stats_first_member"][:1]}
        if "composite" in b:
            c["composite"] = b["composite"]["test_acc_mean"]
            c["composite_minus_lr_raw"] = b["composite_minus_lr"]
            n2 += c["composite_minus_lr_raw"]["diff"] >= -0.010
        n1 += c["single_minus_lr_readable"]["diff"] >= -0.005
        if "oos_auroc_maxsum_mean" in b["tm"]:
            c["oos_tm_maxsum_mean"] = b["tm"]["oos_auroc_maxsum_mean"]
            c["oos_lr_raw_msp"] = b["lr"]["oos_auroc_msp"]
            if "composite" in b and "oos_auroc_maxsum" in b["composite"]:
                c["oos_composite_maxsum"] = b["composite"]["oos_auroc_maxsum"]
        out["datasets"][ds] = c
    out["R7.1_cells_ok"] = int(n1)
    out["R7.2_cells_ok"] = int(n2)
    out["R7.1"] = n1 >= 2
    out["R7.2"] = n2 >= 2
    out["R7_met"] = out["R7.1"] and out["R7.2"]
    return out


if __name__ == "__main__":
    reg = sys.argv[1]
    res = {"R2": r2, "R2c": r2c, "R2e": r2e, "R7": r7, "R7b": r7b, "R2d": lambda: r2c("R2d", "B_trim32_rank"),
           "R7c": lambda: r7b("R7c")}.get(reg, lambda: r4(reg))()
    path = ROOT / "results" / "confirm" / reg / f"rules_{reg}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
