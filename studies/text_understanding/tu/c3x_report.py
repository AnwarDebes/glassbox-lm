#!/usr/bin/env python
"""Step 3c follow-ups, the descriptive report (no bar; (a) and (b') post hoc, (b) the plan's follow-up):

    python -I -B c3x_report.py [--c3x tu/c3x]

Reads tu/c3x/results_c3x_val.jsonl and results_c3x_lr.jsonl (rows of c3x_exists_tm.py and c3x_lr.py). Every arm of a
set comes from one k-means (c3x_feats.py; the metas' kmeans_run_id must agree, which this report checks). Arms:
orig2 (the rebuilt originals), F2 (3c-F's kept paraphrases rebuilt in the same run), F_rk (CLINC150 only: F with the
reader-key and order-free copies of evaluation texts removed), twin (5 EDA noisy copies per original, F1 only), rep5
(orig2's rows repeated 5 times: about 3c-F's updates per epoch; TM only). Gains in points, the TM's as the mean of
seed-paired differences over seeds 4-6, unrounded; also each seed SD and each paired difference's SE.
Comparisons: F2 - orig2 (the 3c gain again, one k-means); F_rk - orig2 and F2 - F_rk (a: the copies' share); twin -
orig2 and F2 - twin (b: the paraphrases' content against noisy copies); rep5 - orig2 and F2 - rep5 (b': the updates).
The replication line (the 11:08Z requirement): per set, F2 - orig2 beside 3c's registered gains
(tu/c3/verdict_c3.json), labelled "replication under a new k-means (post hoc; does not change the registered verdict)".
It is "clearly smaller" on a set when its TM gain is below the registered TM gain by more than twice its own paired SE,
or its LR gain is below the registered LR gain by 0.3 points or more; when any set is, the readout's FIRST line says so.
Fixed-C LR (the 11:38Z and 11:39Z requests, descriptive): every comparison's LR gain also at C 1 and at C 10
for both arms (the rows' acc_by_C), beside the best-C value, since Banking77's registered LR gain depends on the C picked
on val per arm.
Writes c3x/report_c3x.json and .txt; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (4, 5, 6)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def tm_rows(rows, d, arm, rep):
    m = {}
    for r in rows:
        if r["dataset"] == d and r.get("arm") == arm and r.get("repeat", 1) == rep and r.get("tag") == "3cx":
            assert r["seed"] not in m, f"two rows for {d} {arm} rep {rep} seed {r['seed']}"
            m[r["seed"]] = 100.0 * float(r["val_acc"])
    return [m[s] for s in SEEDS] if all(s in m for s in SEEDS) else None


def paired(a, b):
    if a is None or b is None:
        return None
    d = [x - y for x, y in zip(a, b)]
    return {"mean": st.mean(d), "per_seed": d, "se": st.stdev(d) / math.sqrt(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c3x", type=Path, default=HERE / "c3x")
    ap.add_argument("--registered", type=Path, default=HERE / "c3" / "verdict_c3.json")
    a = ap.parse_args()
    oj, ot = a.c3x / "report_c3x.json", a.c3x / "report_c3x.txt"
    for p in (oj, ot):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    val, lr = jsonl(a.c3x / "results_c3x_val.jsonl"), jsonl(a.c3x / "results_c3x_lr.jsonl")
    reg = json.loads(a.registered.read_text())
    res, lines, repl, smaller = {}, [], [], []
    for d in SETS:
        runs = set()
        for arm in ("orig2", "F2", "F_rk", "twin"):
            mp = a.c3x / "feats" / arm / f"{d}__p32tok_K512.meta.json"
            if mp.exists():
                runs.add(json.loads(mp.read_text())["kmeans_run_id"])
        tm = {arm: tm_rows(val, d, arm, 1) for arm in ("orig2", "F2", "F_rk", "twin")}
        tm["rep5"] = tm_rows(val, d, "orig2", 5)
        lrv, lrc = {}, {}
        for arm in ("orig2", "F2", "F_rk", "twin"):
            m = [r for r in lr if r["dataset"] == d and r.get("name") == arm]
            lrv[arm] = 100.0 * float(m[0]["lr_val"]) if len(m) == 1 else None
            lrc[arm] = {c: 100.0 * float(v) for c, v in m[0].get("acc_by_C", {}).items()} if len(m) == 1 else None
        comp = {"F2-orig2": ("F2", "orig2"), "F_rk-orig2": ("F_rk", "orig2"), "F2-F_rk": ("F2", "F_rk"),
                "twin-orig2": ("twin", "orig2"), "F2-twin": ("F2", "twin"), "rep5-orig2": ("rep5", "orig2"), "F2-rep5": ("F2", "rep5")}
        g_tm = {k: paired(tm[x], tm[y]) for k, (x, y) in comp.items()}
        g_lr = {k: (lrv[x] - lrv[y]) if x in lrv and y in lrv and lrv[x] is not None and lrv[y] is not None else None
                for k, (x, y) in comp.items()}
        g_lrc = {c: {k: (lrc[x][c] - lrc[y][c]) if x in lrc and y in lrc and lrc[x] and lrc[y] and c in lrc[x] and c in lrc[y]
                     else None for k, (x, y) in comp.items()} for c in ("1.0", "10.0")}
        res[d] = {"one_kmeans": len(runs) == 1, "kmeans_runs": sorted(runs), "tm": tm, "lr": lrv, "lr_by_C": lrc,
                  "lr_gains_fixed_C": g_lrc,
                  "seed_sd": {k: st.stdev(v) for k, v in tm.items() if v is not None}, "tm_gains": g_tm, "lr_gains": g_lr}
        parts = [f"{k} TM {g_tm[k]['mean']:+.2f} (SE {g_tm[k]['se']:.2f})" + (f", LR {g_lr[k]:+.2f}" if g_lr[k] is not None else "")
                 for k in comp if g_tm[k] is not None]
        lines.append(f"{d}: one k-means {len(runs) == 1}; " + "; ".join(parts))
        fixed = [f"{k} LR at C 1 {g_lrc['1.0'][k]:+.2f}, at C 10 {g_lrc['10.0'][k]:+.2f} (best-C {g_lr[k]:+.2f})"
                 for k in comp if g_lrc["1.0"][k] is not None and g_lrc["10.0"][k] is not None and g_lr[k] is not None]
        if fixed:
            lines.append(f"{d} fixed-C LR: " + "; ".join(fixed))
        r_tm, r_lr = float(reg[d]["tm_gain"]), float(reg[d]["lr_gain"])
        f_tm, f_lr = g_tm["F2-orig2"], g_lr["F2-orig2"]
        if f_tm is not None and f_lr is not None:
            cs = (r_tm - f_tm["mean"] > 2 * f_tm["se"]) or (r_lr - f_lr >= 0.3 - 1e-9)
            res[d]["replication"] = {"registered_tm": r_tm, "registered_lr": r_lr, "F2_minus_orig2_tm": f_tm["mean"],
                                     "F2_minus_orig2_tm_se": f_tm["se"], "F2_minus_orig2_lr": f_lr, "clearly_smaller": cs}
            repl.append(f"{d}: registered TM {r_tm:+.2f} / LR {r_lr:+.2f}; F2 - orig2 TM {f_tm['mean']:+.2f} (SE {f_tm['se']:.2f}) "
                        f"/ LR {f_lr:+.2f}" + (" (clearly smaller)" if cs else ""))
            if cs:
                smaller.append(d)
    out = {"label": "3c follow-ups, descriptive: (a) and (b') post hoc, (b) the plan's follow-up", "sets": res,
           "replication_clearly_smaller_on": smaller}
    first = (f"REPLICATION CLEARLY SMALLER on {', '.join(smaller)}: F2 - orig2 under a new k-means is below 3c's registered gain"
             if smaller else "Replication: F2 - orig2 under a new k-means is not clearly smaller than 3c's registered gain on any set")
    lines = [first, "replication under a new k-means (post hoc; does not change the registered verdict):"] + repl + lines
    lines.append("descriptive only: no bar; val only")
    with open(oj, "x") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    with open(ot, "x") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
