#!/usr/bin/env python
"""W3 step 1 verdict (rules in LOG.md before any run), mechanical.

Scores (intent val, points): TM = the mean of seeds 1-3 of val_acc (the harness's best-epoch val accuracy); LR =
w3_lr.py's val accuracy (the reference rows of ppw_S1000 and potion are w2_lr.py's, the same code, in
w2/results_w2_lr.jsonl). Tables: CF (counter-fitting with Open English WordNet 2025 synonyms and antonyms, the
primary arm), RF (retrofitting, descriptive), CFrand (counter-fitting with the same-degree random graphs, the
control), the reference ppw_S1000 (W1-S rows) and potion (test 2's confirmatory rows).
PASS when, on at least 2 of the 3 sets, CF - ppw_S1000 >= +0.5 with BOTH probes AND CF - CFrand >= +0.3 with BOTH
probes, on the same set. Otherwise FAIL. RF and the share of the gap to potion are descriptive.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
SETS = ("clinc150", "banking77", "hwu64")
SHORT = {"clinc150": "c", "banking77": "b", "hwu64": "h"}
EPS = 1e-9


def rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def tm_mean(R, tag_of, ds):
    vals = []
    for seed in (1, 2, 3):
        hit = [r for r in R if r.get("tag") == tag_of(ds, seed) and r.get("dataset") == ds and r.get("view") == "existsT"]
        if len(hit) != 1:
            return None, []
        vals.append(round(100 * hit[0]["val_acc"], 2))
    return float(np.mean(vals)), vals


def main() -> None:
    w3 = rows(HERE / "results_w3_val.jsonl")
    w1 = rows(EMB / "w1" / "results_w1_val.jsonl")
    conf = rows(EMB / "results_confirm.jsonl")
    lr = rows(HERE / "results_w3_lr.jsonl") + rows(EMB / "w2" / "results_w2_lr.jsonl")
    tm_src = {"ppw_S1000": (w1, lambda ds, s: f"W1S_ppw_S1000_d500_e0.5_{ds}_s{s}"),
              "CF": (w3, lambda ds, s: f"W3_CF_{ds}_s{s}"), "RF": (w3, lambda ds, s: f"W3_RF_{ds}_s{s}"),
              "CFrand": (w3, lambda ds, s: f"W3_CFrand_{ds}_s{s}"), "potion": (conf, lambda ds, s: f"T_{SHORT[ds]}_exT_s{s}")}
    S, out = {}, []
    for ds in SETS:
        for name, (R, f) in tm_src.items():
            S[("TM", name, ds)], S[("TMseeds", name, ds)] = tm_mean(R, f, ds)
            hit = [r for r in lr if r["name"] == name and r["dataset"] == ds]
            S[("LR", name, ds)] = 100 * hit[0]["lr_val"] if hit else None  # w3 rows first, then w2's references
    missing = [k for k, v in S.items() if v is None and k[1] in ("ppw_S1000", "CF", "CFrand")]
    if missing:
        raise SystemExit(f"missing rows: {missing}")
    passed = 0
    for ds in SETS:
        d = {p: S[(p, "CF", ds)] - S[(p, "ppw_S1000", ds)] for p in ("TM", "LR")}
        c = {p: S[(p, "CF", ds)] - S[(p, "CFrand", ds)] for p in ("TM", "LR")}
        ok = all(x >= 0.5 - EPS for x in d.values()) and all(x >= 0.3 - EPS for x in c.values())
        passed += ok
        out.append(f"{ds}:")
        for p in ("TM", "LR"):
            out.append(f"  {p}: " + " / ".join(f"{n} {S[(p, n, ds)]:.2f}" if S[(p, n, ds)] is not None else f"{n} -" for n in tm_src))
        out.append(f"  CF - ppw_S1000: TM {d['TM']:+.2f}, LR {d['LR']:+.2f}; CF - CFrand: TM {c['TM']:+.2f}, LR {c['LR']:+.2f}"
                   f" -> {'meets' if ok else 'misses'} the bar")
        for n in ("CF", "RF"):
            for p in ("TM", "LR"):
                pot, base, v = S[(p, "potion", ds)], S[(p, "ppw_S1000", ds)], S[(p, n, ds)]
                if None not in (pot, base, v) and pot != base:
                    out.append(f"  {n} {p}: {v - base:+.2f} over ppw_S1000, {(v - base) / (pot - base):+.0%} of the gap to potion")
        out.append("  TM seeds: " + "; ".join(f"{n} {S[('TMseeds', n, ds)]}" for n in tm_src))
    word = "PASS" if passed >= 2 else "FAIL"
    out.append(f"words: CF beats ppw_S1000 by +0.5 and CFrand by +0.3 with both probes on {passed} of 3 sets -> {word}")
    txt = "\n".join(out)
    print(txt)
    p = HERE / "verdict_w3.txt"
    if p.exists():
        p = HERE / f"verdict_w3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
    p.write_text(txt + "\n")


if __name__ == "__main__":
    main()
