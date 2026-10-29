#!/usr/bin/env python
"""W2 H-B part 2 verdict (rules in LOG.md before any run), mechanical.

Scores (val, points): TM = the mean of seeds 1-3 of val_acc (the harness's best-epoch val accuracy, as every W1-S row);
LR = w2_lr.py's val accuracy. Reference = the W1 C4-word table ppw_S1000 (TM: its W1-S rows, the same harness and
seeds; LR: w2_lr.py on its W1-S feature caches).
PASS when, on at least 2 of the 3 intent sets, J - ppw_S1000 >= +0.5 with BOTH probes on that set. Otherwise FAIL.
The pair link (not just the document's topic) is credited only when, on at least 2 of the 3 sets, J - Jshift >= +0.5
with both probes. P (the pair block alone) and potion (test 2's confirmatory existsT rows, seeds 1-3; LR on test 2's
cache) are descriptive only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
SETS = ("clinc150", "banking77", "hwu64")
SHORT = {"clinc150": "c", "banking77": "b", "hwu64": "h"}
EPS = 1e-9


def rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def tm_mean(R: list, tag_of, ds: str):
    vals = []
    for seed in (1, 2, 3):
        hit = [r for r in R if r.get("tag") == tag_of(ds, seed) and r.get("dataset") == ds and r.get("view") == "existsT"]
        if len(hit) != 1:
            return None, []
        vals.append(round(100 * hit[0]["val_acc"], 2))
    return float(np.mean(vals)), vals


def main() -> None:
    w2 = rows(HERE / "results_w2_val.jsonl")
    w1 = rows(EMB / "w1" / "results_w1_val.jsonl")
    conf = rows(EMB / "results_confirm.jsonl")
    lr = rows(HERE / "results_w2_lr.jsonl")
    tm_src = {
        "ppw_S1000": (w1, lambda ds, s: f"W1S_ppw_S1000_d500_e0.5_{ds}_s{s}"),
        "J": (w2, lambda ds, s: f"W2P2_J_{ds}_s{s}"),
        "Jshift": (w2, lambda ds, s: f"W2P2_Jshift_{ds}_s{s}"),
        "P": (w2, lambda ds, s: f"W2P2_P_{ds}_s{s}"),
        "potion": (conf, lambda ds, s: f"T_{SHORT[ds]}_exT_s{s}"),
    }
    out, S = [], {}
    for ds in SETS:
        for name, (R, f) in tm_src.items():
            m, v = tm_mean(R, f, ds)
            S[("TM", name, ds)] = m
            S[("TMseeds", name, ds)] = v
        for name in tm_src:
            hit = [r for r in lr if r["name"] == name and r["dataset"] == ds]
            S[("LR", name, ds)] = 100 * hit[-1]["lr_val"] if hit else None
            if len(hit) > 1:
                out.append(f"note: {len(hit)} LR rows for {name} {ds}; the last one is used")
    missing = [k for k, v in S.items() if v is None and k[1] in ("ppw_S1000", "J", "Jshift")]
    if missing:
        raise SystemExit(f"missing rows: {missing}")
    passed, linked = 0, 0
    for ds in SETS:
        d = {p: S[(p, "J", ds)] - S[(p, "ppw_S1000", ds)] for p in ("TM", "LR")}
        k = {p: S[(p, "J", ds)] - S[(p, "Jshift", ds)] for p in ("TM", "LR")}
        ok = all(x >= 0.5 - EPS for x in d.values())
        lk = all(x >= 0.5 - EPS for x in k.values())
        passed += ok
        linked += lk
        line = [f"{ds}:"]
        for p in ("TM", "LR"):
            vals = " / ".join(f"{n} {S[(p, n, ds)]:.2f}" if S[(p, n, ds)] is not None else f"{n} -" for n in tm_src)
            line.append(f"  {p}: {vals}")
        line.append(f"  J - ppw_S1000: TM {d['TM']:+.2f}, LR {d['LR']:+.2f} -> {'meets' if ok else 'misses'} +0.5 with both")
        line.append(f"  J - Jshift:    TM {k['TM']:+.2f}, LR {k['LR']:+.2f} -> {'meets' if lk else 'misses'} +0.5 with both")
        for p in ("TM", "LR"):
            pot, base = S[(p, "potion", ds)], S[(p, "ppw_S1000", ds)]
            if pot is not None and pot != base:
                line.append(f"  {p} gap to potion closed by J: {(S[(p, 'J', ds)] - base) / (pot - base):+.0%} "
                            f"(ppw_S1000 {base - pot:+.2f} from potion, J {S[(p, 'J', ds)] - pot:+.2f})")
        line.append(f"  TM seeds: " + "; ".join(f"{n} {S[('TMseeds', n, ds)]}" for n in tm_src))
        out += line
    word = "PASS" if passed >= 2 else "FAIL"
    out.append(f"words: J - ppw_S1000 >= +0.5 with both probes on {passed} of 3 sets -> {word}; "
               f"pair link (J - Jshift >= +0.5 with both) on {linked} of 3 -> "
               f"{'credited to the matching answer' if linked >= 2 else 'not credited (a gain may come from the document topic)'}")
    txt = "\n".join(out)
    print(txt)
    p = HERE / "verdict_p2.txt"
    if p.exists():  # never overwrite a result file
        from datetime import datetime, timezone
        p = HERE / f"verdict_p2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
    p.write_text(txt + "\n")


if __name__ == "__main__":
    main()
