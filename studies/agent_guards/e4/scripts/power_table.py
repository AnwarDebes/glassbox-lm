"""Power of the registered statistic (no data; review issue 4): with the registered bootstrap (summarize_e4.paired:
10,000 resamples, default_rng(20261021), 95% percentile interval over tasks), how many task flips does each bar need?

A "flip" is a task whose DB outcome differs between the two arms: +1 (favourable to the first arm) or -1 (against).
The pattern is laid out as +1 on the first f tasks, -1 on the next b tasks, 0 on the rest (the interval depends on the
counts and, slightly, on the positions, since the resampled indices are fixed by the seed).
  E4.2 (114 tasks): met iff diff >= +3.0 points and the lower end > 0;
  E4.3 (40 tasks):  met iff diff >= +2.0 points and the lower end > 0;
  E4.5 (40 tasks):  non-inferior iff the lower end > -5.0 points (A2 - A3; b = tasks where A3 is better), and
                    "not informative" when fewer than 4 tasks differ.
Output: for b = 0..6 tasks against, the smallest f that meets the rule, with its diff and interval.
Run: .../taubench/.venv/bin/python -I -B power_table.py --out results/power_table_rev2.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4 / "scripts"))

import numpy as np  # noqa: E402

from summarize_e4 import E42_MIN, E43_MIN, E45_MARGIN, E45_MIN_DIFFER, boot_ci  # noqa: E402


def pattern(n, f, b):
    d = np.zeros(n)
    d[:f] = 100.0
    d[f:f + b] = -100.0
    return d


def min_favourable(n, b, min_pts, fmax=None):
    for f in range(0, (fmax or n - b) + 1):
        d = pattern(n, f, b)
        lo, hi = boot_ci(d)
        if d.mean() >= min_pts and lo > 0:
            return {"against": b, "favourable_needed": f, "diff_points": round(float(d.mean()), 2),
                    "ci95": [round(lo, 2), round(hi, 2)]}
    return {"against": b, "favourable_needed": None}


def noninferior_table(n=40, max_a3=8):
    out = []
    for a2 in range(0, 9):
        for a3 in range(0, max_a3 + 1):
            d = pattern(n, a2, a3)
            lo, hi = boot_ci(d)
            differ = a2 + a3
            if differ < E45_MIN_DIFFER:
                v = "not informative"
            else:
                v = "non-inferior" if lo > E45_MARGIN else "not shown"
            out.append({"A2_better": a2, "A3_better": a3, "diff_points": round(float(d.mean()), 2),
                        "ci95": [round(lo, 2), round(hi, 2)], "verdict": v})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"statistic": "summarize_e4.boot_ci (10,000 resamples, default_rng(20261021), 95% percentile)",
           "E4.2_114_tasks": [min_favourable(114, b, E42_MIN) for b in range(0, 7)],
           "E4.3_40_tasks": [min_favourable(40, b, E43_MIN) for b in range(0, 7)],
           "E4.5_40_tasks_margin_-5": noninferior_table()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k in ("E4.2_114_tasks", "E4.3_40_tasks"):
        print(k, json.dumps(res[k]))
    print("E4.5: smallest A2_better for non-inferior, by A3_better:",
          {b: min((r["A2_better"] for r in res["E4.5_40_tasks_margin_-5"] if r["A3_better"] == b
                   and r["verdict"] == "non-inferior"), default=None) for b in range(0, 9)})


if __name__ == "__main__":
    main()
