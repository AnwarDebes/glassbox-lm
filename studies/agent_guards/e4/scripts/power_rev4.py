"""E4.2's power and E4.3's whole go-bar boundary (registration revision 4; round-3 review items m1 and m5(a)).

m1 (E4.2 had no 80% power figure). The model of scripts/power_e43.py, on E4.2's 114 tasks: each task flips
  independently, +1 (A1 passes, A0 fails) with probability g, -1 (A1 fails, A0 passes) with probability l, 0 otherwise.
  Rule (registered, summarize_e4.paired): E4.2 met iff diff >= +3.0 points and the 95% interval's lower end > 0.
  need42(a) = the smallest number of favourable flips that meets the rule with a flips against, from the registered
  statistic with power_table.py's layout (+1 on the first tasks, then -1, then 0). The search starts at the first f
  whose mean reaches +3.0 points (f - a >= 4 of 114): a smaller f fails the mean part of the rule, so the result is
  the one power_table.min_favourable gives, without the bootstrap calls that cannot matter. a = 0..6 must equal
  results/power_table_rev2.json's E4.2 column (asserted).
  EXACT power = the trinomial sum of power_e43.power with n = 114. Reported: the net effect g - l needed for 80% power
  with l = 0, 0.025 and 0.05 (the round-3 review's three points), and a Monte Carlo check of the 80% point with
  nothing against (the registered statistic on random layouts).
m5(a) (the go-bar boundary). GO iff G >= 8 and G - L >= 4, where G counts A1-failed train tasks and L A1-passed ones, so
  G + L <= 74: the boundary G - L = 4 runs from G 8 (L 4) to G 39 (L 35). E4.3's exact power (power_e43.power_at_counts,
  the pinned need table of results/power_e43_rev3.json) at every boundary point, with the smallest.
Run (CPU, about a minute): offline_env.sh .../taubench/.venv/bin/python -I -B power_rev4.py --out results/power_rev4.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import math
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4 / "scripts"))

N_BASE = 114
GO_G_MIN, GO_NET_MIN, N_TRAIN = 8, 4, 74


def need42_table(n: int = N_BASE, min_pts: float | None = None) -> list:
    """need42[a] for a = 0..n (None when no count of favourable flips meets E4.2's rule)."""
    import numpy as np

    from power_table import pattern
    from summarize_e4 import E42_MIN, boot_ci

    mp = E42_MIN if min_pts is None else min_pts
    out = []
    for a in range(0, n + 1):
        found = None
        f0 = next((f for f in range(0, n - a + 1) if pattern(n, f, a).mean() >= mp), None)
        for f in range(f0, n - a + 1) if f0 is not None else ():
            d = pattern(n, f, a)
            lo, _hi = boot_ci(d)
            if float(np.mean(d)) >= mp and lo > 0:
                found = f
                break
        out.append(found)
    return out


def boundary_points() -> list[tuple[int, int]]:
    """(G, L) on G - L = 4 with G >= 8 and G + L <= 74 (G 8..39)."""
    return [(G, G - GO_NET_MIN) for G in range(GO_G_MIN, N_TRAIN + 1) if G + (G - GO_NET_MIN) <= N_TRAIN]


def main():
    import power_e43 as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--power-table", default=str(E4 / "results/power_table_rev2.json"))
    ap.add_argument("--power-e43", default=str(E4 / "results/power_e43_rev3.json"))
    ap.add_argument("--mc-draws", type=int, default=2000)
    ap.add_argument("--mc-seed", type=int, default=20261021)
    a = ap.parse_args()
    t0 = time.time()
    need42 = need42_table()
    pt = json.loads(Path(a.power_table).read_text())["E4.2_114_tasks"]
    for r in pt:                                         # the published table and this one must agree
        assert need42[r["against"]] == r["favourable_needed"], (r, need42[r["against"]])
    res = {"what": "E4.2 power (114 tasks) and E4.3's whole go-bar boundary; trinomial model; registered rules",
           "statistic": "summarize_e4.boot_ci (10,000 resamples, default_rng(20261021), 95% percentile)",
           "E4.2": {"rule": "met iff diff >= +3.0 points and the interval's lower end > 0 (114 tasks)",
                    "need_favourable_by_against": need42,
                    "agrees_with": {"file": a.power_table, "against": [r["against"] for r in pt]}}}
    eff = []
    for l in (0.0, 0.025, 0.05):  # noqa: E741
        e = P.effect_for_power(0.80, l, need42, n=N_BASE)
        e.pop("net_tasks_of_40", None)                   # power_e43's key names E4.3's 40 tasks
        if e.get("reachable"):
            e["net_tasks_of_114"] = N_BASE * (e["g"] - l)
        eff.append(e)
    res["E4.2"]["effect_for_80pct_power"] = eff
    e0 = eff[0]
    import numpy as np

    from summarize_e4 import E42_MIN, boot_ci

    rng = np.random.default_rng(a.mc_seed)
    met = 0
    for _ in range(a.mc_draws):
        u = rng.random(N_BASE)
        d = np.where(u < e0["g"], 100.0, 0.0)
        lo, _hi = boot_ci(d)
        met += int(d.mean() >= E42_MIN and lo > 0)
    p = met / a.mc_draws
    res["E4.2"]["monte_carlo_check_80pct_l0"] = {"exact": e0["power"], "draws": a.mc_draws, "seed": a.mc_seed,
                                                 "met": met, "power": p, "se": math.sqrt(p * (1 - p) / a.mc_draws)}
    need43 = json.loads(Path(a.power_e43).read_text())["need_favourable_by_against"]
    assert need43 == P.need_table(), "results/power_e43_rev3.json's need table differs from power_e43.need_table()"
    bnd = [{"G": G, "L": L, "power": P.power_at_counts(G, L, need43)} for G, L in boundary_points()]
    low = min(bnd, key=lambda r: r["power"])
    res["E4.3_go_bar_boundary"] = {"what": "G - L = 4, G >= 8, G + L <= 74 (G counts A1-failed, L A1-passed train "
                                           "tasks)", "points": bnd, "first": bnd[0], "last": bnd[-1], "lowest": low,
                                   "need_from": a.power_e43}
    res["seconds"] = round(time.time() - t0, 1)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("need42(a) for a = 0..12:", need42[:13])
    print("E4.2 80% power:", [(e["l"], round(e.get("net_points", float("nan")), 2),
                               round(e.get("net_tasks_of_114", float("nan")), 2)) for e in eff])
    print("E4.2 MC at the 80% point (l = 0):", res["E4.2"]["monte_carlo_check_80pct_l0"])
    print("E4.3 boundary:", [(r["G"], r["L"], round(r["power"], 4)) for r in bnd])
    print("lowest:", low, "seconds", res["seconds"])


if __name__ == "__main__":
    main()
