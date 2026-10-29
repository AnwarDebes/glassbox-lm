"""Exact power of E4.3 (registration revision 3; the decision 1 of 2026-10-21T07:24:12Z; re-review item 0).

E4.3 is LOW POWER. This script states how low, before any Qwen row exists, and gives gobar.py the function that puts
the power at Qwen's observed go-bar rates into gobar.json at STAGE=train_guard.

Model (the review's reading of the go-bar as per-task rates): each of the 40 test tasks flips independently,
  +1 (A2 seed 1 passes, A1 fails) with probability g, -1 (A2 fails, A1 passes) with probability l, 0 otherwise;
  g = G / 74 and l = L / 74 from the go-bar's train-task counts. Both are upper bounds: a learned denial on a task need
  not flip it (G counts failed tasks with a denied non-gold write, L passed tasks with a denied gold write).
Rule (registered, summarize_e4.paired): E4.3 met iff diff >= +2.0 points and the 95% interval's lower end > 0.
need(a) = the smallest number of favourable flips that meets the rule with a flips against, from the registered
  statistic with power_table.py's layout (+1 on the first tasks, then -1, then 0). a = 0..6 must equal
  results/power_table_rev2.json (asserted); a >= 7 comes from the same function (power_table.min_favourable).
EXACT power = sum over (f, a) of the trinomial probability of (f favourable, a against) x [f >= need(a)].
A Monte Carlo check runs the registered statistic itself on random layouts (the interval depends slightly on where
the flips sit, since the resampled indices are fixed by the seed); it is reported beside the exact numbers.
Reported (--out): need(a); the power at the go-bar's threshold points (G = 8 with L = 0..4, then G - L = 4 up to
G = 24), at the review's net-rate reading (G - L = 4 of 74 with nothing against), at the stand-in's observed go-bar
counts (gpt-4.1 trial 0 as A1: G 3, L 5); the net effect needed for 80% power with l = 0, l = 0.025 and l = 4/74.
Run (CPU, seconds): offline_env.sh .../taubench/.venv/bin/python -I -B power_e43.py --out results/power_e43_rev3.json
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

N_TEST, N_TRAIN = 40, 74
GO_G_MIN, GO_NET_MIN = 8, 4


def need_table(n: int = N_TEST, min_pts: float | None = None) -> list:
    """need[a] for a = 0..n (None when no count of favourable flips meets the rule)."""
    from power_table import min_favourable
    from summarize_e4 import E43_MIN

    mp = E43_MIN if min_pts is None else min_pts
    return [min_favourable(n, a, mp)["favourable_needed"] for a in range(0, n + 1)]


def power(g: float, l: float, need: list, n: int = N_TEST) -> float:  # noqa: E741
    """Exact P(rule met) when each task is +1 w.p. g, -1 w.p. l, 0 otherwise (independent tasks)."""
    if g < 0 or l < 0 or g + l > 1 + 1e-12:
        raise ValueError(f"rates out of range: g {g}, l {l}")
    z = max(0.0, 1.0 - g - l)
    tot = 0.0
    for a in range(0, n + 1):
        fa = need[a]
        if fa is None:
            continue
        pa = math.comb(n, a) * (l ** a)
        for f in range(fa, n - a + 1):
            tot += pa * math.comb(n - a, f) * (g ** f) * (z ** (n - a - f))
    return float(min(1.0, tot))


def power_at_counts(G: int, L: int, need: list, n_train: int = N_TRAIN, n: int = N_TEST) -> float:  # noqa: N803
    return power(G / n_train, L / n_train, need, n)


def effect_for_power(target: float, l: float, need: list, n: int = N_TEST) -> dict:  # noqa: E741
    """The smallest favourable rate g (bisection; power rises with g at fixed l) with power >= target; the net effect
    g - l in points and in tasks of 40."""
    lo, hi = l, 1.0 - l
    if power(hi, l, need, n) < target:
        return {"l": l, "reachable": False}
    for _ in range(60):
        mid = (lo + hi) / 2
        if power(mid, l, need, n) >= target:
            hi = mid
        else:
            lo = mid
    return {"l": l, "g": hi, "net_points": 100 * (hi - l), "net_tasks_of_40": n * (hi - l),
            "power": power(hi, l, need, n), "reachable": True}


def power_mc(g: float, l: float, draws: int, seed: int, n: int = N_TEST) -> dict:  # noqa: E741
    """The registered statistic on random layouts: each draw sets every task to +1/-1/0 by the trinomial, then
    summarize_e4.paired decides (diff >= +2.0 and lower end > 0)."""
    import numpy as np

    from summarize_e4 import E43_MIN, boot_ci

    rng = np.random.default_rng(seed)
    met = 0
    for _ in range(draws):
        u = rng.random(n)
        d = np.where(u < g, 100.0, np.where(u < g + l, -100.0, 0.0))
        lo, _hi = boot_ci(d)
        met += int(d.mean() >= E43_MIN and lo > 0)
    p = met / draws
    return {"draws": draws, "seed": seed, "met": met, "power": p, "se": math.sqrt(p * (1 - p) / draws)}


def threshold_points() -> list[tuple[int, int]]:
    """The least favourable GO points: G = 8 with L = 0..4, then G - L = 4 for G = 9..24."""
    pts = [(GO_G_MIN, L) for L in range(0, GO_G_MIN - GO_NET_MIN + 1)]
    pts += [(G, G - GO_NET_MIN) for G in range(GO_G_MIN + 1, 25)]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--power-table", default=str(E4 / "results/power_table_rev2.json"))
    ap.add_argument("--mc-draws", type=int, default=2000)
    ap.add_argument("--mc-seed", type=int, default=20261021)
    a = ap.parse_args()
    t0 = time.time()
    need = need_table()
    pt = json.loads(Path(a.power_table).read_text())["E4.3_40_tasks"]
    for r in pt:                                          # the published table and this one must agree
        assert need[r["against"]] == r["favourable_needed"], (r, need[r["against"]])
    res = {"what": "E4.3 power (low power); trinomial model, g = G/74, l = L/74 (upper bounds); registered rule",
           "statistic": "summarize_e4.boot_ci (10,000 resamples, default_rng(20261021), 95% percentile)",
           "rule": "met iff diff >= +2.0 points and the interval's lower end > 0 (40 test tasks)",
           "need_favourable_by_against": need,
           "agrees_with": {"file": a.power_table, "against": [r["against"] for r in pt]}}
    res["go_bar_threshold"] = [{"G": G, "L": L, "g": G / N_TRAIN, "l": L / N_TRAIN,
                                "power": power_at_counts(G, L, need)} for G, L in threshold_points()]
    corner = power_at_counts(8, 4, need)
    best = power_at_counts(8, 0, need)
    worst = min(r["power"] for r in res["go_bar_threshold"])
    res["at_go_bar_corner_G8_L4"] = corner
    res["at_go_bar_G8_L0"] = best
    res["min_over_threshold_points"] = worst
    res["reviewer_net_rate_reading_4_of_74_nothing_against"] = power(4 / N_TRAIN, 0.0, need)
    res["standin_observed_G3_L5"] = power_at_counts(3, 5, need)
    res["effect_for_80pct_power"] = [effect_for_power(0.80, l, need) for l in (0.0, 0.025, 4 / N_TRAIN)]
    res["monte_carlo_check"] = {
        "G8_L4": {"exact": corner, **power_mc(8 / N_TRAIN, 4 / N_TRAIN, a.mc_draws, a.mc_seed)},
        "G8_L0": {"exact": best, **power_mc(8 / N_TRAIN, 0.0, a.mc_draws, a.mc_seed + 1)},
        "net_4_of_74": {"exact": res["reviewer_net_rate_reading_4_of_74_nothing_against"],
                        **power_mc(4 / N_TRAIN, 0.0, a.mc_draws, a.mc_seed + 2)},
        "80pct_l0": {"exact": res["effect_for_80pct_power"][0]["power"],
                     **power_mc(res["effect_for_80pct_power"][0]["g"], 0.0, a.mc_draws, a.mc_seed + 3)}}
    res["seconds"] = round(time.time() - t0, 1)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(res, indent=1))
    show = {k: v for k, v in res.items() if k not in ("go_bar_threshold", "need_favourable_by_against")}
    print(json.dumps(show, indent=1))
    print("need(a) for a = 0..12:", need[:13])
    print("threshold points:", [(r["G"], r["L"], round(r["power"], 4)) for r in res["go_bar_threshold"]])


if __name__ == "__main__":
    main()
