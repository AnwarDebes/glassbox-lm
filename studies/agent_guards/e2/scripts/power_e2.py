"""Power of E2's rules (registration rev2, section 9.3; review item 4). Decision-layer venv, CPU, seconds:

    .venv-dl/bin/python -I -B scripts/power_e2.py --standin results/offline_e2e_standin.json --out <json>

- (b) and E2.4: the spread of the REGISTERED statistic (report_e2.paired, rng 20261025) on the offline stand-in results
  (scripted agents; the only paired data that exist before E2-1): the gain needed for the interval's lower end to
  clear the bar (point minus lower end), and the true gain needed for 80% power, (1.96 + 0.84) x SE with
  SE = (point - lower end) / 1.96.
- E2.2: the upper end of D4's A1 ASR interval, normal approximation with the design effect seen on the stand-in
  (cluster bootstrap variance / binomial variance), simulated over true ASR values (numpy rng 20261026).
- E2.1(a): "D4 has no more A3 successes than D3r on the same targets", simulated for 40 and 80 targets with
  independent targets; with targets that share an attackability (both arms' success probabilities scaled by one
  Beta(1, 3) draw per target); and coupled (on half the targets both arms' outcomes come from one uniform draw), over
  D3r's ASR and D4's excess (rng 20261026).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "scripts"))
import report_e2 as R

Z975, Z80 = 1.959964, 0.841621


def from_interval(d: dict, bar: float) -> dict:
    lower = d["point"] - d["lo"]
    se = lower / Z975
    return {"point": d["point"], "lo": d["lo"], "hi": d["hi"], "lower_half_width": lower, "se": se,
            "point_needed_for_lower_end_at_bar": bar + lower, "true_gain_for_80pct_power": bar + (Z975 + Z80) * se}


def sim_rule_a(rng, n_targets: int, p_base: float, delta: float, shared, reps: int = 10000) -> float:
    if shared == "coupled":  # on each target, with probability 1/2, both arms draw from ONE uniform (same agent, same
        # candidates: the outcome differs only where the guards differ); otherwise independently
        u = rng.random((reps, n_targets))
        same = rng.random((reps, n_targets)) < 0.5
        ur = np.where(same, u, rng.random((reps, n_targets)))
        xr = (ur < p_base).sum(1)
        x4 = (u < min(1.0, p_base + delta)).sum(1)
        return float((x4 <= xr).mean())
    if shared:
        a = rng.beta(1, 3, size=(reps, n_targets)) * 4  # mean 1: a per-target attackability multiplier
        pr = np.clip(p_base * a, 0, 1)
        p4 = np.clip((p_base + delta) * a, 0, 1)
    else:
        pr = np.full((reps, n_targets), p_base)
        p4 = np.full((reps, n_targets), min(1.0, p_base + delta))
    xr = (rng.random((reps, n_targets)) < pr).sum(1)
    x4 = (rng.random((reps, n_targets)) < p4).sum(1)
    return float((x4 <= xr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eps = R.standin_episodes(a.standin)
    out: dict = {"note": "power from the STAND-IN results (scripted agents); Qwen's variance will differ"}
    b = R.paired(eps, "d4_tm", "d3r", R.A1, "utility")
    out["b_vs_d3r"] = {**from_interval(b, 0.0), "also_needs_point_at_least": R.B_GAIN}
    out["b_vs_d3_already_informed"] = from_interval(R.paired(eps, "d4_tm", "d3", R.A1, "utility"), 0.0)
    out["b_vs_d6_scripted"] = from_interval(R.paired(eps, "d4_tm", "d6_rules", R.A1, "utility"), 0.0)
    e24 = R.e2_4(eps)
    out["e2_4_vs_" + e24["better_twin"]] = from_interval(e24["stat"], R.E24_MARGIN)
    # chance that (b) passes when the true gain equals the stand-in's point, under a normal approximation
    from math import erf, sqrt

    def phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    bb = out["b_vs_d3r"]
    out["b_vs_d3r"]["pass_prob_if_true_gain_equals_standin_point"] = \
        1 - phi((max(bb["lower_half_width"], R.B_GAIN) - bb["point"]) / bb["se"])
    # E2.2: design effect, then the pass probability over true ASR values
    r = R.rate(eps, "d4_tm", R.A1, "attack")
    p, n = r["point"] / 100, r["episodes"]
    se_boot = (r["hi"] - r["point"]) / 100 / Z975
    deff = se_boot ** 2 / (p * (1 - p) / n)
    rng = np.random.default_rng(20261026)
    e22 = {}
    for p_true in (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02):
        ph = rng.binomial(n, p_true, size=20000) / n
        upper = ph + Z975 * np.sqrt(np.maximum(ph * (1 - ph), 1e-12) / n * deff)
        e22[f"{100 * p_true:.2f}%"] = float((upper <= R.E22_UPPER / 100).mean())
    # the largest point estimate that still passes
    grid = np.arange(0, 0.03, 0.0001)
    ok = grid[grid + Z975 * np.sqrt(grid * (1 - grid) / n * deff) <= R.E22_UPPER / 100]
    out["e2_2"] = {"standin_asr": r["point"], "standin_interval": [r["lo"], r["hi"]], "design_effect": deff,
                   "largest_passing_point_estimate_pct": float(100 * ok.max()) if len(ok) else None,
                   "pass_prob_by_true_asr": e22}
    # E2.1(a)
    sims = {}
    for n_t in (40, 80):
        for shared in (False, True, "coupled"):
            for p_base in (0.10, 0.25):
                for delta in (0.0, 0.025, 0.05, 0.10):
                    sims[f"targets={n_t}|shared={shared}|D3r_asr={p_base}|D4_excess={delta}"] = \
                        sim_rule_a(rng, n_t, p_base, delta, shared)
    out["rule_a_pass_prob"] = sims
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rule_a_pass_prob"}, indent=1))
    for k, v in sims.items():
        print(f"{k:60} {v:.3f}")


if __name__ == "__main__":
    main()
