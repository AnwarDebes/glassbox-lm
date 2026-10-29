"""Statistics and gates of A-Struct v2 (PLAN_part3.md section 3), as pure functions on per-position bits.

Paired bootstrap: per-position differences d (arm minus reference, bits per token); the dialogues are resampled
with replacement (10,000 resamples, seed 0) and the statistic is sum(d) / tokens of the resample (the same
estimator as tally floor/stats.block_bootstrap). Gates use 95% intervals; the PASS lines use 97.5% (Bonferroni,
two co-primary arms).
Margins in bits per token: 0.25% perplexity = 0.0036, 0.5% = 0.0072, 1% = 0.0145, 2% = 0.0291, 3% = 0.0439.
"""

from __future__ import annotations

import numpy as np

M_025, M_05, M_1, M_2, M_3 = 0.0036, 0.0072, 0.0145, 0.0291, 0.0439
K1_REL = 1e-3
K9_TOL = 0.0010
N_BOOT, BOOT_SEED = 10_000, 0


def paired_bootstrap(d: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                     level: float = 0.95) -> dict:
    d = np.asarray(d, np.float64)
    ug, inv = np.unique(np.asarray(groups), return_inverse=True)
    inv = inv.reshape(-1)
    S = np.bincount(inv, weights=d)
    n = np.bincount(inv).astype(np.float64)
    ng = len(ug)
    rng = np.random.default_rng(seed)
    stat = np.empty(n_boot)
    for s in range(0, n_boot, 1000):  # chunks keep the weight matrix small
        m = min(1000, n_boot - s)
        Wm = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=m).astype(np.float64)
        stat[s:s + m] = (Wm @ S) / (Wm @ n)
    a = (1 - level) / 2
    lo, hi = np.quantile(stat, [a, 1 - a])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level, "n_boot": n_boot,
            "seed": seed, "groups": int(ng), "tokens": len(d), "boot_sd": float(stat.std(ddof=1))}


def k1(rel_ppl: float) -> dict:
    """K1: |PPL_own / PPL_kenlm - 1| must not exceed 1e-3 on tune-hold."""
    return {"rel_ppl": float(rel_ppl), "limit": K1_REL, "pass": bool(abs(rel_ppl) <= K1_REL)}


def k3(d_trees: float, d_lr2: float, d_lr1: float) -> dict:
    """K3 on tune-hold, per arm (d = arm mix minus B, bits per token; negative = better than B).
    (i) the better of trees and LR2 at least 0.0439 below B; (ii) LR1 at least 0.0291 below B."""
    best = min(d_trees, d_lr2)
    i_ok = best <= -M_3
    ii_ok = d_lr1 <= -M_2
    return {"d_trees": d_trees, "d_lr2": d_lr2, "d_lr1": d_lr1, "best_trees_lr2": best,
            "i_pass": bool(i_ok), "ii_pass": bool(ii_ok), "pass": bool(i_ok and ii_ok)}


def k3c(d_pilot: float, d_trees: float) -> dict:
    """K3c: stop the arm if B minus its better pilot mix is less than a third of B minus its trees mix.
    Amendment 1 (2026-10-21, before any run): the pilot must also be better than B (B minus pilot > 0). Without it, an
    arm that passed K3 (i) through LR2 alone, with its trees mix worse than B, would pass K3c with a pilot worse than B
    (a third of a negative gain is negative)."""
    gain_pilot, gain_trees = -d_pilot, -d_trees
    beats_b = gain_pilot > 0
    third = gain_pilot >= gain_trees / 3.0
    return {"gain_pilot": gain_pilot, "gain_trees": gain_trees, "share": (gain_pilot / gain_trees) if gain_trees else None,
            "pilot_beats_B": bool(beats_b), "third_of_trees_gain": bool(third), "pass": bool(beats_b and third)}


def k8(boot: dict) -> dict:
    """K8 (A/A), arithmetic: the difference within +-0.0036 bits per token, with its 95% interval covering 0.
    Amendment 1 (2026-10-21, before any run): the registered difference is the equal-size A/A, rebuilt(seed 2) minus
    rebuilt(seed 3) (two rebuilds of B, each on 95% of the fit dialogues with its own fold assignment); B minus a
    rebuild also holds the effect of counting on 5% less data and is reported with no rule (s2c_aa.py)."""
    ok = abs(boot["mean"]) <= M_025 and boot["lo"] <= 0 <= boot["hi"]
    return {"mean": boot["mean"], "lo": boot["lo"], "hi": boot["hi"], "pass": bool(ok)}


def k9(b: float, d_hold: float) -> dict:
    """K9: a learner trained on shuffled labels gets b = 0, or a tune-hold mix no more than 0.0010 below B."""
    ok = (b == 0.0) or (d_hold >= -K9_TOL)
    return {"b": b, "d_hold": d_hold, "pass": bool(ok)}


def k2(contracts: dict) -> dict:
    bad = [k for k, v in contracts.items() if not v["pass"]]
    return {"checked": len(contracts), "failed": bad, "pass": not bad}


def k4b(best_tunecal_tm_mix_minus_B: float) -> dict:
    """K4b (GPU stage): the best tune-cal TM mix of the T 8,000 grid at least 0.0145 below B, else stop."""
    return {"d": best_tunecal_tm_mix_minus_B, "pass": bool(best_tunecal_tm_mix_minus_B <= -M_1)}


def pass_lines(d_B: dict, seeds_d_B: list, d_c0: dict, d_e1: dict, d_tree: dict, d_banks: dict, banks_b: dict) -> dict:
    """The four PASS conditions of one co-primary arm on the final split (97.5% intervals; 'TM' = the 10-seed mean of
    per-position bits). d_* are bootstrap dicts of TM minus the named comparator."""
    a = d_B["mean"] <= -M_2 and d_B["hi"] < 0 and all(x < 0 for x in seeds_d_B)
    b = d_c0["mean"] <= -M_1 and d_e1["mean"] <= -M_1 and d_c0["hi"] < 0 and d_e1["hi"] < 0
    c = d_tree["hi"] < 0
    nonvac = any(banks_b[k] > 0 for k in banks_b)
    d = all(d_banks[k]["hi"] < -M_05 for k in d_banks) and nonvac
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d), "pass": bool(a and b and c and d)}
