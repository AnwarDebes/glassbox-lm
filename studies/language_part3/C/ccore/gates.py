"""Gate and bar arithmetic of C-core, fixed before any run (PLAN_part3.md section 4, with the review
amendments). All numbers are bits per word (bpw) with <eos> counted; "diff A - B" < 0 means A is better.
Comparisons allow EPS for floating-point representation only.

Step-6 gates (CPU, valid):
  K0 (sanity): bpw(F) - bpw(best twin system) >= 0.0439 (3% perplexity). A stop points to a bug or poor codes.
  K0b: min(bpw(best LR system), bpw(best LightGBM system)) - bpw(best twin system) <= 0.0072.
  A/A: |bpw(rebuilt best twin system) - bpw(best twin system)| <= 0.0036, else stop.
  Shuffled labels: for every shuffled class model, bpw(its system) - bpw(F) >= -0.0010, else stop.
Later rules (registered now, run after the GPU TM step):
  K1: bpw(best TM system) - bpw(best twin system) <= 0.0218 on valid, else "FAIL at K1 (exploratory)".
  LEVEL = P1 and P2 and P3 on the test file (upper = the upper end of the 95% paired block-bootstrap
  interval of the mean difference of seed-mean per-position bits):
    P1: upper(S_TM - S_twin) < +0.0072 and every seed's point estimate of S_TM - S_twin < +0.0072;
    P2: upper(S_TM - S_uniform), upper(S_TM - S_anchored), upper(S_TM - S_C0), upper(S_TM - S_epoch1)
        all < -0.0072, upper(S_TM - S_tree) < 0, and at least one untrained bank non-vacuous
        (its clauses fire on >= 1% of valid positions and its two members' EM weight, averaged over
        valid positions, is >= 0.01);
    P3: upper(S_TM - F) < -0.0291.
  BEATS: LEVEL and upper(S_TM - S_twin) < -0.0145 and mean bpw(S_TM) <= mean bpw(S_LightGBM).
  ADDS: upper(S_twin+TM - S_twin) < -0.0072, upper(S_twin+TM - S_twin+uniform) < -0.0036,
        upper(S_twin+TM - S_twin+anchored) < -0.0036 and upper(S_twin+TM - S_twin+tree) < -0.0036.
"""
EPS = 1e-12
K0_MIN_GAIN = 0.0439
K0B_MAX_GAP = 0.0072
AA_MAX_ABS = 0.0036
SHUFFLED_MIN_DIFF = -0.0010
K1_MAX_GAP = 0.0218
P1_MAX = 0.0072
P2_MAX = -0.0072
P3_MAX = -0.0291
BEATS_MAX = -0.0145
ADDS_MAX = -0.0072
ADDS_CONTROL_MAX = -0.0036
VACUOUS_FIRE_MIN = 0.01
VACUOUS_WEIGHT_MIN = 0.01


def k0(bpw_F, bpw_twin):
    gain = bpw_F - bpw_twin
    return {"gain_bpw": gain, "min_gain": K0_MIN_GAIN, "pass": bool(gain >= K0_MIN_GAIN - EPS)}


def k0b(bpw_twin, bpw_lr, bpw_lgbm):
    best = min(bpw_lr, bpw_lgbm)
    gap = best - bpw_twin
    return {"best_learner": "LR" if bpw_lr <= bpw_lgbm else "LightGBM", "gap_bpw": gap,
            "max_gap": K0B_MAX_GAP, "pass": bool(gap <= K0B_MAX_GAP + EPS)}


def aa(bpw_twin, bpw_twin_rebuilt):
    d = bpw_twin_rebuilt - bpw_twin
    return {"diff_bpw": d, "max_abs": AA_MAX_ABS, "pass": bool(abs(d) <= AA_MAX_ABS + EPS)}


def shuffled(bpw_F, bpw_shuffled_system):
    d = bpw_shuffled_system - bpw_F
    return {"diff_vs_F_bpw": d, "min_diff": SHUFFLED_MIN_DIFF, "pass": bool(d >= SHUFFLED_MIN_DIFF - EPS)}


def step6_decision(k0r, k0br, aar, shuffled_rs):
    """Order of the stop reasons: shuffled (a leak voids every number), A/A, K0, K0b."""
    if not all(r["pass"] for r in shuffled_rs.values()):
        return "STOP: shuffled-label test failed (find the leak)"
    if not aar["pass"]:
        return "STOP: A/A difference outside +-0.0036 (margins inside the noise)"
    if not k0r["pass"]:
        return "STOP at K0 (sanity): best twin system less than 0.0439 bpw below F (a bug or poor codes)"
    if not k0br["pass"]:
        return "STOP at K0b: LR and LightGBM both more than 0.0072 bpw behind the twin system"
    return "PASS: K0 and K0b passed; the GPU TM step may be scheduled"


def k1(bpw_tm_best, bpw_twin_best):
    gap = bpw_tm_best - bpw_twin_best
    return {"gap_bpw": gap, "max_gap": K1_MAX_GAP, "pass": bool(gap <= K1_MAX_GAP + EPS)}


def non_vacuous(fire_share, member_weight):
    return bool(fire_share >= VACUOUS_FIRE_MIN - EPS and member_weight >= VACUOUS_WEIGHT_MIN - EPS)


def level(up_tm_twin, seed_tm_twin, up_tm_uniform, up_tm_anchored, up_tm_c0, up_tm_epoch1, up_tm_tree,
          banks_non_vacuous, up_tm_F):
    p1 = up_tm_twin < P1_MAX - EPS and all(d < P1_MAX - EPS for d in seed_tm_twin)
    p2 = (all(u < P2_MAX - EPS for u in (up_tm_uniform, up_tm_anchored, up_tm_c0, up_tm_epoch1))
          and up_tm_tree < 0 - EPS and any(banks_non_vacuous))
    p3 = up_tm_F < P3_MAX - EPS
    return {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "LEVEL": bool(p1 and p2 and p3)}


def beats(level_r, up_tm_twin, mean_tm, mean_lgbm):
    return bool(level_r["LEVEL"] and up_tm_twin < BEATS_MAX - EPS and mean_tm <= mean_lgbm + EPS)


def adds(up_add_twin, up_add_uniform, up_add_anchored, up_add_tree):
    return bool(up_add_twin < ADDS_MAX - EPS and up_add_uniform < ADDS_CONTROL_MAX - EPS
                and up_add_anchored < ADDS_CONTROL_MAX - EPS and up_add_tree < ADDS_CONTROL_MAX - EPS)
