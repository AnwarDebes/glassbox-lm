"""Members and systems (PLAN_part3.md section 4; proposal C section 4, step 4). Pure numpy, so it runs
in both environments (tally floor/mix.py imports numpy only).

For a class model X with P_X(c | h) over the 65 classes:
  factored member  f_X(w | h) = P_X(c(w) | h) F(w | h) / F(c(w) | h)
  standard member  s_X(w | h) = P_X(c(w) | h) e(w),  e(w) = (n(w) + 0.5) / (n(c) + 0.5 |c|)
  system           S_X = bucketed MAP-EM mixture of {F, f_X, s_X} fitted on calib (the floor's 40
                   buckets, tau 20, prior = the global EM weights on calib).
A mixture of class models (the twin's chains) uses the same bucketed EM on the class likelihood."""
import numpy as np

from . import paths

paths.setup()
from tally.floor import mix  # noqa: E402

TAU = 20.0
N_BUCKETS = 40
LN2 = np.log(2.0)


class Split:
    """Per-position arrays of one split: true word w, true class y, floor true-word probability pF,
    floor class marginals Fc (n, 65), bucket b, longest match L, block ids, emission of w."""

    def __init__(self, w, y, pF, Fc, b, L, blocks, e):
        self.w, self.y, self.pF, self.Fc, self.b, self.L, self.blocks = w, y, pF, Fc, b, L, blocks
        self.e_t = e[w]
        self.n = len(w)
        self.Fc_t = Fc[np.arange(self.n), y]
        assert (self.Fc_t >= self.pF * (1 - 1e-12)).all(), "F(c) below F(w) for w in c"


def true_members(PX, sp):
    pc = PX[np.arange(sp.n), sp.y]
    return np.stack([sp.pF, pc * sp.pF / sp.Fc_t, pc * sp.e_t], 1)


def fit_mixture(P, b):
    """Bucketed MAP-EM weights for member columns P (n, m) of true-outcome probabilities."""
    lam_g, tr_g = mix.em(P)
    lamb, tr = mix.em_bucketed(P, b, N_BUCKETS, lam_g, tau=TAU)
    assert np.allclose(lamb.sum(1), 1.0)
    return lam_g, lamb, {"em_steps_global": int(len(tr_g)), "em_steps_bucketed": int(len(tr)),
                         "em_monotone": bool(np.all(np.diff(tr) >= -1e-6))}


def mix_classes(Ps, lamb, b):
    """Mixture of class distributions Ps (list of (n, 65)) with per-bucket weights."""
    lb = lamb[b]
    out = np.zeros_like(Ps[0])
    for j, P in enumerate(Ps):
        out += lb[:, j:j + 1] * P
    return out


def fit_class_mixture(Ps_cal, y_cal, b_cal):
    if len(Ps_cal) == 1:
        return None, None, {"members": 1}
    n = len(y_cal)
    P = np.stack([Pm[np.arange(n), y_cal] for Pm in Ps_cal], 1)
    return fit_mixture(P, b_cal)


def fit_system(PX_cal, sp_cal):
    return fit_mixture(true_members(PX_cal, sp_cal), sp_cal.b)


def logp(PX, sp, lamb):
    return np.log((true_members(PX, sp) * lamb[sp.b]).sum(1))


def bpw(lp):
    return float(-np.mean(lp) / LN2)


def bits(lp):
    return -np.asarray(lp) / LN2


def class_bits(PX, sp):
    """65-way class bits per token (a report; floored at 1e-300 for a class absent from a tiny subset)."""
    return float(-np.mean(np.log2(np.maximum(PX[np.arange(sp.n), sp.y], 1e-300))))


def standalone_bpw(PX, sp):
    """The class LM s_X alone, in bits per word (a report)."""
    return float(-np.mean(np.log2(np.maximum(PX[np.arange(sp.n), sp.y] * sp.e_t, 1e-300))))


def rare_split(lp, sp, max_match=2):
    """bpw on positions whose longest KN5 match is at most 2 (rare) and on the others."""
    rare = sp.L <= max_match
    return {"rare_share": float(rare.mean()), "rare_bpw": bpw(lp[rare]) if rare.any() else None,
            "other_bpw": bpw(lp[~rare]) if (~rare).any() else None}


def vectors(PX_rows, D, lamb_rows, cls, e):
    """Full system vectors over V at some positions: D (m, V) floor vectors there, PX_rows (m, 65)."""
    K = PX_rows.shape[1]
    M = np.zeros((D.shape[1], K))
    M[np.arange(D.shape[1]), cls] = 1.0
    Fc = D @ M
    pcw = PX_rows[:, cls]
    f = pcw * D / Fc[:, cls]
    s = pcw * e[None, :]
    return lamb_rows[:, :1] * D + lamb_rows[:, 1:2] * f + lamb_rows[:, 2:3] * s


def contract(PX, sp, lamb, pos, D, cls, e):
    """Section 4 contract on the saved positions: every vector sums to 1 within 1e-6, its minimum is
    above 0, and the fast path equals the vector entry of the true word (|log difference| <= 1e-9)."""
    S = vectors(PX[pos], D, lamb[sp.b[pos]], cls, e)
    fast = logp(PX, sp, lamb)[pos]
    sum_err = float(np.abs(S.sum(1) - 1.0).max())
    pmin = float(S.min())
    fast_err = float(np.abs(np.log(S[np.arange(len(pos)), sp.w[pos]]) - fast).max())
    return {"positions": int(len(pos)), "max_sum_err": sum_err, "min_p": pmin,
            "max_fast_vs_vector_log_err": fast_err,
            "pass": bool(sum_err <= 1e-6 and pmin > 0 and fast_err <= 1e-9)}


def all_positions(PX, lamb, floor_check):
    """Sum and positivity at every position, from the parts: the floor vectors (checked in the class
    marginal pass), the class vectors, the emission sums and the weights; S = sum of weights times
    members that each sum to 1."""
    row_err = float(np.abs(PX.sum(1) - 1.0).max())
    ok = (floor_check["pass"] and row_err <= 1e-6 and float(PX.min()) >= 0
          and np.allclose(lamb.sum(1), 1.0) and float(lamb[:, 0].min()) > 0)
    return {"class_rows_max_sum_err": row_err, "class_min": float(PX.min()),
            "floor_pass": bool(floor_check["pass"]), "min_floor_weight": float(lamb[:, 0].min()), "pass": bool(ok)}


def emission_check(e, cls):
    K = int(cls.max()) + 1
    s = np.bincount(cls, weights=e, minlength=K)
    return {"max_class_sum_err": float(np.abs(s - 1.0).max()), "min": float(e.min()),
            "pass": bool(np.abs(s - 1.0).max() <= 1e-9 and e.min() > 0)}


def null_class_model(y_train, n):
    """Class unigram from train with add-0.5 smoothing, the same at every position (a diagnostic)."""
    cnt = np.bincount(y_train, minlength=65).astype(np.float64)
    p = (cnt + 0.5) / (cnt.sum() + 0.5 * 65)
    return np.tile(p, (n, 1))
