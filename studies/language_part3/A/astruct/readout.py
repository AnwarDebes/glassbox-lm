"""The read-out, the same for every arm (PLAN_part3.md section 3, review 15).

- q = isotonic regression of the arm's score on the tune-cal rows (labels: the candidate is the true token),
  floored at 1e-6 (out-of-bounds scores clipped);
- for w in the candidate set C: P'(w) = P_f(C) [(1 - b) P_f(w) / P_f(C) + b q(w) / sum over C of q], and P' = P_f
  outside C; a position whose q sums to 0 keeps P_f;
- b in {0, 0.05, ..., 0.95}, chosen by the tune-cal bits per token (b = 0 is the floor itself).
Bits are -log2 P'(true token) at every position (words, placeholders and the end token).
"""

from __future__ import annotations

import numpy as np

B_GRID = tuple(np.round(np.arange(0, 0.9501, 0.05), 2))
Q_FLOOR = 1e-6


class Isotonic:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=Q_FLOOR, y_max=1.0, increasing=True)

    def fit(self, score: np.ndarray, y: np.ndarray) -> Isotonic:
        self.m.fit(np.asarray(score, np.float64), np.asarray(y, np.float64))
        return self

    def __call__(self, score: np.ndarray) -> np.ndarray:
        return np.maximum(self.m.predict(np.asarray(score, np.float64)), Q_FLOOR)


class PosTable:
    """What the mix needs per position: floor P of the true token, P_f(C), and the true candidate's row."""

    def __init__(self, rows: dict, p_true: np.ndarray):
        self.off = rows["pos_off"]
        self.n = rows["n_pos"]
        self.pf_true = np.asarray(p_true, np.float64)
        self.pfC = np.add.reduceat(rows["cand_p"], self.off[:-1]) if self.n else np.zeros(0)
        t = np.full(self.n, -1, np.int64)
        yi = np.flatnonzero(rows["y"])
        t[rows["pos"][yi]] = yi
        self.true_row, self.in_C = t, t >= 0
        self.pos = rows["pos"]


def mix_bits(q: np.ndarray, pt: PosTable, b: float) -> np.ndarray:
    q = np.asarray(q, np.float64)
    qs = np.add.reduceat(q, pt.off[:-1])
    ok = pt.in_C & (qs > 0)
    share = np.where(ok, q[np.maximum(pt.true_row, 0)] / np.where(qs > 0, qs, 1.0), 0.0)
    p = np.where(ok, (1 - b) * pt.pf_true + b * pt.pfC * share, pt.pf_true)
    return -np.log2(p)


def choose_b(q_cal: np.ndarray, pt_cal: PosTable) -> tuple[float, list]:
    curve = [float(mix_bits(q_cal, pt_cal, b).mean()) for b in B_GRID]
    i = int(np.argmin(curve))
    return float(B_GRID[i]), curve


def full_mix(PF: np.ndarray, cand: np.ndarray, q: np.ndarray, b: float) -> np.ndarray:
    """The whole vector P' at one position (K2 contract)."""
    out = np.array(PF, np.float64)
    q = np.asarray(q, np.float64)
    qs = q.sum()
    if qs <= 0:
        return out
    s = PF[cand].sum()
    out[cand] = (1 - b) * PF[cand] + b * s * q / qs
    return out


class Arm:
    """Score -> isotonic (tune-cal) -> b (tune-cal) -> bits on any split."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, score_cal: np.ndarray, rows_cal: dict, pt_cal: PosTable) -> Arm:
        self.iso = Isotonic().fit(score_cal, rows_cal["y"])
        self.q_cal = self.iso(score_cal)
        self.b, self.curve = choose_b(self.q_cal, pt_cal)
        self.cal_bits = mix_bits(self.q_cal, pt_cal, self.b)
        return self

    def bits(self, score: np.ndarray, pt: PosTable) -> np.ndarray:
        return mix_bits(self.iso(score), pt, self.b)


def contract(arm_q: np.ndarray, rows: dict, full: dict, pt: PosTable, b: float, x: np.ndarray, fast_bits: np.ndarray) -> dict:
    """K2 on the positions whose full floor vectors were kept: sum to 1 within 1e-6, every entry above 0, and the
    fast path equal to the vector's entry of the true token (|difference| <= 1e-9 bits)."""
    worst_sum, pmin, worst_fast = 0.0, np.inf, 0.0
    for k, PF in full.items():
        lo, hi = pt.off[k], pt.off[k + 1]
        v = full_mix(PF, rows["cand"][lo:hi], arm_q[lo:hi], b)
        worst_sum = max(worst_sum, abs(v.sum() - 1.0))
        pmin = min(pmin, float(v.min()))
        worst_fast = max(worst_fast, abs(-np.log2(v[int(x[k])]) - fast_bits[k]))
    ok = worst_sum <= 1e-6 and pmin > 0 and worst_fast <= 1e-9
    return {"positions": len(full), "max_sum_err": worst_sum, "min_p": pmin, "max_fast_err_bits": worst_fast, "pass": bool(ok)}
