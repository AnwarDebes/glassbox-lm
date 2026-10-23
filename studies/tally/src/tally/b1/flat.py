"""B1 arm 6: tuned flat Tsetlin machine (tmu TMClassifier, CPU).

M clauses in total: two classes with M / 2 clauses each (tmu splits each half into positive and
negative polarity). The machine trains on the same stream, one pass, in chunks. Probability of
y = 1 from the class sums: v = sum(class 1) - sum(class 0), mapped by a logistic (Platt)
calibration q = sigmoid(a v + b) fitted on CALIB_N separate samples drawn once per run (their
count is added to the sample axis). Tuning: grid GRID_S x GRID_T on the tuning game
(seed TUNE_SEED), TUNE_FRAC of the cap, scored by excess bits on a tuning hold-out; one tuning
per team size, shared by all seeds.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from tally.b1 import game

CALIB_N = 2_000
GRID_S = [2.0, 3.9, 8.0]
GRID_T_FRAC = [0.125, 0.25, 0.5]      # T = max(1, round(frac * M))
TUNE_FRAC = 0.25
TUNE_HOLD = 5_000


def make_tm(M, s, T, seed):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    # tmu's C random stream never leaves state 0, so a seed of 0 hangs it: shift by one
    return TMClassifier(max(2, M // 2), T=int(T), s=float(s), platform="CPU", seed=int(seed) + 1)


def bits(X):
    return game.unpack(X).astype(np.uint32)


def class_diff(tm, X):
    _, cs = tm.predict(bits(X), return_class_sums=True)
    return (cs[:, 1] - cs[:, 0]).astype(np.float64)


def calibrate(tm, Xc, yc):
    v = class_diff(tm, Xc).reshape(-1, 1)
    if len(np.unique(yc)) < 2:
        return 0.0, 0.0
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(v, yc)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def predict(tm, ab, X):
    v = class_diff(tm, X)
    return 1.0 / (1.0 + np.exp(-(ab[0] * v + ab[1])))


class Flat:
    def __init__(self, M, s, T, seed):
        self.tm = make_tm(M, s, T, seed)
        self.started = False

    def train(self, X, y):
        self.tm.fit(bits(X), y.astype(np.uint32))
        self.started = True

    def prob(self, X, Xc, yc):
        if not self.started:
            return np.full(len(X), yc.mean() if len(yc) else 0.5)
        ab = calibrate(self.tm, Xc, yc)
        return predict(self.tm, ab, X)


def tune(M, cap, log):
    """Grid search on the tuning game; returns (best (s, T), table)."""
    g = game.Game(M, game.TUNE_SEED)
    st = game.streams(game.TUNE_SEED)
    Xh, _, ph = g.sample(TUNE_HOLD, st["holdout"])
    Xc, yc, _ = g.sample(CALIB_N, st["calib"])
    n_tr = int(TUNE_FRAC * cap)
    Xt, yt, _ = g.sample(n_tr, st["train"])
    table = {}
    for s in GRID_S:
        for f in GRID_T_FRAC:
            T = max(1, int(round(f * M)))
            fl = Flat(M, s, T, game.TUNE_SEED)
            for a in range(0, n_tr, 50_000):
                fl.train(Xt[a:a + 50_000], yt[a:a + 50_000])
            ex = float(game.kl_bits(ph, fl.prob(Xh, Xc, yc)).mean())
            table[f"s={s},T={T}"] = ex
            log(f"tune M={M} s={s} T={T} excess={ex:.4f}")
    best = min(table, key=table.get)
    s, T = best.split(",")
    return (float(s[2:]), int(T[2:])), table
