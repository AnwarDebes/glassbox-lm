"""Fix the member learning constants (s, RHO) on a one-member game, before any registered run.

With a team of one member the team without the member is the background, so the selfish and
marginal rewards are the same number and the shared reward differs only while the member is
silent; the score averages the three rules, so the choice does not favour one reward arm.
Game: one planted rule of K_CAL literals over 256 bits, P(y=1) = 0.9 if it fires else 0.1.
Score: samples until the excess over the entropy is within 0.01 bits at two checkpoints in a row
(cap CAP_CAL), geometric mean over CAL_SEEDS seeds and over the three reward rules; the grid
point with the lowest geometric mean wins.
Seeds 20000+ are used here, never a registered seed.

    python -m tally.b1.calib --label smoke_calib
"""
import argparse
import itertools
import time

import numpy as np

from tally.b1 import game, team
from tally.b1.run import checkpoints, samples_to_target

K_CAL = 4
CAP_CAL = 1_000_000
CAL_SEEDS = list(range(20_000, 20_005))
GRID_S = [2.5, 3.9, 6.0]
GRID_RHO = [0.5, 1.0, 2.0, 4.0]


class OneRule:
    def __init__(self, seed):
        rng = np.random.default_rng([seed, 99])
        self.pos = np.zeros((1, 4), dtype=np.uint64)
        self.neg = np.zeros((1, 4), dtype=np.uint64)
        for v in rng.choice(256, K_CAL, replace=False):
            m = self.pos if rng.integers(0, 2) else self.neg
            m[0, v // 64] |= np.uint64(1) << np.uint64(v % 64)

    def p_true(self, X):
        f = np.ones(len(X), dtype=bool)
        for w in range(4):
            f &= (X[:, w] & self.pos[0, w]) == self.pos[0, w]
            f &= (~X[:, w] & self.neg[0, w]) == self.neg[0, w]
        return np.where(f, game.P_HI, game.P_LO)

    def sample(self, n, rng):
        X = rng.integers(0, np.iinfo(np.uint64).max, size=(n, 4), dtype=np.uint64, endpoint=True)
        p = self.p_true(X)
        return X, (rng.random(n) < p).astype(np.uint8), p


def one(seed, s, rho, rmode=2):
    g = OneRule(seed)
    rng = np.random.default_rng([seed, 5])
    Xh, _, ph = g.sample(20_000, np.random.default_rng([seed, 6]))
    prm = team.default_params()
    prm[team.P_S], prm[team.P_RHO] = s, rho
    T = team.Team(1, seed, prm=prm)
    cps = checkpoints(CAP_CAL)
    curve, done = [], 0
    for c in cps:
        X, y, _ = g.sample(c - done, rng)
        T.train(X, y, rmode, 0)
        done = c
        curve.append(float(game.kl_bits(ph, T.predict(Xh)).mean()))
    return samples_to_target(cps, curve, CAP_CAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="smoke_calib")
    a = ap.parse_args()
    assert a.label.startswith(("smoke", "timing")), "calibration runs are labelled smoke or timing"
    from tally.runlog import Run
    run = Run(f"b1_{a.label}", {"K_CAL": K_CAL, "CAP_CAL": CAP_CAL, "seeds": CAL_SEEDS,
                                "grid_s": GRID_S, "grid_rho": GRID_RHO})
    res = {}
    for s, rho in itertools.product(GRID_S, GRID_RHO):
        t0 = time.time()
        v = [one(sd, s, rho, rm) for rm in (0, 1, 2) for sd in CAL_SEEDS]
        res[f"s={s},rho={rho}"] = {"samples": [x[0] for x in v], "censored": [x[1] for x in v],
                                   "gmean": float(np.exp(np.mean(np.log([x[0] for x in v]))))}
        run.print(f"s={s} rho={rho} samples={[x[0] for x in v]} gmean={res[f's={s},rho={rho}']['gmean']:.0f}"
                  f" ({time.time() - t0:.1f}s)")
    best = min(res, key=lambda k: res[k]["gmean"])
    run.print("best", best)
    run.finish({"grid": res, "best": best})


if __name__ == "__main__":
    main()
