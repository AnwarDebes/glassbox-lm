"""B1 self-checks (no learning claims). Exit code 1 on any failure.

    PYTHONPATH=src NUMBA_NUM_THREADS=1 python -m tally.b1.selfcheck
"""
import sys

import numpy as np

from tally.b1 import game, team
from tally.b1.run import checkpoints, samples_to_target

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def eval_state(T, x, y, rmode):
    M = T.M
    fire = np.zeros(M, dtype=np.bool_)
    active = np.zeros(M, dtype=np.bool_)
    alt = np.zeros(M, dtype=np.int64)
    dsil = np.zeros(M, dtype=np.bool_)
    w = np.zeros(M)
    r = np.zeros(M)
    z, zb, _ = team.team_eval(x, T.pos, T.neg, T.chash, T.has_dup, T.n1, T.n, T.bg,
                              T.prm[team.P_ALPHA], fire, active, alt, dsil, w)
    team.rewards(rmode, y, z, zb, active, alt, dsil, w, T.prm[team.P_WPROBE], r)
    return z, zb, fire, active, w, r


def oracle_team(g, M, seed, extra=0):
    T = team.Team(M, seed)
    T.ta[:] = 0
    T.pos[:] = 0
    T.neg[:] = 0
    T.chash[:] = 0
    T.set_clauses(g.pos, g.neg)
    T.prm[team.P_RHO] = np.inf
    return T


def main():
    # game: exclusivity, P(fire), exact entropy
    for M in (8, 16, 32, 64, 128, 256, 512):
        g = game.Game(M, 3)
        rng = np.random.default_rng(0)
        X, y, p = g.sample(50_000, rng)
        hits = np.zeros(len(X), dtype=int)
        for r in range(g.R):
            f = np.ones(len(X), dtype=bool)
            for w in range(4):
                f &= (X[:, w] & g.pos[r, w]) == g.pos[r, w]
                f &= (~X[:, w] & g.neg[r, w]) == g.neg[r, w]
            hits += f
        pf = (hits > 0).mean()
        check(f"M={M}: at most one planted rule fires; rule set matches p(x)",
              hits.max() <= 1 and np.array_equal(hits > 0, p > 0.5))
        check(f"M={M}: P(fire) = 1/2", abs(pf - 0.5) < 4 * np.sqrt(0.25 / len(X)), f"{pf:.4f}")
        hbar = float(game.hb(p).mean())
        ll = float(game.logloss_bits(y, p).mean())
        se = float(game.logloss_bits(y, p).std() / np.sqrt(len(y)))
        check(f"M={M}: mean Hb(p(x)) equals Hb(0.1) exactly", abs(hbar - game.ENTROPY) < 1e-12)
        check(f"M={M}: oracle log-loss on drawn labels matches H", abs(ll - game.ENTROPY) < 4 * se,
              f"{ll:.4f} vs {game.ENTROPY:.4f}, se {se:.4f}")
        check(f"M={M}: true-p oracle has excess 0", float(game.kl_bits(p, p).mean()) < 1e-12)
        check(f"M={M}: rule length k = log2(M) - 1", g.k == int(np.log2(M)) - 1)

    # counted oracle team reaches the floor; prediction is a probability; CE = H + KL
    M = 32
    g = game.Game(M, 3)
    st = game.streams(3)
    Xh, yh, ph = g.sample(20_000, st["holdout"])
    T = oracle_team(g, M, 3)
    X, y, _ = g.sample(200_000, st["train"])
    T.train(X, y, 2, 0)
    q = T.predict(Xh)
    ex = float(game.kl_bits(ph, q).mean())
    check("counted oracle team (M=32, 200k samples) within 0.001 bits", ex < 1e-3, f"{ex:.5f}")
    check("team prediction is a probability in (0, 1)", bool(np.all((q > 0) & (q < 1)) and np.all(np.isfinite(q))))
    ce = float(np.mean(-(ph * np.log2(q) + (1 - ph) * np.log2(1 - q))))
    check("cross-entropy = mean Hb(p) + mean KL", abs(ce - (game.ENTROPY + ex)) < 1e-9)
    check("no feedback when RHO is infinite (oracle team unchanged)", T.stats[10] == 0 and T.stats[2] == 0)

    # duplicated member: marginal reward exactly 0 for both copies; leave-one-out matches removal
    T2 = team.Team(M + 1, 3)
    T2.ta[:] = 0
    T2.pos[:] = 0
    T2.neg[:] = 0
    T2.chash[:] = 0
    pos = np.vstack([g.pos, g.pos[:1]])
    neg = np.vstack([g.neg, g.neg[:1]])
    n1 = np.concatenate([T.n1[:g.R], T.n1[:1]])
    n = np.concatenate([T.n[:g.R], T.n[:1]])
    T2.set_clauses(pos, neg, n1, n)
    T2.bg[:] = T.bg
    clone = g.R
    zero_ok, fired = True, 0
    for t in range(5000):
        z, zb, fire, active, w, r = eval_state(T2, X[t], y[t], 2)
        if fire[0]:
            fired += 1
            zero_ok &= abs(r[0]) < 1e-12 and abs(r[clone]) < 1e-12
    check("marginal reward of a duplicated member (and of its twin) is 0", zero_ok and fired > 0,
          f"{fired} firings")
    loo_ok = True
    for t in range(2000):
        z, zb, fire, active, w, r = eval_state(T, X[t], y[t], 2)
        for i in np.flatnonzero(active):
            keep = T.pos[i].copy(), T.neg[i].copy(), T.chash[i]
            T.pos[i] = 0
            T.neg[i] = 0
            T.chash[i] = 0
            team.recompute_dup(T.chash, T.has_dup)
            z2, _, _, _, _, _ = eval_state(T, X[t], y[t], 2)
            T.pos[i], T.neg[i], T.chash[i] = keep
            team.recompute_dup(T.chash, T.has_dup)
            want = team.loss_bits(z2, y[t]) - team.loss_bits(z, y[t])
            loo_ok &= abs(want - r[i]) < 1e-9
    check("closed-form marginal bit equals removing the member and recomputing", loo_ok)
    z, zb, fire, active, w, r1 = eval_state(T, X[0], y[0], 1)
    _, _, _, _, _, r0 = eval_state(T, X[0], y[0], 0)
    check("shared reward is the same for every member", np.all(r0 == r0[0]))

    # target rule
    cps = checkpoints(100_000)
    check("checkpoints rise and end at the cap", cps[-1] == 100_000 and all(np.diff(cps) > 0))
    check("samples_to_target needs two checkpoints in a row",
          samples_to_target([1, 2, 3, 4], [0.5, 0.005, 0.02, 0.009], 4) == (4, False)
          and samples_to_target([1, 2, 3], [0.5, 0.005, 0.006], 3) == (2, False)
          and samples_to_target([1, 2], [0.5, 0.2], 2) == (2, True))

    # evaluation count of a team arm: M per sample
    T3 = team.Team(16, 5)
    g3 = game.Game(16, 5)
    X3, y3, _ = g3.sample(1000, np.random.default_rng(1))
    T3.train(X3, y3, 2, 0)
    check("team arm counts M evaluations per sample", T3.stats[1] == 16 * 1000 and T3.stats[0] == 1000)

    print("FAILED: " + ", ".join(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
