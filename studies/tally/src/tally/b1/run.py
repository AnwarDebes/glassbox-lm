"""B1 entry point: one process per (arm, M, seed).

    PYTHONPATH=src NUMBA_NUM_THREADS=1 python -m tally.b1.run --arm marginal --M 128 --seed 0 \
        --label smoke [--cap N] [--registration registrations/B1.md]

Arms: shared, selfish, marginal (Tsetlin team, changes applied directly); dual (marginal with the
dual-state trial); ab (marginal with the fixed-sample A/B test; its length is read from the
finished dual run of the same label, M and seed); greedy; random; flat (tmu, needs the finished
flat_tune run of the same label and M); flat_tune (one per M); oracle.

Every run writes a run folder (tally.runlog.Run) named b1_<label>_<arm>_M<M>_s<seed>, with the
excess-bits curve at the checkpoints and the samples needed to come within TARGET bits.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from tally.b1 import game, team

TARGET = 0.01
HOLDOUT_N = 20_000
CHUNK = 50_000
# sample cap per team size (fixed from the timing run; see the registration)
CAPS = {8: 2_000_000, 16: 2_000_000, 32: 2_000_000, 64: 2_000_000, 128: 2_000_000,
        256: 2_000_000, 512: 2_000_000}
TEAM_ARMS = {"shared": (0, 0), "selfish": (1, 0), "marginal": (2, 0), "dual": (2, 1), "ab": (2, 2)}
ARMS = list(TEAM_ARMS) + ["greedy", "random", "flat", "flat_tune", "oracle"]


def checkpoints(cap):
    """1000 * 2^(j/4) samples, j = 0, 1, ..., and the cap."""
    out, j = [], 0
    while True:
        c = int(round(1000 * 2 ** (j / 4)))
        if c >= cap:
            break
        if not out or c > out[-1]:
            out.append(c)
        j += 1
    out.append(int(cap))
    return out


def samples_to_target(cps, curve, cap, target=TARGET):
    """First checkpoint where the excess is within target there and at the next checkpoint
    (at the last checkpoint: there). Returns (samples, censored)."""
    for j, e in enumerate(curve):
        if e <= target and (j + 1 == len(curve) or curve[j + 1] <= target):
            return int(cps[j]), False
    return int(cap), True


def find_run(label, arm, M, seed):
    from tally.runlog import RESULTS
    hits = []
    for d in sorted(glob.glob(os.path.join(RESULTS, f"*_b1_{label}_{arm}_M{M}_s{seed}"))):
        try:
            with open(os.path.join(d, "run.json")) as f:
                if json.load(f).get("state") != "finished":
                    continue
            with open(os.path.join(d, "metrics.json")) as f:
                hits.append((d, json.load(f)))
        except OSError:
            continue
    if not hits:
        raise SystemExit(f"no finished run b1_{label}_{arm}_M{M}_s{seed}")
    return hits[-1]


def evaluate(q, ph, yh):
    ex = float(game.kl_bits(ph, q).mean())
    ll = float(game.logloss_bits(yh, q).mean() - game.ENTROPY)
    return ex, ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True, help="smoke, timing or reg")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--registration", default=None)
    a = ap.parse_args()
    assert a.label in ("smoke", "timing", "reg") or a.label.startswith(("smoke", "timing")), a.label
    if a.label == "reg":
        assert a.cap is None, "registered runs use the registered caps"
        assert a.registration, "registered runs name their registration"
    cap = a.cap or CAPS[a.M]
    from tally.runlog import Run

    prm = team.default_params()
    cfg = {"arm": a.arm, "M": a.M, "seed": a.seed, "label": a.label, "cap": cap,
           "target_bits": TARGET, "holdout_n": HOLDOUT_N, "entropy_bits": game.ENTROPY,
           "team_params": dict(zip(team.PARAM_NAMES, prm.tolist())),
           "N_states": 64, "n_bits": game.N_BITS,
           "numba_threads": os.environ.get("NUMBA_NUM_THREADS")}
    if a.arm == "ab":
        _, dm = find_run(a.label, "dual", a.M, a.seed)
        prm[team.P_NAB] = max(1.0, round(dm["trial_mean_informative"]))
        cfg["team_params"]["n_ab"] = prm[team.P_NAB]
    flat_cfg = None
    if a.arm == "flat":
        _, tm_ = find_run(a.label, "flat_tune", a.M, game.TUNE_SEED)
        flat_cfg = tm_["best"]
        cfg["flat"] = flat_cfg
    run = Run(f"b1_{a.label}_{a.arm}_M{a.M}_s{a.seed}", cfg, registration=a.registration)

    if a.arm == "flat_tune":
        from tally.b1 import flat
        t0 = time.time()
        (s, T), table = flat.tune(a.M, cap, run.print)
        run.finish({"best": {"s": s, "T": T}, "table": table, "wall_s": time.time() - t0})
        return

    g = game.Game(a.M, a.seed)
    st = game.streams(a.seed)
    Xh, yh, ph = g.sample(HOLDOUT_N, st["holdout"])
    run.print(f"game M={a.M} R={g.R} k={g.k} holdout P(fire)={np.mean(ph > 0.5):.4f} "
              f"H={game.ENTROPY:.6f} mean Hb(p)={float(game.hb(ph).mean()):.6f}")
    cps = checkpoints(cap)
    curve = []
    t_start = time.time()

    def record(samples, evals, q, extra=None):
        ex, ll = evaluate(q, ph, yh)
        row = {"samples": int(samples), "evals": int(evals), "excess": ex, "logloss_minus_H": ll,
               "wall_s": round(time.time() - t_start, 2)}
        if extra:
            row.update(extra)
        curve.append(row)
        run.print(json.dumps(row))

    metrics = {}
    if a.arm in TEAM_ARMS or a.arm == "oracle":
        rmode, cmode = TEAM_ARMS.get(a.arm, (2, 0))
        T = team.Team(a.M, a.seed, prm=prm)
        if a.arm == "oracle":
            # planted clauses as members, counted on the stream, no automaton feedback
            T.ta[:] = 0
            T.pos[:] = 0
            T.neg[:] = 0
            T.chash[:] = 0
            T.set_clauses(g.pos, g.neg)
            T.prm[team.P_RHO] = np.inf
            q_true = g.p_true(Xh)
            metrics["oracle_true_p_excess"] = float(game.kl_bits(ph, q_true).mean())
        done = 0
        for c in cps:
            while done < c:
                n = min(CHUNK, c - done)
                X, y, _ = g.sample(n, st["train"])
                T.train(X, y, rmode, cmode)
                done += n
            sizes = T.clause_sizes()
            record(done, T.stats[1], T.predict(Xh),
                   {"changes": int(T.stats[2]), "in_trial": int((T.tr[:, 0] != 0).sum()),
                    "empty": int((sizes == 0).sum()), "mean_len": float(sizes[sizes > 0].mean())
                    if (sizes > 0).any() else 0.0})
        s = T.stats
        metrics.update({"changes_proposed": int(s[2]), "feedback_events": int(s[10]),
                        "evaluations": int(s[1])})
        if cmode:
            metrics.update({"accepted_new": int(s[3]), "kept_old": int(s[4]),
                            "kept_both": int(s[5]), "truncated": int(s[6]),
                            "trials_finished": int(s[9]),
                            "trial_mean_informative": float(s[7] / max(1, s[9])),
                            "trial_mean_samples": float(s[8] / max(1, s[9]))})
    elif a.arm in ("greedy", "random"):
        from tally.b1 import search
        T = team.Team(a.M, a.seed, prm=prm)
        T.pos[:] = 0
        T.neg[:] = 0
        T.chash[:] = 0
        T.has_dup[:] = False
        S = search.Search(T, a.arm, a.seed)
        last = -1
        for c in cps:
            while T.stats[1] < c * a.M:
                X, y, _ = g.sample(search.B_SEARCH, st["train"])
                S.step(X, y)
            if T.stats[1] == last:
                continue        # one search step can pass several checkpoints: record once
            last = T.stats[1]
            record(T.stats[1] // a.M, T.stats[1], T.predict(Xh),
                   {"fresh_samples": int(T.stats[0]), "changes": int(T.stats[2])})
        metrics.update({"evaluations": int(T.stats[1]), "fresh_samples": int(T.stats[0]),
                        "changes": int(T.stats[2])})
    elif a.arm == "flat":
        from tally.b1 import flat
        Xc, yc, _ = g.sample(flat.CALIB_N, st["calib"])
        F = flat.Flat(a.M, flat_cfg["s"], flat_cfg["T"], a.seed)
        done = 0
        for c in cps:
            while done < c:
                n = min(CHUNK, c - done)
                X, y, _ = g.sample(n, st["train"])
                F.train(X, y)
                done += n
            # calibration samples count toward the sample axis
            record(done + flat.CALIB_N, done * a.M, F.prob(Xh, Xc, yc))
        cps = [c + flat.CALIB_N for c in cps]
    xs = [r["samples"] for r in curve]
    ys = [r["excess"] for r in curve]
    n_t, cens = samples_to_target(xs, ys, xs[-1])
    metrics.update({"samples_to_target": n_t, "censored": cens, "cap": int(xs[-1]),
                    "final_excess": ys[-1], "curve": curve,
                    "wall_s": round(time.time() - t_start, 1),
                    "us_per_sample": 1e6 * (time.time() - t_start) / max(1, xs[-1])})
    run.print(f"samples_to_target={n_t} censored={cens} final_excess={ys[-1]:.4f}")
    run.finish(metrics)


if __name__ == "__main__":
    main()
