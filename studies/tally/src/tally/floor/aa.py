"""Helpers for the A/A noise runs of a deterministic count model.

Two runs of the same code on the same data give identical numbers, so the A/A runs vary what
an honest repeat could vary: which train blocks the counts come from (a seeded subsample of the
train blocks; calib, valid and dev-test never change) and, for members that need out-of-fold
scores on train, the assignment of train blocks to cross-fit folds.

Added 2026-10-25 for scripts/a1_aa_noise.py: repeat() builds the systems of the A1 baseline
check for one train stream (own KN3, KN5 and KN5 + cache with reset per block, the cache
settings chosen on valid and the mixing weights fitted by EM on calib, as in
scripts/a1_baseline_check.py), pair() compares two repeats on the same evaluation tokens with
the block bootstrap, and spread() summarises one number over the repeats.
"""
import numpy as np

from . import cache, mix, stats
from .kn import KN

HALF_LIVES = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)
SYSTEMS = ("kn3", "kn5", "kn5_cache_block")


def subsample_train(ids, blocks, seed, drop_share=0.05):
    """Train stream without a seeded share of its blocks (whole blocks, file order kept)."""
    ub = np.unique(blocks)
    n_drop = int(round(drop_share * len(ub)))
    drop = np.random.default_rng(seed).choice(ub, size=n_drop, replace=False)
    keep = ~np.isin(blocks, drop)
    return ids[keep], blocks[keep], np.sort(drop)


def folds(blocks, k, seed):
    """Fold number (0..k-1) per token, by a seeded assignment of whole blocks."""
    ub, inv = np.unique(blocks, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(ub))
    return (perm % k)[inv]


def repeat(train, V, eos, ids, blocks, half_lives=HALF_LIVES, contract_n=0, contract_seed=7):
    """One A/A repeat: the A1 systems built from the train stream `train`.

    ids, blocks: dicts with the "calib" and "valid" streams (never changed between repeats).
    Returns (lp, info): lp[(system, split)] = natural log probability of every token of the
    split (splits calib and valid); info holds the chosen cache setting, the mixing weights, the
    Kneser-Ney summaries and, if contract_n > 0, the section 4 contract of KN5 + cache on
    contract_n random valid positions."""
    lp, info = {}, {"train_tokens": int(len(train))}
    models = {}
    for N in (3, 5):
        m = KN(train, V, eos, order=N, mode="sent")
        models[N] = m
        for s in ("calib", "valid"):
            lp[(f"kn{N}", s)] = m.logprob(ids[s])
        info[f"kn{N}"] = m.summary()
    p = {s: np.exp(lp[("kn5", s)]) for s in ("calib", "valid")}
    best, grid = None, []
    for hl in half_lives:
        for skip in ((), (eos,)):
            c = cache.Cache(V, hl, "block", skip)
            pc, _ = c.prob(ids["calib"], blocks["calib"], p["calib"])
            lam, _ = mix.em(np.stack([p["calib"], pc], 1))
            pv, _ = c.prob(ids["valid"], blocks["valid"], p["valid"])
            llv = float(np.log(np.stack([p["valid"], pv], 1) @ lam).sum())
            grid.append({"half_life": float(hl), "skip_eos": bool(skip), "valid_bits": -llv / len(pv) / np.log(2)})
            if best is None or llv > best[0]:
                best = (llv, c, lam)
    _, c, lam = best
    for s in ("calib", "valid"):
        pc, _ = c.prob(ids[s], blocks[s], p[s])
        lp[("kn5_cache_block", s)] = mix.logprob(np.stack([p[s], pc], 1), lam)
    info["cache"] = {"chosen": c.config(), "lam": [float(x) for x in lam], "grid": grid}
    if contract_n:
        m5, x, b = models[5], ids["valid"], blocks["valid"]

        def dfn(pos):
            Dk = m5.dists(x, pos)
            return lam[0] * Dk + lam[1] * c.dists(x, b, pos, Dk)
        info["contract_valid"] = stats.contract(dfn, lp[("kn5_cache_block", "valid")], len(x), contract_n,
                                                seed=contract_seed, true_ids=x)
    return lp, info


def pair(lp_a, lp_b, blocks, n_boot=1000, seed=0):
    """Repeat a against repeat b on the same tokens: mean of bits(a) - bits(b) per token with the
    95% block-bootstrap interval, the relative perplexity ppl(a) / ppl(b) - 1, and whether the
    interval excludes 0."""
    r = stats.block_bootstrap(stats.bits_diff(lp_a, lp_b), blocks, n_boot=n_boot, seed=seed)
    r["rel_ppl"] = float(np.exp(np.mean(lp_b) - np.mean(lp_a)) - 1)
    r["excludes_zero"] = bool(r["lo"] > 0 or r["hi"] < 0)
    return r


def spread(values):
    """n, mean, sd (ddof 1), min, max and sd / mean of one number over the repeats."""
    v = np.asarray(values, dtype=np.float64)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": sd, "min": float(v.min()),
            "max": float(v.max()), "rel_sd": sd / float(v.mean())}
