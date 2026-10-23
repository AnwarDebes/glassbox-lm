"""Helpers for the A/A noise runs of a deterministic count model.

Two runs of the same code on the same data give identical numbers, so the A/A runs vary what
an honest repeat could vary: which train blocks the counts come from (a seeded subsample of the
train blocks; calib, valid and dev-test never change) and, for members that need out-of-fold
scores on train, the assignment of train blocks to cross-fit folds.
"""
import numpy as np


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
