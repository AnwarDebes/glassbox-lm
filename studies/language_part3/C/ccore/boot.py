"""Seed-inclusive interval of C-core (reported beside the registered statistic, no rule).

The registered statistic is tally floor/stats.py block_bootstrap on per-position bits averaged over
the seeds of each multi-seed arm (100-sentence blocks, 1,000 resamples, seed 0). Beside it, this
two-level bootstrap resamples the seeds of each multi-seed arm with replacement (independently per arm),
then the blocks (multinomial weights, as block_bootstrap), 1,000 resamples, seed 0, and reports the
percentile interval of sum(d) / tokens."""
import numpy as np


def _block_sums(bits, inv, nb):
    bits = np.atleast_2d(np.asarray(bits, dtype=np.float64))         # (seeds, n)
    return np.stack([np.bincount(inv, weights=b, minlength=nb) for b in bits])   # (seeds, blocks)


def two_level(bits_a, bits_b, blocks, n_boot=1000, seed=0, level=0.95):
    """bits_a, bits_b: (seeds, n) or (n,) per-position bits; the difference is a minus b."""
    ub, inv = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    Sa, Sb = _block_sums(bits_a, inv, nb), _block_sums(bits_b, inv, nb)
    n = np.bincount(inv, minlength=nb).astype(np.float64)
    rng = np.random.default_rng(seed)
    stat = np.empty(n_boot)
    for r in range(n_boot):
        ia = rng.integers(0, Sa.shape[0], Sa.shape[0])
        ib = rng.integers(0, Sb.shape[0], Sb.shape[0])
        W = rng.multinomial(nb, np.full(nb, 1.0 / nb)).astype(np.float64)
        stat[r] = (W @ (Sa[ia].mean(0) - Sb[ib].mean(0))) / (W @ n)
    a = (1 - level) / 2
    lo, hi = np.quantile(stat, [a, 1 - a])
    mean = float((Sa.mean(0) - Sb.mean(0)).sum() / n.sum())
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "level": level, "n_boot": n_boot, "seed": seed,
            "n_blocks": int(nb), "seeds_a": int(Sa.shape[0]), "seeds_b": int(Sb.shape[0]),
            "boot_sd": float(stat.std(ddof=1))}
