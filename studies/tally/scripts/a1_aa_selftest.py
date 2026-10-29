"""Correctness tests of the A/A helpers (floor/aa.py) on small synthetic streams (no PTB data).

    python scripts/a1_aa_selftest.py

- subsample_train: drops round(share x blocks) whole blocks, keeps the file order, same seed
  same blocks, another seed other blocks, share 0 keeps everything;
- folds: every block in one fold, all k folds used, block counts per fold differ by at most 1;
- repeat: with every train block, KN3 and KN5 equal a model built directly, and KN5 + cache
  equals the selection and EM of scripts/a1_baseline_check.py written out here; two calls give
  identical numbers; the section 4 contract passes; a subsample changes the numbers;
- pair: a repeat against itself gives 0 with a point interval; swapping the repeats mirrors the
  mean and the interval; a constant loss of c nats per token gives c / ln 2 bits and a relative
  perplexity of e^c - 1;
- spread: equals numpy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally.floor import aa, cache, mix  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name, flush=True)


def synthetic(rng, n_blocks, per_block, V, eos):
    """Zipf words with three preferred successors, sentences ending in eos, blocks of tokens."""
    z = 1.0 / np.arange(1, V + 1)
    z /= z.sum()
    succ = rng.choice(V, size=(V, 3), p=z)
    n = n_blocks * per_block
    draws = rng.choice(V, size=n, p=z)
    x = np.empty(n, dtype=np.int64)
    x[0] = draws[0]
    for i in range(1, n):
        x[i] = succ[x[i - 1], rng.integers(3)] if rng.random() < 0.5 else draws[i]
    x[rng.random(n) < 0.07] = eos
    b = np.repeat(np.arange(n_blocks), per_block)
    x[per_block - 1::per_block] = eos          # every block ends a sentence
    return x, b


def main():
    rng = np.random.default_rng(3)
    V, eos = 800, 5
    tr, tb = synthetic(rng, 60, 300, V, eos)
    tb = tb * 3 + 1                               # block numbers need not start at 0 or be dense
    ids, blocks = {}, {}
    ids["calib"], blocks["calib"] = synthetic(rng, 8, 250, V, eos)
    ids["valid"], blocks["valid"] = synthetic(rng, 12, 250, V, eos)

    # subsample_train
    x1, b1, d1 = aa.subsample_train(tr, tb, 1, 0.05)
    x1b, _, d1b = aa.subsample_train(tr, tb, 1, 0.05)
    _, _, d2 = aa.subsample_train(tr, tb, 2, 0.05)
    keep = ~np.isin(tb, d1)
    check("subsample: drops round(0.05 x 60) = 3 whole blocks", len(d1) == 3 and np.all(np.isin(d1, tb))
          and not np.isin(d1, b1).any())
    check("subsample: kept tokens are the kept blocks in file order", np.array_equal(x1, tr[keep]))
    check("subsample: same seed same blocks, other seed other blocks",
          np.array_equal(d1, d1b) and np.array_equal(x1, x1b) and not np.array_equal(d1, d2))
    x0, b0, d0 = aa.subsample_train(tr, tb, 1, 0.0)
    check("subsample: share 0 keeps everything", len(d0) == 0 and np.array_equal(x0, tr))

    # folds
    f = aa.folds(tb, 5, 4)
    per_block = {b: set(f[tb == b]) for b in np.unique(tb)}
    counts = np.bincount([next(iter(v)) for v in per_block.values()], minlength=5)
    check("folds: one fold per block, all 5 used, block counts differ by at most 1",
          all(len(v) == 1 for v in per_block.values()) and counts.min() >= 1 and counts.max() - counts.min() <= 1)
    check("folds: another seed gives another assignment", not np.array_equal(f, aa.folds(tb, 5, 5)))

    # repeat with every train block against a direct computation
    lp, info = aa.repeat(tr, V, eos, ids, blocks, contract_n=200)
    ok = True
    for N in (3, 5):
        m = KN(tr, V, eos, order=N, mode="sent")
        for s in ("calib", "valid"):
            ok &= np.array_equal(lp[(f"kn{N}", s)], m.logprob(ids[s]))
    check("repeat: KN3 and KN5 equal models built directly", ok)
    m5 = KN(tr, V, eos, order=5, mode="sent")
    p = {s: np.exp(m5.logprob(ids[s])) for s in ("calib", "valid")}
    best = None
    for hl in aa.HALF_LIVES:                      # scripts/a1_baseline_check.py, tune_cache
        for skip in ((), (eos,)):
            c = cache.Cache(V, hl, "block", skip)
            pc, _ = c.prob(ids["calib"], blocks["calib"], p["calib"])
            lam, _ = mix.em(np.stack([p["calib"], pc], 1))
            pv, _ = c.prob(ids["valid"], blocks["valid"], p["valid"])
            llv = float(np.log(np.stack([p["valid"], pv], 1) @ lam).sum())
            if best is None or llv > best[0]:
                best = (llv, c, lam)
    _, c, lam = best
    pv, _ = c.prob(ids["valid"], blocks["valid"], p["valid"])
    ref = mix.logprob(np.stack([p["valid"], pv], 1), lam)
    check("repeat: KN5 + cache equals the A1 selection and EM written out",
          np.array_equal(lp[("kn5_cache_block", "valid")], ref) and info["cache"]["chosen"] == c.config())
    check("repeat: section 4 contract passes on 200 valid positions", info["contract_valid"]["pass"])
    lp2, _ = aa.repeat(tr, V, eos, ids, blocks)
    check("repeat: two calls give identical numbers", all(np.array_equal(lp[k], lp2[k]) for k in lp))
    lp3, _ = aa.repeat(x1, V, eos, ids, blocks)
    check("repeat: a subsample of the train blocks changes the numbers",
          not np.array_equal(lp[("kn5", "valid")], lp3[("kn5", "valid")]))

    # pair
    a, b = lp[("kn5_cache_block", "valid")], lp3[("kn5_cache_block", "valid")]
    bv = blocks["valid"]
    r0 = aa.pair(a, a, bv)
    check("pair: a repeat against itself gives 0 with a point interval and no exclusion",
          r0["mean"] == 0 and r0["lo"] == 0 and r0["hi"] == 0 and not r0["excludes_zero"] and r0["rel_ppl"] == 0)
    rab, rba = aa.pair(a, b, bv), aa.pair(b, a, bv)
    check("pair: swapping the repeats mirrors mean and interval",
          np.isclose(rab["mean"], -rba["mean"], atol=1e-15) and np.isclose(rab["lo"], -rba["hi"], atol=1e-12)
          and np.isclose(rab["hi"], -rba["lo"], atol=1e-12))
    cst = 0.1
    rc = aa.pair(a - cst, a, bv)
    check("pair: a constant loss of 0.1 nats per token gives 0.1 / ln 2 bits and e^0.1 - 1",
          np.isclose(rc["mean"], cst / np.log(2)) and np.isclose(rc["lo"], cst / np.log(2))
          and np.isclose(rc["rel_ppl"], np.exp(cst) - 1) and rc["excludes_zero"])

    # spread
    v = np.array([3.0, 5.0, 4.0, 10.0])
    s = aa.spread(v)
    check("spread: equals numpy", s["n"] == 4 and np.isclose(s["mean"], v.mean()) and np.isclose(s["sd"], v.std(ddof=1))
          and s["min"] == 3 and s["max"] == 10 and np.isclose(s["rel_sd"], v.std(ddof=1) / v.mean()))
    print("ALL OK" if all(OK) else "SOME FAILED")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
