"""Robustness view (NOT the verdict; the verdict uses the pre-registered split seed 2026): cross-fitted tuned accuracy
over 200 other stratified splits (seeds 0-199), counted in whole pairs, for the L2 runs near the 0.90 bar."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossfit_acc as cf
out = {}
for d in sys.argv[1:]:
    r = cf.one(d)  # rebuilds the scores (and checks them against the log) - reuse its internals for the scores
    import argparse, htm_driver as hd
    log = open(os.path.join(d, "log.txt")).read().splitlines()
    a = argparse.Namespace(**next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG ")))
    X, y, _, _ = hd.load(a, "dev"); keep = np.random.default_rng(1).permutation(len(y))[:a.n_dev or None]; X, y = X[keep], y[keep].astype(np.int64)
    z = np.load(os.path.join(d, "include_bits.npz")); w = 1 - 2 * (np.arange(a.clauses) % 2)
    cs = [np.clip(hd.clause_values(hd.or_sums(np.unpackbits(z[f"class{c}"], axis=-1, count=a.F).astype(bool), X, a.F, a.A, a.K), a.K, False) @ w, -a.T, a.T) for c in (0, 1)]
    score = cs[1] - cs[0]
    vals = np.array([cf.crossfit(score, y, seed=s)[0] for s in range(200)])
    correct = np.rint(vals * len(y)).astype(int)  # whole pairs (both halves have equal size when the label counts are even)
    out[os.path.basename(d)[17:]] = {"seed2026": r["crossfit_tuned_acc"], "mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
                                     "share_at_least_0.90": float((correct >= 0.9 * len(y)).mean()), "n_dev": len(y), "labels": np.bincount(y).tolist()}
    print(os.path.basename(d)[17:], json.dumps(out[os.path.basename(d)[17:]]))
json.dump({k: {"value": v, "label": "derived"} for k, v in out.items()}, open(os.path.join(os.environ.get("TMLM_RUN_DIR", "."), "metrics.json"), "w"), indent=1)
