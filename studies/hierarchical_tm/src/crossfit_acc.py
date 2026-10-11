"""Cross-fitted tuned accuracy (pre-registered on 2 Oct 2026, 07:34 CEST, see docs/registrations.md; a measurement for decision 4, not a gate).

Usage: python crossfit_acc.py RUN_DIR [RUN_DIR ...]   (finished htm_driver.py pair runs; run under run_logged.py)
For each run: rebuild the final model's dev scores in numpy from include_bits.npz (both class machines, counting
inference, clipped to +-T), on the driver's own dev pairs; require that the rebuilt dev AUC and plain accuracy equal
the run's FINAL values exactly; then report plain accuracy, tuned accuracy (cut-off chosen on all dev pairs) and
cross-fitted tuned accuracy:
  - split the dev pairs into two halves, stratified by label, numpy seed 2026;
  - on one half, choose c for "same iff score >= c" over every observed score value plus "never same", by accuracy;
    ties go to the largest c; apply that c unchanged to the other half; swap; report the mean of the two.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htm_driver as hd  # noqa: E402


def best_cut(score, y):
    """(c, accuracy) of the best rule 'same iff score >= c' on these pairs; 'never same' is c = +inf; ties -> largest c."""
    o = np.argsort(-score, kind="stable")
    s, yy = score[o], y[o].astype(np.int64)
    tp, fp, n0 = np.cumsum(yy), np.cumsum(1 - yy), len(y) - int(yy.sum())
    end = np.r_[s[1:] != s[:-1], True]
    acc = np.r_[n0, tp[end] + n0 - fp[end]] / len(y)
    cuts = np.r_[np.inf, s[end]]
    k = int(np.argmax(acc))  # candidates run from the largest c down, so the first maximum is the largest c
    return float(cuts[k]), float(acc[k])


def crossfit(score, y, seed=2026):
    rng, first = np.random.default_rng(seed), np.zeros(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        first[idx[:len(idx) // 2]] = True
    out = []
    for fit, ev in ((first, ~first), (~first, first)):
        c, _ = best_cut(score[fit], y[fit])
        out.append({"cut": c, "held_out_acc": float(((score[ev] >= c).astype(int) == y[ev]).mean())})
    return float(np.mean([o["held_out_acc"] for o in out])), out


def one(d):
    log = open(os.path.join(d, "log.txt")).read().splitlines()
    a = argparse.Namespace(**next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG ")))
    fin = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("FINAL "))
    X, y, _, _ = hd.load(a, "dev")
    keep = np.random.default_rng(1).permutation(len(y))[:a.n_dev or None]  # the driver's dev subset
    X, y = X[keep], y[keep].astype(np.int64)
    z = np.load(os.path.join(d, "include_bits.npz"))
    w = 1 - 2 * (np.arange(a.clauses) % 2)
    cs = [np.clip(hd.clause_values(hd.or_sums(np.unpackbits(z[f"class{c}"], axis=-1, count=a.F).astype(bool),
                                              X, a.F, a.A, a.K), a.K, a.and_group_normalization) @ w, -a.T, a.T)
          for c in (0, 1)]
    score, plain = cs[1] - cs[0], (cs[1] > cs[0]).astype(int)  # np.argmax ties go to class 0
    auc, acc = hd.auc(score, y), float((plain == y).mean())
    exact = bool(auc == fin["dev_auc"] and acc == fin["dev_acc"])
    tuned = hd.tuned_acc(score, y)
    assert abs(best_cut(score, y)[1] - tuned) < 1e-12, "best_cut disagrees with htm_driver.tuned_acc"
    cf, halves = crossfit(score, y)
    return {"run": os.path.basename(d), "s": a.s, "passes": fin["pass"], "n_dev": len(y), "rebuilt_equals_logged": exact,
            "dev_auc": auc, "plain_acc": acc, "pred_pos": float(plain.mean()), "tuned_acc": tuned,
            "crossfit_tuned_acc": cf, "halves": halves}


def main():
    rows = [one(d.rstrip("/")) for d in sys.argv[1:]]
    for r in rows:
        print("CROSSFIT", json.dumps(r))
    print("\n| run | s | passes | rebuilt = logged | AUC | plain | tuned | cross-fitted |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run'][17:]} | {r['s']:g} | {r['passes']} | {r['rebuilt_equals_logged']} | {r['dev_auc']:.4f} | "
              f"{r['plain_acc']:.4f} | {r['tuned_acc']:.4f} | {r['crossfit_tuned_acc']:.4f} |")
    out = os.environ.get("TMLM_RUN_DIR", ".")
    json.dump({r["run"]: {"value": {k: v for k, v in r.items() if k != "run"}, "label": "measured"} for r in rows},
              open(os.path.join(out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
