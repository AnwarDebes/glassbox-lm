"""Train-side noise of the A2 gain (finish pass 2026-10-25, agent tal; dev work on valid and calib, no rule,
no run folder, kept outside the repository).

    cd studies/tally && nice -n 19 ../tm_language_model/.venv-tm/bin/python -I <this file> [--seeds 0-10]

Answers the review point on design rule 9 (10 seeds for a pass or fail claim): the A2 arms are
deterministic, so a repeat is defined as in A1's A/A runs: a seeded 5% of the train blocks is
dropped (floor/aa.py subsample_train, the seeds of run 20261025T113715Z_a1_aa_noise), and
everything built from train is rebuilt from the kept blocks: the exchange code (16/64/256/1024,
codes.exchange_tree), KN5, the four exchange class members; the cache is chosen on valid and the
mixing weights are fitted by EM on calib, as in scripts/a2_run.py. Read per repeat: the gain
1 - ppl(exchange) / ppl(none) on valid, and on calib with the EM weights cross-fitted over two
halves of the calib blocks. Repeat 0 keeps every block and must give the development run's valid
gain (run 20261025T114718Z_a2_valid_dev: 0.08109326649933024). The dev-test stream is dropped
right after loading and never touched.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

STUDY = os.getcwd()
sys.path.insert(0, os.path.join(STUDY, "src"))
from tally import ptb  # noqa: E402
from tally.floor import aa, cache, codes, mix  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

HALF_LIVES = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)
DEV_GAIN = 0.08109326649933024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-10")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    vocab, w2i, ids, blocks = ptb.load()
    for d in (ids, blocks):
        del d["devtest"]
    assert set(ids) == {"train", "calib", "valid"}
    V, eos = len(vocab), w2i["<eos>"]
    cb = blocks["calib"]
    half = np.isin(cb, np.unique(cb)[::2])
    rows = []
    for seed in range(lo, hi + 1):
        t = time.time()
        if seed == 0:
            tr, dropped = ids["train"], []
        else:
            tr, _, dropped = aa.subsample_train(ids["train"], blocks["train"], seed, 0.05)
        C, _ = codes.exchange_tree(tr, V, eos)
        kn5 = KN(tr, V, eos, order=5, mode="sent")
        pk = {s: np.exp(kn5.logprob(ids[s])) for s in ("calib", "valid")}
        best = None
        for hl in HALF_LIVES:
            for skip in ((), (eos,)):
                c = cache.Cache(V, hl, "block", skip)
                lam, _ = mix.em(np.stack([pk["calib"], c.prob(ids["calib"], cb, pk["calib"])[0]], 1))
                ll = float(np.log(np.stack([pk["valid"], c.prob(ids["valid"], blocks["valid"], pk["valid"])[0]], 1) @ lam).sum())
                if best is None or ll > best[0]:
                    best = (ll, c)
        cch = best[1]
        pc = {s: cch.prob(ids[s], blocks[s], pk[s])[0] for s in ("calib", "valid")}
        mem = []
        for l in range(C.shape[1]):
            cm = ClassMember(tr, C[:, l], eos, 5, 0.5)
            mem.append({s: np.exp(cm.logprob(ids[s])) for s in ("calib", "valid")})
        out = {}
        for arm, ms in (("none", []), ("exchange", mem)):
            cal = np.stack([pk["calib"], pc["calib"]] + [m["calib"] for m in ms], 1)
            val = np.stack([pk["valid"], pc["valid"]] + [m["valid"] for m in ms], 1)
            lam, _ = mix.em(cal)
            lcf = np.empty(len(cb))
            for fit in (half, ~half):
                lf, _ = mix.em(cal[fit])
                lcf[~fit] = np.log(cal[~fit] @ lf)
            out[arm] = {"valid_ppl": float(np.exp(-np.log(val @ lam).mean())),
                        "calib_crossfit_ppl": float(np.exp(-lcf.mean())), "lam": [round(float(x), 5) for x in lam]}
        r = {"seed": seed, "dropped_blocks": len(dropped), "train_tokens": int(len(tr)), "cache": cch.config(),
             "none": out["none"], "exchange": out["exchange"],
             "gain_valid": 1 - out["exchange"]["valid_ppl"] / out["none"]["valid_ppl"],
             "gain_calib_crossfit": 1 - out["exchange"]["calib_crossfit_ppl"] / out["none"]["calib_crossfit_ppl"],
             "seconds": round(time.time() - t, 1)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    rep = [r for r in rows if r["seed"] > 0]
    summ = {}
    for k in ("gain_valid", "gain_calib_crossfit"):
        v = np.array([r[k] for r in rep])
        if len(v):
            summ[k] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "min": float(v.min()), "max": float(v.max())}
    r0 = [r for r in rows if r["seed"] == 0]
    if r0:
        summ["repeat0_reproduces_dev_gain"] = bool(abs(r0[0]["gain_valid"] - DEV_GAIN) < 1e-12)
    print("SUMMARY " + json.dumps(summ))


if __name__ == "__main__":
    main()
