#!/usr/bin/env python
"""S2a (CPU): cross-fitted rows for the count twin (every fit position) and for the learned arms (the training
dialogues), tune-cal and tune-hold rows from the full-fit floor and rates; the leak, no-future and duplicate checks.

    chain/part3_cpu_job.sh A tm s2a_rows.py [--s0 DIR] [--s1 DIR] [--out DIR] [--logmd LOG.md]

Refuses to start unless S1 passed K1 and LOG.md holds the SHA-256 of S0's bit_list.json (the frozen bit list must be
logged before the first tune-cal score, review 2).
Checks (each stops the job when it fails):
- the rebuilt full-fit floor gives S1's tune-cal and tune-hold candidates and probabilities exactly;
- review 2 (i): every training row whose act-slot set occurs once in fit has the act-set rate of its back-off; and
  (amendment 1) every S-keyed context lookup of its fold's floor is unseen and S's count there is 0;
- no-future (200 random tune-hold positions, seed 0): the reply cut after the position's token, and the reply with
  every later token replaced by random tokens, give the same candidates, floor numbers and bits at that position;
  the reply with the position's own token replaced (and later tokens random) gives the same candidates, floor
  numbers and bits except the label and the true token's probability.
Duplicates (reported): tune positions inside a span of >= 8 tokens that also occurs in one fit reply.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.build import FOLD_SEED, crossfit, folds_of, make_floor, rows_for, save_rows, subset_rows
from astruct.data import CAL, FIT, HOLD, Corpus
from astruct.positions import Positions
from astruct.rates import Rates

paths.add_study_paths(tally=True)
from tally.floor import stats as tstats

N_NOFUTURE = 200
CHECK_KEYS = ("cand", "cand_p", "rank", "pb", "type5", "due3", "said", "ident", "dset", "rate_da", "rate_set", "bits")


def guard(args) -> tuple[dict, dict]:
    s1d = Path(args.s1)
    done = json.loads((s1d / "done.json").read_text())
    if not done.get("K1"):
        raise SystemExit("S1 did not pass K1: stop and fix the KN code first")
    s0 = json.loads((Path(args.s0) / "done.json").read_text())
    bl = paths.sha256_file(Path(args.s0) / "bit_list.json")
    if bl != s0["bit_list_sha256"]:
        raise SystemExit("bit_list.json changed after S0")
    if bl not in Path(args.logmd).read_text():
        raise SystemExit(f"LOG.md has no line with the bit-list SHA-256 {bl}; log it before S2a")
    return json.loads((s1d / "results.json").read_text()), {"bit_list_sha256": bl}


def duplicates(c: Corpus, fit: np.ndarray, P: Positions) -> np.ndarray:
    Pf = Positions(c, fit)
    blocks = 2 * Pf.local  # windows never cross two fit replies
    return tstats.duplicate_mask(Pf.stream, blocks, P.stream, min_len=8)


def nofuture(c: Corpus, fs, rates, hold: np.ndarray, hold_rows: dict, floor: str, log) -> tuple[dict, dict]:
    """Three variants of each sampled reply: "cut" (nothing after the position's token but the end token),
    "scramble" (every later token random) and "swap" (the position's own token replaced, the rest random).
    cut / scramble: everything at the position is identical, bits included. swap: everything but the label and the
    true token's probability is identical, so no feature reads the token it predicts."""
    rng = np.random.default_rng(0)
    k_all = rng.choice(hold_rows["n_pos"], size=min(N_NOFUTURE, hold_rows["n_pos"]), replace=False)
    items, where = [], []
    words = np.flatnonzero(~c.is_ph & (np.arange(c.V) != c.END))
    for variant in ("cut", "scramble", "swap"):
        for k in k_all:
            r, t = int(hold_rows["pos_reply"][k]), int(hold_rows["pos_t"][k])
            tok = c.tokens(r)
            head = list(map(int, tok[: t + 1]))
            n_after = len(tok) - t - 2  # tokens strictly between the position and the end token
            if variant == "swap":
                other = [int(w) for w in words if w != tok[t]]
                new = head[:t] + [int(rng.choice(other))] + [int(x) for x in rng.choice(words, max(n_after, 0))] + [c.END]
            elif t == len(tok) - 1:  # the end token itself: nothing follows
                new = head
            elif variant == "cut":
                new = head + [c.END]
            else:
                new = head + [int(x) for x in rng.choice(words, n_after)] + [c.END]
            items.append((r, np.array(new)))
            where.append((variant, int(k), t))
    c2 = c.append_replies(items)
    rows_mod, _ = rows_for(c2, fs, rates, np.arange(c.n, c2.n), floor, bits=True)
    starts = np.concatenate([[0], np.cumsum([len(t) for _, t in items])])
    mod_pos = np.array([starts[j] + t for j, (_, _, t) in enumerate(where)], np.int64)
    out, bad = {}, {}
    for variant, keys in (("same", CHECK_KEYS + ("y", "p_true", "top1", "ent")), ("swap", CHECK_KEYS + ("top1", "ent"))):
        sel = np.array([(v == "swap") == (variant == "swap") for v, _, _ in where])
        sm = subset_rows(rows_mod, mod_pos[sel])
        sr = subset_rows(hold_rows, np.array([k for (v, k, _), s_ in zip(where, sel) if s_], np.int64))
        bad[variant] = [key for key in keys if sm[key].shape != sr[key].shape or not np.array_equal(sm[key], sr[key])]
        out[variant] = (sm, sr)
    swapped = out["swap"][0]["pos_x"] != out["swap"][1]["pos_x"]
    res = {"positions": len(k_all), "variants": ["cut", "scramble", "swap"], "mismatched_fields": bad,
           "swap_targets_changed": int(swapped.sum()), "pass": not bad["same"] and not bad["swap"] and bool(swapped.all())}
    log("no-future (floor and bits)", json.dumps(res))
    return res, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=str(paths.OUT / "s0"))
    ap.add_argument("--s1", default=str(paths.OUT / "s1"))
    ap.add_argument("--out", default=str(paths.OUT / "s2a"))
    ap.add_argument("--logmd", default=str(paths.A_DIR.parent / "LOG.md"))
    args = ap.parse_args()
    env = paths.check_env()
    s1, info = guard(args)
    out = paths.OutDir(Path(args.out))
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    c = Corpus.load(Path(args.s0))
    fit, cal, hold = c.replies(FIT), c.replies(CAL), c.replies(HOLD)
    floor = s1["floor"]
    log("S2a start; floor", floor, "picks", json.dumps(s1["picks"]), "| bit list", info["bit_list_sha256"])
    P_cal = Positions(c, cal)
    fold = folds_of(c, fit, FOLD_SEED)
    train_reply = c.a["r_train"][fit]
    cf = crossfit(c, fit, fold, s1, P_cal, train_reply, bits=True, log=log)
    leak = cf["leak"]
    leak["pass"] = (leak["rows"] == leak["equal"] and leak["S_lookups"] == leak["S_lookups_unseen"]
                    and leak["S_positions"] == leak["S_positions_count_zero"])
    log("leak test (singleton act-slot sets)", json.dumps(leak))
    if not leak["pass"]:
        raise SystemExit("review 2 leak test failed")
    fs = make_floor(c, fit, s1, P_cal, log=log)
    rates = Rates(c, fit)
    cal_rows, _ = rows_for(c, fs, rates, cal, floor, bits=True)
    hold_rows, _ = rows_for(c, fs, rates, hold, floor, bits=True)
    for nm, rws in (("cal", cal_rows), ("hold", hold_rows)):
        z = np.load(Path(args.s1) / f"summ_{nm}.npz")
        if not (np.array_equal(z["cand"], rws["cand"]) and np.array_equal(z["cand_p"], rws["cand_p"])
                and np.array_equal(z["p_true"], rws["p_true"])):
            raise SystemExit(f"the rebuilt floor does not reproduce S1 on tune-{nm}")
    log("rebuilt full-fit floor reproduces S1 exactly on tune-cal and tune-hold")
    nf, nfrows = nofuture(c, fs, rates, hold, hold_rows, floor, log)
    if not nf["pass"]:
        raise SystemExit(f"no-future test failed: {nf['mismatched_fields']}")
    dup_hold = duplicates(c, fit, Positions(c, hold))
    dup_cal = duplicates(c, fit, Positions(c, cal))
    log(f"duplicates: tune-hold {dup_hold.mean():.4f} of {len(dup_hold)} tokens, tune-cal {dup_cal.mean():.4f}")
    save_rows(out / "twin_rows.npz", cf["twin"])
    save_rows(out / "train_rows.npz", cf["train"])
    save_rows(out / "cal_rows.npz", cal_rows)
    save_rows(out / "hold_rows.npz", hold_rows)
    for variant, (sm, sr) in nfrows.items():  # for the arm-level checks of S2b and K3c
        save_rows(out / f"nofuture_{variant}_mod.npz", sm)
        save_rows(out / f"nofuture_{variant}_ref.npz", sr)
    np.savez_compressed(out / "dup.npz", hold=dup_hold, cal=dup_cal)
    np.savez_compressed(out / "folds.npz", fit=fit, fold=fold)
    res = {"floor": floor, "picks": s1["picks"], "rows": {"twin": len(cf["twin"]["y"]), "train": len(cf["train"]["y"]),
           "cal": len(cal_rows["y"]), "hold": len(hold_rows["y"])},
           "positions": {"twin": cf["twin"]["n_pos"], "train": cf["train"]["n_pos"], "cal": cal_rows["n_pos"], "hold": hold_rows["n_pos"]},
           "recall_of_C": {"cal": float(np.bincount(cal_rows["pos"], weights=cal_rows["y"], minlength=cal_rows["n_pos"]).mean()),
                           "hold": float(np.bincount(hold_rows["pos"], weights=hold_rows["y"], minlength=hold_rows["n_pos"]).mean())},
           "leak_test": leak, "nofuture": nf, "duplicates": {"hold_share": float(dup_hold.mean()), "cal_share": float(dup_cal.mean())},
           "fold_seed": FOLD_SEED, "seconds": round(time.time() - t0, 1), "env": env, "code_sha256": paths.code_hashes(), **info}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    log("S2a done", json.dumps(res["rows"]), "recall of C", json.dumps(res["recall_of_C"]))
    out.done({"utc_end": paths.utc(), "rows": res["rows"]})


if __name__ == "__main__":
    main()
