"""A1 A/A noise runs: how far the numbers of the count floor move under an honest repeat.

    python scripts/a1_aa_noise.py [--seeds 1-10] [--drop 0.05]

A baseline measurement, not a registered run; no Tsetlin part. The count models are
deterministic, so two runs on the same data agree exactly. An A/A repeat therefore drops a
seeded share (5%) of the train blocks (floor/aa.py, subsample_train; calib and valid never
change) and rebuilds the systems of scripts/a1_baseline_check.py from the kept blocks: own KN3
and KN5 (sentence mode) and KN5 + cache with a reset per block, whose half-life and <eos> skip
are chosen on valid and whose mixing weights are fitted by EM on calib (floor/aa.py, repeat).
Repeat 0 keeps every train block and must reproduce the valid perplexities of the A1 baseline
check (run 20261015T072513Z_a1_baseline_check: own KN3 162.3057, KN5 152.7531, KN5 + cache with
a reset per block 132.9300, to the printed digits).

Reported on valid (73,760 tokens incl. <eos>, V = 10,000), per system:
- the spread of perplexity over the repeats 1..n (mean, sd, min, max);
- for every pair of repeats, the paired difference in bits per token with a 95% interval from
  1,000 resamples of the valid blocks (stats.block_bootstrap, seed 0), the relative perplexity,
  and whether the interval excludes 0; summarised over all pairs and over the disjoint pairs
  (1, 2), (3, 4), ...;
- every repeat against repeat 0 (the cost of dropping the blocks; not an A/A pair).
The block bootstrap resamples evaluation text only. The share of A/A pairs whose interval
excludes 0 shows how much a change on the train side moves a number beyond what that interval
covers. The section 4 contract runs on 1,000 random valid positions of KN5 + cache in every
repeat. ptb.load() builds every split; the dev-test stream is dropped at once and never scored.
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally import ptb  # noqa: E402
from tally.floor import aa  # noqa: E402
from tally.runlog import Run  # noqa: E402

A1_RUN = "20261015T072513Z_a1_baseline_check"
A1_VALID_PPL = {"kn3": 162.3056848209348, "kn5": 152.75306445934226, "kn5_cache_block": 132.92997374986763}


def ppl(lp):
    return float(np.exp(-np.mean(lp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10", help="A/A repeats, as lo-hi")
    ap.add_argument("--drop", type=float, default=0.05, help="share of train blocks dropped per repeat")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    cfg = {"repeats": seeds, "repeat_0": "every train block (reproduction of the A1 baseline check)",
           "drop_share": a.drop, "drop_rule": "floor/aa.py subsample_train(seed): whole train blocks",
           "systems": list(aa.SYSTEMS), "mode": "sent", "cache_reset": "block",
           "half_lives": [float(h) for h in aa.HALF_LIVES], "cache_select": "valid", "em_fit": "calib",
           "bootstrap": {"n_boot": 1000, "level": 0.95, "seed": 0, "unit": "valid blocks of 100 sentences"},
           "contract_positions": 1000, "eval_split": "valid", "devtest": "dropped after loading, never scored",
           "reproduce": {"run": A1_RUN, "valid_ppl": A1_VALID_PPL}}
    run = Run("a1_aa_noise", cfg, data_files=[ptb.path("train"), ptb.path("valid")])
    T = {}
    t = time.time()
    vocab, w2i, ids, blocks = ptb.load()
    for d in (ids, blocks):
        del d["devtest"]
    assert set(ids) == set(blocks) == {"train", "calib", "valid"}
    T["load"] = time.time() - t
    V, eos = len(vocab), w2i["<eos>"]
    run.print("tokens", {s: int(len(ids[s])) for s in ids}, "V", V, "train blocks",
              int(len(np.unique(blocks["train"]))), "valid blocks", int(len(np.unique(blocks["valid"]))))

    LP, R = {}, {}
    for seed in [0] + seeds:
        t = time.time()
        if seed == 0:
            tr, dropped = ids["train"], np.zeros(0, dtype=np.int64)
        else:
            tr, _, dropped = aa.subsample_train(ids["train"], blocks["train"], seed, a.drop)
        lp, info = aa.repeat(tr, V, eos, ids, blocks, contract_n=1000)
        LP[seed] = {k: v for k, v in lp.items() if k[1] == "valid"}
        R[seed] = {"dropped_blocks": [int(x) for x in dropped], "train_tokens": info["train_tokens"],
                   "valid_ppl": {s: ppl(lp[(s, "valid")]) for s in aa.SYSTEMS},
                   "valid_bits": {s: float(-np.mean(lp[(s, "valid")]) / np.log(2)) for s in aa.SYSTEMS},
                   "calib_ppl": {s: ppl(lp[(s, "calib")]) for s in aa.SYSTEMS},
                   "cache_chosen": info["cache"]["chosen"], "lam": info["cache"]["lam"],
                   "contract_valid": info["contract_valid"], "kn5_discounts": info["kn5"]["discounts"],
                   "seconds": round(time.time() - t, 2)}
        assert info["contract_valid"]["pass"], f"contract failed in repeat {seed}: {info['contract_valid']}"
        run.print(f"repeat {seed}: dropped {len(dropped)} blocks, train {info['train_tokens']} tokens, valid ppl "
                  + ", ".join(f"{s} {R[seed]['valid_ppl'][s]:.4f}" for s in aa.SYSTEMS)
                  + f"; cache {info['cache']['chosen']}; contract {info['contract_valid']['pass']}"
                  + f" ({R[seed]['seconds']} s)")
    T["repeats"] = sum(R[s]["seconds"] for s in R)
    repro = {s: {"a1": A1_VALID_PPL[s], "here": R[0]["valid_ppl"][s],
                 "abs_diff": abs(R[0]["valid_ppl"][s] - A1_VALID_PPL[s])} for s in aa.SYSTEMS}
    repro_ok = all(r["abs_diff"] < 1e-6 * r["a1"] for r in repro.values())
    run.print("repeat 0 against the A1 baseline check:", repro, "reproduced" if repro_ok else "NOT REPRODUCED")

    t = time.time()
    bv = blocks["valid"]
    M = {"repeats": {str(k): v for k, v in R.items()}, "reproduction_of_a1": dict(repro, ok=repro_ok),
         "spread": {}, "pairs": {}, "vs_repeat_0": {}, "summary": {}}
    disjoint = {(seeds[i], seeds[i + 1]) for i in range(0, len(seeds) - 1, 2)}
    for s in aa.SYSTEMS:
        M["spread"][s] = {"valid_ppl": aa.spread([R[k]["valid_ppl"][s] for k in seeds]),
                          "valid_bits": aa.spread([R[k]["valid_bits"][s] for k in seeds])}
        prs = {}
        for i, j in itertools.combinations(seeds, 2):
            prs[f"{i}-{j}"] = dict(aa.pair(LP[i][(s, "valid")], LP[j][(s, "valid")], bv), disjoint=(i, j) in disjoint)
        M["pairs"][s] = prs
        M["vs_repeat_0"][s] = {str(k): aa.pair(LP[k][(s, "valid")], LP[0][(s, "valid")], bv) for k in seeds}
        allp = list(prs.values())
        dis = [p for p in allp if p["disjoint"]]
        half = [(p["hi"] - p["lo"]) / 2 for p in allp]
        M["summary"][s] = {
            "pairs": len(allp), "pairs_interval_excludes_0": int(sum(p["excludes_zero"] for p in allp)),
            "disjoint_pairs": len(dis), "disjoint_interval_excludes_0": int(sum(p["excludes_zero"] for p in dis)),
            "max_abs_rel_ppl": float(max(abs(p["rel_ppl"]) for p in allp)),
            "sd_rel_ppl_over_pairs": float(np.std([p["rel_ppl"] for p in allp], ddof=1)),
            "max_abs_mean_bits": float(max(abs(p["mean"]) for p in allp)),
            "median_interval_half_width_bits": float(np.median(half)),
            "mean_rel_ppl_vs_repeat_0": float(np.mean([M["vs_repeat_0"][s][str(k)]["rel_ppl"] for k in seeds]))}
    T["pairs"] = time.time() - t
    run.print("")
    run.print(f"{'system':16s} {'mean ppl':>9s} {'sd':>7s} {'min':>9s} {'max':>9s} {'rel sd':>8s}  "
              f"{'pairs CI excl. 0':>17s} {'disjoint':>9s} {'max |rel ppl|':>14s} {'median half-width bits':>23s}"
              f"   (valid, {len(ids['valid'])} tokens, {len(seeds)} repeats, {a.drop:.0%} of train blocks dropped)")
    for s in aa.SYSTEMS:
        sp, sm = M["spread"][s]["valid_ppl"], M["summary"][s]
        run.print(f"{s:16s} {sp['mean']:9.4f} {sp['sd']:7.4f} {sp['min']:9.4f} {sp['max']:9.4f} {sp['rel_sd']:8.5f}  "
                  f"{sm['pairs_interval_excludes_0']:>8d} of {sm['pairs']:<6d} {sm['disjoint_interval_excludes_0']:>3d} of "
                  f"{sm['disjoint_pairs']:<3d} {sm['max_abs_rel_ppl']:14.5f} {sm['median_interval_half_width_bits']:23.6f}")
    M["timings_s"] = {k: round(v, 2) for k, v in T.items()}
    run.print("timings", M["timings_s"])
    run.finish(M)
    if not repro_ok:
        sys.exit("repeat 0 does not reproduce the A1 baseline check")


if __name__ == "__main__":
    main()
