"""C-core Step 0 (PLAN_part3.md section 4; section 6, step 2): free and exploratory, no claim, no gate.

    scripts/run_stage.sh step0          # real: needs the frozen R1 record; scores valid and dev-test
    scripts/run_stage.sh step0 --dry    # pseudo splits from train blocks; synthetic T0 bits

Order: the T0 arrays are checked against T0's logged bits per word and the tally block ids; the floor F
is built (counts from train, weights on calib) and its contract runs on 1,000 positions of every scored
split; only then is any number logged. Each T0 arm is mixed with F per bucket (weights on calib).
Outputs: data/step0_bits.npz (per-position bits of F and of every arm's seed-mean mixture) and the
run record under runs/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import data, hashing, runlog, step0, systems  # noqa: E402
from ccore import floor as floor_mod  # noqa: E402
from ccore.floor import Floor  # noqa: E402
from tally.floor import stats  # noqa: E402

CONTRACT_SEEDS = {"calib": 21, "valid": 22, "devtest": 23}
BOOT = {"n_boot": 1000, "seed": 0, "level": 0.95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run("step0", a.dry)
    P = data.load_ptb()
    cfg = {"stage": "step0", "label": "exploratory, no claim, no gate", "dry": a.dry,
           "frozen_utc": rec["frozen_utc"] if rec else None, "new_files_since_freeze": new,
           "arms": {k: [v[0], list(v[1])] for k, v in step0.ARMS.items()}, "seeds": list(hashing.SEEDS),
           "mixture": "F and P_T0, bucketed MAP-EM on calib (40 buckets, tau 20)",
           "scored": list(step0.SCORED), "bootstrap": BOOT, "contract_seeds": CONTRACT_SEEDS,
           "floor": "KN5 sent + cache hl 100 block reset, <eos> skipped; bucketed EM on calib"}
    run = runlog.Run("step0", cfg, dry=a.dry)

    def body():
        T0, chk = step0.load_t0(P, log=run.print)
        Q = data.dry_splits(P) if a.dry else P
        ids, bl = Q["ids"], Q["blocks"]
        fl = Floor(ids, bl, Q["V"], Q["eos"])
        F, contract = {}, {}
        for s in step0.SPLITS:
            pF, b, _ = fl.prob(ids[s], bl[s])
            F[s] = (pF, b)
            r = stats.contract(lambda pos, s=s, b=b: fl.dists(ids[s], bl[s], pos, b), np.log(pF), len(ids[s]),
                               1000, seed=CONTRACT_SEEDS[s], true_ids=ids[s])
            contract[s] = r
            if not r["pass"]:
                raise SystemExit(f"floor contract failed on {s}: {r}")
        run.print("floor contract passed on", list(contract))
        a1 = floor_mod.a1_check(fl, ids, bl, Q["V"], Q["eos"], compare=not a.dry)
        if not a.dry:
            run.print("floor code reproduces A1's own valid numbers:", {k: v["diff"] for k, v in a1.items()})
        if a.dry:                       # synthetic T0 bits of the pseudo splits (the real ones are checked above)
            rng = np.random.default_rng(0)
            T0 = {k: {f"{key.split('__')[0]}__{s}": -np.log2(np.minimum(F[s][0] * rng.uniform(0.2, 3.0, len(F[s][0])), 1.0))
                      for key in v for s in step0.SPLITS} for k, v in T0.items()}
        res, mean_bits, F_bits = step0.run(
            T0, F, hashing.SEEDS, {s: bl[s] for s in step0.SCORED},
            lambda d, blk: stats.block_bootstrap(d, blk, BOOT["n_boot"], BOOT["seed"], BOOT["level"]), log=run.print)
        out = runlog.out_dir(a.dry) / "step0_bits.npz"
        np.savez(out, **{f"F__{s}": F_bits[s] for s in step0.SCORED},
                 **{f"{arm}__{s}": mean_bits[arm][s] for arm in mean_bits for s in step0.SCORED})
        run.add_output(out)
        run.finish({"t0_checks": chk, "floor": fl.summary(), "floor_contract": contract, "a1_check": a1, **res})

    runlog.main_wrapper(run, body)


if __name__ == "__main__":
    main()
