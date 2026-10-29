"""C-core stage gates (PLAN_part3.md section 4: K0, K0b, A/A, shuffled labels; section 6 step 6).

    scripts/run_stage.sh gates           # real: needs the frozen R1 record and the twin and learners outputs
    scripts/run_stage.sh gates --dry

Recomputes every system's valid bpw from the saved per-position log probabilities and checks it against
the stage records (|difference| <= 1e-12), applies the registered gate arithmetic (ccore/gates.py),
and reports paired block-bootstrap intervals (100-sentence blocks, 1,000 resamples, seed 0), the
duplicate share of valid (spans of 8 or more tokens that occur in train) with every system's bpw with
those tokens removed, and the rare-context split. Writes data/step6_decision.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import gates, hashing, runlog, systems  # noqa: E402
from tally.floor import stats  # noqa: E402

BOOT = {"n_boot": 1000, "seed": 0, "level": 0.95}


def stage_record(stage, dry):
    """The finished run of a stage: exactly one for a real run (a second look would void the gates);
    the latest one for a dry run."""
    runs = runlog.finished_runs(stage, dry)
    if not runs or (not dry and len(runs) != 1):
        raise SystemExit(f"expected {'one' if not dry else 'a'} finished {stage} run, found {runs}")
    d = runs[-1]
    return d, json.load(open(Path(d) / "metrics.json")), json.load(open(Path(d) / "outputs.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run("gates", a.dry)
    cfg = {"stage": "gates", "dry": a.dry, "frozen_utc": rec["frozen_utc"] if rec else None,
           "new_files_since_freeze": new, "bootstrap": BOOT, "dup_min_len": 8,
           "gates": {"K0_min_gain": gates.K0_MIN_GAIN, "K0b_max_gap": gates.K0B_MAX_GAP,
                     "AA_max_abs": gates.AA_MAX_ABS, "shuffled_min_diff": gates.SHUFFLED_MIN_DIFF}}
    run = runlog.Run("gates", cfg, dry=a.dry)

    def body():
        d = runlog.out_dir(a.dry)
        used = runlog.check_inputs(["prep", "twin", "learners"], a.dry)   # hashes of what those runs wrote
        run.print("inputs:", used)
        tw_dir, tw, _ = stage_record("twin", a.dry)
        le_dir, le, _ = stage_record("learners", a.dry)
        z = np.load(d / "prep.npz")
        T = np.load(d / "twin.npz")
        L = np.load(d / "learners.npz")
        assert np.array_equal(T["F_valid"], L["F_valid"])
        lp = {"F": T["F_valid"], "twin": T["best_valid"], "LR": L["lr_valid"], "LightGBM": L["lgbm_valid"],
              "twin_AA": T["aa_valid"], "twin_shuffled": T["shuffled_valid"], "LR_shuffled": L["lr_shuffled_valid"],
              "LightGBM_shuffled": L["lgbm_shuffled_valid"], "class_unigram_diagnostic": T["null_valid"]}
        bpw = {k: systems.bpw(v) for k, v in lp.items()}
        ref = {"F": tw["F_valid_bpw"], "twin": tw["trials"][tw["selected_index"]]["valid_bpw"],
               "LR": le["best_lr_bpw"], "LightGBM": le["best_lgbm_bpw"], "twin_AA": tw["AA"]["rebuilt_bpw"],
               "twin_shuffled": tw["shuffled"]["bpw"], "LR_shuffled": le["shuffled"]["LR"]["valid_bpw"],
               "LightGBM_shuffled": le["shuffled"]["LightGBM"]["valid_bpw"],
               "class_unigram_diagnostic": tw["class_unigram_diagnostic"]["bpw"]}
        mism = {k: (bpw[k], ref[k]) for k in ref if abs(bpw[k] - ref[k]) > 1e-12}
        if mism:
            raise SystemExit(f"recomputed bpw differ from the stage records: {mism}")
        k0r = gates.k0(bpw["F"], bpw["twin"])
        k0br = gates.k0b(bpw["twin"], bpw["LR"], bpw["LightGBM"])
        aar = gates.aa(bpw["twin"], bpw["twin_AA"])
        shr = {k: gates.shuffled(bpw["F"], bpw[k]) for k in ("twin_shuffled", "LR_shuffled", "LightGBM_shuffled")}
        decision = gates.step6_decision(k0r, k0br, aar, shr)
        blk = z["blocks_valid"]

        def ci(x, y):
            return stats.block_bootstrap(stats.bits_diff(lp[x], lp[y]), blk, BOOT["n_boot"], BOOT["seed"], BOOT["level"])
        intervals = {"twin - F": ci("twin", "F"), "LR - twin": ci("LR", "twin"), "LightGBM - twin": ci("LightGBM", "twin"),
                     "LR - F": ci("LR", "F"), "LightGBM - F": ci("LightGBM", "F"), "twin_AA - twin": ci("twin_AA", "twin"),
                     "twin_shuffled - F": ci("twin_shuffled", "F"), "LR_shuffled - F": ci("LR_shuffled", "F"),
                     "LightGBM_shuffled - F": ci("LightGBM_shuffled", "F"),
                     "class_unigram_diagnostic - F": ci("class_unigram_diagnostic", "F")}
        rep, dup = stats.duplicate_report(z["w_train"], z["blocks_train"], z["w_valid"], "valid", 8)
        dup_removed = {k: systems.bpw(v[~dup]) for k, v in lp.items()}
        rare = z["L_valid"] <= 2
        rare_split = {k: {"rare": systems.bpw(v[rare]), "other": systems.bpw(v[~rare])} for k, v in lp.items()}
        ppl = {k: float(2 ** v) for k, v in bpw.items()}
        res = {"decision": decision, "K0": k0r, "K0b": k0br, "AA": aar, "shuffled": shr, "valid_bpw": bpw,
               "valid_ppl": ppl, "intervals_bits": intervals, "duplicates": rep, "dup_removed_bpw": dup_removed,
               "rare_share": float(rare.mean()), "rare_split_bpw": rare_split, "twin_run": tw_dir,
               "learners_run": le_dir, "selected": {"twin": tw["selected"], "LR_C": le["selected_lr_C"],
                                                    "LightGBM": le["selected_lgbm"]},
               "tokens_valid": int(len(blk)), "label": "valid reading for the CPU gates; not a verdict"}
        out = d / "step6_decision.json"
        hashing.write_json(out, res)
        run.add_output(out)
        for k in bpw:
            run.print(f"{k:28s} valid {bpw[k]:.5f} bpw  ppl {ppl[k]:8.3f}  dup-removed {dup_removed[k]:.5f}")
        for k, v in intervals.items():
            run.print(f"{k:32s} {v['mean']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}] bits")
        run.print("K0", k0r, "\nK0b", k0br, "\nA/A", aar, "\nshuffled", shr)
        run.print("DECISION:", decision)
        run.finish(res)

    runlog.main_wrapper(run, body)


if __name__ == "__main__":
    main()
