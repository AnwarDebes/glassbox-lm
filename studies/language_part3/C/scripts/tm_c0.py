"""C-core stage tm_c0 (R1 [C-core] section 6, the C0 row: the picked setting on a seeded 1% of the train positions,
10 seeds, the same epoch rule; then the step-9 collection: per-position bits averaged over seeds for the TM, its
epoch-1 checkpoints and C0; PLAN_part3.md section 6, step 9). GPU. Written under the step-9 amendment.

    scripts/run_gpu_stage.sh tm_c0          # real: needs finished tm_grid (PASS) and tm_seeds (PASS) runs
    scripts/run_gpu_stage.sh tm_c0 --dry

C0: one 1% subset of the train positions (subset seed 1), TM seeds 1-10, 20 epochs, the read-out on calib after each
epoch, valid scored once at each seed's kept checkpoint. Collection (valid, reported, no rule; the bars are on the
test file in step 10): the TM arm = the grid's pick at seed 1 and seeds 2-10; the early-checkpoint arm = their
epoch-1 checkpoints; C0 = its 10 seeds. For each arm: per-position bits averaged over its seeds (bits, never
probabilities), the seed-mean bpw and every seed's bpw; differences against the twin, F, LR and LightGBM; the
seed-inclusive two-level interval (ccore/boot.py) beside them. C-K3 on the measured total. Exit 3 on a HOLD.
Outputs: data/tm/c0_seed<k>.npz, data/tm/step9_valid.npz (the per-position bits for step 10's report) and
data/tm/c0_decision.json (with the step-9 summary)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import tmcore  # noqa: E402

DRY = "--dry" in sys.argv[1:]
MODE = tmcore.guard(DRY)
import numpy as np  # noqa: E402

from ccore import boot, hashing, runlog, systems, tmtorch  # noqa: E402

STAGE = "tm_c0"


def load_lp(d, name, key):
    z = np.load(Path(d) / name)
    return z[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_stage = time.time()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run(STAGE, a.dry)
    prof = MODE.profile
    cfg = {"stage": STAGE, "dry": a.dry, "mode": MODE.describe(), "frozen_utc": rec["frozen_utc"] if rec else None,
           "amendment": rec.get("amendment") if rec else None, "new_files_since_freeze": new,
           "profile": tmcore.profile_dict(prof), "c0": prof.c0, "c0_seeds": list(prof.c0_seeds),
           "max_epochs": prof.max_epochs, "clauses": prof.clauses, "K3_limit_min": tmcore.K3_LIMIT_MIN,
           "collection": "per-position bits averaged over seeds; seed-inclusive two-level interval, no rule"}
    run = tmcore.TMRun(STAGE, cfg, dry=a.dry)

    def body():
        torch = tmtorch.setup_torch(MODE.device)
        run.print("torch", torch.__version__, "device", MODE.device, "profile", prof.name)
        used = runlog.check_inputs(["prep", "twin", "learners", "gates"], a.dry)
        tmcore.require_pass("tm_smoke", a.dry)
        dec_grid, m_grid, d_grid = tmcore.require_pass("tm_grid", a.dry)
        dec_seeds, m_seeds, d_seeds = tmcore.require_pass("tm_seeds", a.dry)
        T, s = m_grid["pick_setting"]["T"], m_grid["pick_setting"]["s"]
        if m_seeds["pick_setting"] != m_grid["pick_setting"]:
            raise SystemExit("tm_seeds ran another setting than the grid's pick")
        run.print("inputs:", used, "grid:", d_grid, "seeds:", d_seeds, "pick T", T, "s", s)
        Z = tmcore.load_prep(a.dry)
        e = Z["emission"]
        sp = {k: tmcore.split(Z, k, e) for k in ("calib", "valid")}
        dec6, ref = tmcore.step6(a.dry, Z, e, sp)
        out = tmcore.tm_dir(a.dry)
        idx = tmcore.c0_subset(len(Z["y_train"]), prof)
        run.print(f"C0 subset: {len(idx)} of {len(Z['y_train'])} train positions "
                  f"(subset seed {prof.c0['subset_seed']})")
        trained = []
        for k in prof.c0_seeds:
            r = tmtorch.train(Z["X_train"][idx], Z["y_train"][idx], Z["X_calib"], Z["y_calib"], T, s, k, prof.clauses,
                              prof.max_epochs, MODE.device, log=run.print, tag=f"C0 seed {k}")
            tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": f"C0 seed {k}",
                                 "device": r["device"], "epochs": prof.max_epochs, "batches": r["batches"],
                                 "train_s": r["train_s"], "readout_s": r["readout_s"], "wall_s": r["run_s"]})
            trained.append(r)
        scored = [tmtorch.score_checkpoint(r["cfg"], r["best"], Z, sp, e, MODE.device, f"C0 seed {k}")
                  for k, r in zip(prof.c0_seeds, trained)]
        c0_rows = []
        for k, r, sc in zip(prof.c0_seeds, trained, scored):
            ck = r["best"]
            path = out / f"c0_seed{k}.npz"
            np.savez(path, T=T, s=s, seed=k, idx=idx, ta_best=ck["ta"], w_best=ck["w"], epoch_best=ck["epoch"],
                     ro_best_b=np.asarray(ck["readout"]["b"]), ro_best_a_scaled=ck["readout"]["a_scaled"],
                     ro_best_sigma=ck["readout"]["sigma"],
                     calib_class_bits_per_epoch=np.array([h["calib_class_bits"] for h in r["history"]]),
                     lp_valid_best=sc["lp"], lamb_best=sc["lamb"], pc_valid_best=sc["pc_valid"])
            run.add_output(path)
            c0_rows.append({"seed": k, "epoch_kept": ck["epoch"], "calib_class_bits": ck["calib_class_bits"],
                            "readout": ck["readout"], **sc["info"], "clause_stats": ck["stats"],
                            "length_histogram": tmcore.length_histogram(ck["ta"]), "packed_checks": sc["packed"],
                            "history": r["history"], "cfg": r["cfg"], "train_s": r["train_s"],
                            "readout_s": r["readout_s"], "batches": r["batches"]})

        # the step-9 collection: the TM (seed 1 = the grid's pick; seeds 2-10), its epoch-1 checkpoints, C0
        pick = m_grid["pick"]
        tm_lp = [load_lp(out, f"grid_t{pick}.npz", "lp_valid_best")]
        ep1_lp = [load_lp(out, f"grid_t{pick}.npz", "lp_valid_ep1")]
        for k in [k for k in prof.seeds if k != tmcore.GRID_SEED]:
            tm_lp.append(load_lp(out, f"seed{k}.npz", "lp_valid_best"))
            ep1_lp.append(load_lp(out, f"seed{k}.npz", "lp_valid_ep1"))
        c0_lp = [sc["lp"] for sc in scored]
        recorded = [m_grid["trials"][pick]["valid_bpw"]] + [ro["valid_bpw"] for ro in m_seeds["seeds"]]
        if any(abs(systems.bpw(lp) - b) > 1e-12 for lp, b in zip(tm_lp, recorded)):
            raise SystemExit("the saved TM per-position log-probs differ from the recorded bpw")
        arms = {}
        bits = {}
        for name, lps, seeds_ in (("TM", tm_lp, list(prof.seeds)), ("epoch1", ep1_lp, list(prof.seeds)),
                                  ("C0", c0_lp, list(prof.c0_seeds))):
            mean, B, per = tmcore.seed_mean_bits(lps)
            bits[name] = B
            arms[name] = {"seeds": seeds_, "seed_mean_bpw": float(mean.mean()), "per_seed_bpw": per}
        refbits = {k: systems.bits(v) for k, v in ref.items()}
        blk = Z["blocks_valid"]
        diffs, intervals = {}, {}
        for x in ("TM", "epoch1", "C0"):
            for y in ("twin", "F", "LR", "LightGBM"):
                diffs[f"{x} - {y}"] = arms[x]["seed_mean_bpw"] - float(refbits[y].mean())
        for x in ("epoch1", "C0"):
            diffs[f"TM - {x}"] = arms["TM"]["seed_mean_bpw"] - arms[x]["seed_mean_bpw"]
        per_seed_tm_minus_twin = [b - float(refbits["twin"].mean()) for b in arms["TM"]["per_seed_bpw"]]
        for x, y, A, Bb in (("TM", "twin", bits["TM"], refbits["twin"]), ("TM", "F", bits["TM"], refbits["F"]),
                            ("TM", "C0", bits["TM"], bits["C0"]), ("TM", "epoch1", bits["TM"], bits["epoch1"])):
            intervals[f"{x} - {y}"] = boot.two_level(A, Bb, blk)
        before, rows = tmcore.measured_minutes(["tm_smoke", "tm_grid", "tm_seeds"], a.dry)
        k3r = tmcore.k3(before + (time.time() - t_stage) / 60.0, 0.0)
        k3r["earlier_stage_wall_s"] = rows
        if not k3r["pass"]:
            decision = (f"HOLD at C-K3: the measured GPU wall time of step 9, {k3r['total_min']:.1f} min, is above "
                        f"{tmcore.K3_LIMIT_MIN:.0f}: C's claim waits or C stops")
        else:
            decision = "PASS: step 9 complete (TM, epoch-1 and C0 arms, per-position bits in data/tm/step9_valid.npz)"
        vpath = out / "step9_valid.npz"
        np.savez(vpath, tm_bits=bits["TM"], epoch1_bits=bits["epoch1"], c0_bits=bits["C0"],
                 tm_seeds=np.array(prof.seeds), c0_seeds=np.array(prof.c0_seeds), blocks_valid=blk,
                 twin_bits=refbits["twin"], F_bits=refbits["F"])
        run.add_output(vpath)
        res_json = {"decision": decision, "device": MODE.device, "pick_setting": {"T": T, "s": s}, "c0": c0_rows,
                    "c0_subset_positions": int(len(idx)), "arms": arms, "differences_bpw": diffs,
                    "per_seed_TM_minus_twin_bpw": per_seed_tm_minus_twin,
                    "seed_inclusive_intervals_bits_no_rule": intervals, "K3": k3r,
                    "grid_decision": dec_grid, "seeds_decision": dec_seeds, "inputs": used,
                    "label": "valid reading of step 9 (TM 10 seeds, epoch-1 and C0 controls); not a verdict; "
                             "the bars are read on the test file in step 10"}
        dpath = out / tmcore.DECISION_FILES[STAGE]
        hashing.write_json(dpath, res_json)
        run.add_output(dpath)
        for ro in c0_rows:
            run.print(f"C0 seed {ro['seed']}: epoch {ro['epoch_kept']}, valid system {ro['valid_bpw']:.5f} bpw")
        for name, ar in arms.items():
            run.print(f"{name}: seed-mean valid {ar['seed_mean_bpw']:.5f} bpw; per seed "
                      f"{[round(v, 5) for v in ar['per_seed_bpw']]}")
        for k, v in diffs.items():
            run.print(f"{k:22s} {v:+.5f} bpw")
        for k, v in intervals.items():
            run.print(f"{k:22s} {v['mean']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}] (seed-inclusive, no rule)")
        run.print(f"C-K3: {k3r['total_min']:.2f} min measured for step 9 (line {tmcore.K3_LIMIT_MIN:.0f}); "
                  f"pass {k3r['pass']}")
        run.print("DECISION:", decision)
        run.finish(res_json)
        return decision

    out = {}
    runlog.main_wrapper(run, lambda: out.setdefault("decision", body()))
    sys.exit(tmcore.exit_code(out["decision"]))


if __name__ == "__main__":
    main()
