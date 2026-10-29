"""C-core stage tm_grid (R1 [C-core] section 6, the TM row: the 8-trial grid at seed 1; section 9, C-K1 and C-K3;
PLAN_part3.md section 6, step 9). GPU. Written under the step-9 amendment.

    scripts/run_gpu_stage.sh tm_grid          # real: needs a finished tm_smoke run that decided PASS
    scripts/run_gpu_stage.sh tm_grid --dry

The 8 trials T {8,000, 16,000} x s {3, 10, 30, 90} (index order = tie order), TM seed 1, all train positions, 20
epochs each, the read-out on calib after every epoch and the epoch rule (lowest calib class log-loss). Every trial is
trained before valid is read; then each kept checkpoint is scored once on valid through its system {F, f, s}
(ccore/systems.py: bucketed MAP-EM on calib, contract and all-position checks): 8 looks. The pick is the lowest valid
system bpw (ties to the lower index). C-K1: bpw(pick) - bpw(twin) > 0.0218 -> STOP, "FAIL at K1 (exploratory)", no
test read. Only if C-K1 passes: the pick's epoch-1 checkpoint is scored on valid (seed 1 of the early-checkpoint
control). C-K3: measured plus projected GPU wall time against 230 minutes. Exit 3 on a STOP or HOLD.
Outputs: data/tm/grid_t<i>.npz (state, read-out and scores of each trial) and data/tm/grid_decision.json."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import tmcore  # noqa: E402

DRY = "--dry" in sys.argv[1:]
MODE = tmcore.guard(DRY)
import numpy as np  # noqa: E402

from ccore import gates, hashing, runlog, systems, tmtorch  # noqa: E402

STAGE = "tm_grid"


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
           "profile": tmcore.profile_dict(prof), "grid": [list(g) for g in prof.grid], "seed": tmcore.GRID_SEED,
           "max_epochs": prof.max_epochs, "clauses": prof.clauses,
           "pick": "lowest valid system bpw, ties to the lower trial index",
           "K1_max_gap": gates.K1_MAX_GAP, "K3_limit_min": tmcore.K3_LIMIT_MIN}
    run = tmcore.TMRun(STAGE, cfg, dry=a.dry)

    def body():
        torch = tmtorch.setup_torch(MODE.device)
        run.print("torch", torch.__version__, "device", MODE.device, "profile", prof.name)
        used = runlog.check_inputs(["prep", "twin", "learners", "gates"], a.dry)
        dec_smoke, m_smoke, d_smoke = tmcore.require_pass("tm_smoke", a.dry)
        run.print("inputs:", used, "smoke:", d_smoke, dec_smoke)
        Z = tmcore.load_prep(a.dry)
        e = Z["emission"]
        sp = {s: tmcore.split(Z, s, e) for s in ("calib", "valid")}
        dec6, ref = tmcore.step6(a.dry, Z, e, sp)
        if not a.dry and not dec6["decision"].startswith("PASS"):
            raise SystemExit(f"step 6 decided {dec6['decision']!r}: no TM run")
        t_first = time.time() - t_stage
        out = tmcore.tm_dir(a.dry)
        trained = []
        for i, (T, s) in enumerate(prof.grid):
            res = tmtorch.train(Z["X_train"], Z["y_train"], Z["X_calib"], Z["y_calib"], T, s, tmcore.GRID_SEED,
                                prof.clauses, prof.max_epochs, MODE.device, log=run.print, tag=f"trial {i} T {T} s {s}")
            tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": f"grid trial {i}",
                                 "device": res["device"], "epochs": prof.max_epochs, "batches": res["batches"],
                                 "train_s": res["train_s"], "readout_s": res["readout_s"], "wall_s": res["run_s"]})
            trained.append(res)
        # valid is read only now, once per trial, at the kept checkpoint
        t = time.time()
        scored = [tmtorch.score_checkpoint(res["cfg"], res["best"], Z, sp, e, MODE.device, f"trial {i}")
                  for i, res in enumerate(trained)]
        t_system = (time.time() - t) / len(scored)
        bpws = [sc["info"]["valid_bpw"] for sc in scored]
        pick = tmcore.pick_trial(bpws)
        k1r = gates.k1(bpws[pick], systems.bpw(ref["twin"]))
        ep1 = None
        if k1r["pass"] or a.dry:
            ep1 = tmtorch.score_checkpoint(trained[pick]["cfg"], trained[pick]["epoch1"], Z, sp, e, MODE.device,
                                           f"trial {pick} epoch 1")
        # C-K3
        t_batch = sum(r["train_s"] for r in trained) / sum(r["batches"] for r in trained)
        t_ro = sum(r["readout_s"] for r in trained) / (len(trained) * prof.max_epochs)
        n_c0 = int(round(prof.c0["share"] * len(Z["y_train"])))
        plan = tmcore.remaining_plan(STAGE, prof, len(Z["y_train"]), n_c0)
        proj = tmcore.project_minutes(plan, t_batch, t_ro, t_system, t_first)
        before, rows = tmcore.measured_minutes(["tm_smoke"], a.dry)
        k3r = tmcore.k3(before + (time.time() - t_stage) / 60.0, proj)
        k3r.update({"unit_times_s": {"per_batch": t_batch, "readout_per_epoch": t_ro, "system_fit": t_system,
                                     "stage_overhead": t_first}, "earlier_stage_wall_s": rows,
                    "plan": [list(p) for p in plan]})
        if not k1r["pass"]:
            decision = (f"STOP: FAIL at K1 (exploratory): the best TM system is {k1r['gap_bpw']:+.5f} bpw from "
                        f"the twin system (more than {gates.K1_MAX_GAP}); no test read")
        elif not k3r["pass"]:
            decision = (f"HOLD at C-K3: measured plus projected GPU wall time {k3r['total_min']:.1f} min is above "
                        f"{tmcore.K3_LIMIT_MIN:.0f}: C's claim waits or C stops")
        else:
            decision = f"PASS: C-K1 passed (gap {k1r['gap_bpw']:+.5f} bpw); seeds 2-10 and C0 may run"
        trials = []
        for i, ((T, s), res, sc) in enumerate(zip(prof.grid, trained, scored)):
            ck, c1 = res["best"], res["epoch1"]
            arr = {"T": T, "s": s, "seed": tmcore.GRID_SEED, "ta_best": ck["ta"], "w_best": ck["w"],
                   "epoch_best": ck["epoch"], "ro_best_b": np.asarray(ck["readout"]["b"]),
                   "ro_best_a_scaled": ck["readout"]["a_scaled"], "ro_best_sigma": ck["readout"]["sigma"],
                   "ta_ep1": c1["ta"], "w_ep1": c1["w"], "ro_ep1_b": np.asarray(c1["readout"]["b"]),
                   "ro_ep1_a_scaled": c1["readout"]["a_scaled"], "ro_ep1_sigma": c1["readout"]["sigma"],
                   "calib_class_bits_per_epoch": np.array([h["calib_class_bits"] for h in res["history"]]),
                   "lp_valid_best": sc["lp"], "lamb_best": sc["lamb"], "pc_valid_best": sc["pc_valid"]}
            if i == pick and ep1 is not None:
                arr.update(lp_valid_ep1=ep1["lp"], lamb_ep1=ep1["lamb"], pc_valid_ep1=ep1["pc_valid"])
            path = out / f"grid_t{i}.npz"
            np.savez(path, **arr)
            run.add_output(path)
            trials.append({"trial": i, "T": T, "s": s, "seed": tmcore.GRID_SEED, "epoch_kept": ck["epoch"],
                           "calib_class_bits": ck["calib_class_bits"], "readout": ck["readout"], **sc["info"],
                           "minus_twin": sc["info"]["valid_bpw"] - systems.bpw(ref["twin"]),
                           "minus_F": sc["info"]["valid_bpw"] - systems.bpw(ref["F"]),
                           "clause_stats": ck["stats"], "length_histogram": tmcore.length_histogram(ck["ta"]),
                           "packed_checks": sc["packed"], "history": res["history"], "cfg": res["cfg"],
                           "train_s": res["train_s"], "readout_s": res["readout_s"], "batches": res["batches"]})
        res_json = {"decision": decision, "device": MODE.device, "trials": trials, "pick": pick,
                    "pick_setting": {"T": prof.grid[pick][0], "s": prof.grid[pick][1]}, "K1": k1r,
                    "twin_bpw": systems.bpw(ref["twin"]), "F_bpw": systems.bpw(ref["F"]),
                    "LR_bpw": systems.bpw(ref["LR"]), "LightGBM_bpw": systems.bpw(ref["LightGBM"]),
                    "epoch1_of_pick": ({**ep1["info"], "epoch": 1, "packed_checks": ep1["packed"]}
                                       if ep1 is not None else None),
                    "K3": k3r, "smoke_decision": dec_smoke, "inputs": used,
                    "label": "valid reading of the grid (8 looks); not a verdict"}
        dpath = out / tmcore.DECISION_FILES[STAGE]
        hashing.write_json(dpath, res_json)
        run.add_output(dpath)
        for tr in trials:
            run.print(f"trial {tr['trial']} T {tr['T']} s {tr['s']}: epoch {tr['epoch_kept']}, calib class bits "
                      f"{tr['calib_class_bits']:.5f}, valid system {tr['valid_bpw']:.5f} bpw "
                      f"({tr['minus_twin']:+.5f} vs twin, {tr['minus_F']:+.5f} vs F), valid class bits "
                      f"{tr['class_bits_valid']:.5f}")
        run.print(f"pick: trial {pick} {prof.grid[pick]}; C-K1 {k1r}")
        if ep1 is not None:
            run.print(f"epoch-1 checkpoint of the pick: valid system {ep1['info']['valid_bpw']:.5f} bpw")
        run.print(f"C-K3: {k3r['measured_min']:.2f} min measured + {k3r['projected_remaining_min']:.2f} projected = "
                  f"{k3r['total_min']:.2f} (line {tmcore.K3_LIMIT_MIN:.0f}); pass {k3r['pass']}")
        run.print("DECISION:", decision)
        run.finish(res_json)
        return decision

    out = {}
    runlog.main_wrapper(run, lambda: out.setdefault("decision", body()))
    sys.exit(tmcore.exit_code(out["decision"]))


if __name__ == "__main__":
    main()
