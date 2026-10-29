"""C-core stage tm_seeds (R1 [C-core] section 6, the TM row: "the pick then runs seeds 2-10"; the early-checkpoint row:
each seed's state after epoch 1 with its own calib read-out; PLAN_part3.md section 6, step 9). GPU. Written under
the step-9 amendment.

    scripts/run_gpu_stage.sh tm_seeds          # real: needs a finished tm_grid run that decided PASS (C-K1 passed)
    scripts/run_gpu_stage.sh tm_seeds --dry

The grid's pick, TM seeds 2-10, all train positions, 20 epochs, the same read-out and epoch rule. Every seed is
trained before valid is read; then each seed's kept checkpoint and its epoch-1 checkpoint are scored on valid through
their systems. C-K3 as in the grid. Exit 3 on a HOLD.
Outputs: data/tm/seed<k>.npz and data/tm/seeds_decision.json."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import tmcore  # noqa: E402

DRY = "--dry" in sys.argv[1:]
MODE = tmcore.guard(DRY)
import numpy as np  # noqa: E402

from ccore import hashing, runlog, systems, tmtorch  # noqa: E402

STAGE = "tm_seeds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_stage = time.time()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run(STAGE, a.dry)
    prof = MODE.profile
    seeds = [k for k in prof.seeds if k != tmcore.GRID_SEED]
    cfg = {"stage": STAGE, "dry": a.dry, "mode": MODE.describe(), "frozen_utc": rec["frozen_utc"] if rec else None,
           "amendment": rec.get("amendment") if rec else None, "new_files_since_freeze": new,
           "profile": tmcore.profile_dict(prof), "seeds": seeds, "max_epochs": prof.max_epochs,
           "clauses": prof.clauses, "K3_limit_min": tmcore.K3_LIMIT_MIN}
    run = tmcore.TMRun(STAGE, cfg, dry=a.dry)

    def body():
        torch = tmtorch.setup_torch(MODE.device)
        run.print("torch", torch.__version__, "device", MODE.device, "profile", prof.name)
        used = runlog.check_inputs(["prep", "twin", "learners", "gates"], a.dry)
        tmcore.require_pass("tm_smoke", a.dry)
        dec_grid, m_grid, d_grid = tmcore.require_pass("tm_grid", a.dry)
        T, s = m_grid["pick_setting"]["T"], m_grid["pick_setting"]["s"]
        if [T, s] != list(prof.grid[m_grid["pick"]]):
            raise SystemExit(f"the grid's pick {T, s} is not trial {m_grid['pick']} of the profile's grid")
        run.print("inputs:", used, "grid:", d_grid, dec_grid, "pick T", T, "s", s)
        Z = tmcore.load_prep(a.dry)
        e = Z["emission"]
        sp = {k: tmcore.split(Z, k, e) for k in ("calib", "valid")}
        dec6, ref = tmcore.step6(a.dry, Z, e, sp)
        t_first = time.time() - t_stage
        out = tmcore.tm_dir(a.dry)
        trained = []
        for k in seeds:
            res = tmtorch.train(Z["X_train"], Z["y_train"], Z["X_calib"], Z["y_calib"], T, s, k, prof.clauses,
                                prof.max_epochs, MODE.device, log=run.print, tag=f"seed {k}")
            tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": f"seed {k}",
                                 "device": res["device"], "epochs": prof.max_epochs, "batches": res["batches"],
                                 "train_s": res["train_s"], "readout_s": res["readout_s"], "wall_s": res["run_s"]})
            trained.append(res)
        t = time.time()
        scored = [(tmtorch.score_checkpoint(r["cfg"], r["best"], Z, sp, e, MODE.device, f"seed {k}"),
                   tmtorch.score_checkpoint(r["cfg"], r["epoch1"], Z, sp, e, MODE.device, f"seed {k} epoch 1"))
                  for k, r in zip(seeds, trained)]
        t_system = (time.time() - t) / (2 * len(scored))
        t_batch = sum(r["train_s"] for r in trained) / sum(r["batches"] for r in trained)
        t_ro = sum(r["readout_s"] for r in trained) / (len(trained) * prof.max_epochs)
        n_c0 = int(round(prof.c0["share"] * len(Z["y_train"])))
        plan = tmcore.remaining_plan(STAGE, prof, len(Z["y_train"]), n_c0)
        proj = tmcore.project_minutes(plan, t_batch, t_ro, t_system, t_first)
        before, rows = tmcore.measured_minutes(["tm_smoke", "tm_grid"], a.dry)
        k3r = tmcore.k3(before + (time.time() - t_stage) / 60.0, proj)
        k3r.update({"unit_times_s": {"per_batch": t_batch, "readout_per_epoch": t_ro, "system_fit": t_system,
                                     "stage_overhead": t_first}, "earlier_stage_wall_s": rows,
                    "plan": [list(p) for p in plan]})
        if not k3r["pass"]:
            decision = (f"HOLD at C-K3: measured plus projected GPU wall time {k3r['total_min']:.1f} min is above "
                        f"{tmcore.K3_LIMIT_MIN:.0f}: C's claim waits or C stops")
        else:
            decision = "PASS: seeds 2-10 trained and scored; C0 may run"
        rows_out = []
        twin_bpw, F_bpw = systems.bpw(ref["twin"]), systems.bpw(ref["F"])
        for k, r, (sb, s1) in zip(seeds, trained, scored):
            ck, c1 = r["best"], r["epoch1"]
            path = out / f"seed{k}.npz"
            np.savez(path, T=T, s=s, seed=k, ta_best=ck["ta"], w_best=ck["w"], epoch_best=ck["epoch"],
                     ro_best_b=np.asarray(ck["readout"]["b"]), ro_best_a_scaled=ck["readout"]["a_scaled"],
                     ro_best_sigma=ck["readout"]["sigma"], ta_ep1=c1["ta"], w_ep1=c1["w"],
                     ro_ep1_b=np.asarray(c1["readout"]["b"]), ro_ep1_a_scaled=c1["readout"]["a_scaled"],
                     ro_ep1_sigma=c1["readout"]["sigma"],
                     calib_class_bits_per_epoch=np.array([h["calib_class_bits"] for h in r["history"]]),
                     lp_valid_best=sb["lp"], lamb_best=sb["lamb"], pc_valid_best=sb["pc_valid"],
                     lp_valid_ep1=s1["lp"], lamb_ep1=s1["lamb"], pc_valid_ep1=s1["pc_valid"])
            run.add_output(path)
            rows_out.append({"seed": k, "epoch_kept": ck["epoch"], "calib_class_bits": ck["calib_class_bits"],
                             "readout": ck["readout"], **sb["info"], "minus_twin": sb["info"]["valid_bpw"] - twin_bpw,
                             "minus_F": sb["info"]["valid_bpw"] - F_bpw,
                             "epoch1": {**s1["info"], "readout": c1["readout"], "minus_twin":
                                        s1["info"]["valid_bpw"] - twin_bpw},
                             "clause_stats": ck["stats"], "length_histogram": tmcore.length_histogram(ck["ta"]),
                             "packed_checks": {"best": sb["packed"], "epoch1": s1["packed"]}, "history": r["history"],
                             "cfg": r["cfg"], "train_s": r["train_s"], "readout_s": r["readout_s"],
                             "batches": r["batches"]})
        res_json = {"decision": decision, "device": MODE.device, "pick_setting": {"T": T, "s": s}, "seeds": rows_out,
                    "twin_bpw": twin_bpw, "F_bpw": F_bpw, "K3": k3r, "grid_decision": dec_grid, "inputs": used,
                    "label": "valid reading of seeds 2-10 and their epoch-1 checkpoints; not a verdict"}
        dpath = out / tmcore.DECISION_FILES[STAGE]
        hashing.write_json(dpath, res_json)
        run.add_output(dpath)
        for ro in rows_out:
            run.print(f"seed {ro['seed']}: epoch {ro['epoch_kept']}, valid system {ro['valid_bpw']:.5f} bpw "
                      f"({ro['minus_twin']:+.5f} vs twin); epoch 1: {ro['epoch1']['valid_bpw']:.5f}")
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
