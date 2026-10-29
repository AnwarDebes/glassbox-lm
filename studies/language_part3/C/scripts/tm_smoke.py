"""C-core stage tm_smoke (R1 [C-core] section 9: C-K2's smoke, the shuffled-label TM at smoke scale, C-K3's timed CPU
epoch and its cost projection; PLAN_part3.md section 6, step 9). GPU. Written under the step-9 amendment.

    scripts/run_gpu_stage.sh tm_smoke          # real: amended frozen record + its LOG.md line, C/.gpu_go, GPU free
    scripts/run_gpu_stage.sh tm_smoke --dry    # data_dry/ pseudo splits (train blocks only); nothing enforced

1. Smoke TM: 10% of the train blocks (subset seed 1), 2 epochs, T 8,000, s 10, TM seed 1; the registered machine,
   read-out and epoch rule. C-K2 on its kept checkpoint (logged, not a result): STOP and diagnose if more than 50% of
   the clauses are empty, the median clause has more than 30 literals, more than 50% of the clauses have no positive
   literal, or the calibrated valid class bits are worse than the class unigram's (a fix is a dated amendment).
2. Shuffled-label TM: the same subset and setting, class targets permuted across its positions (seed 1); its system
   {F, f, s} on valid must be no more than 0.0010 bpw below F, else STOP (find the leak). The step-6 class-unigram
   system is printed beside it (no rule). C-A/A has no TM part in R1 (the twin only, before any TM run; it passed).
3. Timed CPU epoch: one epoch of the smoke setting on the smoke subset on the CPU at 2 threads (C-K3).
4. C-K3: this stage's wall time plus the projected wall time of grid, seeds and C0 (from this stage's per-batch,
   read-out and scoring times) against 230 GPU-minutes; above it: HOLD (C's claim waits or C stops).
Decision order: shuffled, K2, K3. Exit 3 on a STOP or HOLD. Outputs: data/tm/smoke.npz, data/tm/smoke_decision.json."""
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

STAGE = "tm_smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_stage = time.time()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run(STAGE, a.dry)
    prof = MODE.profile
    sm = prof.smoke
    cfg = {"stage": STAGE, "dry": a.dry, "mode": MODE.describe(), "frozen_utc": rec["frozen_utc"] if rec else None,
           "amendment": rec.get("amendment") if rec else None, "new_files_since_freeze": new,
           "profile": tmcore.profile_dict(prof), "clauses": prof.clauses, "smoke": sm,
           "shuffled_perm_seed": tmcore.SHUFFLE_PERM_SEED,
           "readout": "softmax(S / tau + b), tau and 65 biases on calib",
           "epoch_rule": "lowest calib class log-loss of its own read-out, ties to the earlier epoch",
           "K2": {"max_empty_share": tmcore.K2_MAX_EMPTY_SHARE, "max_median_literals": tmcore.K2_MAX_MEDIAN_LITERALS,
                  "max_no_positive_share": tmcore.K2_MAX_NO_POSITIVE_SHARE,
                  "class_unigram": "systems.null_class_model of all train positions"},
           "shuffled_min_diff": gates.SHUFFLED_MIN_DIFF, "K3_limit_min": tmcore.K3_LIMIT_MIN,
           "cotm": {"file": str(tmtorch.TM_FILE), "sha256": tmtorch.TM_SHA256},
           "cpu_tm_fast": {"file": str(tmtorch.FAST_FILE), "sha256": tmtorch.FAST_SHA256}}
    run = tmcore.TMRun(STAGE, cfg, dry=a.dry)

    def body():
        torch = tmtorch.setup_torch(MODE.device)
        run.print("torch", torch.__version__, "device", MODE.device, "profile", prof.name)
        used = runlog.check_inputs(["prep", "twin", "learners", "gates"], a.dry)
        run.print("inputs:", used)
        Z = tmcore.load_prep(a.dry)
        e = Z["emission"]
        sp = {s: tmcore.split(Z, s, e) for s in ("calib", "valid")}
        dec6, ref = tmcore.step6(a.dry, Z, e, sp)
        if not a.dry and not dec6["decision"].startswith("PASS"):
            raise SystemExit(f"step 6 decided {dec6['decision']!r}: no TM run")
        F_bpw = systems.bpw(ref["F"])
        idx = tmcore.smoke_subset(Z["blocks_train"], prof)
        Xs, ys = Z["X_train"][idx], Z["y_train"][idx]
        run.print(f"smoke subset: {len(idx)} positions of {len(Z['y_train'])} "
                  f"({len(np.unique(Z['blocks_train'][idx]))} blocks)")
        t_first = time.time() - t_stage
        out = tmcore.tm_dir(a.dry)

        # 1. the smoke TM and C-K2
        res = tmtorch.train(Xs, ys, Z["X_calib"], Z["y_calib"], sm["T"], sm["s"], sm["seed"], prof.clauses,
                            sm["epochs"], MODE.device, log=run.print, tag="smoke")
        tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": "smoke TM",
                             "device": res["device"], "epochs": sm["epochs"], "batches": res["batches"],
                             "train_s": res["train_s"], "readout_s": res["readout_s"], "wall_s": res["run_s"]})
        ck = res["best"]
        t = time.time()
        S_val = tmtorch.checkpoint_sums(res["cfg"], ck, Z["X_valid"], MODE.device)
        pv = Z["pos_valid"]
        pk = tmtorch.packed_equal(ck, Z["X_valid"][pv], S_val[pv])
        if not pk["equal"]:
            raise SystemExit(f"packed CPU class sums differ from the CoTM's: {pk}")
        PX_val = tmcore.readout_proba(S_val, ck["readout"])
        tm_bits = tmcore.class_logloss_bits(PX_val, Z["y_valid"])
        uni = systems.null_class_model(Z["y_train"], len(Z["y_valid"]))
        uni_bits = systems.class_bits(uni, sp["valid"])
        uni_sub_bits = systems.class_bits(systems.null_class_model(ys, len(Z["y_valid"])), sp["valid"])
        stats_ck = tmcore.clause_stats(ck["ta"])
        if stats_ck != ck["stats"]:
            raise SystemExit(f"clause stats of the stored state differ from CoTM.stats(): {stats_ck} {ck['stats']}")
        k2r = tmcore.k2(stats_ck, tm_bits, uni_bits)
        t_k2 = time.time() - t

        # 2. the shuffled-label TM at smoke scale
        ysh = tmcore.shuffled_targets(ys)
        rsh = tmtorch.train(Xs, ysh, Z["X_calib"], Z["y_calib"], sm["T"], sm["s"], sm["seed"], prof.clauses,
                            sm["epochs"], MODE.device, log=run.print, tag="shuffled")
        tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": "shuffled-label TM",
                             "device": rsh["device"], "epochs": sm["epochs"], "batches": rsh["batches"],
                             "train_s": rsh["train_s"], "readout_s": rsh["readout_s"], "wall_s": rsh["run_s"]})
        t = time.time()
        S_cal = tmtorch.checkpoint_sums(rsh["cfg"], rsh["best"], Z["X_calib"], MODE.device)
        S_v = tmtorch.checkpoint_sums(rsh["cfg"], rsh["best"], Z["X_valid"], MODE.device)
        PX = {"calib": tmcore.readout_proba(S_cal, rsh["best"]["readout"]),
              "valid": tmcore.readout_proba(S_v, rsh["best"]["readout"])}
        sh_lp, sh_lamb, sh_info = tmcore.system_score(PX, sp, Z, e, "shuffled-label TM")
        t_system = time.time() - t
        shr = gates.shuffled(F_bpw, systems.bpw(sh_lp))
        uni_diag = dec6["valid_bpw"].get("class_unigram_diagnostic")

        # 3. the timed CPU epoch (the CPU option of C-K3; no read-out, no score)
        cpu = tmtorch.time_cpu_epoch(Xs, ys, sm["T"], sm["s"], sm["seed"], prof.clauses, log=run.print)
        cpu_full_epoch_s = cpu["s_per_batch"] * tmcore.batches_per_epoch(len(Z["y_train"]))
        tmcore.gpu_time_row({"utc": runlog.utc(), "stage": run.name, "run": run.id, "what": "timed CPU epoch",
                             "device": "cpu", "epochs": 1, "batches": cpu["batches"], "train_s": cpu["seconds"],
                             "readout_s": "", "wall_s": cpu["seconds"]})

        # 4. C-K3
        t_batch = res["train_s"] / res["batches"]
        t_ro = res["readout_s"] / sm["epochs"]
        n_c0 = int(round(prof.c0["share"] * len(Z["y_train"])))
        plan = tmcore.remaining_plan("tm_smoke", prof, len(Z["y_train"]), n_c0)
        proj = tmcore.project_minutes(plan, t_batch, t_ro, t_system, t_first)
        measured = (time.time() - t_stage) / 60.0
        k3r = tmcore.k3(measured, proj)
        k3r.update({"unit_times_s": {"per_batch": t_batch, "readout_per_epoch": t_ro, "system_fit": t_system,
                                     "stage_overhead": t_first, "k2_scoring": t_k2},
                    "plan": [list(p) for p in plan]})
        if not shr["pass"]:
            decision = ("STOP: the shuffled-label TM at smoke scale gives a system more than 0.0010 bpw below F "
                        "(find the leak)")
        elif not k2r["pass"]:
            decision = f"STOP at C-K2 ({', '.join(k2r['stops'])}): diagnose; a fix is a dated amendment"
        elif not k3r["pass"]:
            decision = (f"HOLD at C-K3: measured plus projected GPU wall time {k3r['total_min']:.1f} min is above "
                        f"{tmcore.K3_LIMIT_MIN:.0f} (twice the top of the 50-115 estimate): C's claim waits or C stops")
        else:
            decision = ("PASS: C-K2 and the shuffled-label check passed; C-K3 projection within the line; "
                        "the grid may run")
        if MODE.dry or MODE.test:
            decision_note = "dry or self-test run: the numbers mean nothing" if MODE.dry else "self-test run"
        else:
            decision_note = "logged, not a result"
        path = out / "smoke.npz"
        np.savez(path, idx=idx, ta_best=ck["ta"], w_best=ck["w"], epoch_best=ck["epoch"],
                 ro_best_b=np.asarray(ck["readout"]["b"]), ro_best_a_scaled=ck["readout"]["a_scaled"],
                 ro_best_sigma=ck["readout"]["sigma"], pc_valid_best=tmcore.true_class_prob(PX_val, Z["y_valid"]),
                 sh_ta_best=rsh["best"]["ta"], sh_w_best=rsh["best"]["w"], sh_epoch_best=rsh["best"]["epoch"],
                 sh_lp_valid=sh_lp, sh_lamb=sh_lamb, ysh=ysh)
        run.add_output(path)
        res_json = {"decision": decision, "note": decision_note, "device": MODE.device,
                    "smoke": {"positions": int(len(idx)), "blocks": int(len(np.unique(Z["blocks_train"][idx]))),
                              "epoch_kept": ck["epoch"], "history": res["history"], "readout_kept": ck["readout"],
                              "valid_class_bits": tm_bits, "class_unigram_valid_class_bits": uni_bits,
                              "subset_class_unigram_valid_class_bits_no_rule": uni_sub_bits, "clause_stats": stats_ck,
                              "length_histogram": tmcore.length_histogram(ck["ta"]), "packed_check_valid": pk,
                              "train_s": res["train_s"], "readout_s": res["readout_s"], "batches": res["batches"],
                              "cfg": res["cfg"]},
                    "K2": k2r,
                    "shuffled": {**shr, "bpw": systems.bpw(sh_lp), "F_bpw": F_bpw, "system": sh_info,
                                 "epoch_kept": rsh["best"]["epoch"], "history": rsh["history"],
                                 "class_unigram_system_minus_F_step6_no_rule":
                                     (uni_diag - F_bpw) if uni_diag is not None else None},
                    "AA": {"note": "C-A/A has no TM part in R1 section 9 (the twin only, before any TM run)",
                           "step6": dec6.get("AA")},
                    "cpu_epoch": {**cpu, "projected_full_train_epoch_s": cpu_full_epoch_s,
                                  "projected_cpu_option_h_no_readout": 4 * 8 * cpu_full_epoch_s / 3600.0},
                    "K3": k3r, "step6_decision": dec6["decision"], "inputs": used}
        dpath = out / tmcore.DECISION_FILES[STAGE]
        hashing.write_json(dpath, res_json)
        run.add_output(dpath)
        run.print(f"C-K2 smoke (logged, not a result): kept epoch {ck['epoch']}, valid class bits {tm_bits:.5f} "
                  f"vs class unigram {uni_bits:.5f}; clauses {stats_ck}; checks {k2r['checks']}")
        run.print(f"shuffled-label TM: system {systems.bpw(sh_lp):.5f} bpw, minus F {shr['diff_vs_F_bpw']:+.5f} "
                  f"(needs >= {gates.SHUFFLED_MIN_DIFF}); pass {shr['pass']}")
        run.print(f"timed CPU epoch: {cpu['seconds']:.1f} s for {cpu['batches']} batches; a full train epoch "
                  f"{cpu_full_epoch_s / 60:.1f} min (projection)")
        run.print(f"C-K3: {k3r['measured_min']:.2f} min measured + {k3r['projected_remaining_min']:.2f} projected = "
                  f"{k3r['total_min']:.2f} (line {tmcore.K3_LIMIT_MIN:.0f}); pass {k3r['pass']}")
        run.print("DECISION:", decision, f"({decision_note})")
        run.finish(res_json)
        return decision

    out = {}
    runlog.main_wrapper(run, lambda: out.setdefault("decision", body()))
    sys.exit(tmcore.exit_code(out["decision"]))


if __name__ == "__main__":
    main()
