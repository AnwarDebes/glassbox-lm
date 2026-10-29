"""C-core stage learners (.venv-dl): LR and LightGBM, 8 trials each, on the 60 bits (PLAN_part3.md
section 4, K0b; review 8: the same selection budget as the twin, all scored by the system's valid bpw).

    scripts/run_stage.sh learners           # real: needs the frozen R1 record, data/prep.npz
    scripts/run_stage.sh learners --dry

For every trial: class probabilities on calib and valid, the system {F, f, s} fitted on calib, the
contract (calib, valid) and the all-position check, then the valid score. The best LR and the best
LightGBM trial are selected by valid bpw. Then the shuffled-label runs of the two selected settings
(10% of the train blocks, targets permuted). Output: data/learners.npz and the selected models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import gates, hashing, layout, learners, runlog, systems  # noqa: E402

EVAL = ("calib", "valid")
DRY_LR_MAX_ITER = 30
DRY_LGBM_ROUNDS, DRY_LGBM_PATIENCE = 20, 5


def load(dry):
    z = np.load(runlog.out_dir(dry) / "prep.npz")
    return {k: z[k] for k in z.files}


def split(Z, s, e):
    return systems.Split(Z[f"w_{s}"], Z[f"y_{s}"], Z[f"pF_{s}"], Z[f"Fc_{s}"], Z[f"b_{s}"], Z[f"L_{s}"],
                         Z[f"blocks_{s}"], e)


def score(P, sp, Z, e, name):
    lam_g, lamb, em = systems.fit_system(P["calib"], sp["calib"])
    rep = {}
    for s in EVAL:
        con = systems.contract(P[s], sp[s], lamb, Z[f"pos_{s}"], Z[f"D_{s}"], Z["cls"], e)
        allp = systems.all_positions(P[s], lamb, {"pass": True})
        if not (con["pass"] and allp["pass"]):
            raise SystemExit(f"{name}: contract or all-position check failed on {s}: {con} {allp}")
        rep[s] = {"contract": con, "all_positions": allp}
    lp = systems.logp(P["valid"], sp["valid"], lamb)
    return lp, {"valid_bpw": systems.bpw(lp), "class_bits_valid": systems.class_bits(P["valid"], sp["valid"]),
                "standalone_bpw_valid": systems.standalone_bpw(P["valid"], sp["valid"]),
                "rare_split": systems.rare_split(lp, sp["valid"]), "checks": rep,
                "mean_member_weights_valid": np.round(lamb[sp["valid"].b].mean(0), 5).tolist(), **em}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run("learners", a.dry)
    max_iter = DRY_LR_MAX_ITER if a.dry else learners.LR_MAX_ITER
    rounds = ({"rounds": DRY_LGBM_ROUNDS, "patience": DRY_LGBM_PATIENCE} if a.dry else
              {"rounds": learners.LGBM_ROUNDS, "patience": learners.LGBM_PATIENCE})
    cfg = {"stage": "learners", "dry": a.dry, "frozen_utc": rec["frozen_utc"] if rec else None,
           "new_files_since_freeze": new, "lr_grid": list(learners.LR_GRID), "lr_max_iter": max_iter,
           "lr_tol": learners.LR_TOL, "lr_inputs": f"{layout.LR_N_FEATURES} one-hot tree-node features",
           "lgbm_grid": list(learners.LGBM_GRID), "lgbm_base": learners.LGBM_BASE,
           "lgbm_rounds": rounds["rounds"], "lgbm_patience": rounds["patience"],
           "shuffled": {"subset_share": learners.SUBSET_SHARE, "subset_seed": learners.SUBSET_SEED,
                        "perm_seed": learners.PERM_SEED}, "selection": "lowest valid bpw of the system",
           "K0b_max_gap": gates.K0B_MAX_GAP}
    run = runlog.Run("learners", cfg, dry=a.dry)

    def body():
        import lightgbm
        import sklearn
        run.print("sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__)
        used = runlog.check_inputs(["prep"], a.dry)
        run.print("inputs:", used)
        Z = load(a.dry)
        e = Z["emission"]
        sp = {s: split(Z, s, e) for s in EVAL}
        F_bpw = systems.bpw(np.log(sp["valid"].pF))
        out_dir = runlog.out_dir(a.dry)
        # LR
        Xs = {s: layout.lr_matrix(Z[f"X_{s}"]) for s in ("train",) + EVAL}
        lr_rows, lr_keep = [], {}
        for C, P, coef, info in learners.lr_path(Xs["train"], Z["y_train"], {s: Xs[s] for s in EVAL},
                                                 max_iter=max_iter, log=run.print):
            lp, r = score(P, sp, Z, e, f"LR C={C}")
            lr_rows.append({**info, **r, "minus_F": r["valid_bpw"] - F_bpw})
            lr_keep[C] = (lp, P, coef)
            run.print(f"LR C={C}: valid {r['valid_bpw']:.5f} bpw ({r['valid_bpw'] - F_bpw:+.5f} vs F)")
        bi = min(range(len(lr_rows)), key=lambda i: (lr_rows[i]["valid_bpw"], i))
        bC = lr_rows[bi]["C"]
        lr_lp, lr_P, lr_coef = lr_keep[bC]
        np.savez(out_dir / "lr_selected_model.npz", C=bC, **lr_coef)
        run.add_output(out_dir / "lr_selected_model.npz")
        # LightGBM
        lg_rows, lg_keep = [], {}
        for i, params in enumerate(learners.LGBM_GRID):
            P, bst, info = learners.lgbm_fit(params, Z["X_train"], Z["y_train"], Z["X_calib"], Z["y_calib"],
                                             {s: Z[f"X_{s}"] for s in EVAL}, **rounds)
            lp, r = score(P, sp, Z, e, f"LightGBM {params}")
            lg_rows.append({**params, **info, **r, "minus_F": r["valid_bpw"] - F_bpw})
            lg_keep[i] = (lp, P, bst)
            run.print(f"LightGBM {params}: {info}; valid {r['valid_bpw']:.5f} bpw ({r['valid_bpw'] - F_bpw:+.5f} vs F)")
        bj = min(range(len(lg_rows)), key=lambda i: (lg_rows[i]["valid_bpw"], i))
        lg_lp, lg_P, lg_bst = lg_keep[bj]
        lg_bst.save_model(str(out_dir / "lgbm_selected_model.txt"))
        run.add_output(out_dir / "lgbm_selected_model.txt")
        k0b_pre = {"best_lr_bpw": lr_rows[bi]["valid_bpw"], "best_lgbm_bpw": lg_rows[bj]["valid_bpw"]}
        run.print("selected: LR C", bC, "LightGBM", learners.LGBM_GRID[bj], k0b_pre)
        # shuffled labels: the selected settings on 10% of the train blocks, targets permuted
        sub = learners.subset_blocks(Z["blocks_train"])
        ysh = learners.permuted(Z["y_train"][sub])
        P, info_lr = learners.lr_fit(Xs["train"][np.flatnonzero(sub)], ysh, bC, {s: Xs[s] for s in EVAL},
                                     max_iter=max_iter)
        lr_sh_lp, r_lr_sh = score(P, sp, Z, e, "shuffled LR")
        P, _, info_lg = learners.lgbm_fit(learners.LGBM_GRID[bj], Z["X_train"][sub], ysh, Z["X_calib"], Z["y_calib"],
                                          {s: Z[f"X_{s}"] for s in EVAL}, **rounds)
        lg_sh_lp, r_lg_sh = score(P, sp, Z, e, "shuffled LightGBM")
        sh = {"LR": {**gates.shuffled(F_bpw, r_lr_sh["valid_bpw"]), **info_lr, **r_lr_sh},
              "LightGBM": {**gates.shuffled(F_bpw, r_lg_sh["valid_bpw"]), **info_lg, **r_lg_sh},
              "subset_positions": int(sub.sum()), "subset_blocks": int(len(np.unique(Z["blocks_train"][sub])))}
        run.print(f"shuffled LR: {sh['LR']['diff_vs_F_bpw']:+.5f} vs F; shuffled LightGBM: "
                  f"{sh['LightGBM']['diff_vs_F_bpw']:+.5f} vs F")
        out = out_dir / "learners.npz"
        np.savez(out, F_valid=np.log(sp["valid"].pF), lr_valid=lr_lp, lgbm_valid=lg_lp, lr_shuffled_valid=lr_sh_lp,
                 lgbm_shuffled_valid=lg_sh_lp, PX_lr_calib=lr_P["calib"], PX_lr_valid=lr_P["valid"],
                 PX_lgbm_calib=lg_P["calib"], PX_lgbm_valid=lg_P["valid"])
        run.add_output(out)
        run.finish({"F_valid_bpw": F_bpw, "lr_trials": lr_rows, "lgbm_trials": lg_rows, "selected_lr_C": bC,
                    "selected_lgbm": learners.LGBM_GRID[bj], "selected_lr_index": bi, "selected_lgbm_index": bj,
                    "shuffled": sh, **k0b_pre})

    runlog.main_wrapper(run, body)


if __name__ == "__main__":
    main()
