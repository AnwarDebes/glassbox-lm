"""C-core stage twin (PLAN_part3.md section 4, "Count twin", A/A, shuffled labels, K0; section 6 step 6).

    scripts/run_stage.sh twin           # real: needs the frozen R1 record and data/prep.npz
    scripts/run_stage.sh twin --dry

Counts every chain member on train, checks PairKN against tally KN (orders 3-5, calib and valid, max
abs difference 1e-12), then for each of the 8 trials fits the class mixture and the system on calib,
runs the contract (calib and valid) and the all-position check, and only then scores valid. The trial
with the lowest valid bpw is the twin. Then: the A/A rebuild (95% of the train blocks, aa.py seed 1;
the floor and the codes stay fixed, the twin's counts and the emission are recounted, weights refitted
on calib), the shuffled-label twin (class targets permuted across train positions, seed 1; T-E through
PairKN on the true contexts), and the class-unigram system (a diagnostic, no rule). K0 is computed here
and again, with the other gates, in the gates stage."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import chains, codes, gates, hashing, runlog, systems, twin  # noqa: E402
from tally.floor import aa  # noqa: E402

AA_SEED = 1
AA_DROP = 0.05
SHUF_SEED = 1
EVAL = ("calib", "valid")


def load(dry):
    z = np.load(runlog.out_dir(dry) / "prep.npz")
    return {k: z[k] for k in z.files}


def split(Z, s, e):
    return systems.Split(Z[f"w_{s}"], Z[f"y_{s}"], Z[f"pF_{s}"], Z[f"Fc_{s}"], Z[f"b_{s}"], Z[f"L_{s}"],
                         Z[f"blocks_{s}"], e)


def comps(Z, s, keep=None):
    c = chains.components(Z[f"X_{s}"])
    c["cls"] = Z[f"y_{s}"]
    if keep is not None:
        c = {k: v[keep] for k, v in c.items()}
    return c


def system(PX_cal, PX_val, sp, Z, e, name, run):
    """Fit on calib, contract and all-position checks, then the valid score. Stops on a failed check."""
    lam_g, lamb, em = systems.fit_system(PX_cal, sp["calib"])
    cls = Z["cls"]
    rep = {}
    for s, PX in (("calib", PX_cal), ("valid", PX_val)):
        con = systems.contract(PX, sp[s], lamb, Z[f"pos_{s}"], Z[f"D_{s}"], cls, e)
        allp = systems.all_positions(PX, lamb, {"pass": True})
        if not (con["pass"] and allp["pass"]):
            raise SystemExit(f"{name}: contract or all-position check failed on {s}: {con} {allp}")
        rep[s] = {"contract": con, "all_positions": allp}
    lp = systems.logp(PX_val, sp["valid"], lamb)
    return lp, lamb, {"checks": rep, "lam_global": np.round(lam_g, 5).tolist(), **em,
                      "mean_member_weights_valid": np.round(lamb[sp["valid"].b].mean(0), 5).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run("twin", a.dry)
    cfg = {"stage": "twin", "dry": a.dry, "frozen_utc": rec["frozen_utc"] if rec else None,
           "new_files_since_freeze": new, "trials": [[n, [twin.mname(m) for m in ms]] for n, ms in twin.TRIALS],
           "class_mixture": "bucketed MAP-EM on calib (floor buckets, tau 20)",
           "system": "bucketed MAP-EM of {F, f, s} on calib", "selection": "lowest valid bpw of the system",
           "aa": {"tally": "floor/aa.py subsample_train", "seed": AA_SEED, "drop_share": AA_DROP,
                  "rebuilt": "twin counts and emission; floor and codes fixed"},
           "shuffled": {"seed": SHUF_SEED, "what": "class targets permuted across all train positions"},
           "gates": {"K0_min_gain": gates.K0_MIN_GAIN, "AA_max_abs": gates.AA_MAX_ABS,
                     "shuffled_min_diff": gates.SHUFFLED_MIN_DIFF}}
    run = runlog.Run("twin", cfg, dry=a.dry)

    def body():
        used = runlog.check_inputs(['prep'], a.dry)
        run.print("inputs:", used)
        Z = load(a.dry)
        e = Z["emission"]
        sp = {s: split(Z, s, e) for s in EVAL}
        cp = {s: comps(Z, s) for s in ("train",) + EVAL}
        yt = Z["y_train"]
        members = twin.all_members()
        models = twin.build(members, yt, cp["train"], mode="kn")
        D = {s: twin.dists(models, cp[s]) for s in EVAL}
        # PairKN equals tally KN on the true targets (it carries the shuffled T-E below)
        pk = twin.build([m for m in members if isinstance(m, tuple)], yt, cp["train"], mode="pair")
        pk_err = {}
        for s in EVAL:
            Dp = twin.dists(pk, cp[s])
            pk_err[s] = {n: float(np.abs(Dp[n] - D[s][n]).max()) for n in Dp}
        if max(v for d in pk_err.values() for v in d.values()) > 1e-12:
            raise SystemExit(f"PairKN differs from tally KN: {pk_err}")
        run.print("PairKN equals tally KN on calib and valid:", pk_err)
        F_lp = np.log(sp["valid"].pF)
        F_bpw = systems.bpw(F_lp)
        rows, lps = [], {}
        for name, ms in twin.TRIALS:
            PX_cal, PX_val, info = twin.trial_class_model(ms, D["calib"], D["valid"], sp["calib"].y,
                                                         sp["calib"].b, sp["valid"].b)
            lp, lamb, sinfo = system(PX_cal, PX_val, sp, Z, e, name, run)
            r = {"trial": name, "valid_bpw": systems.bpw(lp), "minus_F": systems.bpw(lp) - F_bpw,
                 "class_bits_valid": systems.class_bits(PX_val, sp["valid"]),
                 "standalone_bpw_valid": systems.standalone_bpw(PX_val, sp["valid"]),
                 "rare_split": systems.rare_split(lp, sp["valid"]), "class_mixture": info, "system": sinfo}
            rows.append(r)
            lps[name] = (lp, PX_cal, PX_val)
            run.print(f"twin trial {name}: valid {r['valid_bpw']:.5f} bpw ({r['minus_F']:+.5f} vs F), "
                      f"class bits {r['class_bits_valid']:.4f}")
        best = min(range(len(rows)), key=lambda i: (rows[i]["valid_bpw"], i))
        bname, bms = twin.TRIALS[best]
        b_lp, b_cal, b_val = lps[bname]
        k0r = gates.k0(F_bpw, rows[best]["valid_bpw"])
        run.print(f"selected twin: {bname}; K0 {k0r}")

        # A/A: the selected trial rebuilt on 95% of the train blocks
        _, _, drop = aa.subsample_train(Z["w_train"], Z["blocks_train"], AA_SEED, AA_DROP)
        keep = ~np.isin(Z["blocks_train"], drop)
        m_aa = twin.build(bms, yt[keep], comps(Z, "train", keep), mode="kn")
        D_aa = {s: twin.dists(m_aa, cp[s]) for s in EVAL}
        e_aa = codes.emission(Z["w_train"][keep], Z["cls"], len(e))
        sp_aa = {s: split(Z, s, e_aa) for s in EVAL}
        Pc, Pv, _ = twin.trial_class_model(bms, D_aa["calib"], D_aa["valid"], sp_aa["calib"].y, sp_aa["calib"].b,
                                           sp_aa["valid"].b)
        aa_lp, _, aa_info = system(Pc, Pv, sp_aa, Z, e_aa, "A/A", run)
        aar = gates.aa(rows[best]["valid_bpw"], systems.bpw(aa_lp))
        run.print(f"A/A: dropped {len(drop)} train blocks; {aar}")

        # shuffled labels: the selected trial counted on permuted class targets
        ysh = yt[np.random.default_rng(SHUF_SEED).permutation(len(yt))]
        m_sh = twin.build(bms, ysh, cp["train"], mode="pair")
        D_sh = {s: twin.dists(m_sh, cp[s]) for s in EVAL}
        Pc, Pv, _ = twin.trial_class_model(bms, D_sh["calib"], D_sh["valid"], sp["calib"].y, sp["calib"].b,
                                           sp["valid"].b)
        sh_lp, _, sh_info = system(Pc, Pv, sp, Z, e, "shuffled twin", run)
        shr = gates.shuffled(F_bpw, systems.bpw(sh_lp))
        run.print(f"shuffled-label twin: {shr}")

        # the class-unigram system (diagnostic, no rule): what a context-free class member adds to F
        Pn = {s: systems.null_class_model(yt, sp[s].n) for s in EVAL}
        nl_lp, _, nl_info = system(Pn["calib"], Pn["valid"], sp, Z, e, "class unigram", run)
        run.print(f"class-unigram system (diagnostic): {systems.bpw(nl_lp) - F_bpw:+.5f} bpw vs F")

        out = runlog.out_dir(a.dry) / "twin.npz"
        np.savez(out, F_valid=F_lp, best_valid=b_lp, aa_valid=aa_lp, shuffled_valid=sh_lp, null_valid=nl_lp,
                 PX_best_calib=b_cal, PX_best_valid=b_val, aa_dropped_blocks=drop)
        run.add_output(out)
        run.finish({"F_valid_bpw": F_bpw, "pairkn_vs_kn_max_abs": pk_err, "trials": rows, "selected": bname,
                    "selected_index": best, "K0": k0r, "AA": {**aar, "rebuilt_bpw": systems.bpw(aa_lp),
                                                              "dropped_blocks": drop.tolist(), "system": aa_info},
                    "shuffled": {**shr, "bpw": systems.bpw(sh_lp), "system": sh_info},
                    "class_unigram_diagnostic": {"bpw": systems.bpw(nl_lp), "minus_F": systems.bpw(nl_lp) - F_bpw,
                                                 "system": nl_info},
                    "members": {n: md.summary() for n, md in models.items() if hasattr(md, "summary")}})

    runlog.main_wrapper(run, body)


if __name__ == "__main__":
    main()
