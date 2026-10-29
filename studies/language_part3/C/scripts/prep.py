"""C-core stage prep (PLAN_part3.md section 6, step 6, part 1): codes, inputs, targets and floor arrays.

    scripts/run_stage.sh prep           # real: needs the frozen R1 record
    scripts/run_stage.sh prep --dry     # pseudo splits from train blocks only

Builds tree E and tree B from train only, the 60 input bits and the 65-class targets of train, calib
and valid, the floor F (counts from train, weights on calib), F's class marginals at every calib and
valid position (with every full vector's sum and minimum checked), F's full vectors at 1,000 fixed
positions per split for the contracts, and runs F's contract and the future-change test on calib.
Output: data/prep.npz (+ codes metadata in the run record)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import chains, codes, data, hashing, layout, runlog, systems  # noqa: E402
from ccore import floor as floor_mod  # noqa: E402
from ccore.floor import Floor  # noqa: E402

EVAL = ("calib", "valid")
CONTRACT_SEEDS = {"calib": 31, "valid": 32}
CONTRACT_N = 1000
FUTURE_CUTS = 6
FUTURE_SEED = 33


def future_change(x, bl, eos, V, fl, path6, path8, cls, te, rng, cuts=FUTURE_CUTS):
    """Change the tokens at and after t; the inputs and every distribution at t must stay the same."""
    X0 = layout.features(x, eos, path6, path8)
    b0 = fl.buckets(x)[0]
    D_te0 = te.dist(cls[x])
    out = []
    for t in np.sort(rng.choice(np.arange(5, len(x) - 5), size=cuts, replace=False)):
        x1 = x.copy()
        x1[t:] = rng.integers(0, V, size=len(x) - t)
        x1[-1] = eos
        X1 = layout.features(x1, eos, path6, path8)
        b1 = fl.buckets(x1)[0]
        pos = np.array([t - 3, t - 1, t])
        same = {"inputs": bool(np.array_equal(X0[:t + 1], X1[:t + 1])),
                "buckets": bool(np.array_equal(b0[:t + 1], b1[:t + 1])),
                "floor_vectors": bool(np.array_equal(fl.dists(x, bl, pos, b0), fl.dists(x1, bl, pos, b1))),
                "class_kn_vectors": bool(np.array_equal(D_te0[pos], te.dist(cls[x1])[pos]))}
        out.append({"t": int(t), **same})
    return {"cuts": out, "pass": bool(all(all(v for k, v in c.items() if k != "t") for c in out))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rec, new = hashing.check_frozen(require=not a.dry)
    runlog.refuse_second_real_run("prep", a.dry)
    P = data.load_ptb()
    Q = data.dry_splits(P) if a.dry else P
    cfg = {"stage": "prep", "dry": a.dry, "frozen_utc": rec["frozen_utc"] if rec else None,
           "new_files_since_freeze": new, "tree_E": "exchange 64 (tally) + balanced merge tree, 6 bits",
           "tree_B": "C4 S1000 vectors, weighted-median PCA bisection, depth 8", "inputs": "60 bits",
           "targets": "65 classes", "contract_seeds": CONTRACT_SEEDS, "contract_positions": CONTRACT_N,
           "future_change": {"split": "calib", "cuts": FUTURE_CUTS, "seed": FUTURE_SEED},
           "ptb_source_hashes": P["source_hashes"], "dry_splits": Q.get("dry")}
    run = runlog.Run("prep", cfg, dry=a.dry)

    def body():
        ids, bl, V, eos, vocab = Q["ids"], Q["blocks"], Q["V"], Q["eos"], Q["vocab"]
        tr = ids["train"]
        counts = np.bincount(tr, minlength=V).astype(np.float64)
        cls, path6, leaf_paths, metaE = codes.tree_E(tr, V, eos, log=run.print)
        Xc4, kind, cov = codes.c4_vectors(vocab, counts, eos)
        path8, metaB = codes.tree_B(Xc4, counts, eos)
        e = codes.emission(tr, cls, V)
        emis = systems.emission_check(e, cls)
        assert emis["pass"], emis
        run.print("codes: E classes", int(cls.max()) + 1, "B leaves used", len(np.unique(path8[np.arange(V) != eos])),
                  "C4 coverage", cov)
        X = {s: layout.features(ids[s], eos, path6, path8) for s in ("train",) + EVAL}
        y = {s: cls[ids[s]] for s in ("train",) + EVAL}
        fl = Floor(ids, bl, V, eos)
        arrays = {"cls": cls, "path6": path6, "path8": path8, "leaf_paths": leaf_paths, "emission": e,
                  "counts": counts, "w_train": tr, "blocks_train": bl["train"], "X_train": X["train"],
                  "y_train": y["train"], "floor_lam": fl.lam, "floor_lam_global": fl.lam_global}
        checks = {}
        for s in EVAL:
            pF, b, L = fl.prob(ids[s], bl[s])
            Fc, chk = fl.class_marginals(ids[s], bl[s], cls, layout.N_CLASSES, b)
            if not chk["pass"]:
                raise SystemExit(f"floor vectors fail on {s}: {chk}")
            pos = np.sort(np.random.default_rng(CONTRACT_SEEDS[s]).choice(len(ids[s]), size=CONTRACT_N, replace=False))
            D = fl.dists(ids[s], bl[s], pos, b)
            fast_err = float(np.abs(np.log(D[np.arange(len(pos)), ids[s][pos]]) - np.log(pF[pos])).max())
            fc_err = float(np.abs(Fc[pos] - D @ np.eye(layout.N_CLASSES)[cls]).max())
            con = {"positions": CONTRACT_N, "max_sum_err": float(np.abs(D.sum(1) - 1).max()), "min_p": float(D.min()),
                   "max_fast_vs_vector_log_err": fast_err, "max_class_marginal_err": fc_err}
            con["pass"] = bool(con["max_sum_err"] <= 1e-6 and con["min_p"] > 0 and fast_err <= 1e-9 and fc_err <= 1e-12)
            if not con["pass"]:
                raise SystemExit(f"floor contract failed on {s}: {con}")
            sp = systems.Split(ids[s], y[s], pF, Fc, b, L, bl[s], e)   # asserts F(c) >= F(w)
            checks[s] = {"all_positions": chk, "contract": con, "tokens": int(sp.n)}
            arrays.update({f"X_{s}": X[s], f"y_{s}": y[s], f"w_{s}": ids[s], f"blocks_{s}": bl[s], f"pF_{s}": pF,
                           f"Fc_{s}": Fc, f"b_{s}": b, f"L_{s}": L, f"pos_{s}": pos, f"D_{s}": D})
        a1 = floor_mod.a1_check(fl, ids, bl, V, eos, compare=not a.dry)   # dry: pseudo splits, no comparison
        if not a.dry:
            run.print("floor code reproduces A1's own valid numbers:", {k: v["diff"] for k, v in a1.items()})
        te = chains.TEModel(y["train"], 5)
        fut = future_change(ids["calib"], bl["calib"], eos, V, fl, path6, path8, cls, te,
                            np.random.default_rng(FUTURE_SEED))
        if not fut["pass"]:
            raise SystemExit(f"future-change test failed: {fut}")
        run.print("floor contract, all-position checks and future-change test passed")
        F_bpw = {s: systems.bpw(np.log(arrays[f"pF_{s}"])) for s in EVAL}
        run.print("floor F bits per word:", {s: round(v, 5) for s, v in F_bpw.items()})
        out = runlog.out_dir(a.dry) / "prep.npz"
        np.savez(out, **arrays)
        run.add_output(out)
        ones = {s: float(X[s].mean()) for s in X}
        run.finish({"floor": fl.summary(), "floor_bpw": F_bpw, "checks": checks, "future_change": fut, "a1_check": a1,
                    "emission": emis, "c4_coverage": cov, "tree_E": metaE, "tree_B": metaB,
                    "class_sizes": np.bincount(cls, minlength=layout.N_CLASSES).tolist(),
                    "share_of_one_bits": ones, "tokens": {s: int(len(ids[s])) for s in ("train",) + EVAL}})

    runlog.main_wrapper(run, body)


if __name__ == "__main__":
    main()
