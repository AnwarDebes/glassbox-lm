#!/usr/bin/env python
"""S2b-arm (CPU): one learner family on one co-primary arm's bits; configurations chosen on tune-cal, tune-hold
reported; at the pick, K9 (retrained on shuffled labels); K2 for every configuration; the no-future check.

    chain/part3_cpu_job.sh A tm s2b_arm.py --arm full|nf --family tree|lr1|lr2|hgb [--s1 DIR] [--s2a DIR] [--twin DIR] [--out DIR]

Arms: A-full reads all 364 bits; A-nf reads the 320 bits that are not count-derived. Families and grids:
astruct/twins.py (tree twin) and astruct/learners.py (LR1, LR2, boosted trees). Exit code 3 when K9 fails.
Refuses to start before S2c (K8) is done, and after a K8 STOP (K8 is decided before any learner number exists).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.data import salted_mod
from astruct.gates import k9
from astruct.learners import HGB_GRID, LR_C, fit_hgb, fit_lr, proba
from astruct.readout import Arm, PosTable, contract, mix_bits
from astruct.rows import pair_features, shuffled_labels
from astruct.store import load_rows
from astruct.twins import TREE_ALPHA, TREE_LEAF, TreeTwin

SHUFFLE_SEED = 1


def es_rows(rows: dict, dialogue_ids: list) -> np.ndarray:
    d = rows["pos_dlg"][rows["pos"]]
    ud = np.unique(d)
    es_d = np.array([salted_mod(dialogue_ids[i], "es", 10) == 0 for i in ud])
    return es_d[np.searchsorted(ud, d)]


class Family:
    """configs() -> names; fit(name, y) -> model; score(model, which) -> scores on 'cal', 'hold' or a rows dict."""

    def __init__(self, fam: str, tr: dict, cal: dict, hold: dict, keep: np.ndarray, names: list, n_pos_bits: int,
                 es: np.ndarray):
        import scipy.sparse as sp
        self.fam, self.keep, self.names, self.n_pos_bits, self.es = fam, keep, names, n_pos_bits, es
        self.raw = {"tr": tr["bits"], "cal": cal["bits"], "hold": hold["bits"]}
        if fam == "lr2":
            self.X = {k: pair_features(v, names, n_pos_bits, keep) for k, v in self.raw.items()}
        elif fam == "lr1":
            self.X = {k: sp.csr_matrix(v[:, keep].astype(np.float32)) for k, v in self.raw.items()}
        else:
            self.X = {k: np.ascontiguousarray(v[:, keep]) for k, v in self.raw.items()}
        self._trees = {}

    def configs(self) -> list:
        if self.fam == "tree":
            return [f"leaf{m}_a{int(a)}" for m in TREE_LEAF for a in TREE_ALPHA]
        if self.fam in ("lr1", "lr2"):
            return [f"{self.fam}_C{c:g}" for c in LR_C]
        return [f"hgb_m{m}_lr{lr:g}_l2{l2:g}" for m, lr, l2 in HGB_GRID]

    def fit(self, name: str, y: np.ndarray, tag: str = ""):
        X = self.X["tr"]
        if self.fam == "tree":
            m, a = name.split("_")
            leaf, alpha = int(m[4:]), float(a[1:])
            key = (leaf, tag)
            tt = TreeTwin(leaf, alpha).fit(X, y, tree=self._trees.get(key))
            self._trees[key] = tt.tree
            return tt
        if self.fam in ("lr1", "lr2"):
            return fit_lr(X, y, float(name.split("_C")[1]))
        i = self.configs().index(name)
        m, lr, l2 = HGB_GRID[i]
        return fit_hgb(X, y, self.es, m, lr, l2)

    def score(self, model, which) -> np.ndarray:
        if isinstance(which, str):
            X = self.X[which]
        else:  # a rows dict (no-future check)
            import scipy.sparse as sp
            b = which["bits"]
            X = (pair_features(b, self.names, self.n_pos_bits, self.keep) if self.fam == "lr2" else
                 sp.csr_matrix(b[:, self.keep].astype(np.float32)) if self.fam == "lr1" else np.ascontiguousarray(b[:, self.keep]))
        return model.predict(X) if self.fam == "tree" else proba(model, X)

    def info(self, model) -> dict:
        if self.fam == "tree":
            return {"leaves": model.n_leaves, "depth": int(model.tree.get_depth())}
        if self.fam in ("lr1", "lr2"):
            return {"n_iter": int(np.max(model.n_iter_)), "converged": bool(model.converged_), "n_features": int(model.coef_.shape[1])}
        return {"n_iter": int(model.n_iter_)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "nf"])
    ap.add_argument("--family", required=True, choices=["tree", "lr1", "lr2", "hgb"])
    ap.add_argument("--s0", default=str(paths.OUT / "s0"))
    ap.add_argument("--s1", default=str(paths.OUT / "s1"))
    ap.add_argument("--s2a", default=str(paths.OUT / "s2a"))
    ap.add_argument("--twin", default=str(paths.OUT / "s2b_twin"))
    ap.add_argument("--aa", default=str(paths.OUT / "s2c_aa"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run-ignore-k8", action="store_true", help="synthetic dry run only: run even after a K8 STOP")
    args = ap.parse_args()
    env = paths.check_env()
    out_path = Path(args.out or paths.OUT / f"s2b_{args.arm}_{args.family}")
    if args.dry_run_ignore_k8 and paths.is_real_output(out_path):
        raise SystemExit("--dry-run-ignore-k8 is for the synthetic dry run only")
    # K8 is decided before any learner's tune-hold number exists: S2c must be done, and its K8 must have passed
    aa_done = Path(args.aa) / "done.json"
    if not aa_done.exists():
        raise SystemExit(f"S2c (A/A, K8) is not done ({aa_done}): run s2c_aa.py before the learners")
    if not json.loads(aa_done.read_text())["K8"] and not args.dry_run_ignore_k8:
        raise SystemExit("K8 STOP (A/A): no learner runs")
    out = paths.OutDir(out_path)
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    s2a = Path(args.s2a)
    bl = json.loads((Path(args.s0) / "bit_list.json").read_text())
    cd = np.array(bl["count_derived"])
    keep = np.ones(len(cd), bool) if args.arm == "full" else ~cd
    meta = json.loads((Path(args.s0) / "meta.json").read_text())
    tr, cal, hold = (load_rows(s2a / f"{n}_rows.npz") for n in ("train", "cal", "hold"))
    nfr = {v: (load_rows(s2a / f"nofuture_{v}_mod.npz"), load_rows(s2a / f"nofuture_{v}_ref.npz")) for v in ("same", "swap")}
    B = np.load(Path(args.twin) / "bits.npz")
    B_hold = B["B_hold"]
    ptc, pth = PosTable(cal, cal["p_true"]), PosTable(hold, hold["p_true"])
    full = []
    for nm in ("cal", "hold"):
        z = np.load(Path(args.s1) / f"summ_{nm}.npz")
        full.append({int(k): z["full_vec"][i] for i, k in enumerate(z["full_pos"])})
    es = es_rows(tr, meta["dialogue_ids"])
    log(f"S2b {args.arm}/{args.family} start; bits {int(keep.sum())} of {len(keep)}; training rows {len(tr['y'])}"
        f" ({int(es.sum())} for early stopping)")
    fam = Family(args.family, tr, cal, hold, keep, bl["names"], bl["n_position_bits"], es)
    res, bits_c, bits_h, models = {}, {}, {}, {}
    for name in fam.configs():
        t1 = time.time()
        m = fam.fit(name, tr["y"])
        sc, sh = fam.score(m, "cal"), fam.score(m, "hold")
        arm = Arm(name).fit(sc, cal, ptc)
        hb = arm.bits(sh, pth)
        k2c = contract(arm.iso(sc), cal, full[0], ptc, arm.b, cal["pos_x"], arm.cal_bits)
        k2h = contract(arm.iso(sh), hold, full[1], pth, arm.b, hold["pos_x"], hb)
        if not (k2c["pass"] and k2h["pass"]):
            raise SystemExit(f"K2 failed for {name}: {k2c} {k2h}")
        res[name] = {"b": arm.b, "tune_cal_bits": float(arm.cal_bits.mean()), "tune_hold_bits": float(hb.mean()),
                     "tune_hold_minus_B": float((hb - B_hold).mean()), "seconds": round(time.time() - t1, 1),
                     **fam.info(m), "K2": {"cal": k2c, "hold": k2h}}
        bits_c[name], bits_h[name], models[name] = arm.cal_bits, hb, (m, arm)
        log(name, json.dumps({k: v for k, v in res[name].items() if k != "K2"}))
    pick = min(res, key=lambda n: (res[n]["tune_cal_bits"], fam.configs().index(n)))
    m_p, arm_p = models[pick]
    # no-future: the pick's per-position bits on the modified replies equal those of the batch rows
    sm, sr = nfr["same"]
    nf_mod = mix_bits(arm_p.iso(fam.score(m_p, sm)), PosTable(sm, sm["p_true"]), arm_p.b)
    nf_ref = mix_bits(arm_p.iso(fam.score(m_p, sr)), PosTable(sr, sr["p_true"]), arm_p.b)
    wm, wr = nfr["swap"]
    nf = {"positions": len(nf_ref), "bits_equal": bool(np.array_equal(nf_mod, nf_ref)),
          "swap_scores_equal": bool(np.array_equal(fam.score(m_p, wm), fam.score(m_p, wr)))}
    nf["pass"] = nf["bits_equal"] and nf["swap_scores_equal"]
    log("no-future (scores and bits at the pick)", json.dumps(nf))
    if not nf["pass"]:
        raise SystemExit("no-future test failed at the arm level")
    ys = shuffled_labels(tr, SHUFFLE_SEED)
    ms = fam.fit(pick, ys, tag="shuffled")
    arm_s = Arm(pick + "_shuffled").fit(fam.score(ms, "cal"), cal, ptc)
    hb_s = arm_s.bits(fam.score(ms, "hold"), pth)
    k9r = k9(arm_s.b, float((hb_s - B_hold).mean()))
    log(f"pick {pick}: tune-cal {res[pick]['tune_cal_bits']:.5f}, tune-hold {res[pick]['tune_hold_bits']:.5f}, "
        f"minus B {res[pick]['tune_hold_minus_B']:+.5f}; K9 {json.dumps(k9r)}")
    np.savez_compressed(out / "bits.npz", **{f"cal_{k}": v for k, v in bits_c.items()},
                        **{f"hold_{k}": v for k, v in bits_h.items()}, shuffled_cal=arm_s.cal_bits, shuffled_hold=hb_s)
    with open(out / "pick.pkl", "wb") as f:
        pickle.dump({"pick": pick, "model": m_p, "arm": arm_p, "keep": keep, "family": args.family}, f)
    summary = {"arm": args.arm, "family": args.family, "pick": pick, "configs": res, "K9": k9r, "nofuture": nf,
               "bits_used": int(keep.sum()), "shuffle_seed": SHUFFLE_SEED, "seconds": round(time.time() - t0, 1),
               "env": env, "code_sha256": paths.code_hashes(), "s2a_done_sha256": paths.sha256_file(s2a / "done.json")}
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    out.done({"utc_end": paths.utc(), "pick": pick, "K9": k9r["pass"]})
    if not k9r["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
