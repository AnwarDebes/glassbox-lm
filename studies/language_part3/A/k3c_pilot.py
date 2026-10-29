#!/usr/bin/env python
"""K3c (CPU): one CPU pilot of the Tsetlin machine for one co-primary arm (PLAN_part3.md section 3, K3c).

    chain/part3_cpu_job.sh A dl k3c_pilot.py --arm full|nf --s 15|40 [--shuffled] [--s0 DIR] [--s2a DIR] [--twin DIR] [--out DIR]

Our CoTM (dl/tm.py, workspace code, not a CAIR library): 2 classes, 4,000 clauses, T 8,000, s 15 or 40, 10 epochs,
batch 128, negation, hardest negative, init half, seed 1, G1's loop on the CPU (astruct/tmrun.py), torch at 2
threads. Rows: the training rows of the dialogues with sha256("pilot:" + dialogue_id) mod 10 == 0 (about 10%).
--shuffled (K9 for the TM): the positive of every pilot position re-drawn uniformly among its candidates (seed 1).
Read-out as every arm: isotonic and b on tune-cal; tune-hold bits reported. --epochs / --clauses are for the dry run.
A real pilot (outputs under A/out/; amendment 1) refuses to start unless its settings are the registered ones (10
epochs, 4,000 clauses, s 15 or 40, its own folder A/out/k3c_<arm>_s<s>[_shuffled]) and A/out/s2d_gates/gates.json has
go_to_K3c[arm] true; the shuffled pilot also needs both pilots done, s equal to the better s (lower tune-cal bits, ties:
s 15) and K3c passed at that s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.data import salted_mod
from astruct.gates import k3c
from astruct.readout import Arm, PosTable, contract, mix_bits
from astruct.rows import shuffled_labels
from astruct.store import load_rows, subset_rows
from astruct.tmrun import clause_report, tm_scores, train_tm

T_PILOT, SEED, SHUFFLE_SEED = 8000, 1, 1
REGISTERED = {"epochs": 10, "clauses": 4000}
PILOT_S = (15, 40)


def tag_of(arm: str, s: float, shuffled: bool) -> str:
    return f"k3c_{arm}_s{s:g}" + ("_shuffled" if shuffled else "")


def real_pilot_problems(arm: str, s: float, shuffled: bool, epochs: int, clauses: int, out_path: Path,
                        root: Path | None = None) -> list:
    """Why a real pilot may not start (amendment 1): the registered settings, the arm's S2d decision and, for the
    shuffled pilot (K9 for the TM), the K3c decision of the two finished pilots."""
    root = Path(root if root is not None else paths.OUT)
    bad = []
    if epochs != REGISTERED["epochs"] or clauses != REGISTERED["clauses"]:
        bad.append(f"--epochs {epochs} / --clauses {clauses}: a real pilot uses the registered "
                   f"{REGISTERED['epochs']} epochs and {REGISTERED['clauses']} clauses (the overrides are for the dry run)")
    if s not in PILOT_S:
        bad.append(f"--s {s:g}: the registered pilots use s {PILOT_S}")
    if Path(out_path).resolve() != (root / tag_of(arm, s, shuffled)).resolve():
        bad.append(f"a real pilot writes to its own folder {root / tag_of(arm, s, shuffled)}, not {out_path}")
    gp = root / "s2d_gates" / "gates.json"
    if not gp.exists():
        bad.append(f"S2d is not done ({gp} missing): no pilot before the S2 gates")
        return bad
    gates = json.loads(gp.read_text())
    if not gates["go_to_K3c"][arm]:
        bad.append(f"A-{arm} stopped at the S2 gates (go_to_K3c false): no pilot")
        return bad
    if shuffled:
        done = {x: root / tag_of(arm, x, False) for x in PILOT_S}
        if not all((d / "done.json").exists() for d in done.values()):
            bad.append("the shuffled pilot runs after both pilots (s 15 and s 40)")
            return bad
        pil = {x: json.loads((d / "results.json").read_text()) for x, d in done.items()}
        best = min(PILOT_S, key=lambda x: (pil[x]["tune_cal_bits"], x))
        if s != best:
            bad.append(f"the shuffled pilot runs at the better s {best}, not {s:g}")
        g = k3c(pil[best]["tune_hold_minus_B"], gates["arms"][arm]["families"]["hgb"]["minus_B"]["mean"])
        if not g["pass"]:
            bad.append("K3c stopped this arm: no shuffled pilot is needed")
    return bad


def pilot_positions(rows: dict, dialogue_ids: list) -> np.ndarray:
    d = rows["pos_dlg"]
    ud = np.unique(d)
    keep = np.array([salted_mod(dialogue_ids[i], "pilot", 10) == 0 for i in ud])
    return np.flatnonzero(keep[np.searchsorted(ud, d)])


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "nf"])
    ap.add_argument("--s", required=True, type=float)
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--s0", default=str(paths.OUT / "s0"))
    ap.add_argument("--s1", default=str(paths.OUT / "s1"))
    ap.add_argument("--s2a", default=str(paths.OUT / "s2a"))
    ap.add_argument("--twin", default=str(paths.OUT / "s2b_twin"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--clauses", type=int, default=4000)
    args = ap.parse_args()
    env = paths.check_env()
    torch.set_num_threads(2)
    tag = tag_of(args.arm, args.s, args.shuffled)
    out_path = Path(args.out or paths.OUT / tag)
    if paths.is_real_output(out_path):
        bad = real_pilot_problems(args.arm, args.s, args.shuffled, args.epochs, args.clauses, out_path)
        if bad:
            raise SystemExit("real pilot refused:\n  " + "\n  ".join(bad))
    out = paths.OutDir(out_path)
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    s2a = Path(args.s2a)
    bl = json.loads((Path(args.s0) / "bit_list.json").read_text())
    cd = np.array(bl["count_derived"])
    keep = np.ones(len(cd), bool) if args.arm == "full" else ~cd
    meta = json.loads((Path(args.s0) / "meta.json").read_text())
    tr = load_rows(s2a / "train_rows.npz")
    tr = subset_rows(tr, pilot_positions(tr, meta["dialogue_ids"]))
    cal, hold = load_rows(s2a / "cal_rows.npz"), load_rows(s2a / "hold_rows.npz")
    B_hold = np.load(Path(args.twin) / "bits.npz")["B_hold"]
    y = shuffled_labels(tr, SHUFFLE_SEED) if args.shuffled else tr["y"]
    log(f"{tag}: pilot rows {len(y)} ({tr['n_pos']} positions), bits {int(keep.sum())}, torch threads {torch.get_num_threads()}")
    tm = train_tm(tr["bits"][:, keep], y, T_PILOT, args.s, SEED, device="cpu", log=log, epochs=args.epochs,
                  n_clauses=args.clauses)
    sc, sh = tm_scores(tm, cal["bits"][:, keep]), tm_scores(tm, hold["bits"][:, keep])
    ptc, pth = PosTable(cal, cal["p_true"]), PosTable(hold, hold["p_true"])
    arm = Arm(tag).fit(sc, cal, ptc)
    hb = arm.bits(sh, pth)
    # K2 before any number is logged: the mix sums to 1 and is positive at the kept tune positions
    k2 = {}
    for nm, rows_, s_, pt_, bits_ in (("cal", cal, sc, ptc, arm.cal_bits), ("hold", hold, sh, pth, hb)):
        z = np.load(Path(args.s1) / f"summ_{nm}.npz")
        full = {int(k): z["full_vec"][i] for i, k in enumerate(z["full_pos"])}
        k2[nm] = contract(arm.iso(s_), rows_, full, pt_, arm.b, rows_["pos_x"], bits_)
    if not (k2["cal"]["pass"] and k2["hold"]["pass"]):
        raise SystemExit(f"K2 failed for {tag}: {k2}")
    # no-future: the pilot's bits on the modified replies equal those on the batch rows
    sm, sr = load_rows(s2a / "nofuture_same_mod.npz"), load_rows(s2a / "nofuture_same_ref.npz")
    nf_mod = mix_bits(arm.iso(tm_scores(tm, sm["bits"][:, keep])), PosTable(sm, sm["p_true"]), arm.b)
    nf_ref = mix_bits(arm.iso(tm_scores(tm, sr["bits"][:, keep])), PosTable(sr, sr["p_true"]), arm.b)
    wm, wr = load_rows(s2a / "nofuture_swap_mod.npz"), load_rows(s2a / "nofuture_swap_ref.npz")
    if not (np.array_equal(nf_mod, nf_ref) and np.array_equal(tm_scores(tm, wm["bits"][:, keep]), tm_scores(tm, wr["bits"][:, keep]))):
        raise SystemExit("no-future test failed for the TM pilot")
    cr = clause_report(tm)
    res = {"tag": tag, "arm": args.arm, "s": args.s, "T": T_PILOT, "clauses": args.clauses, "epochs": args.epochs,
           "seed": SEED, "shuffled": args.shuffled, "pilot_rows": len(y), "b": arm.b,
           "tune_cal_bits": float(arm.cal_bits.mean()), "tune_hold_bits": float(hb.mean()),
           "tune_hold_minus_B": float((hb - B_hold).mean()), "epoch_seconds": tm.epoch_seconds,
           "clauses_report": cr, "K2": k2, "nofuture": {"positions": len(nf_ref), "pass": True},
           "engine": "our CoTM (dl/tm.py, workspace code, not a CAIR library)",
           "seconds": round(time.time() - t0, 1), "env": env, "code_sha256": paths.code_hashes()}
    log(f"{tag}: b {arm.b}, tune-cal {res['tune_cal_bits']:.5f}, tune-hold {res['tune_hold_bits']:.5f}, minus B "
        f"{res['tune_hold_minus_B']:+.5f}; empty {cr['empty']}, median literals {cr['median_literals']}, "
        f"no positive literal {cr.get('no_positive_literal')}")
    st = tm.state()
    np.savez_compressed(out / "state.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"])
    np.savez_compressed(out / "bits.npz", cal=arm.cal_bits, hold=hb, score_cal=sc, score_hold=sh)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    out.done({"utc_end": paths.utc(), "tune_cal_bits": res["tune_cal_bits"]})


if __name__ == "__main__":
    main()
