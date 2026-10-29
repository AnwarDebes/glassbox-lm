#!/usr/bin/env python
"""K3c decision for one arm, from its two CPU pilots (s 15 and 40) and, when K3c passes, its shuffled-label pilot
(K9 for the TM).

    chain/part3_cpu_job.sh A tm k3c_gate.py --arm full|nf [--root DIR] [--out DIR] [--gates FILE]

The better pilot is the one with the lower tune-cal bits (ties: s 15). K3c: stop the arm if B minus that pilot's
tune-hold mix is less than a third of B minus the arm's boosted-trees mix (S2d). K9 for the TM: the shuffled pilot at
the better s gets b = 0, or a tune-hold mix no more than 0.0010 below B.
Order: run this after the two pilots. K3c STOP -> the decision is written (exit 3); no shuffled pilot is needed.
K3c pass and no shuffled pilot yet -> nothing is written; the next command is printed (exit 5). With the shuffled
pilot -> the final decision is written (exit 0: the arm may go to the GPU stage; exit 3: it stops).
Amendment 1 (2026-10-21, before any run): K3c also needs the pilot to be better than B (astruct/gates.k3c); a real
decision (outputs under A/out/) reads only A/out/ and its S2d gates.json, and refuses pilots whose settings are not the
registered ones (arm, s, 4,000 clauses, 10 epochs, T 8,000, seed 1, shuffled or not).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.gates import k3c, k9

PILOT_S = (15, 40)
ENGINE = "our CoTM (dl/tm.py, workspace code, not a CAIR library)"
REGISTERED = {"clauses": 4000, "epochs": 10, "T": 8000, "seed": 1}


def pilot_problems(res: dict, arm: str, s: int, shuffled: bool) -> list:
    """A pilot's results.json against its registered settings (amendment 1)."""
    want = {**REGISTERED, "arm": arm, "s": s, "shuffled": shuffled}
    return [f"{res.get('tag')}: {k} is {res.get(k)!r}, registered {v!r}" for k, v in want.items() if res.get(k) != v]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "nf"])
    ap.add_argument("--root", default=str(paths.OUT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--gates", default=None, help="the S2d gates.json (default: <root>/s2d_gates/gates.json)")
    args = ap.parse_args()
    paths.check_env()
    R = Path(args.root)
    out_path = Path(args.out or R / f"k3c_gate_{args.arm}")
    real = paths.is_real_output(out_path)
    if real and (R.resolve() != paths.OUT.resolve() or args.gates is not None
                 or out_path.resolve() != (paths.OUT / f"k3c_gate_{args.arm}").resolve()):
        raise SystemExit("a real K3c decision reads only A/out/ and its own S2d gates.json, and writes A/out/k3c_gate_<arm>")
    gates = json.loads(Path(args.gates or R / "s2d_gates" / "gates.json").read_text())
    if not gates["go_to_K3c"][args.arm]:
        raise SystemExit(f"A-{args.arm} stopped at the S2 gates; no pilot decision")
    pil = {s: json.loads((R / f"k3c_{args.arm}_s{s}" / "results.json").read_text()) for s in PILOT_S}
    if real:
        bad = [x for s in PILOT_S for x in pilot_problems(pil[s], args.arm, s, False)]
        if bad:
            raise SystemExit("pilot settings differ from the registered ones:\n  " + "\n  ".join(bad))
    best = min(PILOT_S, key=lambda s: (pil[s]["tune_cal_bits"], s))
    d_trees = gates["arms"][args.arm]["families"]["hgb"]["minus_B"]["mean"]
    g = k3c(pil[best]["tune_hold_minus_B"], d_trees)
    sh_dir = R / f"k3c_{args.arm}_s{best}_shuffled"
    if g["pass"] and not (sh_dir / "done.json").exists():
        print(f"K3c passes for A-{args.arm} (better s {best}); K9 needs the shuffled pilot first:\n"
              f"  chain/part3_cpu_job.sh A dl k3c_pilot.py --arm {args.arm} --s {best} --shuffled")
        sys.exit(5)
    k9r = None
    if g["pass"]:
        sh = json.loads((sh_dir / "results.json").read_text())
        if real:
            bad = pilot_problems(sh, args.arm, best, True)
            if bad:
                raise SystemExit("shuffled pilot settings differ from the registered ones:\n  " + "\n  ".join(bad))
        k9r = k9(sh["b"], sh["tune_hold_minus_B"])
    ok = bool(g["pass"] and k9r["pass"])
    out = paths.OutDir(out_path)
    log = paths.Log(out / "log.txt")
    res = {"arm": args.arm, "better_s": best, "pilots": {s: {k: pil[s][k] for k in ("b", "tune_cal_bits", "tune_hold_bits",
           "tune_hold_minus_B")} for s in PILOT_S}, "K3c": g, "K9_tm": k9r, "go_to_GPU": ok, "engine": ENGINE}
    txt = (f"A-{args.arm} K3c ({ENGINE}): better pilot s {best} (tune-cal {pil[best]['tune_cal_bits']:.5f}); "
           f"B minus pilot {g['gain_pilot']:+.5f} (must be > 0), B minus trees {g['gain_trees']:+.5f} bits per token "
           f"(the pilot needs a third of it) -> "
           + ("pass" if g["pass"] else "STOP") + "; K9 (TM, shuffled) "
           + ("not needed" if k9r is None else ("pass" if k9r["pass"] else "FAIL"))
           + f" -> {'may go to the GPU stage' if ok else 'no GPU job'}")
    log(txt)
    (out / "k3c.json").write_text(json.dumps(res, indent=1))
    (out / "words.txt").write_text(txt + "\n")
    out.done({"utc_end": paths.utc(), "go_to_GPU": ok})
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
