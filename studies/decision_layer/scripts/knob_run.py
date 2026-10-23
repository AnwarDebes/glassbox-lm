#!/usr/bin/env python
"""Run a study script with TM settings that its command line does not expose, and keep its dev predictions (e83;
added 2026-10-21).

    knob_run.py [--neg-sampling hardest|random|mix:Q] [--clause-drop P] [--preds-out FILE] -- SCRIPT [ARGS...]

Plan: logs/plan_e83_learning_dynamics.md. tm.py and every run script stay byte for byte. This wrapper patches
dl.tm.CoTM in its own process, then runs SCRIPT there (runpy, as __main__), the way newtable_run.py and table_run.py
chain:
- CoTM.__init__ sets the given knobs on the TMConfig the script built, before the original __init__ runs. So the row's
  cfg (asdict(cfg) in abcd_explore.py) records them.
- CoTM.fit notes the y_val it is given.
- CoTM.predict appends a CPU copy of every prediction it returns. In abcd_explore.py these are the 20 evaluations
  inside fit, then the final call on the best-dev snapshot.
- --preds-out FILE writes preds [calls, n] (int16), y_val [n] (int16) and the knobs to FILE, only after SCRIPT ends
  without an error. An existing FILE is refused before SCRIPT starts.
With no knob and no --preds-out, nothing is patched. Tested by tests/test_knob_run.py. The GPU knob-off identity (a row
through this wrapper equals the plain row) is checked in the smoke stage of scripts/e83_pick.py.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import tm as dltm  # noqa: E402


def neg_rule(v: str) -> str:
    if v in ("hardest", "random"):
        return v
    m = re.fullmatch(r"mix:(0\.\d+)", v)
    if m and 0.0 < float(m.group(1)) < 1.0:
        return v
    raise argparse.ArgumentTypeError(f"{v!r}: one of hardest, random, mix:Q with 0 < Q < 1")


def drop_p(v: str) -> float:
    try:
        p = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v!r}: not a number") from None
    if not 0.0 <= p < 1.0:
        raise argparse.ArgumentTypeError(f"{v}: needs 0 <= P < 1")
    return p


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit("knob_run: put -- before the script to run")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="knob_run.py")
    ap.add_argument("--neg-sampling", type=neg_rule)
    ap.add_argument("--clause-drop", type=drop_p)
    ap.add_argument("--preds-out", type=Path)
    args = ap.parse_args(argv[:i])
    rest = argv[i + 1 :]
    if not rest or not rest[0].endswith(".py"):
        raise SystemExit("knob_run: no script after --")
    return args, rest


def main() -> None:
    args, rest = parse(sys.argv[1:])
    knobs = {}
    if args.neg_sampling is not None:
        knobs["neg_sampling"] = args.neg_sampling
    if args.clause_drop is not None:
        knobs["clause_drop"] = args.clause_drop
    if args.preds_out is not None and args.preds_out.exists():
        raise SystemExit(f"knob_run: {args.preds_out} exists; not overwriting")
    state: dict = {"preds": [], "y_val": None}
    if knobs or args.preds_out is not None:
        orig_init, orig_fit, orig_predict = dltm.CoTM.__init__, dltm.CoTM.fit, dltm.CoTM.predict

        def init(self, n_classes, n_features, cfg, device="cuda"):
            for k, v in knobs.items():
                setattr(cfg, k, v)
            orig_init(self, n_classes, n_features, cfg, device=device)

        def fit(self, X, y, X_val=None, y_val=None, log=None):
            state["y_val"] = None if y_val is None else np.asarray(y_val).astype(np.int16)
            return orig_fit(self, X, y, X_val, y_val, log)

        def predict(self, X):
            out = orig_predict(self, X)
            state["preds"].append(out.detach().cpu().numpy().astype(np.int16))
            return out

        dltm.CoTM.__init__, dltm.CoTM.fit, dltm.CoTM.predict = init, fit, predict
    print(f"knob_run: knobs {json.dumps(knobs, sort_keys=True)}; preds-out {args.preds_out}; running {' '.join(rest)}", flush=True)
    sys.argv = list(rest)
    try:
        runpy.run_path(str(Path(rest[0]).resolve()), run_name="__main__")
    except SystemExit as e:  # a script that ends with sys.exit(0) has still finished; any other exit stops here
        if e.code not in (None, 0):
            raise
    if args.preds_out is not None:
        if not state["preds"] or state["y_val"] is None:
            raise SystemExit("knob_run: the script made no prediction after a fit with dev labels; no preds file")
        args.preds_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.preds_out, preds=np.stack(state["preds"]), y_val=state["y_val"],
                            knobs=np.array(json.dumps(knobs, sort_keys=True)), argv=np.array(json.dumps(rest)))
        print(f"knob_run: wrote {args.preds_out} ({len(state['preds'])} predictions of {state['y_val'].shape[0]} dev rows)", flush=True)


if __name__ == "__main__":
    main()
