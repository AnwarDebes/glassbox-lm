#!/usr/bin/env python
"""e82's R5 (plan logs/plan_e82_refined_table_agent.md): e72d's checks and guards on the refined-table agent's five ABCD
models at e82-R3's s*. Property runs start only after the R5 note (logs/plan_e82_r5_guards.md) is frozen.

    verify_policy_a3B.py --s S --prepare                       # the model folder, from e82's Stage C models
    verify_policy_a3B.py --s S --check encoding --threads 1 --tag e82r5
    verify_policy_a3B.py --s S --check properties [--guard] --threads 1 --tag e82r5
    verify_policy_a3B.py --s S --check guardcost --tag e82r5    # open loop and closed loop

A thin wrapper of scripts/verify_policy_c4.py (unedited), as verify_policy_c4_e72e.py is. Three things change:
- it registers c4words_a3B_v1 through scripts/newtable_run.register and sets vpc.TABLE to that name, so the view's
  bits are built on the a3B table, never on the C4 table;
- vpc.SRC is results/explore/r5_a3B_s<S>/ (model_seed<k>.npz and bit_names.json). --prepare fills it from e82's ABCD
  models at 4,000 clauses and s* (results/explore/abcd_models/<tag>_seed<k>.npz, k = 1..5; byte-for-byte copies). The
  bit names come from the models' own name arrays, and verify_policy_c4 asserts that they equal the view's;
- vpc.OUT is results/explore/verify_e82r5.jsonl.
Everything else is e72d's and, through it, e70's: the 50 checks, the exact MILP encoding and HiGHS options, the
level-B domain rows from the bit names (xcust/xrecent keep their names, so the thermometer chains apply), and the
guards with M recomputed per model. Added 2026-10-21.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import e82_pick  # noqa: E402
import verify_policy_c4 as vpc  # noqa: E402
from newtable_run import register  # noqa: E402

NAME = "c4words-a3B-v1"
TABLE_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"


def prepare(s: int) -> None:
    dst = ROOT / "results" / "explore" / f"r5_a3B_s{s}"
    if dst.exists():
        raise SystemExit(f"{dst} exists; not overwriting")
    tag = e82_pick.tag("abcd", 4000, s)
    dst.mkdir(parents=True)
    names = None
    for k in range(1, 6):
        src = ROOT / "results" / "explore" / "abcd_models" / f"{tag}_seed{k}.npz"
        shutil.copy2(src, dst / f"model_seed{k}.npz")
        n = [str(x) for x in np.load(src)["names"]]
        assert names is None or n == names, f"{src}: other bit names"
        names = n
        print(f"{dst.name}/model_seed{k}.npz {hashlib.sha256(src.read_bytes()).hexdigest()} (from abcd_models/{src.name})", flush=True)
    (dst / "bit_names.json").write_text(json.dumps(names))
    print(f"{dst.name}/bit_names.json {len(names)} names", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--prepare", action="store_true")
    a, rest = ap.parse_known_args()
    if a.prepare:
        prepare(a.s)
        return
    if "properties" in rest and not (ROOT / "logs" / "plan_e82_r5_guards.frozen.json").exists():
        raise SystemExit("R5 property runs start only after logs/plan_e82_r5_guards.md is frozen")
    if "--tag" not in rest:
        raise SystemExit("pass --tag e82r5")
    if register(Path("data/tables/c4words_a3B_v1"), NAME) != TABLE_SHA:
        raise SystemExit("not the registered a3B table")
    vpc.TABLE = NAME
    vpc.SRC = ROOT / "results" / "explore" / f"r5_a3B_s{a.s}"
    vpc.OUT = ROOT / "results" / "explore" / "verify_e82r5.jsonl"
    sys.argv = [sys.argv[0]] + rest
    vpc.main()


if __name__ == "__main__":
    main()
