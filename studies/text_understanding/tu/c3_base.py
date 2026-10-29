#!/usr/bin/env python
"""Step 3c, the base by R2 (tu/PLAN_3c.md "Base stack", fixed before 3a's verdict): writes tu/c3/base.json from the rule.

    python -I c3_base.py --fa on|off

- 3a PASS (tu/a3/verdict_a3.json): the a3B table at lambda* (tu/a3/tables/a3B_d768.npz, cache tu/a3/feats/a3B_d768),
  the reader's s from R3 (tu/r3/r3_picks.json; R3 must be done), the TM reference = R3's seed rows (tag r3-seeds), the
  LR reference = 3a's a3B LR row.
- 3a FAIL: the C4 counting table ppw_S1000 (w1/tables/ppw_S1000_d500_e0.5.npz, cache w1/feats/ppw_S1000_d500_e0.5),
  the reader's s from 0b (tu/b0_picks.json), the TM reference = 0b's seed rows (tag 0b-seeds), the LR reference = 3a's
  fresh ppw_S1000 LR row.
- Both: potion's TM rows at its own 0b pick (descriptive gap share); -fa as the GPU smoke test passed it.
Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
SETS = ["clinc150", "banking77", "hwu64"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", choices=["on", "off"], required=True)
    args = ap.parse_args()
    out = HERE / "c3" / "base.json"
    assert not out.exists(), f"{out} exists"
    v = json.loads((HERE / "a3" / "verdict_a3.json").read_text())["verdict"]
    b0 = json.loads((HERE / "b0_picks.json").read_text())
    pot = {"file": str(HERE / "results_0b_val.jsonl"), "tag": "0b-seeds", "s": {d: b0[f"potion-base-32M|{d}"]["pick"] for d in SETS}}
    if v == "PASS":
        r3 = json.loads((HERE / "r3" / "r3_picks.json").read_text())
        assert (HERE / "r3" / "runs_r3.log").read_text().rstrip().endswith("R3 done"), "R3 is not done"
        base = {"base": "a3B_d768", "table_npz": str(HERE / "a3" / "tables" / "a3B_d768.npz"),
                "ref_dir": str(HERE / "a3" / "feats" / "a3B_d768"), "s": {d: r3[f"a3B_d768|{d}"]["pick"] for d in SETS},
                "ref_tm": {"file": str(HERE / "r3" / "results_r3_val.jsonl"), "tag": "r3-seeds"}}
    else:
        assert v == "FAIL", v
        base = {"base": "ppw_S1000_d500_e0.5", "table_npz": str(EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"),
                "ref_dir": str(EMB / "w1" / "feats" / "ppw_S1000_d500_e0.5"),
                "s": {d: b0[f"ppw_S1000_d500_e0.5|{d}"]["pick"] for d in SETS},
                "ref_tm": {"file": str(HERE / "results_0b_val.jsonl"), "tag": "0b-seeds"}}
    base.update({"rule": f"3a {v}", "feats_F": str(HERE / "c3" / "feats" / "F"),
                 "ref_lr": {"file": str(HERE / "a3" / "results_a3_lr.jsonl")}, "potion_tm": pot, "fa": args.fa})
    out.write_text(json.dumps(base, indent=1))
    print(json.dumps(base))


if __name__ == "__main__":
    main()
