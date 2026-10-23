#!/usr/bin/env python
"""e81x (dated note logs/plan_e81_note2_2026-10-21.md; declared post hoc, dev only, no rule): LR on MultiWOZ's view-B
bits below the old C grid's edge (0.001), reported beside e81's B2, never in its place. Read only.

    e81x_report.py     # after logs/jobs_e81x_lr_lowC.txt and e81_pick.py --stage final

For the refined (a3B) and the C4 bit sets it gives:
- the dev accuracy at C 1e-3, 3e-4 and 1e-4. The C 1e-3 job reruns the old edge value, and whether it reproduces the old
  LR row exactly is printed;
- the best of the three, with ties going to the larger C, and whether that best is at the new edge (1e-4).
Beside B2 it prints the refined TM at e81's pick (16k, mean of seeds 1-2, the value e81_pick.py's B2 line uses) minus
the best LR, and the same against the old LR. For C4 it prints the TM at T = clauses (16k, mean of seeds 1-2) minus the
best LR.
Every LR row must be on the TM rows' dev bits, with its one C, written after the note's freeze.
Tested by tests/test_e81x_report.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e81_pick as P  # noqa: E402

CS = [("C1e-3", 0.001), ("C3e-4", 0.0003), ("C1e-4", 0.0001)]
OLD = {"a3B": "dry_e82mwoz_lr_mem_a3B", "C4": "dry_e77mwoz_lr_mem_c4"}


def lr_rows(R: P.Rows, agent: str, frozen: str) -> tuple[dict, float]:
    h = R.get(P.AGENTS[agent]["ref4k"]["multiwoz"], 1, agent, "multiwoz", 4000, 4000)["bits_sha256"]["dev"]
    acc = {}
    for cname, c in CS:
        name = f"dry_e81x_mwoz_lr_{agent}_{cname}"
        j = json.loads((ROOT / "results" / "explore" / "dryrun" / name / "seed0.json").read_text())
        if j["arm"] != "lr" or j["bits_sha256"]["dev"] != h or j["grid_c"] != [c] or j["C"] != c or j["time"] <= frozen:
            raise SystemExit(f"{name}: not LR at C {c} on the TM rows' dev bits, written after the note's freeze")
        acc[c] = j["dev_acc"]
    old = json.loads((ROOT / "results" / "explore" / "dryrun" / OLD[agent] / "seed0.json").read_text())
    if old["bits_sha256"]["dev"] != h or old["C"] != 0.001:
        raise SystemExit(f"{OLD[agent]}: not the old edge row")
    return acc, old["dev_acc"]


def b2_tm(R: P.Rows) -> float:
    """The refined TM value of e81_pick.py's B2 line: its MultiWOZ pick at 16k, mean of seeds 1-2."""
    acc4 = {r: P.row4k(R, "a3B", "multiwoz", r)["acc"] for r in P.RATIOS}
    best = max(P.RATIOS, key=lambda r: (acc4[r], -P.RATIOS.index(r)))
    r = best if acc4[best] - acc4[1.0] >= 0.003 - 1e-12 else 1.0
    if r == 1.0:
        rows = [R.get(t, sd, "a3B", "multiwoz", 16000, 16000) for t, sd in P.AGENTS["a3B"]["reffull"]["multiwoz"]]
    else:
        T = int(16000 * r)
        rows = [R.get(P.tag("multiwoz", "a3B", 16000, 8, T), sd, "a3B", "multiwoz", 16000, T) for sd in (1, 2)]
    return (rows[0]["acc"] + rows[1]["acc"]) / 2


def main() -> None:
    frozen = json.loads((ROOT / "logs" / "plan_e81_note2_2026-10-21.frozen.json").read_text())["frozen_utc"]
    R = P.Rows()
    tm = {"a3B": b2_tm(R)}
    c4 = [R.get(t, sd, "C4", "multiwoz", 16000, 16000) for t, sd in P.AGENTS["C4"]["reffull"]["multiwoz"]]
    tm["C4"] = (c4[0]["acc"] + c4[1]["acc"]) / 2
    for agent in ("a3B", "C4"):
        acc, old = lr_rows(R, agent, frozen)
        best = max(CS, key=lambda kc: (acc[kc[1]], -CS.index(kc)))
        rep = "reproduces the old row" if acc[0.001] == old else f"does NOT reproduce the old row ({old:.4f})"
        print(f"{agent:4} LR on MultiWOZ dev: " + ", ".join(f"C {n[1:]} {acc[c]:.4f}" for n, c in CS) + f" (C 1e-3 {rep}); best C {best[0][1:]} "
              f"{acc[best[1]]:.4f}" + (" (again at the grid's edge)" if best[1] == 0.0001 else " (inside the extended grid)"))
        what = "TM at e81's pick, 16k, mean of seeds 1-2" if agent == "a3B" else "TM at T = clauses, 16k, mean of seeds 1-2"
        print(f"{'B2' if agent == 'a3B' else 'C4'} beside it (extended LR, no rule): {what} {tm[agent]:.4f} minus the best LR "
              f"{100 * (tm[agent] - acc[best[1]]):+.2f} (against the old LR at C 1e-3: {100 * (tm[agent] - old):+.2f})")


if __name__ == "__main__":
    main()
