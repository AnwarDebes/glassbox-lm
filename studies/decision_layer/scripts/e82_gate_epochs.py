#!/usr/bin/env python
"""Beside e82's go-bars and the R24 gate (no rule; a note of 2026-10-21 03:52 UTC): the same differences at the
final logged epoch and as the mean of the last 10 logged dev evaluations, so the reader sees how much best-epoch
selection adds. The gates themselves are judged on the frozen statistic (the best-dev snapshot), by e82_pick.py.

    e82_gate_epochs.py      # after e82_pick.py --stage final (it reads s* the same way)

Reads results/explore/abcd.jsonl (each row's history: val_acc every 2 epochs). Read only. Added 2026-10-21.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e82_pick as P  # noqa: E402


def three(d: dict) -> tuple[float, float, float]:
    v = [x["val_acc"] for x in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / len(v[-10:])


def main() -> None:
    R = P.Rows()
    picks = {ds: P.pick(R, ds) for ds in ("star", "abcd", "multiwoz")}
    star = {ds: P.s_star(R, ds, picks[ds]) for ds in ("star", "abcd", "multiwoz")}
    for ds, bar in (("star", 1.0), ("abcd", 0.7)):
        k = [three(R.get(P.tag(ds, P.FULL[ds], star[ds]), sd, ds, P.FULL[ds], star[ds])) for sd in (1, 2)]
        c = [three(R.get(t, sd, ds, P.STAGE_A[ds][0], P.STAGE_A[ds][1], refined=False)) for t, sd in P.C4_REF[ds]]
        f = lambda i: 100 * ((k[0][i] + k[1][i]) / 2 - (c[0][i] + c[1][i]) / 2)  # noqa: E731
        print(f"go-bar {ds} (bar +{bar:.1f}, judged on best-dev): best-dev {f(0):+.2f}; final epoch {f(1):+.2f}; last-10 mean {f(2):+.2f}")
    m = [three(R.get(P.tag("multiwoz", 16000, star["multiwoz"]), sd, "multiwoz", 16000, star["multiwoz"])) for sd in (1, 2)]
    b = three(R.r[P.BGE_MWOZ][0])
    for i, name in enumerate(("best-dev", "final epoch", "last-10 mean")):
        print(f"R24 gate ({name}): refined seed 1 minus bge-large seed 1 {100 * (m[0][i] - b[i]):+.2f}; mean of refined seeds 1-2 {100 * ((m[0][i] + m[1][i]) / 2 - b[i]):+.2f}")


if __name__ == "__main__":
    main()
