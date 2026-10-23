#!/usr/bin/env python
"""e82 (plan logs/plan_e82_refined_table_agent.md): e82-R3's picks, Stage C's jobs, and the go-bars and the R24 gate.
Read only, except that it writes job files and refuses to overwrite them. Written before any e82 number.

    e82_pick.py --stage screen    # after the 4k screens: the pick per dataset; writes logs/jobs_e82r3b.txt (16k jobs)
    e82_pick.py --stage final     # after e82r3b: s* per dataset; writes logs/jobs_e82c.txt (seed 2, and ABCD seeds 3-5)
    e82_pick.py --stage gobars    # after Stage C: the dev go-bars, the R24 gate, and which one test

Rules (fixed in the plan, copied from e74d / e75 / e77):
- the grid is [8, 12, 16, 24] at 4,000 clauses, seed 1; the pick is the highest 4k dev, if it is at least 0.3 points
  above s 8's in-sweep row, otherwise s 8; an exact tie goes to the first in the grid;
- STAR: 16k at s 8 and at the pick; s* is the pick, unless its 16k dev is below s 8's 16k dev, then s 8;
- ABCD: no 16k step, s* is the pick;
- MultiWOZ: 16k at s 8 and at the pick; s* as for STAR.
A row whose settings equal Stage A's (STAR 16k s 16, ABCD 4k s 16, MultiWOZ 16k s 8) is Stage A's row, with Stage A's
tag; no duplicate job is run for it. Every e82 row must have the a3B table's file hash in static_info and must have been
written after the plan's freeze.
Go-bars: STAR (refined at s*, 16k) minus the C4 agent (e74c and e78 rows, s 16), mean of seeds 1-2, at least +1.0;
ABCD (refined at s*, 4k) minus the C4 agent (s 16), at least +0.7. The R24 gate: the refined agent at its MultiWOZ s*
(16k, seed 1) minus e77_mwoz_bge_c16000_s8 (0.3359) at least -1.0. The precedence: R24-refined if the gate passes;
otherwise R25 on the datasets whose go-bars pass; otherwise no test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = [8, 12, 16, 24]
TABLE_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"
B = ["mem", "words", "xcust", "xrecent"]
STAGE_A = {"star": (16000, 16), "abcd": (4000, 16), "multiwoz": (16000, 8)}
FULL = {"star": 16000, "abcd": 4000, "multiwoz": 16000}  # the clauses at which s* is used
C4_REF = {"star": [("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)],
          "abcd": [("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)]}
BGE_MWOZ = ("e77_mwoz_bge_c16000_s8", 1)
RUN = ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 "
       ".venv-dl/bin/python scripts/newtable_run.py --table-dir data/tables/c4words_a3B_v1 --name c4words-a3B-v1 scripts/abcd_explore.py "
       "--encoder none --arms tm --parts mem,words,xcust,xrecent --save")


def tag(ds: str, clauses: int, s: int) -> str:
    """Stage A's tag where the settings equal Stage A's, else the e82-R3 tag."""
    if STAGE_A[ds] == (clauses, s):
        return f"e82_{ds}_Bt_c{clauses}_s{s}"
    return f"e82r3_{ds}_Bt_c{clauses}_s{s}"


def job(ds: str, clauses: int, s: int, seed: int) -> str:
    t = tag(ds, clauses, s)
    return f"{RUN} --dataset {ds} --clauses {clauses} --s {s} --seed {seed} --tag {t} >> logs/{t}_seed{seed}.log 2>&1"


class Rows:
    def __init__(self) -> None:
        self.frozen = json.loads((ROOT / "logs" / "plan_e82_refined_table_agent.frozen.json").read_text())["frozen_utc"]
        self.r: dict = {}
        for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)

    def has(self, t: str, seed: int) -> bool:
        return len(self.r.get((t, seed), [])) == 1

    def get(self, t: str, seed: int, ds: str, clauses: int, s: int, refined: bool = True) -> dict:
        got = self.r.get((t, seed), [])
        if len(got) != 1:
            raise SystemExit(f"{t} seed {seed}: {len(got)} rows (need exactly 1)")
        d = got[0]
        ok = (d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == B and d["lex_k"] == 1000 and d["cfg"]["n_clauses"] == clauses
              and d["cfg"]["T"] == clauses and d["cfg"]["s"] == float(s) and d["cfg"]["seed"] == seed)
        if not ok:
            raise SystemExit(f"{t} seed {seed}: settings differ from {ds} {clauses} s {s}")
        if refined:
            if not all(d["static_info"][k]["table_file_sha256"] == TABLE_SHA for k in ("xcust", "xrecent")):
                raise SystemExit(f"{t} seed {seed}: not built on the a3B table {TABLE_SHA[:16]}")
            if d["time"] <= self.frozen:
                raise SystemExit(f"{t} seed {seed}: written before the plan's freeze")
        return d


def pick(R: Rows, ds: str) -> int:
    acc = {s: R.get(tag(ds, 4000, s), 1, ds, 4000, s)["acc"] for s in GRID}
    best = max(GRID, key=lambda s: (acc[s], -GRID.index(s)))
    p = best if acc[best] - acc[8] >= 0.003 - 1e-12 else 8
    print(f"{ds} 4k: " + ", ".join(f"s {s} {acc[s]:.4f}" for s in GRID) + f"; best s {best} ({100 * (acc[best] - acc[8]):+.2f} over s 8); pick s {p}")
    return p


def s_star(R: Rows, ds: str, p: int) -> int:
    if ds == "abcd":
        return p
    a8 = R.get(tag(ds, 16000, 8), 1, ds, 16000, 8)["acc"]
    ap = a8 if p == 8 else R.get(tag(ds, 16000, p), 1, ds, 16000, p)["acc"]
    s = p if ap >= a8 - 1e-12 else 8
    print(f"{ds} 16k: s 8 {a8:.4f}; pick s {p} {ap:.4f} -> s* {s}")
    return s


def write(path: Path, header: str, lines: list[str]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; not overwriting")
    path.write_text("\n".join([header] + lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(lines)} jobs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen", "final", "gobars"])
    args = ap.parse_args()
    R = Rows()
    picks = {ds: pick(R, ds) for ds in ("star", "abcd", "multiwoz")}
    if args.stage == "screen":
        lines = []
        for ds in ("star", "multiwoz"):
            for s in sorted({8, picks[ds]}):
                if not R.has(tag(ds, 16000, s), 1):
                    lines.append(job(ds, 16000, s, 1))
        write(ROOT / "logs" / "jobs_e82r3b.txt", "# e82r3b (plan logs/plan_e82_refined_table_agent.md): the 16k runs of e82-R3 (STAR and MultiWOZ, s 8 and the pick, "
              "seed 1; rows equal to Stage A's are reused). Written by scripts/e82_pick.py --stage screen. One at a time through seq_queue.sh.", lines)
        return
    star = {ds: s_star(R, ds, picks[ds]) for ds in ("star", "abcd", "multiwoz")}
    print("s*:", star)
    if args.stage == "final":
        lines = []
        for ds in ("star", "abcd", "multiwoz"):
            if not R.has(tag(ds, FULL[ds], star[ds]), 2):
                lines.append(job(ds, FULL[ds], star[ds], 2))
        for seed in (3, 4, 5):
            if not R.has(tag("abcd", 4000, star["abcd"]), seed):
                lines.append(job("abcd", 4000, star["abcd"], seed))
        write(ROOT / "logs" / "jobs_e82c.txt", "# e82c (plan logs/plan_e82_refined_table_agent.md): Stage C, seed 2 at each dataset's s*, and ABCD seeds 3-5 at its s* "
              "(R5's five models). Written by scripts/e82_pick.py --stage final. One at a time through seq_queue.sh.", lines)
        return
    out = {}
    for ds, bar in (("star", 1.0), ("abcd", 0.7)):
        k = [R.get(tag(ds, FULL[ds], star[ds]), sd, ds, FULL[ds], star[ds])["acc"] for sd in (1, 2)]
        c = [R.get(t, sd, ds, STAGE_A[ds][0], STAGE_A[ds][1], refined=False)["acc"] for t, sd in C4_REF[ds]]
        gap = 100 * (sum(k) / 2 - sum(c) / 2)
        out[ds] = gap >= bar - 1e-9
        print(f"go-bar {ds}: refined at s* {star[ds]} {k[0]:.4f} / {k[1]:.4f}; C4 at s 16 {c[0]:.4f} / {c[1]:.4f}; difference {gap:+.2f} "
              f"(needs >= +{bar:.1f}): {'GO' if out[ds] else 'stop (dev-only line; test unread)'}")
    m = [R.get(tag("multiwoz", 16000, star["multiwoz"]), sd, "multiwoz", 16000, star["multiwoz"])["acc"] for sd in (1, 2)]
    b = R.r[BGE_MWOZ][0]["acc"]
    gate = 100 * (m[0] - b)
    print(f"R24 gate: refined at s* {star['multiwoz']} seed 1 {m[0]:.4f} minus bge-large {b:.4f}: {gate:+.2f} (needs >= -1.0); "
          f"mean of seeds 1-2 {100 * (sum(m) / 2 - b):+.2f} (reported)")
    if gate >= -1.0 - 1e-9:
        print("PRECEDENCE: the R24 gate passes -> R24-refined is the one registered test; R25 does not run")
    elif out["star"] or out["abcd"]:
        print(f"PRECEDENCE: R25 on {[ds for ds in ('star', 'abcd') if out[ds]]}")
    else:
        print("PRECEDENCE: no registered test")


if __name__ == "__main__":
    main()
