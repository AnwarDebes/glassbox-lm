#!/usr/bin/env python
"""e74 pick (plan logs/plan_e74_s_sweep.md): read the six STAR 4k screen rows, apply the rule, and write the e74b jobs.

    e74_pick.py          # prints the six rows and the pick; writes logs/jobs_e74b.txt unless the pick is s 8

Rule (fixed in the plan):
- the s with the highest 4k dev score;
- if that score is less than 0.3 points above s 8's 4k score, the pick is s 8;
- an exact tie goes to the s that comes first in [2, 3, 4, 6, 8, 12].
Read only, except that it writes the job file; it refuses to overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = [2, 3, 4, 6, 8, 12]
JOB = ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/table_run.py "
       "--table c4words-S1000-v1 scripts/abcd_explore.py --encoder none --arms tm --seed 1 --parts mem,words,xcust,xrecent")


def tag(s) -> str:
    return f"e74_star_Bc4_c4000_s{s}"


def main() -> None:
    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag", "").startswith("e74_star_Bc4_c4000_s"):
            rows.setdefault(d["tag"], []).append(d)
    frozen = json.loads((ROOT / "logs" / "plan_e74_s_sweep.frozen.json").read_text())["frozen_utc"]
    acc = {}
    for s in S:
        got = rows.get(tag(s), [])
        if len(got) != 1:
            raise SystemExit(f"{tag(s)}: {len(got)} rows (need exactly 1)")
        d = got[0]
        assert d["cfg"]["s"] == float(s) and d["cfg"]["n_clauses"] == 4000 and d["cfg"]["T"] == 4000 and d["encoder"] == "none", d["tag"]
        assert d["time"] > frozen, f"{d['tag']} before the freeze"
        acc[s] = d["acc"]
        print(f"s {s:>2}: dev {d['acc']:.4f} (best epoch {d['best_epoch']}, {d['train_s']} s)")
    best = max(S, key=lambda s: (acc[s], -S.index(s)))
    pick = best if acc[best] - acc[8] >= 0.003 - 1e-12 else 8
    print(f"best s {best} ({acc[best]:.4f}); s 8 {acc[8]:.4f}; gap {100 * (acc[best] - acc[8]):+.2f}; pick s {pick}")
    if pick == 8:
        print("the pick is s 8: e74b does not run (s 8 stays)")
        return
    out = ROOT / "logs" / "jobs_e74b.txt"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(
        f"# e74b (plan logs/plan_e74_s_sweep.md): the pick s {pick} at R21's 16,000 clauses on STAR, then on ABCD at 4,000 clauses. Written by scripts/e74_pick.py.\n"
        f"{JOB} --dataset star --clauses 16000 --s {pick} --save --tag e74_star_Bc4_c16000_s{pick} >> logs/e74_star_Bc4_c16000_s{pick}.log 2>&1\n"
        f"{JOB} --dataset abcd --clauses 4000 --s {pick} --save --tag e74_abcd_Bc4_c4000_s{pick} >> logs/e74_abcd_Bc4_c4000_s{pick}.log 2>&1\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
