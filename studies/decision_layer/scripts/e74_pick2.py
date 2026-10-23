#!/usr/bin/env python
"""e74c / e74d picks (plan logs/plan_e74cd.md): read the 4k screen rows, apply the rule, write the follow-up jobs.

    e74_pick2.py --stage c     # the C4 agent: s 12 (e74a) against 16 and 24; follow-ups in logs/jobs_e74c2.txt
    e74_pick2.py --stage d     # the bge-large agent: s 8, 12, 16, 24; follow-up in logs/jobs_e74d2.txt
    e74_pick2.py --stage final # after the follow-ups: s* for R22 and the tuned dev gap

Rules (fixed in the plan):
- stage c: the highest 4k dev score among [12, 16, 24]; it must be at least 0.3 points above s 12's, else s 12; an
  exact tie goes to the first in the list. If the pick is not 12, its STAR 16k and ABCD 4k jobs are written. s* is the
  pick, unless its STAR 16k dev is below 0.6519 or its ABCD 4k dev is below 0.8267; then s* = 12.
- stage d: the same over [8, 12, 16, 24] against s 8's in-sweep row; if the pick is not 8, its STAR 16k job is written.
Read only, except that it writes the job files; it refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C4 = ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/table_run.py "
      "--table c4words-S1000-v1 scripts/abcd_explore.py --encoder none --arms tm --seed 1 --parts mem,words,xcust,xrecent")
BGE = ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/abcd_explore.py "
       "--dataset star --encoder bge-large --parts mem,intent,step,words --arms tm --seed 1")
TAGS = {"c": {12: "e74_star_Bc4_c4000_s12", 16: "e74c_star_Bc4_c4000_s16", 24: "e74c_star_Bc4_c4000_s24"},
        "d": {8: "e74d_star_bge_c4000_s8", 12: "e74d_star_bge_c4000_s12", 16: "e74d_star_bge_c4000_s16", 24: "e74d_star_bge_c4000_s24"}}
REF = {"c": 12, "d": 8}


def rows() -> dict:
    out = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag", "").startswith(("e74", "e46", "e72")):
            out.setdefault(d["tag"], []).append(d)
    return out


def one(r: dict, tag: str, frozen: str | None = None) -> dict:
    got = r.get(tag, [])
    if len(got) != 1:
        raise SystemExit(f"{tag}: {len(got)} rows (need exactly 1)")
    if frozen is not None and got[0]["time"] <= frozen:
        raise SystemExit(f"{tag}: written before the freeze")
    return got[0]


def pick(stage: str) -> int:
    r = rows()
    frozen = json.loads((ROOT / "logs" / "plan_e74cd.frozen.json").read_text())["frozen_utc"]
    S = list(TAGS[stage])
    acc = {}
    for s in S:
        d = one(r, TAGS[stage][s], None if (stage == "c" and s == 12) else frozen)
        assert d["cfg"]["s"] == float(s) and d["cfg"]["n_clauses"] == 4000 and d["cfg"]["T"] == 4000, d["tag"]
        acc[s] = d["acc"]
        print(f"{stage} s {s:>2}: dev {d['acc']:.4f} (best epoch {d['best_epoch']}, {d['train_s']} s)")
    ref = REF[stage]
    best = max(S, key=lambda s: (acc[s], -S.index(s)))
    p = best if acc[best] - acc[ref] >= 0.003 - 1e-12 else ref
    print(f"best s {best} ({acc[best]:.4f}); reference s {ref} {acc[ref]:.4f}; gap {100 * (acc[best] - acc[ref]):+.2f}; pick s {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["c", "d", "final"])
    args = ap.parse_args()
    if args.stage in ("c", "d"):
        p = pick(args.stage)
        if p == REF[args.stage]:
            print(f"the pick is the reference s {p}: no follow-up")
            return
        out = ROOT / "logs" / f"jobs_e74{args.stage}2.txt"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        if args.stage == "c":
            body = (f"{C4} --dataset star --clauses 16000 --s {p} --save --tag e74c_star_Bc4_c16000_s{p} >> logs/e74c_star_Bc4_c16000_s{p}.log 2>&1\n"
                    f"{C4} --dataset abcd --clauses 4000 --s {p} --save --tag e74c_abcd_Bc4_c4000_s{p} >> logs/e74c_abcd_Bc4_c4000_s{p}.log 2>&1\n")
        else:
            body = f"{BGE} --clauses 16000 --s {p} --save --tag e74d_star_bge_c16000_s{p} >> logs/e74d_star_bge_c16000_s{p}.log 2>&1\n"
        out.write_text(f"# e74{args.stage}2 (plan logs/plan_e74cd.md): the follow-up of the pick s {p}. Written by scripts/e74_pick2.py.\n" + body)
        print(f"wrote {out.relative_to(ROOT)}")
        return
    # final: s* for R22 and the tuned dev gap at 16k
    r = rows()
    pc, pd = pick("c"), pick("d")
    s_star = 12
    if pc != 12:
        st = one(r, f"e74c_star_Bc4_c16000_s{pc}")["acc"]
        ab = one(r, f"e74c_abcd_Bc4_c4000_s{pc}")["acc"]
        s_star = pc if (st >= 0.6519 - 1e-12 and ab >= 0.8267 - 1e-12) else 12
        print(f"c pick s {pc}: STAR 16k {st:.4f} (s 12: 0.6519), ABCD 4k {ab:.4f} (bar 0.8267) -> s* {s_star}")
    c16 = one(r, "e74_star_Bc4_c16000_s12")["acc"] if s_star == 12 else one(r, f"e74c_star_Bc4_c16000_s{s_star}")["acc"]
    b16 = one(r, "e46_star_mem_c16000")["acc"] if pd == 8 else one(r, f"e74d_star_bge_c16000_s{pd}")["acc"]
    print(f"s* (C4) = {s_star}: STAR 16k dev {c16:.4f}; bge-large at its pick s {pd}: STAR 16k dev {b16:.4f}; "
          f"tuned gap C4 - bge-large {100 * (c16 - b16):+.2f} points")


if __name__ == "__main__":
    main()
