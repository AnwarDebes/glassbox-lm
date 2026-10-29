#!/usr/bin/env python
"""M2 code check, the comparison (tu/PLAN_M2.md "The report step"): the m2_tm.py and m2_lr.py rows of the code check
on 3c's CLINC150 rebuilt-original cache (test := a copy of val; no test text or label is used) must reproduce 3c's
rows exactly. No MTOP file is read.

    python -I -B m2_codecheck.py --c3 tu/c3 --ck tu/m2/check

- TM (<ck>/results_m2_tm.jsonl, tag m2-check) against 3c's 3c-ref row (tu/c3/results_c3_val.jsonl, clinc150, seed 4):
  val_acc equal; test_acc equal to val_acc (test := val); best_epoch equal (40); the full val curve equal (20 points).
- LR (<ck>/results_m2_lr.jsonl, name check) against 3c's ref_rebuilt row (tu/c3/results_c3_lr.jsonl, clinc150): lr_val
  equal; lr_test equal to lr_val; C equal (1.0).
- Exactly one row on each side. Exit 0 when every comparison holds, else 1 with every failed comparison printed.

This replaces the inline comparison of m1_chain.sh (and of draft 1's m2_chain.sh and m2_dry_gpu.sh), which compared
val_acc, test_acc and lr_val only; best_epoch, the curve and C were added after the review of draft 1 (design MINOR 10).
M1's own code-check rows (tu/m1/check) pass it (tests/test_m2.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rows(path: Path, **want) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if all(r.get(k) == v for k, v in want.items()):
                out.append(r)
    return out


def compare(c3: Path, ck: Path) -> list[str]:
    a = rows(ck / "results_m2_tm.jsonl", tag="m2-check")
    b = rows(c3 / "results_c3_val.jsonl", tag="3c-ref", dataset="clinc150", seed=4)
    la = rows(ck / "results_m2_lr.jsonl", name="check")
    lb = rows(c3 / "results_c3_lr.jsonl", name="ref_rebuilt", dataset="clinc150")
    n = {"m2 TM (m2-check)": len(a), "3c-ref seed 4": len(b), "m2 LR (check)": len(la), "3c ref_rebuilt": len(lb)}
    errs = [f"{k}: {v} rows, expected 1" for k, v in n.items() if v != 1]
    if errs:
        return errs
    a, b, la, lb = a[0], b[0], la[0], lb[0]
    if a["val_acc"] != b["val_acc"]:
        errs.append(f"TM val_acc {a['val_acc']} != 3c-ref {b['val_acc']}")
    if a["test_acc"] != a["val_acc"]:
        errs.append(f"TM test_acc {a['test_acc']} != its val_acc {a['val_acc']} (test is a copy of val)")
    if a.get("best_epoch") != b.get("best_epoch"):
        errs.append(f"TM best_epoch {a.get('best_epoch')} != 3c-ref {b.get('best_epoch')}")
    ca, cb = [list(p) for p in a.get("curve", [])], [list(p) for p in b.get("curve", [])]
    if not cb or ca != cb:
        diff = [(x, y) for x, y in zip(ca, cb) if x != y][:3]
        errs.append(f"TM curve differs ({len(ca)} vs {len(cb)} points; first differences {diff})")
    if la["lr_val"] != lb["lr_val"]:
        errs.append(f"LR lr_val {la['lr_val']} != ref_rebuilt {lb['lr_val']}")
    if la["lr_test"] != la["lr_val"]:
        errs.append(f"LR lr_test {la['lr_test']} != its lr_val {la['lr_val']} (test is a copy of val)")
    if la.get("C") != lb.get("C"):
        errs.append(f"LR C {la.get('C')} != ref_rebuilt {lb.get('C')}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--ck", type=Path, required=True)
    a = ap.parse_args()
    errs = compare(a.c3, a.ck)
    if errs:
        print("code check FAILED:\n" + "\n".join(errs))
        return 1
    tm = rows(a.ck / "results_m2_tm.jsonl", tag="m2-check")[0]
    lr = rows(a.ck / "results_m2_lr.jsonl", name="check")[0]
    print(f"code check ok: TM val {tm['val_acc']} test {tm['test_acc']} best epoch {tm['best_epoch']} curve "
          f"{len(tm['curve'])} points equal to 3c-ref; LR val {lr['lr_val']} test {lr['lr_test']} C {lr['C']} equal to ref_rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
