#!/usr/bin/env python
"""Tests for c3_handcheck_sample.py on synthetic files (a plain script, no pytest):

    OMP_NUM_THREADS=1 nice -n 19 $PY -I tu/tests/test_c3_handcheck_sample.py

Each case has its own folder in tu/tests/runs/<UTC stamp>_handcheck/. Nothing real is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TU = Path(__file__).resolve().parent.parent
SCRIPT = TU / "c3_handcheck_sample.py"
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_handcheck")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def wj(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make(name: str, n_src: dict, extra_kept: bool = False, orphan_c: bool = False) -> Path:
    """n_src: sources per set; each source has 5 paraphrases: p0, p1 kept (F), p2 dropped by F3, p3 p4 dropped before
    F3; source 0 of each set repeats p0 twice (a duplicate kept once, dropped once before F3)."""
    d = RUNS / name
    gen, kc, kf = [], [], []
    for s, n in n_src.items():
        for i in range(n):
            src = f"{s} source {i}"
            paras = [f"{s} para {i}.{j}" for j in range(5)]
            if i == 0:
                paras[4] = paras[0]
            gen.append({"set": s, "index": i, "label": f"{s}_lab{i % 3}", "text": src, "paraphrases": paras})
            for j in (0, 1, 2):
                c = {"set": s, "index": i, "y": i % 3, "label": f"{s}_lab{i % 3}", "source": src, "text": paras[j],
                     "cos": 0.8 + j / 100, "near_eval": 0.5, "f3_pred": f"{s}_lab{i % 3}" if j < 2 else f"{s}_other"}
                kc.append(c)
                if j < 2:
                    kf.append(c)
    orphan = {"set": "hwu64", "index": 999, "y": 0, "label": "x", "source": "nowhere", "text": "not generated",
              "cos": 0.9, "near_eval": 0.1, "f3_pred": "x"}
    if extra_kept:  # in kept_C and kept_F, but not generated
        kf.append(orphan)
        kc.append(orphan)
    if orphan_c:  # in kept_F only
        kf.append(orphan)
    wj(d / "gen.jsonl", gen)
    wj(d / "kept_C.jsonl", kc)
    wj(d / "kept_F.jsonl", kf)
    return d


def run(d: Path, out_name: str = "sample.tsv") -> tuple[int, str]:
    cmd = [sys.executable, "-I", str(SCRIPT), "--gen", str(d / "gen.jsonl"), "--kept-c", str(d / "kept_C.jsonl"),
           "--kept-f", str(d / "kept_F.jsonl"), "--out", str(d / out_name)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    return p.returncode, p.stdout + p.stderr


def rows(p: Path) -> list[list[str]]:
    return [line.split("\t") for line in p.read_text().splitlines()[1:]]


# 1. Big pools: 30 kept and 30 dropped per set, labelled right.
d = make("big", {"clinc150": 40, "banking77": 40, "hwu64": 40})
e, out = run(d)
check("big: exit 0", e == 0, out)
r = rows(d / "sample.tsv")
for s in ("clinc150", "banking77", "hwu64"):
    k = [x for x in r if x[0] == s and x[4] == "kept"]
    dr = [x for x in r if x[0] == s and x[4] != "kept"]
    check(f"big: {s} has 30 kept and 30 dropped", len(k) == 30 and len(dr) == 30, f"{len(k)} kept, {len(dr)} dropped")
    check(f"big: {s} kept rows are p0 or p1", all(x[3].split(".")[-1] in ("0", "1") for x in k), str(k[:3]))
    check(f"big: {s} F3 drops are p2 with the predicted label", all(x[3].endswith(".2") and x[4] == f"F3:{s}_other"
          for x in dr if x[4].startswith("F3:")) and any(x[4].startswith("F3:") for x in dr), str(dr[:3]))
    check(f"big: {s} pre-F3 drops are p3, p4 or the duplicate p0", all(x[3].split(".")[-1] in ("3", "4", "0")
          for x in dr if x[4] == "F1/F1b/F2"), str(dr[:3]))
summary = json.loads(out.strip().splitlines()[-1])["per_set"]
check("big: the pools count 2 kept and 3 dropped per source (the duplicate counted once each way)",
      summary["clinc150"]["kept_pool"] == 80 and summary["clinc150"]["dropped_pool"] == 120
      and summary["clinc150"]["dropped_f3_pool"] == 40, str(summary))
check("big: the judgement and note columns are empty", all(x[6] == "" and x[7] == "" for x in r), str(r[:2]))
check("big: kept rows carry the cosine", all(x[5] != "" for x in r if x[4] == "kept"), str(r[:2]))

# 2. The same input gives the same sample (fixed seed); a second run never overwrites.
e2, out2 = run(d, "sample2.tsv")
check("same seed: identical sample", e2 == 0 and (d / "sample2.tsv").read_text() == (d / "sample.tsv").read_text(), out2)
e3, out3 = run(d)
check("an existing output is refused (exit 2)", e3 == 2 and "refusing to overwrite" in out3, out3)

# 3. Small pools: the whole pool is taken.
d = make("small", {"clinc150": 3, "banking77": 1, "hwu64": 2})
e, out = run(d)
r = rows(d / "sample.tsv")
check("small: exit 0", e == 0, out)
check("small: CLINC150 takes all 6 kept and all 9 dropped", sum(1 for x in r if x[0] == "clinc150" and x[4] == "kept") == 6
      and sum(1 for x in r if x[0] == "clinc150" and x[4] != "kept") == 9, str(r))

# 4. A kept row that is not in gen.jsonl is an error (exit 1, no output).
d = make("orphan", {"clinc150": 3, "banking77": 3, "hwu64": 3}, extra_kept=True)
e, out = run(d)
check("orphan: exit 1", e == 1 and "not found in gen.jsonl" in out and not (d / "sample.tsv").exists(), out)

# 5. A kept_F row that is not in kept_C is an error too.
d = make("orphan_c", {"clinc150": 3, "banking77": 3, "hwu64": 3}, orphan_c=True)
e, out = run(d)
check("orphan in kept_F only: exit 1", e == 1 and "not in kept_C.jsonl" in out and not (d / "sample.tsv").exists(), out)

print(f"handcheck sample tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
