#!/usr/bin/env python
"""Tests for b3_verdict.py on synthetic rows (a plain script, no pytest; CPU only, no GPU):

    OMP_NUM_THREADS=1 nice -n 19 $PY -I -B tu/tests/test_b3_verdict.py

Cases: PASS; FAIL by part 1, part 2 (linear) and part 3 (no teacher), each with its reason; the +0.5 and +0.3 edges;
a missing row; the refusal to overwrite. Each case has its own folder in tu/tests/runs/<UTC stamp>_b3verdict/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TU = Path(__file__).resolve().parent.parent
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3verdict")
SETS = ("clinc150", "banking77", "hwu64")
S = {"clinc150": 8.0, "banking77": 8.0, "hwu64": 16.0}
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def wj(p: Path, rows) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


BASE = {"clinc150": 0.9450, "banking77": 0.9150, "hwu64": 0.8930}
LRB = {"clinc150": 0.9400, "banking77": 0.9100, "hwu64": 0.8900}


def make(name, tm_add, lr_add, drop=None):
    """tm_add / lr_add: {arm: {set: points}} added to the base (TM per seed, LR once)."""
    d = RUNS / name
    d.mkdir(parents=True)
    wj(d / "r3.jsonl", [{"tag": "r3-seeds", "table": "a3B_d768", "dataset": k, "s": S[k], "seed": sd, "val_acc": BASE[k] + 0.001 * (sd - 5)}
                        for k in SETS for sd in (4, 5, 6)])
    wj(d / "a3_lr.jsonl", [{"table": "a3B_d768", "dataset": k, "lr_val": LRB[k]} for k in SETS])
    wj(d / "0b.jsonl", [{"tag": "0b-seeds", "table": "potion-base-32M", "dataset": k, "s": 8.0, "seed": sd, "val_acc": BASE[k] + 0.01}
                        for k in SETS for sd in (4, 5, 6)])
    (d / "base.json").write_text(json.dumps({"base": "a3B_d768", "s": S, "ref_tm": {"file": str(d / "r3.jsonl"), "tag": "r3-seeds"},
                                             "ref_lr": {"file": str(d / "a3_lr.jsonl")},
                                             "potion_tm": {"file": str(d / "0b.jsonl"), "tag": "0b-seeds", "s": {k: 8.0 for k in SETS}}}))
    val, lr = [], []
    for arm in ("3b-T", "3b-L", "3b-I", "3b-C", "3b-ceil"):
        for k in SETS:
            for sd in (4, 5, 6):
                if drop == (arm, k, sd):
                    continue
                val.append({"tag": "3b-tm", "arm": arm, "dataset": k, "s": S[k], "seed": sd,
                            "val_acc": BASE[k] + 0.001 * (sd - 5) + tm_add.get(arm, {}).get(k, 0.0) / 100})
            lr.append({"name": arm, "dataset": k, "lr_val": LRB[k] + lr_add.get(arm, {}).get(k, 0.0) / 100})
    wj(d / "results_b3_val.jsonl", val)
    wj(d / "results_b3_lr.jsonl", lr)
    return d


def run(d):
    p = subprocess.run([sys.executable, "-I", "-B", str(TU / "b3_verdict.py"), "--b3", str(d)], capture_output=True, text=True,
                       timeout=120, check=False)
    v = json.loads((d / "verdict_b3.json").read_text()) if (d / "verdict_b3.json").exists() else {}
    return p.returncode, p.stdout + p.stderr, v


def all3(x):
    return {k: x for k in SETS}


good_tm = {"3b-T": all3(1.0), "3b-L": all3(0.5), "3b-I": all3(0.4)}
good_lr = {"3b-T": all3(1.0), "3b-L": all3(0.6), "3b-I": all3(0.5)}
e, out, v = run(make("pass", good_tm, good_lr))
check("PASS: exit 0", e == 0, out)
check("PASS: verdict and 3 sets", v.get("verdict") == "PASS" and v.get("sets_meeting_bar") == 3, out)
check("PASS: the TM gain over base is +1.00 with paired seeds", abs(v["sets"]["clinc150"]["tm_gains"]["3b-T-base"]["mean"] - 1.0) < 1e-9)
check("PASS: the paired SE of identical shifts is 0", abs(v["sets"]["clinc150"]["tm_gains"]["3b-T-base"]["se"]) < 1e-9)
check("PASS: the text file has the verdict line", (RUNS / "pass" / "verdict_b3.txt").read_text().strip().endswith("with both probes)"))

e, out, v = run(make("fail1", {"3b-T": all3(0.4), "3b-L": all3(0.0), "3b-I": all3(0.0)}, good_lr))
check("FAIL part 1: reason named", v.get("verdict") == "FAIL" and "part 1" in (v.get("reason") or ""), str(v.get("reason")))
e, out, v = run(make("fail2", {"3b-T": all3(1.0), "3b-L": all3(0.9), "3b-I": all3(0.0)}, good_lr))
check("FAIL part 2: a linear re-encoding", v.get("verdict") == "FAIL" and "linear" in (v.get("reason") or ""), str(v.get("reason")))
e, out, v = run(make("fail3", {"3b-T": all3(1.0), "3b-L": all3(0.0), "3b-I": all3(0.9)}, good_lr))
check("FAIL part 3: the second set of concept bits", v.get("verdict") == "FAIL" and "concept bits" in (v.get("reason") or ""), str(v.get("reason")))
lr_weak = {"3b-T": all3(1.0), "3b-L": all3(0.6), "3b-I": all3(0.5)}
lr_weak["3b-T"] = {"clinc150": 1.0, "banking77": 0.3, "hwu64": 0.3}
e, out, v = run(make("lrprobe", good_tm, lr_weak))
check("both probes are needed: an LR gain of +0.3 over base fails part 1 on 2 sets", v.get("verdict") == "FAIL"
      and v.get("sets_meeting_bar") == 1, str(v.get("sets_meeting_bar")))
e, out, v = run(make("edge", {"3b-T": all3(0.5), "3b-L": all3(0.2), "3b-I": all3(0.2)},
                     {"3b-T": all3(0.5), "3b-L": all3(0.2), "3b-I": all3(0.2)}))
check("edges: +0.5 over base and +0.3 over both twins pass", v.get("verdict") == "PASS", json.dumps(v.get("sets", {}).get("clinc150", {}).get("parts")))
e, out, v = run(make("missing", good_tm, good_lr, drop=("3b-I", "hwu64", 6)))
check("a missing row stops the verdict (exit 1, no file)", e != 0 and "rows match" in out and not v, out[-300:])
e, out, v = run(RUNS / "pass")
check("an existing verdict is refused (exit 2)", e == 2 and "refusing to overwrite" in out, out)

print(f"b3_verdict tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
