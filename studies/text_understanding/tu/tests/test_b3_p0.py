#!/usr/bin/env python
"""End-to-end smoke test of b3_p0.py on tiny real-format data (a plain script, no pytest; CPU only):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_b3_p0.py --codes tu/b3/codes.npz

C4 only, as P0 itself: the concepts from the first 20,000 training sentences, 3,000 training and 500 held-out rows,
one s, 50 clauses, 1 epoch; then the pool check. A choice file with pick SH is written in the run folder (the real
pick comes from the code choice). Checks the verdict's keys and rule, the fidelity fields, the restart (a second run
skips the done steps and refuses the existing verdict), and the pool check's output. Each run has its own folder in
tu/tests/runs/<UTC stamp>_b3p0/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3p0")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


ap = argparse.ArgumentParser()
ap.add_argument("--codes", type=Path, required=True)
a = ap.parse_args()
RUNS.mkdir(parents=True, exist_ok=True)
choice = RUNS / "choice.json"
choice.write_text(json.dumps({"pick": "SH"}))
out = RUNS / "p0"
base = [sys.executable, "-I", "-B", str(TU / "b3_p0.py"), "--choice", str(choice), "--codes", str(a.codes), "--out-dir", str(out),
        "--n-train", "3000", "--n-held", "500", "--fit-limit", "20000", "--s", "8", "--clauses", "50", "--epochs", "1"]
t0 = time.time()
p = subprocess.run(base, capture_output=True, text=True, timeout=3600, check=False)
check(f"P0 smoke: exit 0 ({time.time() - t0:.0f} s)", p.returncode == 0, (p.stdout + p.stderr)[-1200:])
v = json.loads((out / "p0_verdict.json").read_text()) if (out / "p0_verdict.json").exists() else {}
check("verdict: the keys of the rule", all(k in v for k in ("code", "best_s", "tm_best_balanced_acc", "tm_best_fidelity", "linear",
                                                            "counting", "difference_tm_minus_linear", "go", "files_sha16")), str(v)[:400])
check("verdict: GO follows the rule", v.get("go") == (v.get("tm_best_balanced_acc", 0) >= v.get("linear", {}).get("balanced_acc", 1e9) - 2.0 - 1e-9),
      str({k: v.get(k) for k in ("go", "tm_best_balanced_acc", "difference_tm_minus_linear")}))
check("verdict: fidelity (bitwise agreement) for every student", "bit_acc" in v.get("linear", {}) and "bit_acc" in v.get("counting", {})
      and v.get("tm_best_fidelity") is not None)
check("verdict: the rows asked for", v.get("n_train") == 3000 and v.get("n_held") == 500 and v.get("code") == "SH")
tr = json.loads((out / "tm_s8.json").read_text()) if (out / "tm_s8.json").exists() else {}
check("the TM run used the pilot's settings", tr.get("cfg", {}).get("n_clauses") == 50 and tr.get("cfg", {}).get("T") == 50
      and tr.get("R") == 4 and tr.get("device") == "cpu" and tr.get("n_train") == 3000, str(tr.get("cfg")))
z64 = np.load(out / "train64.npz") if (out / "train64.npz").exists() else None
check("the TM and the twins read 64 fixed code bits", z64 is not None and int(z64["n_outputs"]) == 64
      and v.get("bits") == sorted(np.random.default_rng(20261031).choice(768, 64, replace=False).tolist())
      and "train64.npz" in json.dumps(tr.get("inputs_sha256", {})), str(v.get("bits"))[:200])
conc = json.loads(str(np.load(out / "c4_concepts_cpu.npz")["meta"])) if (out / "c4_concepts_cpu.npz").exists() else {}
check("the concepts used the fit limit and K 512", conc.get("lines") == 20000 and conc.get("K") == 512, str(conc)[:300])
p2 = subprocess.run(base, capture_output=True, text=True, timeout=600, check=False)
check("a second run refuses the existing verdict (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr[-300:])
log = (out / "p0.log").read_text() if (out / "p0.log").exists() else ""
check("the log has every step with its exit 0", log.count(" exit 0") == 5 and "exit 1" not in log, log[-600:])
t1 = time.time()
q = subprocess.run(base + ["--pool-check"], capture_output=True, text=True, timeout=3600, check=False)
check(f"pool check: exit 0 ({time.time() - t1:.0f} s)", q.returncode == 0, (q.stdout + q.stderr)[-1200:])
pc = json.loads((out / "p0_pool_check.json").read_text()) if (out / "p0_pool_check.json").exists() else {}
check("pool check: 32 fixed bits, both arms with fidelity", len(pc.get("bits", [])) == 32 and pc.get("bits") == sorted(pc.get("bits", []))
      and "bit_acc" in pc.get("shared", {}).get("held", {}) and "mean_bit_acc" in pc.get("per_bit", {}), str(pc)[:400])
check("pool check: matched clauses (1,024 shared against 32 x 32)", pc.get("shared", {}).get("clauses") == 1024
      and pc.get("per_bit", {}).get("clauses_each") == 32)
q2 = subprocess.run(base + ["--pool-check"], capture_output=True, text=True, timeout=600, check=False)
check("pool check: refuses its existing output (exit 2)", q2.returncode == 2, q2.stderr[-300:])

print(f"b3_p0 smoke tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
