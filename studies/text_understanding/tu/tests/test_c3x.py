#!/usr/bin/env python
"""Tests for the 3c follow-up code (a plain script, no pytest; CPU only, no GPU):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_c3x.py [--feats]

- c3x_twin.eda_copy: deterministic per seed, a deletion rate near 0.1, never empty, one adjacent swap.
- c3x_rk.keys_of: the reader key ignores order, repeats and words with no table row; the order-free key sorts the
  normalised words.
- c3x_report.py on synthetic rows: the gains, the paired SE, the one-k-means check, the refusal to overwrite.
- with --feats: c3x_feats.py on the CPU for CLINC150 with a 5-row kept file (about a minute): both arms share one
  k-means run id, the arm's original rows equal orig2's bit for bit, and the 5 rows are appended with their labels.
Each run has its own folder in tu/tests/runs/<UTC stamp>_c3x/.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
sys.path.insert(0, str(TU.parent / "w1"))
import c3x_twin as TW

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_c3x")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


RUNS.mkdir(parents=True, exist_ok=True)
words = [f"w{i}" for i in range(20)]
r1, r2 = np.random.default_rng(TW.SEED), np.random.default_rng(TW.SEED)
a = [TW.eda_copy(words, r1) for _ in range(200)]
b = [TW.eda_copy(words, r2) for _ in range(200)]
check("eda_copy is deterministic for a seed", a == b)
dele = 1 - np.mean([len(x) for x in a]) / len(words)
check(f"deletion rate near 0.1 ({dele:.3f})", 0.07 < dele < 0.13, str(dele))
check("never empty: a one-word text keeps its word", all(TW.eda_copy(["only"], np.random.default_rng(s)) == ["only"] for s in range(50)))


def one_swap(src, out):
    kept = [w for w in src if w in out]
    if sorted(kept) != sorted(out) or len(out) < 2:
        return False
    diff = [i for i, (x, y) in enumerate(zip(kept, out)) if x != y]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and kept[diff[0]] == out[diff[1]] and kept[diff[1]] == out[diff[0]]


check("each copy of distinct words is the kept words with exactly one adjacent swap", all(one_swap(words, x) for x in a))

import c3x_rk as RK
from c3_common import B
from wordtab_feats import glove_tokens

live = {"please", "say", "that", "again", "what", "is", "my", "name"}
r1, o1 = RK.keys_of("please say that again", live, B, glove_tokens)
r2, o2 = RK.keys_of("Say that again, please!", live, B, glove_tokens)
r3, _ = RK.keys_of("please please say that zorbleflux again", live, B, glove_tokens)
check("reader key: order, case, punctuation, repeats and table-less words do not matter", r1 == r2 == r3, f"{r1} {r2} {r3}")
check("order-free key: the sorted normalised words", o1 == o2 == tuple(sorted(["please", "say", "that", "again"])), f"{o1} {o2}")
check("different words give different keys", RK.keys_of("what is my name", live, B, glove_tokens)[0] != r1)

# c3x_report.py on synthetic rows
d = RUNS / "report"
for arm in ("orig2", "F2", "twin", "F_rk"):
    for s in ("clinc150", "banking77", "hwu64"):
        if arm == "F_rk" and s != "clinc150":
            continue
        p = d / "feats" / arm / f"{s}__p32tok_K512.meta.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kmeans_run_id": "same" if s != "hwu64" or arm != "twin" else "other"}))
base = {"clinc150": 0.945, "banking77": 0.915, "hwu64": 0.893}
add = {"orig2": 0.0, "F2": 0.008, "twin": 0.003, "F_rk": 0.007}
val = []
for s, b0 in base.items():
    for sd in (4, 5, 6):
        for arm, x in add.items():
            if arm == "F_rk" and s != "clinc150":
                continue
            val.append({"tag": "3cx", "arm": arm, "dataset": s, "seed": sd, "repeat": 1, "val_acc": b0 + x + 0.001 * (sd - 5)})
        val.append({"tag": "3cx", "arm": "orig2", "dataset": s, "seed": sd, "repeat": 5, "val_acc": b0 + 0.002 + 0.001 * (sd - 5)})
    val.append({"tag": "3cx-check", "arm": "orig", "dataset": s, "seed": 4, "repeat": 1, "val_acc": 0.1})
(d / "results_c3x_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
lr = [{"name": arm, "dataset": s, "lr_val": 0.9 + x, "acc_by_C": {"1.0": 0.9 + x, "10.0": 0.8 + 2 * x, "100.0": 0.7}}
      for s in base for arm, x in add.items() if arm != "F_rk" or s == "clinc150"]
(d / "results_c3x_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))
reg = d / "verdict_reg.json"
# the synthetic rows have a paired SE of 0, so any TM shortfall counts as clearly smaller: every registered TM gain here
# is at most the synthetic +0.80
reg.write_text(json.dumps({"clinc150": {"tm_gain": 0.79, "lr_gain": 0.77}, "banking77": {"tm_gain": 0.79, "lr_gain": 0.69},
                           "hwu64": {"tm_gain": 0.79, "lr_gain": -0.37}}))
p = subprocess.run([sys.executable, "-I", "-B", str(TU / "c3x_report.py"), "--c3x", str(d), "--registered", str(reg)],
                   capture_output=True, text=True, timeout=120, check=False)
check("report: exit 0", p.returncode == 0, p.stderr[-500:])
txt = (d / "report_c3x.txt").read_text().splitlines() if (d / "report_c3x.txt").exists() else [""]
check("report: first line says the replication is not clearly smaller", txt[0].startswith("Replication: F2 - orig2") , txt[0])
check("report: the replication lines sit beside the registered gains", any("registered TM +0.79 / LR +0.77; F2 - orig2 TM +0.80" in x
      for x in txt), "\n".join(txt[:5]))
rep = json.loads((d / "report_c3x.json").read_text()) if (d / "report_c3x.json").exists() else {}
cl = rep.get("sets", {}).get("clinc150", {})
check("report: F2 - orig2 TM is +0.80 with SE 0", cl and abs(cl["tm_gains"]["F2-orig2"]["mean"] - 0.8) < 1e-9
      and abs(cl["tm_gains"]["F2-orig2"]["se"]) < 1e-9, str(cl.get("tm_gains", {}).get("F2-orig2")))
check("report: rep5 - orig2 TM is +0.20 and F2 - rep5 +0.60", cl and abs(cl["tm_gains"]["rep5-orig2"]["mean"] - 0.2) < 1e-9
      and abs(cl["tm_gains"]["F2-rep5"]["mean"] - 0.6) < 1e-9)
check("report: LR F2 - twin is +0.50", cl and abs(cl["lr_gains"]["F2-twin"] - 0.5) < 1e-9, str(cl.get("lr_gains")))
check("report: the check rows are left out (the TM rows are 3 seeds each)", cl and len(cl["tm"]["orig2"]) == 3)
fx = cl.get("lr_gains_fixed_C", {}) if cl else {}
check("report: fixed-C LR gains (F2 - orig2: +0.80 at C 1, +1.60 at C 10)", fx and abs(fx["1.0"]["F2-orig2"] - 0.8) < 1e-9
      and abs(fx["10.0"]["F2-orig2"] - 1.6) < 1e-9, str(fx)[:300])
check("report: the fixed-C line is in the text", any(x.startswith("clinc150 fixed-C LR:") for x in txt), "\n".join(txt[-6:]))
check("report: one k-means on CLINC150, not on HWU64 (a twin meta with another run id)",
      rep["sets"]["clinc150"]["one_kmeans"] and not rep["sets"]["hwu64"]["one_kmeans"])
check("report: no F_rk on Banking77", rep["sets"]["banking77"]["tm_gains"]["F_rk-orig2"] is None)
p2 = subprocess.run([sys.executable, "-I", "-B", str(TU / "c3x_report.py"), "--c3x", str(d), "--registered", str(reg)],
                    capture_output=True, text=True, timeout=120, check=False)
check("report: refuses to overwrite (exit 2)", p2.returncode == 2, p2.stderr[-200:])
d2 = RUNS / "report_smaller"
shutil.copytree(d, d2, ignore=shutil.ignore_patterns("report_c3x.*"))
reg2 = d2 / "verdict_reg.json"
reg2.write_text(json.dumps({"clinc150": {"tm_gain": 1.5, "lr_gain": 0.77}, "banking77": {"tm_gain": 0.79, "lr_gain": 0.69},
                            "hwu64": {"tm_gain": 0.79, "lr_gain": -0.37}}))
p3 = subprocess.run([sys.executable, "-I", "-B", str(TU / "c3x_report.py"), "--c3x", str(d2), "--registered", str(reg2)],
                    capture_output=True, text=True, timeout=120, check=False)
t3 = (d2 / "report_c3x.txt").read_text().splitlines() if (d2 / "report_c3x.txt").exists() else [""]
check("report: a clearly smaller replication is named in the FIRST line, and only that set", p3.returncode == 0
      and t3[0].startswith("REPLICATION CLEARLY SMALLER on clinc150:"), t3[0])

if "--feats" in sys.argv:
    kept = RUNS / "tiny_kept.jsonl"
    sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
    from dl.data import load

    ds = load("clinc150")
    rows = [{"set": "clinc150", "index": i, "y": int(ds.labels["train"][i]), "label": ds.intents[int(ds.labels["train"][i])],
             "source": ds.texts["train"][i], "text": ds.texts["train"][i] + " please"} for i in range(5)]
    kept.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = RUNS / "feats"
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", "-B", str(TU / "c3x_feats.py"), "--dataset", "clinc150", "--table",
                        "wordtab:" + str(TU / "a3" / "tables" / "a3B_d768.npz"), "--arms", f"orig2:none,tiny:{kept}",
                        "--out-root", str(out), "--device", "cpu"], capture_output=True, text=True, timeout=1800, check=False)
    check(f"feats: exit 0 ({time.time() - t0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        za, zb = np.load(out / "orig2" / "clinc150__p32tok_K512.npz"), np.load(out / "tiny" / "clinc150__p32tok_K512.npz")
        ma = json.loads((out / "orig2" / "clinc150__p32tok_K512.meta.json").read_text())
        mb = json.loads((out / "tiny" / "clinc150__p32tok_K512.meta.json").read_text())
        n = int(za["n_orig"])
        check("feats: both arms share one k-means run", ma["kmeans_run_id"] == mb["kmeans_run_id"] and ma["thresholds"] == mb["thresholds"])
        check("feats: the arm's original rows equal orig2's bit for bit", all(np.array_equal(za[k], zb[k][:n]) for k in za.files
                                                                              if k.startswith("train__"))
              and all(np.array_equal(za[k], zb[k]) for k in za.files if k.startswith("val__")))
        check("feats: 5 rows appended with their labels", zb["y_train"].shape[0] == n + 5
              and list(zb["y_train"][n:]) == [r["y"] for r in rows] and mb["n_aug"] == 5 and ma["n_aug"] == 0)
else:
    print("feats part not run (no --feats)")

print(f"c3x tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
