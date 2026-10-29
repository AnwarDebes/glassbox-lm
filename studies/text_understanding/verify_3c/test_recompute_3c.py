#!/usr/bin/env python
"""Tests for recompute_3c.py on synthetic files (a plain script, no pytest):

    OMP_NUM_THREADS=1 nice -n 19 $PY -I test_recompute_3c.py

Each case writes its own folder under verify_3c/test_runs/<UTC stamp>/ (base.json, the reference, potion and W2 files,
the F metas, the 3c TM and LR rows) and runs the recompute as the real run will. Nothing real is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECOMPUTE = HERE / "recompute_3c.py"
RUNS = HERE / "test_runs" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
SETS = ("clinc150", "banking77", "hwu64")
S = {"clinc150": 8.0, "banking77": 8.0, "hwu64": 16.0}
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def wj(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make(name: str, tm_ref: dict, tm_f: dict, lr_ref: dict, lr_f: dict, lr_c: dict, identical=None, rebuilt_tm=None,
         rebuilt_lr=None, dup_f=None, drop_f=None, check_val=None, lr_new_ref=None, pot=None) -> Path:
    d = RUNS / name
    identical = identical or {k: True for k in SETS}
    pot = pot or {"clinc150": 0.9600, "banking77": 0.9300, "hwu64": 0.9117}
    lpot = {"clinc150": 0.9300, "banking77": 0.9000, "hwu64": 0.9000}
    base = {"base": "a3B_d768", "s": S, "ref_tm": {"file": str(d / "r3.jsonl"), "tag": "r3-seeds"},
            "ref_lr": {"file": str(d / "a3_lr.jsonl")}, "potion_tm": {"file": str(d / "0b.jsonl"), "tag": "0b-seeds",
            "s": {k: 8.0 for k in SETS}}, "feats_F": str(d / "feats" / "F"), "ref_dir": str(d / "ref")}
    d.mkdir(parents=True, exist_ok=True)
    (d / "base.json").write_text(json.dumps(base))
    r3 = [{"tag": "r3-seeds", "table": "a3B_d768", "dataset": k, "s": S[k], "seed": sd, "val_acc": tm_ref[k][i],
           "emb_dir": base["ref_dir"]} for k in SETS for i, sd in enumerate((4, 5, 6))]
    r3 += [{"tag": "r3-seeds", "table": "a3B_d768", "dataset": k, "s": 32.0, "seed": 4, "val_acc": 0.1} for k in SETS]
    r3 += [{"tag": "r3-screen", "table": "a3B_d768", "dataset": k, "s": S[k], "seed": 4, "val_acc": 0.2} for k in SETS]
    wj(d / "r3.jsonl", r3)
    wj(d / "a3_lr.jsonl", [{"name": "a3B_d768", "table": "a3B_d768", "dataset": k, "lr_val": lr_ref[k]} for k in SETS]
       + [{"name": "ppw_S1000", "table": "ppw_S1000_d500_e0.5", "dataset": k, "lr_val": 0.3} for k in SETS])
    wj(d / "0b.jsonl", [{"tag": "0b-seeds", "table": "potion-base-32M", "dataset": k, "s": 8.0, "seed": sd, "val_acc": pot[k]}
                        for k in SETS for sd in (4, 5, 6)]
       + [{"tag": "0b-seeds", "table": "ppw_S1000_d500_e0.5", "dataset": k, "s": 8.0, "seed": sd, "val_acc": 0.5}
          for k in SETS for sd in (4, 5, 6)])
    wj(d / "w2_lr.jsonl", [{"name": "potion", "table": "potion-base-32M", "dataset": k, "view": "existsT", "lr_val": lpot[k]}
                           for k in SETS] + [{"name": "potion", "table": "potion-base-32M", "dataset": k, "view": "exists",
                                              "lr_val": 0.4} for k in SETS])
    for k in SETS:
        p = d / "feats" / "F" / f"{k}__p32tok_K512.meta.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"arm": "F", "n_orig": 100, "n_aug": 300, "equivalence": {"identical": identical[k]}}))
    val = [{"tag": "3c-check", "arm": "orig", "dataset": "clinc150", "seed": 4, "s": 8.0, "n_orig": 100, "n_train": 100,
            "val_acc": tm_ref["clinc150"][0] if check_val is None else check_val, "emb_dir": base["ref_dir"]}]
    for k in SETS:
        for i, sd in enumerate((4, 5, 6)):
            if drop_f == (k, sd):
                continue
            row = {"tag": "3c-tm", "arm": "F", "dataset": k, "seed": sd, "s": S[k], "n_orig": 100, "n_train": 400,
                   "val_acc": tm_f[k][i], "emb_dir": base["feats_F"]}
            val.append(row)
            if dup_f == (k, sd):
                val.append(dict(row))
            if not identical[k]:
                val.append({"tag": "3c-ref", "arm": "orig", "dataset": k, "seed": sd, "s": S[k], "n_orig": 100,
                            "n_train": 100, "val_acc": rebuilt_tm[k][i], "emb_dir": str(d / "c3feats" / "orig")})
    wj(d / "results_c3_val.jsonl", val)
    lr = []
    for k in SETS:
        lr += [{"name": "ref", "arm": "orig", "dataset": k, "lr_val": lr_ref[k] if lr_new_ref is None else lr_new_ref[k]},
               {"name": "F", "arm": "F", "dataset": k, "lr_val": lr_f[k]}, {"name": "C", "arm": "C", "dataset": k, "lr_val": lr_c[k]}]
        if not identical[k]:
            lr.append({"name": "ref_rebuilt", "arm": "orig", "dataset": k, "lr_val": rebuilt_lr[k]})
    wj(d / "results_c3_lr.jsonl", lr)
    return d


def run(d: Path, out_name: str = "recompute.json", compare: Path | None = None) -> tuple[int, str, dict | None]:
    out = d / out_name
    cmd = [sys.executable, "-I", str(RECOMPUTE), "--base", str(d / "base.json"), "--val", str(d / "results_c3_val.jsonl"),
           "--lr", str(d / "results_c3_lr.jsonl"), "--w2-lr", str(d / "w2_lr.jsonl"), "--out", str(out)]
    if compare is not None:
        cmd += ["--compare", str(compare)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    (d / (out_name + ".stdout.txt")).write_text(p.stdout + p.stderr)
    res = json.loads(out.read_text()) if out.exists() and p.returncode != 2 else None
    return p.returncode, p.stdout + p.stderr, res


def failed_names(res: dict | None) -> list[str]:
    return [c["check"] for c in (res or {}).get("checks", []) if not c["ok"]]


REF = {"clinc150": [0.9450, 0.9480, 0.9420], "banking77": [0.9150, 0.9160, 0.9170], "hwu64": [0.8930, 0.8920, 0.8940]}
LREF = {"clinc150": 0.9400, "banking77": 0.9100, "hwu64": 0.8900}


def plus(x: dict, by: dict) -> dict:
    return {k: [v + by[k] for v in x[k]] if isinstance(x[k], list) else x[k] + by[k] for k in x}


# 1. PASS: CLINC150 and Banking77 meet the bar with both probes; HWU64 only with LR.
d = make("pass", REF, plus(REF, {"clinc150": 0.010, "banking77": 0.008, "hwu64": 0.002}), LREF,
         plus(LREF, {"clinc150": 0.010, "banking77": 0.007, "hwu64": 0.010}), plus(LREF, {"clinc150": 0.012, "banking77": 0.0, "hwu64": -0.001}))
e, out, res = run(d)
check("pass: exit 0", e == 0, out)
check("pass: verdict PASS with 2 sets", res is not None and res["verdict"] == "PASS" and res["sets_meeting_bar"] == 2, out)
check("pass: the CLINC150 TM gain is +1.00", res is not None and abs(res["sets"]["clinc150"]["tm_gain"] - 1.0) < 1e-9, out)
check("pass: the HWU64 TM gain +0.20 misses the bar", res is not None and res["sets"]["hwu64"]["meets_bar"] is False, out)
check("pass: 3c-C's LR gain is +1.20 on CLINC150", res is not None and abs(res["sets"]["clinc150"]["lr_gain_C"] - 1.2) < 1e-9, out)
check("pass: the share of the TM gap is gain / (potion - reference)", res is not None and
      abs(res["sets"]["clinc150"]["tm_share_of_gap"] - 1.0 / (96.0 - 100 * sum(REF["clinc150"]) / 3)) < 1e-6, out)
check("pass: every input file is hashed", res is not None and len(res["inputs_sha256"]) == 7 + 3, out)
e2, out2, _ = run(d)
check("an existing output is never overwritten (exit 2)", e2 == 2 and "refusing to overwrite" in out2, out2)

# 2. FAIL: only CLINC150 meets both; Banking77 meets the TM bar but its LR gain is +0.40.
d = make("fail", REF, plus(REF, {"clinc150": 0.010, "banking77": 0.008, "hwu64": 0.002}), LREF,
         plus(LREF, {"clinc150": 0.010, "banking77": 0.004, "hwu64": 0.010}), LREF)
e, out, res = run(d)
check("fail: exit 0 (the checks pass; the verdict is FAIL)", e == 0, out)
check("fail: verdict FAIL with 1 set", res is not None and res["verdict"] == "FAIL" and res["sets_meeting_bar"] == 1, out)

# 3. The bar's edge: exactly +0.5 in decimal counts, +0.499 does not; potion below the reference gives no share.
ref_e = {"clinc150": [0.9450] * 3, "banking77": [0.9150] * 3, "hwu64": [0.8930] * 3}
lr_e = {"clinc150": 0.9273, "banking77": 0.9100, "hwu64": 0.8900}
d = make("edge", ref_e, {"clinc150": [0.9500] * 3, "banking77": [0.9200] * 3, "hwu64": [0.89799] * 3}, lr_e,
         {"clinc150": 0.9323, "banking77": 0.9150, "hwu64": 0.8950}, lr_e,
         pot={"clinc150": 0.9400, "banking77": 0.9300, "hwu64": 0.9117})
e, out, res = run(d)
check("edge: +0.5 exactly meets the bar on CLINC150 and Banking77", res is not None and res["sets"]["clinc150"]["meets_bar"]
      and res["sets"]["banking77"]["meets_bar"], out)
check("edge: +0.499 does not (HWU64)", res is not None and res["sets"]["hwu64"]["meets_bar"] is False, out)
check("edge: verdict PASS with 2 sets", res is not None and res["verdict"] == "PASS" and res["sets_meeting_bar"] == 2, out)
check("edge: no share of the gap when potion is below the reference", res is not None and
      res["sets"]["clinc150"]["tm_share_of_gap"] is None, out)

# 4. The equivalence check fails on HWU64: the rebuilt originals are the reference, and the verdict changes.
d = make("rebuilt", REF, plus(REF, {"clinc150": 0.010, "banking77": 0.002, "hwu64": 0.010}), LREF,
         plus(LREF, {"clinc150": 0.010, "banking77": 0.010, "hwu64": 0.010}), LREF,
         identical={"clinc150": True, "banking77": True, "hwu64": False},
         rebuilt_tm={"hwu64": [v + 0.009 for v in REF["hwu64"]]}, rebuilt_lr={"hwu64": LREF["hwu64"] + 0.002})
e, out, res = run(d)
check("rebuilt: exit 0", e == 0, out)
check("rebuilt: HWU64 uses the rebuilt originals", res is not None and res["sets"]["hwu64"]["reference_rows"] == "rebuilt originals", out)
check("rebuilt: HWU64's TM gain is +0.10 against the rebuilt rows (not +1.00)", res is not None and
      abs(res["sets"]["hwu64"]["tm_gain"] - 0.1) < 1e-9, out)
check("rebuilt: HWU64's LR gain is +0.80 against ref_rebuilt", res is not None and abs(res["sets"]["hwu64"]["lr_gain"] - 0.8) < 1e-9, out)
check("rebuilt: verdict FAIL (1 set; with the base rows it would be PASS)", res is not None and res["verdict"] == "FAIL", out)

# 5. A duplicated F row, a missing seed, a failed code check and a failed LR code check: each gives exit 1.
good = (REF, plus(REF, {"clinc150": 0.010, "banking77": 0.008, "hwu64": 0.002}), LREF,
        plus(LREF, {"clinc150": 0.010, "banking77": 0.007, "hwu64": 0.010}), LREF)
e, out, res = run(make("dup", *good, dup_f=("clinc150", 5)))
check("dup: exit 1", e == 1, out)
check("dup: the row check names it", "one row: 3c-tm F clinc150 seed 5" in failed_names(res), out)
e, out, res = run(make("missing", *good, drop_f=("hwu64", 6)))
check("missing: exit 1", e == 1, out)
check("missing: the verdict is INCOMPLETE, not FAIL", res is not None and res["verdict"] == "INCOMPLETE", out)
e, out, res = run(make("codecheck", *good, check_val=0.9999))
check("code check: exit 1", e == 1, out)
check("code check: named", "code check: the 3c-check row equals the base's row" in failed_names(res), out)
e, out, res = run(make("lrcheck", *good, lr_new_ref={"clinc150": 0.9401, "banking77": LREF["banking77"], "hwu64": LREF["hwu64"]}))
check("LR code check: exit 1", e == 1, out)
check("LR code check: named", "LR code check clinc150: c3_lr.py's ref row equals the base's" in failed_names(res), out)

# 6. --compare against a registered verdict file in c3_verdict.py's keys.
d = make("compare", *good)
e, out, res = run(d)
v = {"verdict": res["verdict"], "sets_meeting_bar": res["sets_meeting_bar"]}
for k, x in res["sets"].items():
    v[k] = {kk: (round(x[kk], 2) if isinstance(x[kk], float) and "share" not in kk else x[kk]) for kk in
            ("s", "reference_rows", "n_orig", "n_aug_F", "tm_F", "tm_ref", "tm_gain", "lr_F", "lr_ref", "lr_gain", "lr_C",
             "lr_gain_C", "tm_potion", "lr_potion")}
    v[k]["tm_paired"] = [round(p, 2) for p in x["tm_paired"]]
    for kk in ("tm_share_of_gap", "lr_share_of_gap"):
        v[k][kk] = None if x[kk] is None else round(x[kk], 3)
(d / "verdict_same.json").write_text(json.dumps(v))
e, out, res2 = run(d, "recompute_cmp.json", d / "verdict_same.json")
check("compare: a matching verdict file passes (exit 0)", e == 0 and res2 is not None and res2["checks_failed"] == 0, out)
v2 = json.loads(json.dumps(v))
v2["banking77"]["tm_gain"] = round(v2["banking77"]["tm_gain"] + 0.02, 2)
(d / "verdict_off.json").write_text(json.dumps(v2))
e, out, res3 = run(d, "recompute_off.json", d / "verdict_off.json")
check("compare: a tm_gain 0.02 off fails (exit 1)", e == 1 and "compare banking77: tm_gain" in failed_names(res3), out)
v3 = json.loads(json.dumps(v))
v3["verdict"] = "FAIL"
(d / "verdict_word.json").write_text(json.dumps(v3))
e, out, res4 = run(d, "recompute_word.json", d / "verdict_word.json")
check("compare: a different verdict word fails (exit 1)", e == 1 and "compare: verdict" in failed_names(res4), out)

print(f"recompute_3c tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
