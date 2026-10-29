#!/usr/bin/env python
"""Tests for c3_injection.py (a plain script, no pytest):

    OMP_NUM_THREADS=1 nice -n 19 $PY -I tu/tests/test_c3_injection.py

Unit cases on synthetic rows, then the end-to-end check: the script must give the pilots' logged counts (registration
entry 2026-10-20T23:19:40Z: v3 70 of 298, 13 / 39 / 18; v4 63 of 296, 14 / 34 / 15) from tu/c3/pilot_v3 and tu/c3/pilot.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
import c3_injection as I

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_injection")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


check("label words: '_' is a space, stop words and short words go", I.label_words("activate_my_card") == {"activate", "card"},
      str(I.label_words("activate_my_card")))
check("label words: HWU64 style labels with a space", I.label_words("iot hue_lightoff") == {"hue", "lightoff"},
      str(I.label_words("iot hue_lightoff")))
check("a label word the source lacks is injected", I.injected("card_arrival", "where is it", "where is my card") == {"card"})
check("a label word already in the source is not", not I.injected("card_arrival", "where is my card", "my card, where is it"))
check("case and punctuation do not matter", I.injected("card_arrival", "Where is it?", "WHERE is my CARD?!") == {"card"})
check("a stop word of the label is never counted", not I.injected("balance_not_updated", "it is wrong", "it is not right"))
c = I.count([("x", "card_arrival", "where is it", "where is my card"), ("x", "card_arrival", "where is it", "where is it now"),
             ("y", "alarm_set", "wake me", "set an alarm")])
check("count: totals and per set", c["paraphrases"] == 3 and c["injected"] == 2 and c["per_set"]["x"] == {"paraphrases": 2, "injected": 1}
      and c["per_set"]["y"] == {"paraphrases": 1, "injected": 1}, str(c))

RUNS.mkdir(parents=True, exist_ok=True)
expect = {"pilot_v3": (70, 298, {"clinc150": 13, "banking77": 39, "hwu64": 18}),
          "pilot": (63, 296, {"clinc150": 14, "banking77": 34, "hwu64": 15})}
for v, (inj, tot, per) in expect.items():
    out = RUNS / f"{v}.json"
    p = subprocess.run([sys.executable, "-I", str(TU / "c3_injection.py"), "--gen", str(TU / "c3" / v / "gen_pilot.jsonl"),
                        "--kept", str(TU / "c3" / v / "kept_F.jsonl"), "--out", str(out)],
                       capture_output=True, text=True, timeout=120, check=False)
    r = json.loads(out.read_text()) if out.exists() else {}
    g = r.get("all_generated", {})
    check(f"{v}: exit 0", p.returncode == 0, p.stderr)
    check(f"{v}: {inj} of {tot} as logged", g.get("injected") == inj and g.get("paraphrases") == tot, str(g))
    check(f"{v}: per set {per}", {k: x["injected"] for k, x in g.get("per_set", {}).items()} == per, str(g.get("per_set")))
    check(f"{v}: the kept rows are counted too", r.get("kept_rows", {}).get("paraphrases", 0) > 0, str(r.get("kept_rows")))
    p2 = subprocess.run([sys.executable, "-I", str(TU / "c3_injection.py"), "--gen", str(TU / "c3" / v / "gen_pilot.jsonl"),
                         "--out", str(out)], capture_output=True, text=True, timeout=120, check=False)
    check(f"{v}: an existing output is refused (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr)

print(f"injection tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
