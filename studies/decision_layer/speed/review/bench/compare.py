#!/usr/bin/env python
"""Compare two fingerprints (fingerprint.py output): the verdict of the no-loss rule for one benchmark.

    python -I -B compare.py BEFORE.json AFTER.json [--time-field train_s] [--min-saving 0.01] [--json OUT.json]

- RESULT: IDENTICAL only if the "result" parts are equal value for value (no tolerance; floats compared as their JSON
  text, so 0.1 + 0.2 != 0.3 is a difference). Any difference prints every differing path and gives DIFFERENT.
- TIME (if --time-field names a "timing" field present in both, e.g. train_s, or seconds_total for milp): before,
  after, saving in seconds and as a fraction of before. FASTER only if saving >= --min-saving x before.
- VERDICT: ACCEPT only if RESULT is IDENTICAL and (no time field asked, or FASTER); else REJECT.
Exit codes: 0 ACCEPT, 1 REJECT, 2 unusable input (other kinds, missing fields).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)


def diff(a, b, path="result", out=None, limit=50):
    """Every path where a and b differ (dict keys, list positions, leaf values by canonical JSON text)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'after' if k not in a else 'before'}")
            else:
                diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif canon(a) != canon(b):
        out.append(f"{path}: {canon(a)[:120]} != {canon(b)[:120]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--time-field", default="")
    ap.add_argument("--min-saving", type=float, default=0.01)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    A, B = json.loads(a.before.read_text()), json.loads(a.after.read_text())
    if A.get("kind") != B.get("kind") or "result" not in A or "result" not in B:
        print("unusable: kinds differ or no result part", file=sys.stderr)
        raise SystemExit(2)
    d = diff(A["result"], B["result"])
    identical = not d and canon(A["result"]) == canon(B["result"])
    rep = {"before": str(a.before), "after": str(a.after), "kind": A["kind"],
           "result": "IDENTICAL" if identical else "DIFFERENT", "differences": d,
           "result_sha256": {"before": A.get("result_sha256"), "after": B.get("result_sha256")}}
    faster = True
    if a.time_field:
        ta, tb = A.get("timing", {}).get(a.time_field), B.get("timing", {}).get(a.time_field)
        if not isinstance(ta, (int, float)) or not isinstance(tb, (int, float)) or ta <= 0:
            print(f"unusable: timing field {a.time_field!r} missing or not positive", file=sys.stderr)
            raise SystemExit(2)
        saving = ta - tb
        faster = saving >= a.min_saving * ta
        rep["time"] = {"field": a.time_field, "before": ta, "after": tb, "saving": round(saving, 3),
                       "saving_fraction": round(saving / ta, 5), "min_saving": a.min_saving,
                       "faster": faster}
    rep["verdict"] = "ACCEPT" if identical and faster else "REJECT"
    print(json.dumps(rep, indent=1))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n")
    raise SystemExit(0 if rep["verdict"] == "ACCEPT" else 1)


if __name__ == "__main__":
    main()
