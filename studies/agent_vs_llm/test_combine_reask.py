"""Tests of combine_reask.py on synthetic files (standard library only).

run: ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -I test_combine_reask.py
"""
import json
import os
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import combine_reask as C  # noqa: E402

OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])


def write(p, recs):
    with open(p, "w") as h:
        h.writelines(json.dumps(r) + "\n" for r in recs)


with tempfile.TemporaryDirectory() as tmp:
    base = [{"i": i, "gold": "a", "pred": ("a" if i % 3 else ""), "error": None} for i in range(500)]
    base[1]["pred"] = None
    bp = os.path.join(tmp, "base.jsonl")
    write(bp, base)
    rp = os.path.join(tmp, "rows.json")
    C.main(["rows", "--base", bp, "--out", rp])
    rows = json.load(open(rp))
    want = sorted([i for i in range(500) if i % 3 == 0] + [1])
    check(f"rows: the {len(want)} rows with an empty or missing answer", rows == want)
    re = [{"i": i, "gold": "a", "pred": ("a" if i % 2 else "b"), "error": None} for i in rows]
    ap = os.path.join(tmp, "re.jsonl")
    write(ap, re)
    od = os.path.join(tmp, "out")
    C.main(["combine", "--base", bp, "--reask", ap, "--rows", rp, "--out", od])
    out = [json.loads(x) for x in open(os.path.join(od, "preds.jsonl"))]
    res = json.load(open(os.path.join(od, "result.json")))
    n_right = sum(1 for i in range(500) if i % 3 and i != 1) + sum(1 for i in rows if i % 2)
    check("combined: 500 rows in base order, listed rows replaced", [r["i"] for r in out] == list(range(500))
          and all(out[i]["pred"] == ("a" if i % 2 else "b") for i in rows))
    check(f"accuracy {res['accuracy']:.3f} equals the hand count {n_right}/500", abs(res["accuracy"] - n_right / 500) < 1e-12)
    check("result marks the composite and the sources", res["composite"] and len(res["sources"]) == 3 and res["reasked"] == len(rows))
    write(ap, re[:-1])
    try:
        C.main(["combine", "--base", bp, "--reask", ap, "--rows", rp, "--out", os.path.join(tmp, "o2")])
        check("a re-ask missing a row is refused", False)
    except SystemExit as e:
        check("a re-ask missing a row is refused", "exactly the listed rows" in str(e))
    re2 = [dict(r) for r in re]
    re2[0]["error"] = "HTTPError 500"
    write(ap, re2)
    try:
        C.main(["combine", "--base", bp, "--reask", ap, "--rows", rp, "--out", os.path.join(tmp, "o3")])
        check("a re-ask with a request error is refused", False)
    except SystemExit as e:
        check("a re-ask with a request error is refused", "request error" in str(e))
    write(bp, base[:-1])
    try:
        C.main(["rows", "--base", bp, "--out", os.path.join(tmp, "r2.json")])
        check("a base of 499 rows is refused", False)
    except SystemExit as e:
        check("a base of 499 rows is refused", "not 500" in str(e))
n_fail = sum(x.startswith("FAIL") for x in OUT)
print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
print("PASSED" if n_fail == 0 else "FAILED")
sys.exit(1 if n_fail else 0)
