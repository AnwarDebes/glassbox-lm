"""Build a 500-row dev result from a base run and a re-ask of some of its rows (NOTE_review_fixes.md, L35t16 on STAR).

usage: python3.12 -I combine_reask.py rows --base runs/llm_q3_L35t/star/preds.jsonl --out runs/llm_q3_L35t16/star_reask_rows.json
       python3.12 -I combine_reask.py combine --base runs/llm_q3_L35t/star/preds.jsonl \
           --reask runs/llm_q3_L35t16/star_reask/preds.jsonl --rows runs/llm_q3_L35t16/star_reask_rows.json \
           --out runs/llm_q3_L35t16/star
"rows": the rows of the base run with no usable answer (pred empty or None), as a sorted JSON list.
"combine": the base rows outside that list are kept as they are; the listed rows are replaced by the re-asked ones.
Refuses when the re-ask does not hold exactly the listed rows, holds a request error, or the base is not 500 rows.
Writes OUT/preds.jsonl and OUT/result.json (composite: true, the sources and their sha256). Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import time


def load(p):
    with open(p) as h:
        return [json.loads(x) for x in h if x.strip()]


def sha(p):
    with open(p, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def empty_rows(recs):
    return sorted(r["i"] for r in recs if not (r.get("pred") or "").strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rows", "combine"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--reask")
    ap.add_argument("--rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = load(a.base)
    if len(base) != 500 or len({r["i"] for r in base}) != 500:
        sys.exit("refused: the base run is not 500 distinct rows")
    if a.what == "rows":
        rows = empty_rows(base)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "x") as h:
            json.dump(rows, h)
        print(f"{len(rows)} rows without a usable answer -> {a.out}")
        return 0
    rows = json.load(open(a.rows))
    re = load(a.reask)
    if sorted(r["i"] for r in re) != sorted(rows):
        sys.exit("refused: the re-ask does not hold exactly the listed rows")
    if any(r.get("error") is not None for r in re):
        sys.exit("refused: the re-ask holds a request error")
    if rows != empty_rows(base):
        sys.exit("refused: the listed rows are not the base run's rows without a usable answer")
    by = {r["i"]: r for r in re}
    out = [by.get(r["i"], r) for r in base]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "preds.jsonl"), "x") as h:
        h.writelines(json.dumps(r) + "\n" for r in out)
    acc = sum(r["pred"] == r["gold"] for r in out) / len(out)
    res = {"composite": True, "note": "NOTE_review_fixes.md", "n": len(out), "accuracy": acc,
           "errors": sum(r.get("error") is not None for r in out), "reasked": len(rows),
           "reasked_now_usable": sum(1 for r in re if (r.get("pred") or "").strip()),
           "reasked_correct": sum(r["pred"] == r["gold"] for r in re),
           "base_accuracy": sum(r["pred"] == r["gold"] for r in base) / len(base),
           "sources": {a.base: sha(a.base), a.reask: sha(a.reask), a.rows: sha(a.rows)},
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(a.out, "result.json"), "x") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
