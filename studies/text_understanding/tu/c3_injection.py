#!/usr/bin/env python
"""Step 3c, descriptive: the label-word injection rate (tu/PLAN_3c.md, Bar, "Descriptive"; the registration entry of
2026-10-20T23:19:40Z reports it for pilots v3 and v4). A paraphrase injects when it contains a word of its intent
label that its source request lacks. The definition is the one the pilots were counted with on 2026-10-20 (22:24Z and
23:18Z), unchanged:
- words: the lowercase runs of a-z of a text, as a set;
- label words: the words of the label with '_' read as a space, minus STOP and minus words of 2 letters or fewer;
- injected: (label words & the paraphrase's words) - the source's words is not empty.

    python -I c3_injection.py --gen tu/c3/gen.jsonl [--kept tu/c3/kept_F.jsonl] [--out <new json>]

--gen counts every generated paraphrase (gen.jsonl rows: set, label, text, paraphrases), as the pilots did; --kept
also counts the kept rows (set, label, source, text). The output file, when given, is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STOP = {"a", "an", "the", "of", "to", "by", "in", "on", "for", "my", "is", "and", "or", "via", "not", "qa", "general", "iot"}


def tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def label_words(label: str) -> set[str]:
    return {w for w in tok(label.replace("_", " ")) if w not in STOP and len(w) > 2}


def injected(label: str, source: str, para: str) -> set[str]:
    return (label_words(label) & tok(para)) - tok(source)


def count(triples) -> dict:
    per: dict[str, list[int]] = {}
    for s, label, source, para in triples:
        c = per.setdefault(s, [0, 0])
        c[0] += 1
        c[1] += bool(injected(label, source, para))
    tot, inj = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"paraphrases": tot, "injected": inj, "share": round(inj / tot, 4) if tot else None,
            "per_set": {k: {"paraphrases": v[0], "injected": v[1]} for k, v in per.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--kept", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.out is not None and a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    with open(a.gen) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    res = {"gen": str(a.gen), "all_generated": count((r["set"], r["label"], r["text"], p) for r in rows for p in r["paraphrases"])}
    if a.kept is not None:
        with open(a.kept) as f:
            kept = [json.loads(line) for line in f if line.strip()]
        res["kept"] = str(a.kept)
        res["kept_rows"] = count((c["set"], c["label"], c["source"], c["text"]) for c in kept)
    text = json.dumps(res, indent=1)
    if a.out is not None:
        with open(a.out, "x") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
