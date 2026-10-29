#!/usr/bin/env python
"""Step 3c, descriptive: the sample for the hand check of kept and dropped paraphrases (tu/PLAN_3c.md, Bar,
"Descriptive": 30 kept and 30 dropped paraphrases per set; judged by an AI model, not a person). The sample is fixed
here, before any filter output exists (2026-10-21):

    python -I c3_handcheck_sample.py --gen tu/c3/gen.jsonl --kept-c tu/c3/kept_C.jsonl --kept-f tu/c3/kept_F.jsonl \
        --out tu/c3/handcheck_3c_sample.tsv

- Kept: the registered arm's paraphrases (kept_F.jsonl) of the set.
- Dropped: every paraphrase of the set in gen.jsonl that is not in kept_F.jsonl, counted as multisets over
  (set, source index, text). Its outcome is "F3:<predicted label>" when it is in kept_C.jsonl minus kept_F.jsonl
  (dropped by the label filter) and "F1/F1b/F2" otherwise (c3_filter.py saves those drops apart only with --pilot).
- One numpy default_rng(20261024), used in the order CLINC150, Banking77, HWU64, kept then dropped: 30 of each pool
  without replacement (the whole pool when it is smaller), in pool order. Pools are in file order.
- The TSV has an empty "judgement" column (yes / no / unclear: a fluent utterance with the source's intent) and a
  "note" column. The output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SETS = ("clinc150", "banking77", "hwu64")
N = 30
SEED = 20261024


def jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def clean(s: str) -> str:
    return " ".join(str(s).split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--kept-c", type=Path, required=True)
    ap.add_argument("--kept-f", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    gen, kc, kf = jsonl(a.gen), jsonl(a.kept_c), jsonl(a.kept_f)
    rng = np.random.default_rng(SEED)
    lines = ["set\tlabel\tsource\tparaphrase\toutcome\tcos\tjudgement\tnote"]
    summary = {}
    for name in SETS:
        kept = [c for c in kf if c["set"] == name]
        left = Counter((c["set"], c["index"], c["text"]) for c in kept)
        kf_left, in_c = Counter(left), {}  # in_c: kept_C minus kept_F as multisets, i.e. the F3 drops
        for c in kc:
            if c["set"] != name:
                continue
            k = (c["set"], c["index"], c["text"])
            if kf_left[k] > 0:
                kf_left[k] -= 1
            else:
                in_c.setdefault(k, []).append(c)
        if any(v > 0 for v in kf_left.values()):
            print(f"{name}: kept_F rows not in kept_C.jsonl: {sum(kf_left.values())}", file=sys.stderr)
            return 1
        dropped = []
        for r in gen:
            if r["set"] != name:
                continue
            for p in r["paraphrases"]:
                k = (name, r["index"], p)
                if left[k] > 0:
                    left[k] -= 1
                    continue
                cs = in_c.get(k)
                if cs:
                    c = cs.pop(0)
                    dropped.append({**c, "outcome": f"F3:{c['f3_pred']}"})
                else:
                    dropped.append({"set": name, "index": r["index"], "label": r["label"], "source": r["text"], "text": p,
                                    "outcome": "F1/F1b/F2"})
        if any(v > 0 for v in left.values()):
            print(f"{name}: kept_F rows not found in gen.jsonl: {sum(left.values())}", file=sys.stderr)
            return 1
        picks = {}
        for group, pool in (("kept", kept), ("dropped", dropped)):
            idx = np.sort(rng.choice(len(pool), size=min(N, len(pool)), replace=False)) if pool else np.zeros(0, int)
            picks[group] = [pool[i] for i in idx]
        for group in ("kept", "dropped"):
            for c in picks[group]:
                outcome = "kept" if group == "kept" else c["outcome"]
                cos = f"{c['cos']:.3f}" if "cos" in c else ""
                lines.append("\t".join([name, clean(c["label"]), clean(c["source"]), clean(c["text"]), outcome, cos, "", ""]))
        summary[name] = {"kept_pool": len(kept), "dropped_pool": len(dropped),
                         "dropped_f3_pool": sum(1 for c in dropped if c["outcome"].startswith("F3:")),
                         "sampled_kept": len(picks["kept"]), "sampled_dropped": len(picks["dropped"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"out": str(a.out), "seed": SEED, "per_set": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
