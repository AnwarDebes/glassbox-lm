#!/usr/bin/env python
"""P9 sizing (dev only; CPU; read only; no rule): how many of the tuned STAR agent's dev errors an API-result fact could
touch, before any fact bit is built. For the text-stack map in the project notes (not published), P9.

    p9_size.py --tag e74c_star_Bc4_c16000_s16

The tag's dev decisions are rebuilt as scripts/e77_breakdown.py does (the C4 view on the CPU; bits and accuracy checked
against the row). Then every dev error is put in one class, first match wins:
- "result pair": gold and prediction are both result-like labels (failed / successful / available / unavailable /
  nothing found / correct / incorrect ...) of the same task prefix; a fact about the last query's result separates
  them;
- "query or inform": one of the two is query:<api> and the other an inform label of the same task; a fact "a query
  was made since the customer's last utterance" separates them;
- "result-like gold, other": the gold is result-like, the prediction anything else;
- "other".
Also counted: how many dev examples follow a knowledge-base return in the same dialogue, and which messages they saw.
Writes results/explore/p9_size_<tag>.json (refuses to overwrite). Added 2026-10-19.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text, cpu_tm_fast  # noqa: E402

RESULT = re.compile(r"fail|success|unavail|avail|nothing_found|not_available|cannot|conflict|incorrect|correct")


def task(label: str) -> str:
    return label.split(":", 1)[1].split("_")[0] if label.startswith("query:") else label.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e74c_star_Bc4_c16000_s16")
    args = ap.parse_args()
    out_path = ROOT / "results" / "explore" / f"p9_size_{args.tag}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; not overwriting")
    r = [d for d in map(json.loads, (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines())
         if d.get("arm") == "tm" and d.get("tag") == args.tag]
    assert len(r) == 1 and r[0]["encoder"] == "none" and r[0]["dataset"] == "star", "one C4 STAR row needed"
    r = r[0]
    count_text.patch(r["static_info"]["xcust"]["table"])
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", "star")
    dv, _ = load_examples("dev", "star")
    E = np.load(ROOT / "data" / "emb" / "star__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=tuple(r["parts"]), lex_k=r["lex_k"], seed=0).fit(tr, emb["train"], blist)
    X = view.transform(dv, emb["dev"])
    assert hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest() == r["bits_sha256"]["dev"], "dev bits differ"
    z = np.load(ROOT / "results" / "explore" / "abcd_models" / f"{args.tag}_seed{r['seed']}.npz")
    p = cpu_tm_fast.class_sums(cpu_tm_fast.pack(z["ta"], z["w"]), X).argmax(1)
    y = np.array([e["y"] for e in dv])
    assert float((p == y).mean()) == r["acc"], "CPU accuracy differs from the row"
    cls, pairs = Counter(), Counter()
    for gi, pi in zip(y, p):
        if gi == pi:
            continue
        g, q = blist[gi], blist[pi]
        if RESULT.search(g) and RESULT.search(q) and task(g) == task(q):
            c = "result pair"
        elif (g.startswith("query:") or q.startswith("query:")) and task(g) == task(q) and not (g.startswith("query:") and q.startswith("query:")):
            c = "query or inform"
        elif RESULT.search(g):
            c = "result-like gold, other"
        else:
            c = "other"
        cls[c] += 1
        if c != "other":
            pairs[(c, g, q)] += 1
    n_err = int((p != y).sum())
    out = {"what": "P9 sizing on STAR dev (no rule)", "tag": args.tag, "n_dev": len(dv), "acc": r["acc"], "n_errors": n_err,
           "classes": {c: {"n": n, "share_of_errors": round(n / n_err, 4), "points_of_accuracy": round(100 * n / len(dv), 2)} for c, n in cls.most_common()},
           "result_like_labels": sorted(b for b in blist if RESULT.search(b)),
           "top_pairs": [[c, g, q, n] for (c, g, q), n in pairs.most_common(25)]}
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("result_like_labels", "top_pairs")}, indent=1))
    for c, g, q, n in out["top_pairs"][:15]:
        print(f"{n:>4}  {c:24} {g} -> {q}")
    print("wrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
