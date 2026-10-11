"""Purity and coverage of learned OR-groups on an EQUAL FOOTING across runs (diagnostic, not a gate).

Group recovery (gates.md) excludes always-true groups, so runs with many always-true groups are judged on a selected
subset. This script compares runs on the same footing:
  - top8: the OR-groups of the 8 highest-ranked positive clauses (the order of rules.txt: dev fire-rate difference);
  - all:  the OR-groups of all positive clauses.
In both, a group's words are the words of its NON-EMPTY options (an empty option adds no word), so always-true
groups are not dropped. A group needs at least 2 distinct words.
  purity   = share of a group's words that come from its main hidden group (random groups of ~25 words: about 0.24)
  coverage = share of that hidden group's words (m = V/G) that the group contains
Usage: python pcp_purity.py RUN_DIR [RUN_DIR ...]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcp_world import RUNGS  # noqa: E402


def groups_of(inc, clauses, A, K):
    out = []
    for c in clauses:
        for j in range(2 * K):
            opts = inc[c, j * A:(j + 1) * A]
            words = np.flatnonzero(opts[opts.any(1)].any(0))
            if len(words) >= 2:
                out.append(words)
    return out


def stats(groups, G, m):
    if not groups:
        return {"n": 0, "purity": None, "coverage": None}
    pur, cov = [], []
    for w in groups:
        cnt = np.bincount(w % G, minlength=G)
        pur.append(cnt.max() / len(w))
        cov.append(cnt.max() / m)
    return {"n": len(groups), "purity": round(float(np.mean(pur)), 3), "coverage": round(float(np.mean(cov)), 3),
            "mean_words": round(float(np.mean([len(w) for w in groups])), 1)}


def main():
    rows = []
    for d in sys.argv[1:]:
        cfg = next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(d, "log.txt")) if l.startswith("CONFIG "))
        rung = re.search(r"pairs_(L\d)_s\d+", cfg["data"]).group(1)
        V, G = RUNGS[rung]["V"], RUNGS[rung]["G"]
        inc = np.unpackbits(np.load(os.path.join(d, "include_bits.npz"))["class1"], axis=-1, count=cfg["F"]).astype(bool)
        top = [int(m.group(1)) for m in re.finditer(r"^#\d+ clause (\d+) \(\+\)", open(os.path.join(d, "rules.txt")).read(), re.M)][:8]
        allpos = list(range(0, inc.shape[0], 2))
        r = {"run": os.path.basename(d)[17:], "s": cfg["s"], "top8": stats(groups_of(inc, top, cfg["A"], cfg["K"]), G, V // G),
             "all": stats(groups_of(inc, allpos, cfg["A"], cfg["K"]), G, V // G)}
        rows.append(r)
        print(json.dumps(r))
    print("\n| run | s | top-8 purity | top-8 coverage | all purity | all coverage | groups (all) |\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['s']} | {r['top8']['purity']} | {r['top8']['coverage']} | {r['all']['purity']} | "
              f"{r['all']['coverage']} | {r['all']['n']} |")
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        json.dump(rows, open(os.path.join(run_dir, "purity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
