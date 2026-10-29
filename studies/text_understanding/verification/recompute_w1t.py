"""Independent recompute of REGISTRATION_W1T.md's bars (read only).

Word arm: glassbox-lm/studies/text_understanding/w1/results_w1t_test.jsonl (view existsT_c4words, mode test, seeds 1-5).
References: N7's registered test rows (glassbox-lm/studies/text_understanding/results_n7_test.jsonl): existsT_potion, existsT_glove, bow.
Statistic: S2 = test_at_best_val (primary), S1 = test_mean_last50 beside it; paired by seed; means of 5.
- word - potion and word - GloVe: REACHES = mean diff >= -1.0 on >= 3 of 5 sets; NEAR = >= -2.0 on >= 3 of 5; else BELOW.
- word - bow: N7's D rule: on a set, "adds" when the mean > 0 and >= 4 of 5 paired seed diffs > 0; met on >= 3 of 5.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import json
import statistics as st
from pathlib import Path

E = Path(f"{_REPO_ROOT}/studies/text_understanding")
SETS = ("trec", "mr", "sst2", "r52", "r8")
SEEDS = (1, 2, 3, 4, 5)
EPS = 1e-9


def load(path, keep):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") != "test" or not keep(r):
            continue
        out.setdefault((r["dataset"], r["view"], int(r["seed"])), r)
    return out


word = load(E / "w1/results_w1t_test.jsonl", lambda r: r.get("registration") == "REGISTRATION_W1T.md")
ref = load(E / "results_n7_test.jsonl", lambda r: r.get("registration") == "REGISTRATION_N7.md")
print(f"word rows {len(word)}, N7 reference rows {len(ref)}")


def diffs(ds, other_view, key):
    d = [word[(ds, "existsT_c4words", s)][key] - ref[(ds, other_view, s)][key] for s in SEEDS
         if (ds, "existsT_c4words", s) in word and (ds, other_view, s) in ref]
    return d if len(d) == 5 else None


for name, view in (("potion", "existsT_potion"), ("GloVe 42B", "existsT_glove")):
    within1 = within2 = 0
    parts = []
    for ds in SETS:
        d2, d1 = diffs(ds, view, "test_at_best_val"), diffs(ds, view, "test_mean_last50")
        if d2 is None:
            parts.append(f"{ds}: incomplete")
            continue
        m2 = st.mean(d2)
        within1 += m2 >= -1.0 - EPS
        within2 += m2 >= -2.0 - EPS
        parts.append(f"{ds} S2 {m2:+.2f} (S1 {st.mean(d1):+.2f})")
    verdict = "REACHES" if within1 >= 3 else ("NEAR" if within2 >= 3 else "BELOW")
    print(f"word - {name}: {verdict} (within 1.0 on {within1}, within 2.0 on {within2} of 5)  [" + "; ".join(parts) + "]")

n_add = 0
parts = []
for ds in SETS:
    d = diffs(ds, "bow", "test_at_best_val")
    if d is None:
        parts.append(f"{ds}: incomplete")
        continue
    adds = st.mean(d) > EPS and sum(x > EPS for x in d) >= 4
    n_add += adds
    parts.append(f"{ds} {st.mean(d):+.2f} ({sum(x > EPS for x in d)}/5){' adds' if adds else ''}")
print(f"word - bow (D rule): adds on {n_add} of 5 -> {'MET' if n_add >= 3 else 'NOT MET'}  [" + "; ".join(parts) + "]")
for ds in SETS:
    a = [word[(ds, "existsT_c4words", s)]["test_at_best_val"] for s in SEEDS if (ds, "existsT_c4words", s) in word]
    if a:
        print(f"  {ds}: word arm S2 mean {st.mean(a):.2f} over {len(a)} seeds")
