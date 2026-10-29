"""PLAN_zipf_synth.md readings from runs/zipf/*.json (mean over the seeds present).

f50 for a learner: walk the patterns from the most to the least frequent (training share, mean over seeds); f50 is
the share of the first pattern whose mean recall is below 0.5 (None if no pattern falls below 0.5).
Reading 1 (threshold set by s): at 320 clauses, f50 decreases as s grows and f50 * s stays within a factor of 4 over
the s values where f50 is defined. Reading 2 (budget crowding): at each s, f50 rises when clauses fall from 320 to 20.
usage: python -I zipf_report.py --out runs/zipf_report.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))


def f50(recall, freq):
    for j in sorted(range(len(freq)), key=lambda j: -freq[j]):
        if recall[j] < 0.5:
            return freq[j]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(H, "runs/zipf/*.json")):
        r = json.load(open(f))
        key = ("lr", None, None) if r["learner"] == "lr" else ("tm", r["s"], r["clauses"])
        groups[key].append(r)
    table = {}
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2] or 0, kv[0][1] or 0)):
        k = len(rs[0]["recall_by_pattern"])
        rec = [sum(r["recall_by_pattern"][j] for r in rs) / len(rs) for j in range(k)]
        freq = [sum(r["f_train"][j] for r in rs) / len(rs) for j in range(k)]
        name = "lr" if key[0] == "lr" else f"tm s{key[1]:g} c{key[2]}"
        table[name] = {"seeds": sorted(r["seed"] for r in rs), "acc": sum(r["test_acc"] for r in rs) / len(rs),
                       "fpr": sum(r["fpr_negatives"] for r in rs) / len(rs), "f50": f50(rec, freq),
                       "recall_top5": sum(rec[:5]) / 5, "recall_21_60": sum(rec[20:]) / max(1, len(rec[20:])),
                       "s": key[1], "clauses": key[2]}
    svals = sorted({v["s"] for v in table.values() if v["s"] is not None})
    big = [(s, table.get(f"tm s{s:g} c320", {}).get("f50")) for s in svals]
    defined = [(s, f) for s, f in big if f is not None]
    dec = all(defined[i + 1][1] <= defined[i][1] for i in range(len(defined) - 1)) if len(defined) > 1 else False
    prods = [s * f for s, f in defined]
    within4 = bool(prods) and max(prods) / min(prods) <= 4.0
    crowd = []
    for s in svals:
        f320, f20 = table.get(f"tm s{s:g} c320", {}).get("f50"), table.get(f"tm s{s:g} c20", {}).get("f50")
        if f320 is not None and f20 is not None:
            crowd.append(f20 > f320)
    res = {"plan": "PLAN_zipf_synth.md", "table": table,
           "reading1_threshold_set_by_s": bool(dec and within4), "f50_at_320": big, "f50_times_s": prods,
           "reading2_budget_crowding": bool(crowd) and all(crowd), "crowding_by_s": crowd}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    for name, v in table.items():
        f = "-" if v["f50"] is None else f"{v['f50']:.4f}"
        print(f"{name:14s} seeds {v['seeds']} acc {v['acc']:.3f} fpr {v['fpr']:.3f} f50 {f} recall top5 "
              f"{v['recall_top5']:.2f} rare(21-60) {v['recall_21_60']:.2f}")
    print("reading 1 (threshold set by s):", res["reading1_threshold_set_by_s"], "| f50 at 320:", big)
    print("reading 2 (budget crowding):", res["reading2_budget_crowding"], crowd)


if __name__ == "__main__":
    main()
