"""Check 2c, accuracy part (gates.md; pre-registered on 2 Oct 2026, 06:06 CEST, see docs/registrations.md; counted in whole texts per the 07:38 check).

Hierarchical accuracy is within 2 points of a flat TM while using fewer rules (mean of 3 seeds), read one-sidedly:
pass if the hierarchical mean dev accuracy at pass 50 over seeds 1-3 is at least the flat mean minus 0.02.
In whole texts: sum of correct dev texts over the 3 hierarchical runs >= the flat sum - 0.02 * (3 * n_dev).
Seed 1 of each model is its screen run at the chosen s; seeds 2-3 are the e5a_{model}_s{S}_seed{2,3} runs.
Usage: python e5a_check2c.py S   (the chosen s, the same for both models here)"""
import glob
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
S = sys.argv[1]


def final(d):
    log = open(os.path.join(d, "log.txt")).read().splitlines()
    cfg = next(json.loads(l[7:]) for l in log if l.startswith("CONFIG "))
    fin = next(json.loads(l[6:]) for l in log if l.startswith("FINAL "))
    return cfg, fin


out = {}
for m in ("hier", "flat"):
    dirs = [sorted(glob.glob(f"{R}/*_e5a_screen_{m}_s{S}"))[-1]] + [sorted(glob.glob(f"{R}/*_e5a_{m}_s{S}_seed{k}"))[-1] for k in (2, 3)]
    rows = []
    for d in dirs:
        cfg, fin = final(d)
        assert fin["pass"] == 50 and not fin.get("stopped"), (d, fin["pass"], fin.get("stopped"))
        n = cfg["n_dev"]
        k = round(fin["dev_acc"] * n)
        assert abs(k / n - fin["dev_acc"]) < 1e-12, "dev_acc is not a whole number of texts"
        rows.append({"run": os.path.basename(d), "seed": cfg["seed"], "s": cfg["s"], "rules_per_topic": cfg["clauses"],
                     "n_dev": n, "correct": k, "dev_acc": fin["dev_acc"], "dev_macro_f1": fin["dev_macro_f1"]})
    out[m] = {"runs": rows, "correct_total": sum(r["correct"] for r in rows), "texts_total": sum(r["n_dev"] for r in rows)}
    out[m]["mean_acc"] = out[m]["correct_total"] / out[m]["texts_total"]
assert out["hier"]["texts_total"] == out["flat"]["texts_total"]
N = out["hier"]["texts_total"]
allow = round(0.02 * N)  # 600 for 3 x 10,000 dev texts
verdict = out["hier"]["correct_total"] >= out["flat"]["correct_total"] - allow
res = {"s": S, "hier_correct": out["hier"]["correct_total"], "flat_correct": out["flat"]["correct_total"], "texts": N,
       "allowed_shortfall_texts": allow, "shortfall_texts": out["flat"]["correct_total"] - out["hier"]["correct_total"],
       "hier_mean_acc": out["hier"]["mean_acc"], "flat_mean_acc": out["flat"]["mean_acc"],
       "rules_per_topic": {"hier": out["hier"]["runs"][0]["rules_per_topic"], "flat": out["flat"]["runs"][0]["rules_per_topic"]},
       "check2c_accuracy_part_pass": bool(verdict)}
for m in ("hier", "flat"):
    for r in out[m]["runs"]:
        print("RUN", m, json.dumps(r))
print("CHECK2C_ACC", json.dumps(res))
json.dump({"runs": {"value": out, "label": "measured"}, **{k: {"value": v, "label": "derived"} for k, v in res.items()}},
          open(os.path.join(os.environ.get("TMLM_RUN_DIR", "."), "metrics.json"), "w"), indent=1)
