"""E3 direct-label fallback: the s choice (pre-registered on 2 Oct 2026, 22:28 CEST; the E3 rule of 2 Oct 2026,
06:15 CEST; see docs/registrations.md), applied to the 3 seed-1 screen runs with slide10.check2b --split dev (288 view):
  most Check 2b criteria met on dev (coverage >= 0.9; highest firing on negated / near-miss / broken <= 0.05;
  brackets recovered >= 4); ties -> higher coverage minus highest firing (dev); then higher final dev accuracy
  (FINAL dev_acc of the run); then lower s. The evaluation split is not touched.
Usage: python e3fb_choice.py RUN_S5 RUN_S21 RUN_S100"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slide10
out_dir = os.environ.pop("TMLM_RUN_DIR", ".")
shutil.copy(os.path.abspath(__file__), out_dir)
rows = []
for r in sys.argv[1:]:
    log = open(os.path.join(r, "log.txt")).read().splitlines()
    cfg = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG "))
    fin = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("FINAL "))
    print("==", os.path.basename(r), "s", cfg["s"], "final pass", fin["pass"])
    c = slide10.check2b(r, None, "dev")
    crit = [c["coverage_dev_288"] >= 0.9, c["fires_dev_max"] <= 0.05, c["brackets_recovered_288"] >= 4]
    rows.append({"run": os.path.basename(r), "s": cfg["s"], "seed": cfg["seed"], "final_pass": fin["pass"],
                 "criteria_met": sum(crit), "criteria": crit, "cov_minus_fire": c["coverage_dev_288"] - c["fires_dev_max"],
                 "dev_acc": fin["dev_acc"], "coverage_dev_288": c["coverage_dev_288"], "fires_dev_max": c["fires_dev_max"],
                 "brackets_recovered_288": c["brackets_recovered_288"]})
assert all(r["seed"] == 1 for r in rows) and len({r["s"] for r in rows}) == 3
best = max(rows, key=lambda r: (r["criteria_met"], r["cov_minus_fire"], r["dev_acc"], -r["s"]))
for r in rows:
    print("ROW", json.dumps(r))
print("CHOICE", json.dumps({"s": best["s"], "run": best["run"]}))
json.dump({"rows": {"value": rows, "label": "measured"}, "chosen_s": {"value": best["s"], "label": "derived"}},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
