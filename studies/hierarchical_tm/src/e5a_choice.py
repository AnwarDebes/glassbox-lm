"""E5a s choice (pre-registered on 2 Oct 2026, 06:06 CEST, see docs/registrations.md): separately for hier and flat, the s with the highest mean dev
accuracy over passes 46-50 (FINAL 'last5_mean_dev_acc'); differences below 0.005 count as ties; ties go to the lower s;
a run stopped early by rule is not eligible."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
out = {}
for m in ("hier", "flat"):
    rows = []
    for s in ("21", "60", "230"):
        d = sorted(glob.glob(f"{R}/*_e5a_screen_{m}_s{s}"))[-1]
        f = next(json.loads(l[6:]) for l in open(f"{d}/log.txt") if l.startswith("FINAL "))
        rows.append({"s": s, "run": os.path.basename(d), "last5_mean_dev_acc": f["last5_mean_dev_acc"], "stopped": f.get("stopped") or ""})
    ok = [r for r in rows if r["last5_mean_dev_acc"] is not None and not r["stopped"]]
    best = max(r["last5_mean_dev_acc"] for r in ok)
    ties = [r for r in ok if best - r["last5_mean_dev_acc"] < 0.005]
    choice = min(ties, key=lambda r: float(r["s"]))
    out[m] = {"rows": rows, "choice_s": choice["s"], "within_tie_band": [r["s"] for r in ties]}
    print("CHOICE", m, json.dumps(out[m]))
json.dump({k: {"value": v, "label": "derived"} for k, v in out.items()}, open(os.path.join(os.environ.get("TMLM_RUN_DIR", "."), "metrics.json"), "w"), indent=1)
