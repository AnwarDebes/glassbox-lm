"""Apply a pre-registered selection rule from gates.md to finished runs, exactly as written. Run under run_logged.py.

  apply_rules.py headline-s RUN_S100 RUN_S200 RUN_S400   (rule of Fri 2 Oct about 02:58)
      The s in {100, 200, 400} with the highest dev AUC at pass 100 whose passes 96-100 each meet Check 3's per-pass
      stability conditions (dev accuracy >= 0.58; pred_pos in [0.2, 0.8]). Ties go to the lower s.
      A run without pass 100 (e.g. stopped early by rule) is not eligible.
  apply_rules.py pass-P CURVE_RUN                          (rule of Fri 2 Oct about 02:30)
      P = 100, unless the dev AUC at pass 150 or 200 is at least 0.01 above the dev AUC at pass 100 AND passes P-4..P
      each meet the same stability conditions; then P is the smallest such pass count.
Writes metrics.json (measured: per-run values read from the logs; derived: the choice) and prints RULE {json}."""
import json
import os
import sys


def hist(run):
    return {h["pass"]: h for h in (json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(run, "log.txt"))
                                   if l.startswith("EPOCH "))}


def window(h, P):
    """Passes P-4..P with Check 3's per-pass stability conditions."""
    win = [h.get(p) for p in range(P - 4, P + 1)]
    ok = all(w is not None and w["dev_acc"] >= 0.58 and 0.2 <= w["pred_pos"] <= 0.8 for w in win)
    return ok, [[p, w["dev_acc"], w["pred_pos"]] if w else [p, None, None] for p, w in zip(range(P - 4, P + 1), win)]


def cfg(run):
    return next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(run, "log.txt")) if l.startswith("CONFIG "))


def headline_s(runs):
    rows = []
    for s_expected, run in zip((100, 200, 400), runs):
        c, h = cfg(run), hist(run)
        if float(c["s"]) != s_expected or c["data"] != "sa" or c["seed"] != 1 or c["clauses"] != 50:
            raise SystemExit(f"{run}: not the expected screen run for s = {s_expected}")
        stable, win = window(h, 100)
        rows.append({"s": s_expected, "run": os.path.basename(run), "has_pass_100": 100 in h,
                     "auc_at_100": h[100]["dev_auc"] if 100 in h else None, "window_96_100": win, "stable": stable,
                     "last_pass": max(h)})
    ok = [r for r in rows if r["has_pass_100"] and r["stable"]]
    best = max(ok, key=lambda r: (r["auc_at_100"], -r["s"])) if ok else None   # ties -> lower s
    ranked = sorted((r["auc_at_100"] for r in ok), reverse=True)
    return rows, {"headline_s": best["s"] if best else None,
                  "margin_to_next_eligible": round(ranked[0] - ranked[1], 6) if len(ranked) > 1 else None}


def pass_P(run):
    h = hist(run)
    a100 = h[100]["dev_auc"]
    rows = []
    for P in (150, 200):
        stable, win = window(h, P)
        gain = h[P]["dev_auc"] - a100 if P in h else None
        rows.append({"P": P, "auc": h[P]["dev_auc"] if P in h else None, "gain_over_100": gain,
                     "window": win, "stable": stable,
                     # the gain is a difference of two floats: a true gain of exactly 0.01 must not fail by rounding
                     "qualifies": gain is not None and gain >= 0.01 - 1e-12 and stable})
    P = next((r["P"] for r in rows if r["qualifies"]), 100)
    stable100, win100 = window(h, 100)
    return rows, {"P": P, "auc_at_100": a100, "window_96_100": win100, "stable_at_100": stable100}


def main():
    mode, runs = sys.argv[1], sys.argv[2:]
    rows, choice = headline_s(runs) if mode == "headline-s" else pass_P(runs[0])
    out = os.environ.get("TMLM_RUN_DIR", ".")
    m = {"rows": {"value": rows, "label": "measured"}, **{k: {"value": v, "label": "derived"} for k, v in choice.items()}}
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("RULE", json.dumps({"mode": mode, **choice, "rows": rows}))


if __name__ == "__main__":
    main()
