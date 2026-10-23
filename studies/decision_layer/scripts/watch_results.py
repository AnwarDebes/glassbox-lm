#!/usr/bin/env python
"""Read JSON result lines from stdin (tail -F of the result files) and print one short line each."""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if "learner" in r and "router_acc" in r:
        ev = r.get("eval_result", {})
        print("R6", r.get("tag"), r["learner"], "seed", r["seed"], "val_ret", round(r.get("val", {}).get("mean_return", -9), 4),
              "eval_ret", round(ev.get("mean_return", -9), 4), "ask", round(ev.get("share_ask", -1), 3),
              "route0", round(ev.get("share_route0", -1), 3), "esc0", round(ev.get("share_escalate0", -1), 3),
              "router", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r["router_acc"].items()}, flush=True)
    elif "learner" in r:
        print("BANDIT", r["tag"], r["dataset"], r["target"], r["learner"], "eps", r.get("eps"), "topk", r.get("topk"),
              "replay", r.get("replay"), "alpha", r.get("alpha"), "cum", round(r.get("cum_online_acc", -1), 4),
              "final", round(r["final_eval_acc"], 4), [round(c["eval_acc"], 3) for c in r.get("curve", [])], r.get("seconds"), flush=True)
    elif "stats" in r and "view" in r:
        s = r["stats"]
        print("TM", r["tag"], r["dataset"], r["encoder"], r["view"], "val", r["val_acc"], "ep", r["best_epoch"], "t", r["train_s"],
              "med", s["median_literals"], "max", s["max_literals"], "nopos", s.get("no_positive_literal"), "empty", s["empty"],
              "oos", r.get("oos_auroc_val"), "bitsLR", r.get("bits_logreg_val"), "ens", r.get("ensemble_val"),
              r.get("member_val"), "ro", r.get("readout"), flush=True)
