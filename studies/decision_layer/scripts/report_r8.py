#!/usr/bin/env python
"""R8 report: mean test return per learner (seeds averaged per episode) and paired bootstrap
differences over test episodes; applies the frozen R8 rules.

    report_r8.py [--folder results/confirm/R8] [--tag R8]

Each learner's per-episode returns are in r8_returns_<learner>[_nomem]_seed<k>_<tag>_test.npy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boot(d: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results/confirm/R8")
    ap.add_argument("--tag", default="R8")
    args = ap.parse_args()
    folder = ROOT / args.folder
    rows = [json.loads(l) for l in open(folder / "r8.jsonl") if l.strip()]
    rows = [r for r in rows if r["tag"] == args.tag and r["eval"] == "test"]
    arms = {}
    for r in rows:
        name = r["learner"] + ("" if r.get("memory", True) else "_nomem")
        f = folder / f"r8_returns_{name}_seed{r['seed']}_{args.tag}_test.npy"
        arms.setdefault(name, {})[r["seed"]] = (np.load(f), r)
    out = {"arms": {}}
    per_ep = {}
    for name, seeds in sorted(arms.items()):
        R = np.stack([v[0] for _, v in sorted(seeds.items())])  # [seeds, episodes]
        per_ep[name] = R.mean(0)
        assert R.shape[1] == 5500, "CLINC150 test: 4,500 in-scope episodes, then 1,000 out of scope"
        is_oos = np.r_[np.zeros(4500, bool), np.ones(1000, bool)]
        ev = [v[1]["eval_result"] for _, v in sorted(seeds.items())]
        out["arms"][name] = {"seeds": sorted(seeds), "mean_return": float(R.mean()), "per_seed": [float(x) for x in R.mean(1)],
                             "inscope": float(R[:, ~is_oos].mean()), "oos": float(R[:, is_oos].mean()),
                             "mean_asks": float(np.mean([e["mean_asks"] for e in ev])),
                             "end_step_shares": [float(np.mean([e["end_step_shares"][t] for e in ev])) for t in range(4)],
                             "oos_escalated": float(np.mean([e.get("oos_escalated", np.nan) for e in ev]))}
    # the rule: the better on val of rule and rule2 (deterministic: one entry each)
    vals = {nm: arms[nm][min(arms[nm])][1]["val"]["mean_return_at_oos_share"] for nm in ("rule", "rule2") if nm in arms}
    best_rule = max(vals, key=vals.get) if vals else None
    out["best_rule_on_val"] = {"name": best_rule, "val_at_15pct": vals}
    comps = {"tm_q_nomem-tm_myopic_nomem": ("tm_q_nomem", "tm_myopic_nomem"), "tm_q_nomem-ask_route": ("tm_q_nomem", "ask_route"),
             "tm_q_nomem-best_rule": ("tm_q_nomem", best_rule), "tm_q_nomem-lin_q": ("tm_q_nomem", "lin_q"),
             "tm_q-tm_q_nomem": ("tm_q", "tm_q_nomem"), "lin_q-best_rule": ("lin_q", best_rule)}
    out["diffs"] = {}
    for k, (a, b) in comps.items():
        if a in per_ep and b in per_ep:
            d = per_ep[a] - per_ep[b]
            out["diffs"][k] = {"all": boot(d), "inscope": boot(d[~is_oos]), "oos": boot(d[is_oos])}
    D = out["diffs"]
    rules = {}
    if "tm_q_nomem-tm_myopic_nomem" in D:
        x = D["tm_q_nomem-tm_myopic_nomem"]["all"]
        rules["R8.1"] = x["diff"] >= 0.10 and x["lo"] > 0
    if "tm_q_nomem-ask_route" in D:
        rules["R8.2"] = D["tm_q_nomem-ask_route"]["all"]["lo"] > 0
    rules["R8_met"] = bool(rules.get("R8.1")) and bool(rules.get("R8.2"))
    out["rules"] = rules
    (folder / f"report_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"best_rule_on_val": out["best_rule_on_val"], "rules": rules}, indent=1))
    for name, a in out["arms"].items():
        print(f"{name:14s} {a['mean_return']:+.4f}  in {a['inscope']:+.4f}  oos {a['oos']:+.4f}  asks {a['mean_asks']:.2f}  seeds {len(a['seeds'])}")
    for k, v in D.items():
        print(f"{k:20s} all {v['all']['diff']:+.4f} [{v['all']['lo']:+.4f}, {v['all']['hi']:+.4f}]  in {v['inscope']['diff']:+.4f}  oos {v['oos']['diff']:+.4f}")


if __name__ == "__main__":
    main()
