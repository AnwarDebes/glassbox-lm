"""Descriptive, call-level: the deterministic arms (W1-W2, D2, D3, D3r) on labelled rows (no pipeline, no LLM).
On ground-truth rows this is scripted behaviour: plumbing, not a result. Pure Python (either venv):
    python -I -B scripts/rule_rows.py --rows data/rows_gt_v1.2.2.jsonl --out results/rule_rows_gt.json"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
from e2guard.facts import ToolTable, facts_from_json
from e2guard.rules import D2Rules, D3Rope, decide_d3, decide_w

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tasks = json.loads((E2 / "data" / "tasks_v1.2.2.json").read_text())
T = ToolTable()
d2, d3r = D2Rules(), D3Rope(T)
cnt = defaultdict(Counter)
named_attack = Counter()
for line in Path(a.rows).read_text().splitlines():
    r = json.loads(line)
    if r["label"] not in ("ALLOW", "DENY") or r["facts"]["kind"] is None:
        continue
    f = facts_from_json(r["facts"])
    on = set(r["facts_on"])
    q = tasks[r["suite"]]["user_tasks"][r["user_task"]]["prompt"]
    ctx = "benign" if r["injection_task"] is None else "attacked"
    for arm, (allow, _, _) in {"W1-W2": decide_w(f, on), "D2": d2.decide(f, on, q), "D3": decide_d3(f, on),
                               "D3r": d3r.decide(f, on)}.items():
        for scope in ("ALL", r["suite"]):
            cnt[(arm, scope)][f"{r['label']}|{ctx}|{'blocked' if not allow else 'allowed'}"] += 1
    if r["label"] == "DENY":
        av = set(r.get("attacker_values") or [])
        provs = {t.prov for x in f.args for t in x.atoms if t.shown in av}
        for scope in ("ALL", r["suite"]):
            named_attack[(scope, "attacker value in a named source" if "named_source" in provs else
                          ("attacker value in the query" if "user_query" in provs else "other"))] += 1
res = {"note": "ground-truth replay rows (scripted behaviour): call-level counts; plumbing, not an E2 result",
       "arms": {f"{arm}|{scope}": dict(c) for (arm, scope), c in sorted(cnt.items())},
       "deny_rows_by_attacker_value_provenance": {f"{s}|{k}": v for (s, k), v in sorted(named_attack.items())}}
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res["arms"].items():
    if k.endswith("|ALL"):
        al = v.get("ALLOW|benign|blocked", 0) + v.get("ALLOW|attacked|blocked", 0)
        an = al + v.get("ALLOW|benign|allowed", 0) + v.get("ALLOW|attacked|allowed", 0)
        dl = v.get("DENY|attacked|blocked", 0)
        dn = dl + v.get("DENY|attacked|allowed", 0)
        print(f"{k:12} user-task calls blocked {al}/{an} ({al/an:.3f}); injection calls blocked {dl}/{dn} ({dl/dn:.3f})")
print({k: v for k, v in res["deny_rows_by_attacker_value_provenance"].items()})
