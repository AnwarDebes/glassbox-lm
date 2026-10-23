#!/usr/bin/env python
"""e71 v2 step 2 on a fixed sample (dated note logs/plan_e71_v2_note2_2026-10-19.md; decided 10:17 UTC): k = 3 and 5 by
the decision-version MILP for 50 sampled (state, group) pairs certified at k = 2 (logs/e71_k35_sample.json).

    e71_k35_sample.py --seed 1          # tag e71v2_k35, as verify_enum.py --check milp35

The same call as verify_enum.py's milp35 mode (verify_local.solve_group with budget = k, the reduced exact encoding with its
bigram superset, 120 s, 2 threads), on the sampled pairs only.
- k = 5 runs only where k = 3 is certified. That also holds on a resume: an existing k = 3 row that is not certified
  stops the pair. (verify_enum's resume would go on to k = 5; that path never ran: the milp35 run stopped at 10:18:03
  with 0 rows.)
- A pair that already has a row for a budget is not rerun.
- Rows carry "sampled": true and the sample's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import cpu_tm_fast  # noqa: E402
from verify_local import OUT, TextModel, groups_for, solve_group  # noqa: E402
from verify_policy import load_tm  # noqa: E402

SAMPLE = ROOT / "logs" / "e71_k35_sample.json"
SAMPLE_SHA = "95d3dbbfa08fde2cd2bb4bfedb2017ce1eaa5ca3dca23d6a823a5945306557c6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="e71v2_k35")
    args = ap.parse_args()
    if hashlib.sha256(SAMPLE.read_bytes()).hexdigest() != SAMPLE_SHA:
        raise SystemExit("the sample file is not the one in the dated note")
    if not (ROOT / "logs" / "plan_e71_v2_note2_2026-10-19.frozen.json").exists():
        raise SystemExit("the sampled MILP runs start only after the dated note is frozen")
    from lineb_counts import properties  # noqa: E402

    sample = [tuple(p) for p in json.loads(SAMPLE.read_text())["sample"]]
    tmod = TextModel()
    tm = load_tm(args.seed)
    m = tm["pack"]
    props = properties(tmod.blist)
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    cert2 = {(d["state"], d["group"]) for d in rows if d.get("tag") == "e71v2" and d.get("kind") == "enum" and d.get("seed") == args.seed
             and d.get("outcome") == "certified at k = 2"}
    assert all(p in cert2 for p in sample), "a sampled pair is not certified at k = 2"
    done = {(d["state"], d["group"], d["budget"]): d for d in rows if d.get("tag") == args.tag and d.get("kind") == "milp_k" and d.get("seed") == args.seed}
    for i, group in sample:
        r = tmod.dv[i]
        x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
        g = next(g for g in groups_for(r, props, tmod.blist) if g["group"] == group)
        for k in (3, 5):
            prev = done.get((i, group, k))
            if prev is not None:
                if prev["outcome"] != f"certified at k = {k}":
                    break  # not certified at k: no larger k
                continue
            res = solve_group(tmod, tm, r, x0, g["buttons"], g["against"], args.time_limit, args.threads, budget=k)
            rec = {"tag": args.tag, "kind": "milp_k", "model": "tm", "seed": args.seed, "state": i, "group": group, "budget": k,
                   "sampled": True, "sample_sha256": SAMPLE_SHA,
                   **{a: res.get(a) for a in ("status", "seconds", "edits", "inserted", "deleted", "n_rows", "nnz", "nodes", "solver")},
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if res.get("edits") is not None:
                r2 = tmod.edited_row(r, res["keep"], res["inserted"])
                x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                c2 = int(cpu_tm_fast.class_sums(m, x2[None])[0].argmax())
                rec.update(replay_choice=tmod.blist[c2], replay_breaks_rule=bool(c2 in g["buttons"]))
            rec["outcome"] = {"infeasible": f"certified at k = {k}", "optimal": f"an edit set of size <= {k} found",
                              "time limit": "unknown"}.get(res["status"], res["status"])
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({a: rec.get(a) for a in ("state", "group", "budget", "outcome", "seconds", "replay_breaks_rule")}), flush=True)
            if res["status"] != "infeasible":
                break  # not certified at 3: k = 5 cannot be certified either


if __name__ == "__main__":
    main()
