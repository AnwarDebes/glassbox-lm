"""Read-only: what is known before e81 (the plan's 'What I knew' table) and the dev-curve swings, through e81_pick's checks."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import json, statistics, sys
from pathlib import Path
S = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(S / "scripts"))
import e81_pick as P
class R0(P.Rows):
    def __init__(self):
        self.frozen = "9999"; self.bits = {}; self.r = {}
        for line in (S / "results/explore/abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)
R = R0()
for agent, a in P.AGENTS.items():
    for ds in ("star", "abcd", "multiwoz"):
        d4 = R.get(a["ref4k"][ds], 1, agent, ds, 4000, 4000)
        c = P.FULL[ds]
        full = [R.get(t, sd, agent, ds, c, c) for t, sd in a["reffull"][ds] if R.r.get((t, sd))]
        tm = sum(d["acc"] for d in full) / len(full)
        ep = [P.epochs(d) for d in full]
        lr = P.lr_acc(R, agent, ds)
        sw = [abs(x["val_acc"] - y["val_acc"]) for d in [d4] + full for x, y in zip(d["history"], d["history"][1:])]
        print(f"{agent:4} {ds:8} 4k seed 1 {d4['acc']:.4f} (best ep {d4['best_epoch']}); {c} clauses, {len(full)} seed(s): best-dev {tm:.4f}, "
              f"final {sum(e[1] for e in ep) / len(ep):.4f}, last-10 {sum(e[2] for e in ep) / len(ep):.4f}; LR "
              + (f"{lr:.4f}; TM-LR {100 * (tm - lr):+.2f}" if lr is not None else "not run")
              + f"; swing between evaluations: median {100 * statistics.median(sw):.2f}, max {100 * max(sw):.2f} points")
