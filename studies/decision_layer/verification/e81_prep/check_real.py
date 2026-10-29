"""Read-only: the real reference rows and LR rows through e81_pick's checks (no frozen file yet, no e81 rows yet)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import json, sys
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
n = 0
for agent, a in P.AGENTS.items():
    for ds in ("star", "abcd", "multiwoz"):
        R.get(a["ref4k"][ds], 1, agent, ds, 4000, 4000); n += 1
        for t, sd in a["reffull"][ds]:
            if R.r.get((t, sd)):
                R.get(t, sd, agent, ds, P.FULL[ds], P.FULL[ds]); n += 1
        lr = P.lr_acc(R, agent, ds)
        print(f"{agent:4} {ds:8} bits {R.bits[(agent, ds)][:8]} LR {lr if lr is None else round(lr, 4)}")
today = [R.get(t, sd, "a3B", "multiwoz", 16000, 16000)["acc"] for t, sd in P.AGENTS["a3B"]["reffull"]["multiwoz"]]
assert P.lr_acc(R, "a3B", "multiwoz") == P.B2_LR and (today[0] + today[1]) / 2 == P.B2_TODAY
print(f"{n} reference row reads passed every check; B2 constants equal the rows; B2 today {100 * (P.B2_TODAY - P.B2_LR):+.3f}, bar {50 * (P.B2_TODAY - P.B2_LR):+.3f}")
