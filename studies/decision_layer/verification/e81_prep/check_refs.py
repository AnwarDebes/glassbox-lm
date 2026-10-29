"""Read-only: check e81's reference rows through e81_pick's own get() (no frozen file yet), LR rows, and queue-log durations."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import json, re, sys
from datetime import datetime
from pathlib import Path
S = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(S / "scripts"))
import e81_pick as P

class R0(P.Rows):
    def __init__(self):
        self.frozen = "9999"  # no e81 row may exist yet; get() refuses e81_ rows written before this
        self.r = {}
        for line in (S / "results/explore/abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)
R = R0()
print("e81 rows now:", sum(1 for (t, s) in R.r if t.startswith("e81")))
for agent, a in P.AGENTS.items():
    for ds in ("star", "abcd", "multiwoz"):
        d = R.get(a["ref4k"][ds], 1, agent, ds, 4000, 4000)
        print(f"ref4k {agent:4} {ds:8} {a['ref4k'][ds]:28} acc {d['acc']:.4f} train_s {d['train_s']} time {d['time']}")
        c = P.FULL[ds]
        for t, sd in a["reffull"][ds]:
            if len(R.r.get((t, sd), [])) == 0:
                print(f"  full {t} seed {sd}: none yet")
                continue
            d = R.get(t, sd, agent, ds, c, c)
            print(f"  full {t:34} seed {sd} acc {d['acc']:.4f} train_s {d['train_s']} time {d['time']}")
for k, n in P.LR.items():
    p = S / "results/explore/dryrun" / n / "seed0.json"
    if p.exists():
        j = json.loads(p.read_text())
        print("LR", k, n, j.get("dev_acc"), j.get("parts"), j.get("encoder"), j.get("C") or j.get("best_c") or {kk: j[kk] for kk in j if "c" in kk.lower()}.keys())
    else:
        print("LR", k, n, "missing")
# queue-log durations
fmt = "%Y-%m-%dT%H:%M:%SZ"
for q in sorted((S / "logs").glob("*_queue.log")):
    lines = q.read_text().splitlines()
    cur = None
    for ln in lines:
        m = re.match(r"(\S+) start \(try (\d+)\): (.*)", ln)
        if m:
            cur = (m.group(1), m.group(3))
            continue
        m = re.match(r"(\S+) exit (\d+) \(try (\d+)\)", ln)
        if m and cur:
            t0, cmd = cur
            sec = (datetime.strptime(m.group(1), fmt) - datetime.strptime(t0, fmt)).total_seconds()
            tg = re.search(r"--(?:tag|name) (\S+)", cmd)
            ds = re.search(r"--dataset (\S+)", cmd)
            cl = re.search(r"--clauses (\d+)", cmd)
            enc = re.search(r"--encoder (\S+)", cmd)
            print(f"{q.name:22} {tg.group(1) if tg else '?':40} {ds.group(1) if ds else '?':8} {cl.group(1) if cl else '-':6} {enc.group(1) if enc else '?':10} exit {m.group(2)} {sec/60:6.1f} min")
            cur = None
