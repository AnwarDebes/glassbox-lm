#!/usr/bin/env python
"""Write logs/jobs_R6.txt (one r6_seed.sh line per seed) from the R6 selection."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sel = json.loads(subprocess.run([sys.executable, str(ROOT / "scripts" / "select_r5r6.py"), "R6"], capture_output=True, text=True, check=True).stdout)
tq_tag, tq, _ = sel["tm_q"]["chosen"]
lq_tag, lq, _ = sel["lin_q"]["chosen"]
def args(d, extra):
    a = []
    if d.get("fqi"):
        a += ["--fqi", f"--fqi-iters {d['fqi_iters']}", f"--fqi-epochs {d['fqi_epochs']}"]
    else:
        a += [f"--updates {d['updates']}"]
    a += [f"--{k} {d[k]}" for k in extra if d.get(k) is not None]
    return " ".join(a)
tq_args = args(tq, ["kappa", "clauses", "T", "s"])
lq_args = args(lq, ["lr"])
lines = [f'scripts/r6_seed.sh {s} "{tq_args}" "{lq_args}"' for s in range(1, 11)]
(ROOT / "logs" / "jobs_R6.txt").write_text("\n".join(lines) + "\n")
print("TM-Q from", tq_tag, "->", tq_args)
print("Lin-Q from", lq_tag, "->", lq_args)
