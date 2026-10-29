#!/usr/bin/env python
"""Step 3b, P0's capacity-matched line (descriptive; the note of 07:50Z): P0 is conservative in feedback per
output (about 18,750 row-feedbacks against about 104,000 in the GPU job) but generous in clauses per output (1,000 for
64 outputs = 15.6, against 4,000 for 768 = 5.2). This run gives the 64 outputs the GPU job's ratio: --clauses (default
333) and T = clauses, at P0's best s, on P0's own 64-bit files, with P0's R, batch, epochs and seed.

    python -I -B b3_p0_capacity.py --p0 tu/b3/p0 [--clauses 333]

Reads p0_verdict.json (best s, epochs, the 1,000-clause line and the ridge line), runs b3_student.py, and writes
p0_capacity_check.json beside them; never overwrites. Its reading line states both sides: "conservative in feedback per
output, generous in clauses per output", and the capacity-matched result next to the main one. If this line falls well
below the main one, the GPU job's clause count is raised, or the reason not to is written, before the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p0", type=Path, required=True)
    ap.add_argument("--clauses", type=int, default=333)
    a = ap.parse_args()
    out = a.p0 / "p0_capacity_check.json"
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    v = json.loads((a.p0 / "p0_verdict.json").read_text())
    s, nb, ep = v["best_s"], v["n_bits"], v["epochs"]
    tag = f"tm_cap{a.clauses}_s{s}"
    js = a.p0 / f"{tag}.json"
    if not js.exists():
        cmd = [sys.executable, "-I", "-B", str(HERE / "b3_student.py"), "train", "--train", str(a.p0 / f"train{nb}.npz"),
               "--held", str(a.p0 / f"held{nb}.npz"), "--out-dir", str(a.p0), "--tag", tag, "--clauses", str(a.clauses),
               "--T", str(a.clauses), "--s", s, "--R", "4", "--epochs", str(ep), "--batch", "1024", "--seed", "1", "--device", "cpu"]
        with open(a.p0 / "p0.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} run: {' '.join(cmd)}\n")
            f.flush()
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} exit {p.returncode}\n")
        if p.returncode != 0:
            print(f"the capacity run failed (exit {p.returncode}); see {a.p0 / 'p0.log'}", file=sys.stderr)
            return 1
    r = json.loads(js.read_text())
    cap, main_ = r["curve"][-1], v["tm_by_s"][s]
    per_out_main, per_out_cap = v["clauses"] / nb, a.clauses / nb
    res = {"clauses": a.clauses, "outputs": nb, "clauses_per_output": round(per_out_cap, 2), "s": s, "epochs": ep,
           "capacity_matched": cap, "seconds": r["seconds"], "main": {"clauses": v["clauses"], "clauses_per_output": round(per_out_main, 2),
                                                                         "held": main_},
           "linear": v["linear"], "gpu_job_clauses_per_output": round(4000 / 768, 2),
           "difference_cap_minus_main": round(cap["balanced_acc"] - main_["balanced_acc"], 3),
           "difference_cap_minus_linear": round(cap["balanced_acc"] - v["linear"]["balanced_acc"], 3),
           "reading": (f"P0 is conservative in feedback per output (about 18,750 row-feedbacks against about 104,000 in the GPU "
                       f"job) and generous in clauses per output ({per_out_main:.1f} against {4000 / 768:.1f}); at the GPU job's "
                       f"ratio ({per_out_cap:.1f}) the TM student's balanced bit accuracy is {cap['balanced_acc']} against "
                       f"{main_['balanced_acc']} at {v['clauses']} clauses and {v['linear']['balanced_acc']} for ridge"),
           "inputs_sha16": {"p0_verdict.json": sha16(a.p0 / "p0_verdict.json"), f"train{nb}.npz": sha16(a.p0 / f"train{nb}.npz"),
                            f"held{nb}.npz": sha16(a.p0 / f"held{nb}.npz")},
           "b3_p0_capacity.py": sha16(Path(__file__).resolve()), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(res["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
