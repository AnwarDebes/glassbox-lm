#!/usr/bin/env python
"""Step 3b, P0-T: the one declared revision round after the P0 NO-GO (post hoc, dev only, C4 only; the message of 2026-10-21 09:42Z; the frozen note tu/b3/NOTE_2026-10-21_P0T.md says what it may and may not do).

    python -I -B b3_p0t.py --p0 tu/b3/p0 [--T 22,60,200]

For each T of --T, b3_student.py trains on P0's own 64-bit files (train64.npz, held64.npz) with everything else as P0's
s 8 row: 1,000 clauses, s 8, R 4, batch 1,024, 3 epochs, seed 1, CPU. T 1,000 is P0's existing s 8 row (tm_s8.json),
read, not rerun. The pick is the T with the best final-epoch mean balanced bit accuracy on the C4 held-out file; the GO
rule is P0's, unchanged: the pick must reach the ridge twin's balanced accuracy minus 2.0 (twins.twins.json). Writes
p0t_verdict.json beside P0's files, with every T's curve per epoch; a T whose record exists is not rerun; nothing is
overwritten.
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
GO_MARGIN = 2.0


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p0", type=Path, required=True)
    ap.add_argument("--T", default="22,60,200")
    a = ap.parse_args()
    out = a.p0 / "p0t_verdict.json"
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    v = json.loads((a.p0 / "p0_verdict.json").read_text())
    nb, clauses, ep = v["n_bits"], v["clauses"], v["epochs"]
    rows = {str(clauses): json.loads((a.p0 / "tm_s8.json").read_text())}
    assert rows[str(clauses)]["cfg"]["T"] == clauses and rows[str(clauses)]["cfg"]["s"] == 8.0, "P0's s 8 row is not T = clauses"
    for T in a.T.split(","):
        tag = f"tm_T{T}_s8"
        js = a.p0 / f"{tag}.json"
        if not js.exists():
            cmd = [sys.executable, "-I", "-B", str(HERE / "b3_student.py"), "train", "--train", str(a.p0 / f"train{nb}.npz"),
                   "--held", str(a.p0 / f"held{nb}.npz"), "--out-dir", str(a.p0), "--tag", tag, "--clauses", str(clauses),
                   "--T", T, "--s", "8", "--R", "4", "--epochs", str(ep), "--batch", "1024", "--seed", "1", "--device", "cpu"]
            with open(a.p0 / "p0.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} run (P0-T): {' '.join(cmd)}\n")
                f.flush()
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} exit {p.returncode}\n")
            if p.returncode != 0:
                print(f"P0-T run T {T} failed (exit {p.returncode}); see {a.p0 / 'p0.log'}", file=sys.stderr)
                return 1
        rows[T] = json.loads(js.read_text())
    tw = json.loads((a.p0 / "twins.twins.json").read_text())
    lin = tw["linear"]["held"]["balanced_acc"]
    final = {T: r["curve"][-1]["balanced_acc"] for T, r in rows.items()}
    pick = max(final, key=lambda T: final[T])
    res = {"label": "P0-T, post hoc after the P0 NO-GO, dev only, C4 only", "T_grid": list(rows), "pick_T": pick,
           "pick_balanced_acc": final[pick], "linear_balanced_acc": lin, "counting_balanced_acc": tw["counting"]["held"]["balanced_acc"],
           "difference_pick_minus_linear": round(final[pick] - lin, 3), "go": bool(final[pick] >= lin - GO_MARGIN - 1e-9),
           "rule": f"GO when the best T's final balanced accuracy >= ridge - {GO_MARGIN} (P0's rule, unchanged)",
           "curves": {T: r["curve"] for T, r in rows.items()}, "seconds": {T: r["seconds"] for T, r in rows.items()},
           "settings": {"clauses": clauses, "s": 8, "R": 4, "batch": 1024, "epochs": ep, "seed": 1, "bits": nb},
           "inputs_sha16": {k: sha16(a.p0 / k) for k in ("p0_verdict.json", "twins.twins.json", "tm_s8.json", f"train{nb}.npz", f"held{nb}.npz")},
           "b3_p0t.py": sha16(Path(__file__).resolve()), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("pick_T", "pick_balanced_acc", "linear_balanced_acc", "difference_pick_minus_linear", "go")}))
    print(json.dumps({T: [c["balanced_acc"] for c in r] for T, r in res["curves"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
