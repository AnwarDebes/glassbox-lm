#!/usr/bin/env python
"""Benchmark BC4 (MILP verdicts): e72e's 50 property checks per model on R23's five ABCD C4 models (s 16), rerun
through the decision_layer UNEDITED scripts/verify_policy_c4.py with only its two module paths set, as the decision_layer own thin wrapper
scripts/verify_policy_c4_e72e.py does: SRC = results/confirm/R23abcd/tm_mem_c4_s16 (read-only) and OUT = --out (a
fresh file in glassbox-lm/studies/decision_layer/speed; never the decision_layer results/explore/verify_e72e.jsonl). Same settings as e72e's runs: tag
e72e, --threads 1, time limit 600 s, the default 500 / 50 votes. Dev and proof work only (no test split is read).

    python -I -B milp_bench.py --out OUT.jsonl [--guard] [--seeds 1 2 3 4 5]

Then: fingerprint.py milp --rows OUT.jsonl --tag e72e --guarded yes|no [--seeds ...] and compare.py against the
baseline fingerprint of the stored rows (out/baseline/M1.json guarded, M2.json raw).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import sys
from pathlib import Path

DL = Path(f"{_REPO_ROOT}/studies/decision_layer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    out = a.out.resolve()
    if DL in out.parents:
        raise SystemExit("refused: --out must be outside the decision_layer study folder")
    if out.exists() and out.stat().st_size > 0:
        raise SystemExit(f"refused: {out} exists (the checker skips rows already in its OUT file; use a fresh file)")
    out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DL / "src"))
    sys.path.insert(1, str(DL / "scripts"))
    import verify_policy_c4 as vpc  # the decision_layer file, imported read-only (run with -B: no .pyc written)

    vpc.SRC = DL / "results" / "confirm" / "R23abcd" / "tm_mem_c4_s16"
    vpc.OUT = out
    sys.argv = ["verify_policy_c4.py", "--check", "properties", *(["--guard"] if a.guard else []),
                "--seeds", *map(str, a.seeds), "--threads", "1", "--tag", "e72e"]
    vpc.main()


if __name__ == "__main__":
    main()
