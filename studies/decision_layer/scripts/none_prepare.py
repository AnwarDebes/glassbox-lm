#!/usr/bin/env python
"""An empty encoder cache: data/emb/<dataset>__none.npz (added 2026-10-19 for e72; plan logs/plan_e72_count_table.md).

It has the 9 keys of the encoder caches ({train,dev,test}__{cust,recent,full}), each an empty float32 array of shape
(0, 0), so that a run with ``--encoder none`` loads no encoder vector of any kind. A view part that needs one (intent,
step, fstep) then fails. No text is read. Refuses to overwrite.

    none_prepare.py --dataset abcd star
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["abcd", "star"])
    args = ap.parse_args()
    for ds in args.dataset:
        p = ROOT / "data" / "emb" / f"{ds}__none.npz"
        if p.exists():
            raise SystemExit(f"{p} exists; not overwriting")
        arrays = {f"{s}__{f}": np.zeros((0, 0), np.float32) for s in ("train", "dev", "test") for f in ("cust", "recent", "full")}
        np.savez(p, **arrays)
        print(p.relative_to(ROOT), hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
