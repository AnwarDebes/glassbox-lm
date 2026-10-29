#!/usr/bin/env python
"""M1 code check input (tu/PLAN_M1.md; no test text or label is read): 3c's CLINC150 rebuilt-original cache
(tu/c3/feats/orig) copied into M1's format, with "test" := a copy of val. m1_tm.py and m1_lr.py on it, with the stand-in
dataset m1_standin.py:load, must give 3c's 3c-ref (seed 4, s 8) and ref_rebuilt rows exactly on val, and the same
number again on this "test".

    python -I -B m1_check_cache.py --src tu/c3/feats/orig --out tu/m1/check

Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="clinc150")
    a = ap.parse_args()
    name = f"{a.dataset}__p32tok_K512"
    dst, dmeta = a.out / f"{name}.npz", a.out / f"{name}.meta.json"
    for p in (dst, dmeta):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    z = np.load(a.src / f"{name}.npz")
    assert not any(k.startswith("test__") for k in z.files) and "y_test" not in z.files, "the source has test rows"
    out = {k: z[k] for k in z.files}
    for k in z.files:
        if k.startswith("val__"):
            out["test__" + k[len("val__"):]] = z[k]
    out["y_test"] = z["y_val"]
    meta = json.loads((a.src / f"{name}.meta.json").read_text())
    meta["m1_code_check"] = {"src": str(a.src), "test_is_a_copy_of_val": True}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    dmeta.write_text(json.dumps(meta))
    print(json.dumps({"out": str(dst), "keys": len(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
