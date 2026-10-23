#!/usr/bin/env python
"""e77 smoke test (CPU; no model is trained, no score is computed): the C4 agent's view B on MultiWOZ train and dev,
built exactly as abcd_explore.py builds it under table_run.py (dl.count_text.patch on c4words-S1000-v1, the empty
"none" cache, ABCDView(parts mem,words,xcust,xrecent, lex_k 1000, seed 0) fit on train). Prints the bit count, the
bits' SHA-256 per split (abcd_explore.py's bits_sha256, so the e77a rows can be compared) and the share of rows with
no bit set per part. Added 2026-10-19.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text  # noqa: E402


def main() -> None:
    t0 = time.time()
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", "multiwoz")
    dv, _ = load_examples("dev", "multiwoz")
    E = np.load(ROOT / "data" / "emb" / "multiwoz__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], blist)
    X = {"train": view.transform(tr, emb["train"]), "dev": view.transform(dv, emb["dev"])}
    names = np.array(view.names)
    out = {"what": "e77 view smoke test (C4 view B, MultiWOZ)", "table": meta.get("name", "c4words-S1000-v1"), "n_bits": int(X["train"].shape[1]),
           "n_labels": len(blist), "bits_sha256": {s: hashlib.sha256(np.ascontiguousarray(x, dtype=np.uint8).tobytes()).hexdigest() for s, x in X.items()},
           "static_info": view.static_info(), "per_part": {}}
    for part in ("mem", "words", "xcust", "xrecent"):
        idx = [i for i, n in enumerate(names) if str(n).startswith(part) or (part == "mem" and str(n).split("(")[0] in ("done", "last", "k>="))]
        out["per_part"][part] = {"n_bits_name_prefix": len(idx)}
    out["first_names"] = [str(n) for n in names[:5]] + ["..."] + [str(n) for n in names[-5:]]
    out["mean_bits_on"] = {s: round(float(x.sum(1).mean()), 1) for s, x in X.items()}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
