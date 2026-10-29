#!/usr/bin/env python
"""Step 3c, the near-copy rate (descriptive only; tu/PLAN_3c.md "Bar"; it decides nothing).

    python -I c3_nearcopy.py --kept tu/c3/kept_F.jsonl --out tu/c3/nearcopy_F.json --device cuda

A memorising LLM could write a val text almost word for word, and F1 only drops exact keys and shared 8-grams. So per
set: the share of kept paraphrases whose bge-base cosine (a3_teacher.py's encoder) to some val text of the same set is
at least 0.98, beside the same share for the original training texts (natural near-duplicates between train and val).
Val texts are read without labels. Also the 20 closest paraphrase / val pairs per set, for reading (judged by an AI model,
not a person). Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import a3_teacher as TEA  # noqa: E402
import c3_common as C  # noqa: E402
from dl.data import load  # noqa: E402

LINE = 0.98


def max_cos(a: np.ndarray, b: np.ndarray, device: str):
    B = torch.as_tensor(b.astype(np.float32), device=device)
    best, arg = np.zeros(len(a), np.float32), np.zeros(len(a), np.int64)
    for s in range(0, len(a), 4096):
        S = torch.as_tensor(a[s : s + 4096].astype(np.float32), device=device) @ B.T
        v, i = S.max(1)
        best[s : s + 4096], arg[s : s + 4096] = v.cpu().numpy(), i.cpu().numpy()
    return best, arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    kept = [json.loads(l) for l in args.kept.read_text().splitlines()]
    TEA.verify_files()
    tok, model = TEA.load(args.device, torch.float16 if args.device == "cuda" else torch.float32)
    res = {"kept": str(args.kept), "line": LINE}
    for name in C.SETS:
        ds = load(name)
        va, tr = ds.texts["val"], ds.texts["train"]
        para = [c["text"] for c in kept if c["set"] == name]
        ev, _ = TEA.encode(tok, model, va, args.device, batch=256)
        et, _ = TEA.encode(tok, model, tr, args.device, batch=256)
        bt, _ = max_cos(et, ev, args.device)
        r = {"val": len(va), "train": len(tr), "train_share_ge_line": round(float((bt >= LINE).mean()), 4), "paraphrases": len(para)}
        if para:
            ep, _ = TEA.encode(tok, model, para, args.device, batch=256)
            bp, ip = max_cos(ep, ev, args.device)
            r["paraphrase_share_ge_line"] = round(float((bp >= LINE).mean()), 4)
            top = np.argsort(-bp)[:20]
            r["closest"] = [{"paraphrase": para[i], "val_text": va[ip[i]], "cos": round(float(bp[i]), 4)} for i in top]
        res[name] = r
        print(json.dumps({name: {k: v for k, v in r.items() if k != "closest"}}), flush=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
