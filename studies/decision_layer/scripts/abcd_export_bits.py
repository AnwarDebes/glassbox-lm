#!/usr/bin/env python
"""Export the readable ABCD view as bit matrices (for learners in other environments, e.g. the
hierarchical TM package in .venv-htm). Writes data/abcd/bits_<name>_<split>.npz with X (uint8),
y, convo_id, k, the AST weights, the bit names and the block sizes of the view's parts.
The view is fitted on train only, as everywhere else."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl.abcd import ABCDView, load_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="mem,intent,step,words")
    ap.add_argument("--name", default="short")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--encoder", default="bge-large")
    args = ap.parse_args()
    parts = tuple(args.parts.split(","))
    tr, blist = load_examples("train")
    E = np.load(ROOT / "data" / "emb" / f"abcd__{args.encoder}.npz")
    em = lambda s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")}  # noqa: E731
    view = ABCDView(parts=parts, seed=0).fit(tr, em("train"), blist)
    sizes = []
    for p in parts:
        one = ABCDView(parts=(p,), seed=0).fit(tr, em("train"), blist)
        sizes.append(len(one.names))
    assert sum(sizes) == len(view.names)
    for s in args.splits:
        rows = tr if s == "train" else load_examples(s)[0]
        X = view.transform(rows, em(s))
        np.savez_compressed(ROOT / "data" / "abcd" / f"bits_{args.name}_{s}.npz", X=X, y=np.array([r["y"] for r in rows]),
                            convo_id=np.array([r["convo_id"] for r in rows]), k=np.array([r["k"] for r in rows]),
                            w_bert=np.array([r["ast_count_bert"] for r in rows]), names=np.array(view.names),
                            sizes=np.array(sizes), parts=np.array(parts), buttons=np.array(blist))
        print(s, X.shape, sizes)


if __name__ == "__main__":
    main()
