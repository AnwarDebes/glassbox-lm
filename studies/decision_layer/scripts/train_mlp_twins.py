#!/usr/bin/env python
"""Line B (plan logs/plan_e70_verify.md, section "The MLP part"): the MLP twins of R19abcd's TM, for the
verification comparison. DEV only: trained on train, best dev epoch; the test split is not read.

    train_mlp_twins.py --hidden 64 128 512 --seed 1

dl.mlp.fit_mlp (one hidden ReLU layer, dropout 0.2, AdamW, best dev epoch; the study's MLP head) on view B's bits
(mem, words, xcust, xrecent; potion-32m), whose dev hash must equal R19's. Writes
results/explore/mlp_twins/mlp_h<H>_seed<k>.npz (W1, b1, W2, b2 as float32; eval mode) and .json (dev accuracy).
Refuses to overwrite. Added 2026-10-19 after the plan's freeze (a recorded deviation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402
from dl.mlp import fit_mlp, mlp_scores  # noqa: E402

DEV_SHA = "2f3223b2d9ca726ce7102990fca2cba1e0cbcb5ec5cf4cd727095b99503ce343"
TRAIN_SHA = "1f81cb23d740ea066e5aaf0f7e65a2ce995ce85fd9b7434dcacf8c9e3095cc6c"


def sha(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 128, 512])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.15)
    out = ROOT / "results" / "explore" / "mlp_twins"
    out.mkdir(parents=True, exist_ok=True)
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    Xtr, Xdv = view.transform(tr, em["train"]), view.transform(dv, em["dev"])
    assert sha(Xtr) == TRAIN_SHA and sha(Xdv) == DEV_SHA, "bits differ from R19's view B"
    assert view.names == json.loads((ROOT / "results/confirm/R19abcd/tm_mem_static/bit_names.json").read_text())
    ytr, ydv = np.array([r["y"] for r in tr]), np.array([r["y"] for r in dv])
    for h in args.hidden:
        f = out / f"mlp_h{h}_seed{args.seed}.npz"
        if f.exists():
            raise SystemExit(f"exists: {f}")
        t0 = time.time()
        net, info = fit_mlp(Xtr, ytr, Xdv, ydv, len(blist), hidden=h, seed=args.seed)
        lin1, lin2 = net[0], net[3]
        W1, b1 = lin1.weight.detach().cpu().numpy(), lin1.bias.detach().cpu().numpy()
        W2, b2 = lin2.weight.detach().cpu().numpy(), lin2.bias.detach().cpu().numpy()
        dev_acc = float((mlp_scores(net, Xdv).argmax(1).cpu().numpy() == ydv).mean())
        np.savez_compressed(f, W1=W1, b1=b1, W2=W2, b2=b2)
        rec = {"script": "train_mlp_twins.py", "hidden": h, "seed": args.seed, "dev_acc": dev_acc, **info,
               "train_s": round(time.time() - t0, 1), "bits_sha256": {"train": TRAIN_SHA, "dev": DEV_SHA},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        f.with_suffix(".json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
