#!/usr/bin/env python
"""e87 A0 (plan logs/plan_e87_receipts.md): the C4 agent's train and dev bits, and the L2 LR on them, saved for the
CPU readability step (scripts/e87_readability.py). GPU: the view and the LR run on cuda, as in the registered runs.
No test file is read.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/e87_bits.py --dataset abcd

The view is built as scripts/table_run.py + scripts/abcd_explore.py build it: count_text.patch on the table
c4words-S1000-v1, then ABCDView(parts mem, words, xcust, xrecent; lex_k 1000; seed 0) fitted on train, with encoder
none. Checks:
- the train and dev bits' SHA-256 equal the registered C4 arm's (bits_sha256 in its seed1.json: R24mwoz, R22star,
  R23abcd; only that key and the dev bits are read from it);
- LR refitted on the train bits at the C of the dataset's LR dev row (results/explore/dryrun/<row>/seed0.json) gives
  exactly that row's dev accuracy.
Writes results/explore/e87/<dataset>_c4.npz and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text  # noqa: E402

TABLE = "c4words-S1000-v1"
PARTS = ("mem", "words", "xcust", "xrecent")
REG = {"multiwoz": "results/confirm/R24mwoz/tm_mem_c4_nrand", "star": "results/confirm/R22star/tm_mem_c4_s16",
       "abcd": "results/confirm/R23abcd/tm_mem_c4_s16"}
LR_ROW = {"multiwoz": "dry_e77mwoz_lr_mem_c4", "star": "dry_R21star_lr_mem_c4", "abcd": "dry_R21abcd_lr_mem_c4"}


def bits_sha256(X: np.ndarray) -> str:
    """As scripts/abcd_explore.py: SHA-256 of the uint8 bit matrix in C order."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(REG))
    a = ap.parse_args()
    out = ROOT / "results" / "explore" / "e87" / f"{a.dataset}_c4.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    meta = count_text.patch(TABLE)
    from dl.abcd import ABCDView, load_examples
    from dl.linear import fit_logreg, logits

    t0 = time.time()
    tr, blist = load_examples("train", a.dataset)
    dv, _ = load_examples("dev", a.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{a.dataset}__none.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, em["train"], blist)
    Xtr, Xdv = view.transform(tr, em["train"]), view.transform(dv, em["dev"])
    ytr, ydv = np.array([r["y"] for r in tr]), np.array([r["y"] for r in dv])
    got = {"train": bits_sha256(Xtr), "dev": bits_sha256(Xdv)}
    reg = json.loads((ROOT / REG[a.dataset] / "seed1.json").read_text())["bits_sha256"]
    if (got["train"], got["dev"]) != (reg["train"], reg["dev"]):
        raise SystemExit(f"bits differ from the registered C4 arm's: train {got['train'][:8]} / {reg['train'][:8]}, "
                         f"dev {got['dev'][:8]} / {reg['dev'][:8]}")
    row = json.loads((ROOT / "results" / "explore" / "dryrun" / LR_ROW[a.dataset] / "seed0.json").read_text())
    if row["arm"] != "lr" or row["bits_sha256"]["dev"] != got["dev"]:
        raise SystemExit(f"{LR_ROW[a.dataset]}: not LR on these dev bits")
    W, b = fit_logreg(Xtr, ytr, len(blist), row["C"])
    pred = logits(Xdv, W, b).argmax(1).cpu().numpy()
    acc = float((pred == ydv).mean())
    if acc != row["dev_acc"]:
        raise SystemExit(f"LR refit at C {row['C']}: dev accuracy {acc!r}, the row has {row['dev_acc']!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_train=Xtr.astype(np.uint8), y_train=ytr, X_dev=Xdv.astype(np.uint8), y_dev=ydv,
                        names=np.array(view.names), buttons=np.array(blist), lr_W=W.cpu().numpy(), lr_b=b.cpu().numpy(),
                        lr_C=np.float64(row["C"]), lr_pred_dev=pred, bits_sha256=json.dumps(got), table=json.dumps(meta))
    print(json.dumps({"e87_bits": a.dataset, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
                      "train": list(Xtr.shape), "dev": list(Xdv.shape), "density_train": round(float(Xtr.mean()), 4),
                      "bits_sha256": got, "lr_C": row["C"], "lr_dev_acc": acc, "out": str(out.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
