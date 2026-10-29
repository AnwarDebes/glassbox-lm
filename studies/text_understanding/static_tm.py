#!/usr/bin/env python
"""Exploratory run (VAL only): static sentence embeddings as Tsetlin Machine input.

    static_tm.py --dataset clinc150 --encoder potion-8m [--view thermo:8] [--emb-dir DIR]

A static model is a word -> vector table distilled from a transformer; a sentence is the
average of its token vectors (no attention). The TM, the boolean views and the logistic
regression come read-only from the decision-layer study, with the settings of its
exploratory script (scripts/tm_explore.py), so the rows compare with its results.
Test splits are never embedded or read. Appends one JSON line to results.jsonl here.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DL / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import accuracy, auroc, fit_logreg, logits  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "oos_train", "oos_val")  # never test, never oos_test
STATIC = {
    # short name: (HF id, pinned revision)
    "potion-8m": ("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23"),
    "potion-32m": ("minishlab/potion-base-32M", "1e5a03f8eeb2c98b928fbbd846f22f816360919f"),
    "static-retr": ("sentence-transformers/static-retrieval-mrl-en-v1", "f60985c706f192d45d218078e49e5a8b6f15283a"),
}


def embed(dataset: str, enc: str, emb_dir: Path) -> dict[str, np.ndarray]:
    path = emb_dir / f"{dataset}__{enc}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from sentence_transformers import SentenceTransformer

    hf_id, rev = STATIC[enc]
    model = SentenceTransformer(hf_id, device="cpu", revision=rev)
    ds = load(dataset)
    out: dict[str, np.ndarray] = {}
    for split in SPLITS:
        if split in ds.texts:
            out[split] = model.encode(ds.texts[split], batch_size=1024, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    for split in ("train", "val"):
        out[f"y_{split}"] = ds.labels[split]
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def best_logreg(Xtr, ytr, Xva, yva, C, grid):
    best = (-1.0, None)
    for Cc in grid:
        W, b = fit_logreg(Xtr.astype(np.float32), ytr, C, Cc)
        acc = accuracy(logits(Xva.astype(np.float32), W, b), yva)
        if acc > best[0]:
            best = (acc, Cc)
    return round(best[0], 4), best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True, choices=list(STATIC))
    ap.add_argument("--view", default="thermo:8")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    d = embed(args.dataset, args.encoder, args.emb_dir)
    if args.embed_only:
        print(json.dumps({"dataset": args.dataset, "encoder": args.encoder, "dim": int(d["train"].shape[1]),
                          "embed_s": round(time.time() - t0, 1)}))
        return
    ds = load(args.dataset)
    C = len(ds.intents)
    ytr, yva = d["y_train"], d["y_val"]
    view = make_view(args.view, seed=args.seed)
    view.fit(d["train"], ytr, texts=ds.texts["train"], class_names=ds.intents)
    Btr, Bva = view.transform(d["train"], ds.texts["train"]), view.transform(d["val"], ds.texts["val"])
    cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.epochs, batch_size=128,
                   neg_sampling="hardest", negation=True, max_literals=0, budget_mode="exact",
                   budget_select="threshold", max_negated=0, clause_drop=0.0, half=True,
                   seed=args.seed, eval_every=2)
    tm = make_tm(C, Btr.shape[1], cfg, "half")
    tm.fit(Btr, ytr, Bva, yva)
    rec = {
        "tag": args.tag, "dataset": args.dataset, "encoder": args.encoder, "view": args.view,
        "dim": int(d["train"].shape[1]), "n_bits": int(Btr.shape[1]), "cfg": asdict(cfg),
        "val_acc": round(tm.best_val, 4), "best_epoch": getattr(tm, "best_epoch", None),
        "train_s": round(tm.train_seconds, 1),
        "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats(),
    }
    rec["bits_logreg_val"], rec["bits_logreg_C"] = best_logreg(Btr, ytr, Bva, yva, C, [1.0, 10.0, 100.0])
    rec["raw_logreg_val"], rec["raw_logreg_C"] = best_logreg(d["train"], ytr, d["val"], yva, C,
                                                             [1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    if "oos_val" in d:
        Xin = torch.as_tensor(Bva, dtype=torch.uint8, device="cuda")
        Xoo = torch.as_tensor(view.transform(d["oos_val"], ds.texts["oos_val"]), dtype=torch.uint8, device="cuda")
        S = torch.cat([tm.class_sums(Xin), tm.class_sums(Xoo)]).cpu().numpy()
        is_oos = np.r_[np.zeros(len(Bva)), np.ones(len(d["oos_val"]))]
        rec["oos_auroc_val_max_sum"] = round(auroc(is_oos, -np.sort(S, 1)[:, -1]), 4)
    rec["total_s"] = round(time.time() - t0, 1)
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats")}))


if __name__ == "__main__":
    main()
