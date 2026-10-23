"""Cache of the readable view's bits for train and dev (fitting the view costs minutes of CPU).

``view_bits(dataset, parts)`` fits ``ABCDView`` on train exactly as scripts/abcd_explore.py and
scripts/abcd_confirm.py do (lex_k 1000, seed 0, bge-large) and stores the bits of train and dev
in data/<dataset>/bits_<parts>_<split>.npz. The test split is never cached here: registered runs
build their own view (scripts/abcd_confirm.py). Added 2026-10-17 after R18, for dev exploration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dl.abcd import ABCDView, load_examples

ROOT = Path(__file__).resolve().parents[2]


def _save(path: Path, **arrays) -> None:
    """Write atomically (temp file, then rename), so two jobs that build the same cache cannot corrupt it."""
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def view_bits(dataset: str, parts=("mem", "intent", "step", "words"), encoder: str = "bge-large", lex_k: int = 1000):
    """Returns (X_train, y_train, X_dev, y_dev, names, buttons, rows_train, rows_dev)."""
    tag = "-".join(parts) + ("" if encoder == "bge-large" else f"__{encoder}") + ("" if lex_k == 1000 else f"__lex{lex_k}")
    d = ROOT / "data" / dataset
    paths = {s: d / f"bits_{tag}_{s}.npz" for s in ("train", "dev")}
    tr, blist = load_examples("train", dataset)
    dv, _ = load_examples("dev", dataset)
    if all(p.exists() for p in paths.values()):
        z = {s: np.load(p) for s, p in paths.items()}
        names = json.loads(str(z["train"]["names"]))
        assert json.loads(str(z["train"]["buttons"])) == blist
        return z["train"]["X"], z["train"]["y"], z["dev"]["X"], z["dev"]["y"], names, blist, tr, dv
    E = np.load(ROOT / "data" / "emb" / f"{dataset}__{encoder}.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=tuple(parts), lex_k=lex_k, seed=0).fit(tr, em["train"], blist)
    out = {}
    for s, rows in (("train", tr), ("dev", dv)):
        X = view.transform(rows, em[s])
        y = np.array([r["y"] for r in rows])
        _save(paths[s], X=X, y=y, names=json.dumps(view.names), buttons=json.dumps(blist))
        out[s] = (X, y)
    return out["train"][0], out["train"][1], out["dev"][0], out["dev"][1], view.names, blist, tr, dv


def free_view_bits(dataset: str, parts=("mem", "words", "rclust", "cclust"), encoder: str = "bge-large", splits=("train", "dev")):
    """The label-free view of R10 (ABCDFreeView, lex_k 1000, k 64, seed 0), cached like view_bits.
    Returns ({split: X}, names, buttons, {split: rows}). Test only when asked for by name."""
    from dl.abcd import ABCDFreeView

    tag = "free-" + "-".join(parts) + ("" if encoder == "bge-large" else f"__{encoder}")
    d = ROOT / "data" / dataset
    rows = {s: load_examples(s, dataset)[0] for s in splits}
    blist = load_examples("train", dataset)[1]
    paths = {s: d / f"bits_{tag}_{s}.npz" for s in splits}
    if all(p.exists() for p in paths.values()):
        z = {s: np.load(p) for s, p in paths.items()}
        assert all(json.loads(str(z[s]["buttons"])) == blist for s in splits)
        return {s: z[s]["X"] for s in splits}, json.loads(str(z[splits[0]]["names"])), blist, rows
    tr = rows["train"] if "train" in rows else load_examples("train", dataset)[0]
    E = np.load(ROOT / "data" / "emb" / f"{dataset}__{encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent")}
    view = ABCDFreeView(parts=tuple(parts), seed=0).fit(tr, etr, blist)
    X = {}
    for s in splits:
        es = {f: E[f"{s}__{f}"] for f in ("cust", "recent")}
        X[s] = view.transform(rows[s], es)
        _save(paths[s], X=X[s], names=json.dumps(view.names), buttons=json.dumps(blist))
    return X, view.names, blist, rows
