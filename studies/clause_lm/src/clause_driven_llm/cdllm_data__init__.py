"""CLINC150 and Banking77 loaders (REBUILT for logical-llm M5 step 1).

The original module is missing from the public repository: the unanchored
``.gitignore`` line ``data/`` also matched ``src/cdllm/data/``. This file was
rebuilt on 2026-10-02 to the interface used by every call site:

* ``scripts/precompute_embeddings.py``: ``load_dataset(name, data_dir)``,
  ``ds.summary()``, ``ds.train_texts / train_labels``, ``ds.val_texts /
  val_labels``, ``ds.test_texts / test_labels``, ``ds.oos.items()`` (split ->
  list of texts; empty lists are skipped there), ``ds.intents``.
* ``scripts/eval_receipts.py``: ``load_dataset(name, ROOT / "data")``,
  ``ds.test_texts[i]``.
* ``tests/test_data.py``: ``load_clinc150(path)``, ``load_banking77(train_csv,
  test_csv)``, ``ds.n_classes``, ``ds.oos["test"]``, labels as numpy arrays.

Label order: label ``k`` is the ``k``-th intent name in **sorted** order. The
one alternative natural order (first appearance in the train split) can be
selected with ``label_order="first_appearance"`` or the environment variable
``CDLLM_LABEL_ORDER=first_appearance``; the default is ``"sorted"``.

CLINC150 (``data_full.json`` from clinc/oos-eval) uses the file's own
``train`` (15,000), ``val`` (3,000) and ``test`` (4,500) in-scope splits, in
file order, and ``oos_train`` / ``oos_val`` / ``oos_test`` (100 / 100 / 1,000)
as ``ds.oos["train" | "val" | "test"]``.

Banking77 has no official validation split. The split used upstream is
unknown; this rebuild holds out a stratified 10% of each class of the official
training CSV with a fixed seed. It is NOT verified against the original and is
not used by M5 step 1.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["IntentDataset", "load_clinc150", "load_banking77", "load_dataset"]

_LABEL_ORDERS = ("sorted", "first_appearance")


@dataclass
class IntentDataset:
    """In-scope train/val/test splits plus an optional out-of-scope pool."""

    name: str
    intents: list[str]
    train_texts: list[str]
    train_labels: np.ndarray
    val_texts: list[str]
    val_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    oos: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.intents)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_classes": self.n_classes,
            "train": len(self.train_texts),
            "val": len(self.val_texts),
            "test": len(self.test_texts),
            "oos": {k: len(v) for k, v in self.oos.items()},
        }


def _resolve_label_order(label_order: str | None) -> str:
    order = label_order or os.environ.get("CDLLM_LABEL_ORDER", "sorted")
    if order not in _LABEL_ORDERS:
        raise ValueError(f"label_order must be one of {_LABEL_ORDERS}, got {order!r}")
    return order


def _intent_names(train_intents: list[str], label_order: str) -> list[str]:
    if label_order == "sorted":
        return sorted(set(train_intents))
    return list(dict.fromkeys(train_intents))  # first appearance in the train split


def _encode(intents: list[str], names: list[str]) -> np.ndarray:
    index = {n: i for i, n in enumerate(names)}
    return np.array([index[n] for n in intents], dtype=np.int64)


def load_clinc150(path, label_order: str | None = None) -> IntentDataset:
    """Load CLINC150 from the clinc/oos-eval ``data_full.json`` file."""
    order = _resolve_label_order(label_order)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def split(key: str) -> tuple[list[str], list[str]]:
        rows = raw[key]
        return [str(t) for t, _ in rows], [str(lab) for _, lab in rows]

    tr_x, tr_y = split("train")
    va_x, va_y = split("val")
    te_x, te_y = split("test")
    names = _intent_names(tr_y, order)
    oos = {s: split(f"oos_{s}")[0] for s in ("train", "val", "test")}
    return IntentDataset(
        name="clinc150",
        intents=names,
        train_texts=tr_x,
        train_labels=_encode(tr_y, names),
        val_texts=va_x,
        val_labels=_encode(va_y, names),
        test_texts=te_x,
        test_labels=_encode(te_y, names),
        oos=oos,
    )


def _read_banking_csv(path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts.append(str(row["text"]))
            labels.append(str(row["category"]))
    return texts, labels


def load_banking77(
    train_csv,
    test_csv,
    val_fraction: float = 0.1,
    seed: int = 0,
    label_order: str | None = None,
) -> IntentDataset:
    """Load Banking77 from the PolyAI CSVs; carve a stratified val split from train.

    The upstream split is unknown (see the module docstring); this one is a
    reconstruction and is not verified against the original.
    """
    order = _resolve_label_order(label_order)
    tr_x_all, tr_y_all = _read_banking_csv(train_csv)
    te_x, te_y = _read_banking_csv(test_csv)
    names = _intent_names(tr_y_all, order)
    y_all = _encode(tr_y_all, names)

    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y_all), dtype=bool)
    for c in range(len(names)):
        idx = np.flatnonzero(y_all == c)
        n_val = int(round(val_fraction * len(idx)))
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True

    tr_idx = np.flatnonzero(~val_mask)
    va_idx = np.flatnonzero(val_mask)
    return IntentDataset(
        name="banking77",
        intents=names,
        train_texts=[tr_x_all[i] for i in tr_idx],
        train_labels=y_all[tr_idx],
        val_texts=[tr_x_all[i] for i in va_idx],
        val_labels=y_all[va_idx],
        test_texts=te_x,
        test_labels=_encode(te_y, names),
        oos={},
    )


def load_dataset(name: str, data_dir, **kwargs) -> IntentDataset:
    """Load ``name`` in ``{"clinc150", "banking77"}`` from ``data_dir`` (the
    file names written by ``scripts/download_data.py``)."""
    data_dir = Path(data_dir)
    name = name.lower()
    if name == "clinc150":
        return load_clinc150(data_dir / "clinc150_data_full.json", **kwargs)
    if name == "banking77":
        return load_banking77(data_dir / "banking77_train.csv", data_dir / "banking77_test.csv", **kwargs)
    raise ValueError(f"unknown dataset: {name}")
