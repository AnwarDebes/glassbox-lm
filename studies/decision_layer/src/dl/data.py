"""Intent datasets for the decision-layer study: CLINC150, Banking77, HWU64.

Splits:
- CLINC150 (clinc/oos-eval data_full.json): the file's own train/val/test in-scope
  splits (15,000 / 3,000 / 4,500) and oos_train / oos_val / oos_test (100 / 100 / 1,000).
- Banking77 (legacy-datasets/banking77 parquet, the official 10,003 / 3,080 split): val is a
  stratified 10% of each class of the official train split, permutation seed 0; the rest is train.
- HWU64 (FastFit/hwu_64 parquet, the DialoGLUE split 8,954 / 1,076 / 1,076).

Labels are indices into the sorted list of intent names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASETS = ("clinc150", "banking77", "hwu64")


@dataclass
class IntentData:
    name: str
    intents: list[str]
    texts: dict[str, list[str]]  # split -> texts (train, val, test, and oos_* for clinc150)
    labels: dict[str, np.ndarray]  # split -> int labels (in-scope splits only)
    sources: dict[str, str] = field(default_factory=dict)  # file -> sha256

    def summary(self) -> dict:
        return {"name": self.name, "n_classes": len(self.intents), **{k: len(v) for k, v in self.texts.items()}}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name: str) -> IntentData:
    if name == "clinc150":
        p = RAW / "clinc150_data_full.json"
        d = json.loads(p.read_text())
        intents = sorted({lab for split in ("train", "val", "test") for _, lab in d[split]})
        idx = {k: i for i, k in enumerate(intents)}
        texts, labels = {}, {}
        for split in ("train", "val", "test"):
            texts[split] = [t for t, _ in d[split]]
            labels[split] = np.array([idx[lab] for _, lab in d[split]], dtype=np.int64)
        for split in ("oos_train", "oos_val", "oos_test"):
            texts[split] = [t for t, _ in d[split]]
        return IntentData(name, intents, texts, labels, {p.name: _sha(p)})

    import pandas as pd

    if name == "banking77":
        ptr, pte = RAW / "banking77_train.parquet", RAW / "banking77_test.parquet"
        tr, te = pd.read_parquet(ptr), pd.read_parquet(pte)
        # Label ids are the official order; the names come from the parquet's HF metadata
        # (extracted to banking77_label_names.json).
        intents = json.loads((RAW / "banking77_label_names.json").read_text())
        assert len(intents) == 77
        ytr_all = tr["label"].to_numpy().astype(np.int64)
        rng = np.random.default_rng(0)
        val_mask = np.zeros(len(tr), dtype=bool)
        for c in range(77):
            ids = np.flatnonzero(ytr_all == c)
            ids = ids[rng.permutation(len(ids))]
            val_mask[ids[: int(round(0.1 * len(ids)))]] = True
        texts = {
            "train": tr["text"][~val_mask].tolist(),
            "val": tr["text"][val_mask].tolist(),
            "test": te["text"].tolist(),
        }
        labels = {
            "train": ytr_all[~val_mask],
            "val": ytr_all[val_mask],
            "test": te["label"].to_numpy().astype(np.int64),
        }
        return IntentData(name, intents, texts, labels, {ptr.name: _sha(ptr), pte.name: _sha(pte)})

    if name == "hwu64":
        ps = {s: RAW / f"hwu64_{s}.parquet" for s in ("train", "val", "test")}
        dfs = {s: pd.read_parquet(p) for s, p in ps.items()}
        intents = sorted(set(dfs["train"]["label"]) | set(dfs["val"]["label"]) | set(dfs["test"]["label"]))
        idx = {k: i for i, k in enumerate(intents)}
        texts = {s: df["text"].tolist() for s, df in dfs.items()}
        labels = {s: np.array([idx[x] for x in df["label"]], dtype=np.int64) for s, df in dfs.items()}
        return IntentData(name, intents, texts, labels, {p.name: _sha(p) for p in ps.values()})

    if name in ("massive_en", "massive_nb"):
        # MASSIVE 1.1 (Amazon, CC BY 4.0), en-US or nb-NO: 60 intents, the file's own
        # train / dev / test partitions (11,514 / 2,033 / 2,974; the nb-NO file is a translation
        # of the same utterances). Added 2026-10-16 for phase 2; read from data/untrusted/massive.
        loc = {"massive_en": "en-US", "massive_nb": "nb-NO"}[name]
        p = RAW.parent / "untrusted" / "massive" / f"{loc}.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        intents = sorted({r["intent"] for r in rows})
        idx = {k: i for i, k in enumerate(intents)}
        part = {"train": "train", "val": "dev", "test": "test"}
        texts = {s: [r["utt"] for r in rows if r["partition"] == part[s]] for s in part}
        labels = {s: np.array([idx[r["intent"]] for r in rows if r["partition"] == part[s]], dtype=np.int64) for s in part}
        return IntentData(name, intents, texts, labels, {p.name: _sha(p)})

    raise ValueError(f"unknown dataset {name!r}")
