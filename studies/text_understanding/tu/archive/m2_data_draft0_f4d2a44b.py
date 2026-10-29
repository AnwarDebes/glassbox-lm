"""M2 (tu/PLAN_M2.md): the MTOP English loader, as a dl.data.IntentData, for the --load hook of the M2 scripts.

The official zip (fresh/NOTE_2026-10-23_mtop.md) is read in memory, never extracted, with its sha256 checked first.
Only three of the eight TSV columns are read: 1 (ID), 2 (the top-level intent) and 4 (the utterance).

- load(name): name "mtop_en". Splits train (en/train), val (en/eval) and test (en/test). The classes are the sorted
  distinct intents of en/train; an eval or test item whose intent is not among them gets label -1, which no model can
  predict, so it counts as an error (the count is in sources["unseen_intents"]). For the registered run only: it reads
  the test labels.
- load_standin(name): the same texts and IDs, but the labels are a stand-in (the sha256 of the text, mod 20) and no
  label column is read at all; test := a copy of val. For the tests and the dry run before the freeze.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import IntentData

ZIP = Path(f"{_REPO_ROOT}/studies/text_understanding/data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip")
ZIP_SHA256 = "0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b"
FILES = {"train": "mtop/en/train.txt", "val": "mtop/en/eval.txt", "test": "mtop/en/test.txt"}
ID, INTENT, UTT = 0, 1, 3


def _rows(z: zipfile.ZipFile, member: str) -> list[list[str]]:
    rows = [ln.split("\t") for ln in z.read(member).decode("utf-8").split("\n") if ln.strip()]
    assert all(len(r) == 8 for r in rows), f"{member}: a row without 8 columns"
    return rows


def _open(zip_path: Path) -> zipfile.ZipFile:
    h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert h == ZIP_SHA256 or zip_path != ZIP, f"{zip_path}: sha256 {h} is not the pinned one"
    return zipfile.ZipFile(zip_path)


def load(name: str, zip_path: Path = ZIP) -> IntentData:
    assert name == "mtop_en", name
    z = _open(zip_path)
    rows = {sp: _rows(z, m) for sp, m in FILES.items()}
    intents = sorted({r[INTENT] for r in rows["train"]})
    idx = {k: i for i, k in enumerate(intents)}
    texts = {sp: [r[UTT] for r in rs] for sp, rs in rows.items()}
    labels = {sp: np.array([idx.get(r[INTENT], -1) for r in rs], dtype=np.int64) for sp, rs in rows.items()}
    unseen = {sp: int((labels[sp] < 0).sum()) for sp in ("val", "test")}
    src = {Path(zip_path).name: hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(), "unseen_intents": str(unseen)}
    return IntentData(name, intents, texts, labels, src)


def load_standin(name: str, zip_path: Path = ZIP, n_classes: int = 20) -> IntentData:
    """No label column is read: labels = sha256(text) mod n_classes; test := val."""
    assert name == "mtop_en", name
    z = _open(zip_path)
    texts = {sp: [r[UTT] for r in _rows(z, m)] for sp, m in FILES.items() if sp != "test"}
    lab = lambda t: int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % n_classes
    labels = {sp: np.array([lab(t) for t in ts], dtype=np.int64) for sp, ts in texts.items()}
    texts["test"], labels["test"] = list(texts["val"]), labels["val"].copy()
    return IntentData(name, [f"standin_{i}" for i in range(n_classes)], texts, labels, {"standin": "labels are text hashes"})
