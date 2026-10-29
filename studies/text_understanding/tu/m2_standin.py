"""M2 stand-in datasets for the code check, the dry runs and the CPU tests only (tu/PLAN_M2.md).

tu/m1_standin.py (d295f9ddba944c4d) with one addition and nothing else changed:
- load(name) and load_small(name) are M1's, for the code check on 3c's CLINC150 rows: dl.data.load loads every split
  into memory; the stand-in then replaces the test split with a copy of val (texts and labels) before any code uses
  it. So a run on a stand-in uses no test text and no test label, and its "test" accuracy must equal its val accuracy.
- added: mtop_small(name): m2_data.load_standin (real MTOP en/train and en/eval texts, labels = sha256(text) mod 20,
  the intent column never used, en/test.txt never opened, test := val) with train cut to its first 1,500 rows, for
  the CPU smoke of the MTOP data path. The full-size stand-in is m2_data.py:load_standin itself.
- added after the review of draft 1 (2026-10-25; design MINOR 6, data MINOR 3): mtop_unseen(name) and
  mtop_small_unseen(name): the same stand-ins with every 40th en/eval label set to -1 (an intent not in train), and
  test := val again, so the -1 path runs through m2_feats.py, m2_tm.py and m2_lr.py in the dry runs; sources records
  the counts as m2_data.load does ("unseen_intents").
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import sys
from pathlib import Path

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl.data import IntentData
from dl.data import load as _load


def load(name: str) -> IntentData:
    ds = _load(name)
    ds.texts["test"] = list(ds.texts["val"])
    ds.labels["test"] = ds.labels["val"].copy()
    ds.texts.pop("oos_test", None)
    return ds


def load_small(name: str, n: int = 1500) -> IntentData:
    ds = load(name)
    ds.texts["train"] = ds.texts["train"][:n]
    ds.labels["train"] = ds.labels["train"][:n]
    return ds


def mtop_small(name: str, n: int = 1500) -> IntentData:
    import m2_data

    ds = m2_data.load_standin(name)
    ds.texts["train"] = ds.texts["train"][:n]
    ds.labels["train"] = ds.labels["train"][:n]
    ds.ids["train"] = ds.ids["train"][:n]
    return ds


def _with_unseen(ds: IntentData, every: int = 40) -> IntentData:
    """Every `every`-th val label := -1; test := val (texts, labels, IDs); the counts in sources."""
    ds.labels["val"] = ds.labels["val"].copy()
    ds.labels["val"][::every] = -1
    ds.texts["test"], ds.labels["test"] = list(ds.texts["val"]), ds.labels["val"].copy()
    ds.ids["test"] = list(ds.ids["val"])
    k = int((ds.labels["val"] < 0).sum())
    ds.sources["unseen_intents"] = str({"val": k, "test": k})
    return ds


def mtop_unseen(name: str) -> IntentData:
    import m2_data

    return _with_unseen(m2_data.load_standin(name))


def mtop_small_unseen(name: str, n: int = 1500) -> IntentData:
    return _with_unseen(mtop_small(name, n))
