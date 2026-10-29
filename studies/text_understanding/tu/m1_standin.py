"""M1 stand-in datasets for the code check and the CPU tests only (tu/PLAN_M1.md). dl.data.load loads every split into
memory; the stand-in then replaces the test split with a copy of val (texts and labels) before any code uses it. So a
run on a stand-in uses no test text and no test label, and its "test" accuracy must equal its val accuracy.

- load(name): the full train and val; test := val.
- load_small(name): train cut to its first 1,500 rows (for the CPU tests); val as is; test := val.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import sys

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
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
