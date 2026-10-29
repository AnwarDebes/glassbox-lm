"""The end-to-end smoke script (scripts/smoke_e2e.py) passes on the CPU, and its golden-hash check can fail."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("smoke_e2e", ROOT / "scripts" / "smoke_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_passes_on_the_cpu(tmp_path):
    smoke = _load()
    assert smoke.main(["--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ok"] and set(summary["steps"]) == {"env", "train", "learn", "receipts"}
    assert summary["steps"]["train"]["golden_checked"]


def test_a_wrong_golden_hash_fails_the_train_step():
    smoke = _load()
    smoke.GOLDEN_TOY_STATE_HASH = "0" * 64
    assert not smoke.step_train()["ok"]
