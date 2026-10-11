"""experiments/configs/*.json are the pre-registered run definitions (docs/13, docs/05).

Checks: valid JSON; the experiment id has a heading in docs/05 (or is an M3 gate of docs/04);
every pre-registered hypothesis id exists in docs/13; seeds come from the fixed lists of docs/13
section 3; engine overrides build a valid EngineConfig; the named script path is under
experiments/. No config is executed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tmlm.engine.config import EngineConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "experiments" / "configs").glob("*.json"))
M3_GATES = {"tiny_lm", "parity_tmu", "planted_rules"}
# M2 items of docs/04 that are pipelines, not experiments of docs/05: the data build and
# the Track C baseline runs (docs/06 "Baselines").
M2_ITEMS = {"M2_data", "M2_baselines"}
ALLOWED_SEEDS = {1, 2, 3, 4, 5}


def _headings(doc: str) -> set[str]:
    text = (ROOT / "docs" / doc).read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^## ([A-Za-z0-9.]+)\b", text, re.M)}


DOCS05 = _headings("05-experiments.md")
PREREG_IDS = set(re.findall(r"^\| (H\d\.\d) \|", (ROOT / "docs" / "13-preregistration.md").read_text(), re.M))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_is_consistent_with_the_docs(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("experiment", "docs", "milestone", "lane", "status", "prereg", "seeds", "script", "params"):
        assert key in cfg, f"{path.name}: missing {key}"
    exp = cfg["experiment"]
    assert exp in DOCS05 or exp in M3_GATES or exp in M2_ITEMS or exp == "T0.2", f"{path.name}: {exp} has no heading in docs/05"
    for hid in cfg["prereg"]:
        assert hid in PREREG_IDS, f"{path.name}: {hid} is not in docs/13"
    assert cfg["seeds"] and set(cfg["seeds"]) <= ALLOWED_SEEDS, f"{path.name}: seeds must come from docs/13 section 3"
    assert cfg["status"] == "planned"
    assert cfg["script"].startswith("experiments/")
    if "engine" in cfg:
        EngineConfig(**cfg["engine"]).validate()


def test_every_docs05_experiment_has_a_config():
    covered = {json.loads(p.read_text())["experiment"] for p in CONFIGS}
    missing = {h for h in DOCS05 if h not in covered and h not in {"E0"}}  # E0 is already logged
    assert not missing, f"docs/05 experiments without a config: {sorted(missing)}"
