"""Shared pytest setup: make src/ importable without installing the package, and keep
tests away from the real results/ directory, the real GPU lock, and the real GPU."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FAKE_NVIDIA_SMI = """#!/bin/sh
case "$*" in
    *memory.used*) echo "1234" ;;
    *) echo "Fake GPU 9000, 1.2.3, 32768 MiB" ;;
esac
"""


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def scripts_dir() -> Path:
    return ROOT / "scripts"


@pytest.fixture(autouse=True)
def _isolated_tooling_env(tmp_path, monkeypatch):
    """No test may write into the real results/ or data/, or wait on the real lock files."""
    monkeypatch.setenv("TMLM_RESULTS_DIR", str(tmp_path / "_results"))
    monkeypatch.setenv("TMLM_LOCK_DIR", str(tmp_path / "_locks"))
    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "_data"))  # never touch data/ (protocol rule 9)
    for name in ("TMLM_RUN_DIR", "TMLM_RUN_NAME", "TMLM_CPU_SLOTS", "TMLM_CPU_SLOT_FDS",
                 "TMLM_GPU_LOCK_FD", "TMLM_GPU_LOCK_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch) -> Path:
    """Put a fake nvidia-smi first on PATH: memory.used is always 1234 MiB."""
    bin_dir = tmp_path / "_fakebin"
    bin_dir.mkdir()
    tool = bin_dir / "nvidia-smi"
    tool.write_text(FAKE_NVIDIA_SMI)
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tool


@pytest.fixture
def no_nvidia_smi(tmp_path, monkeypatch) -> None:
    """A PATH on which nvidia-smi (and everything else) cannot be found."""
    empty = tmp_path / "_emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
