"""Shared pytest setup: make src/ importable without installing the package, and keep
tests away from the real results/ directory, the real GPU lock, and the real GPU.

The GPU (since 2026-10-25, finish pass, TLM-B3; before, only the folders were moved and the GPU
stayed visible): unless this process holds the GPU lock of scripts/gpu_run.sh, this file hides
every CUDA device (CUDA_VISIBLE_DEVICES set to the empty string) before any test module is
imported, so neither a test nor a collection-time device probe can create a CUDA context, and
every test marked gpu is skipped with the reason. Holding the lock means: TMLM_GPU_LOCK_MODE is
shared or exclusive and the descriptor TMLM_GPU_LOCK_FD is the very file <lock dir>/gpu.lock,
which gpu_run.sh passes on. `make test` runs `-m "not gpu and not slow"` with the GPU hidden;
`make test-gpu` runs the gpu-marked tests under `scripts/gpu_run.sh shared`.
"""

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


def gpu_lock_held(env=None) -> bool:
    """True when this process holds the GPU lock taken by scripts/gpu_run.sh."""
    env = os.environ if env is None else env
    if env.get("TMLM_GPU_LOCK_MODE") not in ("shared", "exclusive"):
        return False
    lock_file = Path(env.get("TMLM_LOCK_DIR") or ROOT / "results" / ".locks") / "gpu.lock"
    try:
        want = os.stat(lock_file)
    except OSError:
        return False
    for part in (env.get("TMLM_GPU_LOCK_FD") or "").split():
        if part.isdigit() and int(part) > 2:
            try:
                have = os.fstat(int(part))
            except OSError:
                continue
            if (have.st_dev, have.st_ino) == (want.st_dev, want.st_ino):
                return True
    return False


GPU_LOCK_HELD = gpu_lock_held()
if not GPU_LOCK_HELD:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_collection_modifyitems(config, items):
    if GPU_LOCK_HELD:
        return
    skip = pytest.mark.skip(reason="marked gpu: runs only under scripts/gpu_run.sh (make test-gpu)")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)


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
