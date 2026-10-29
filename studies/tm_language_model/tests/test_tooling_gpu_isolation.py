"""The test suite keeps off the GPU unless it holds the GPU lock (2026-10-25, finish pass, TLM-B3).

Four checks: tests/conftest.py hides every CUDA device outside the lock; its lock test accepts only
a descriptor that is the very file <lock dir>/gpu.lock; a gpu-marked test is skipped outside the
lock (the test below would fail if it ran); `make test` deselects the gpu and slow tests with the
GPU hidden, and `make test-gpu` runs the gpu-marked tests under scripts/gpu_run.sh. A static scan
also finds every test function that asks for a CUDA device and checks that it carries the gpu marker.

    .venv-tm/bin/python -m pytest -q tests/test_tooling_gpu_isolation.py
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import conftest

TESTS = Path(__file__).resolve().parent
# Source fragments that mean "this test needs a CUDA device" in this suite.
GPU_USE = ("_gpu_or_skip(", "getDeviceCount", "Device(0)", 'backend="cupy"', "backend='cupy'", "CudaEngine(",
           "in BACKENDS", "BACKENDS[", "_cupy_or_skip(", "_cuda_engine_or_skip(")


def test_cuda_devices_are_hidden_outside_the_gpu_lock():
    if conftest.GPU_LOCK_HELD:
        pytest.skip("this run holds the GPU lock")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


@pytest.mark.gpu
def test_a_gpu_marked_test_never_runs_outside_the_gpu_lock():
    assert conftest.GPU_LOCK_HELD, "a gpu-marked test ran without the GPU lock"


def test_gpu_lock_held_needs_the_real_lock_descriptor(tmp_path):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock = lock_dir / "gpu.lock"
    lock.touch()
    other = tmp_path / "other.txt"
    other.touch()
    fd = os.open(lock, os.O_RDONLY)
    fd_other = os.open(other, os.O_RDONLY)
    try:
        good = {"TMLM_GPU_LOCK_MODE": "shared", "TMLM_GPU_LOCK_FD": str(fd), "TMLM_LOCK_DIR": str(lock_dir)}
        assert conftest.gpu_lock_held(good)
        assert conftest.gpu_lock_held({**good, "TMLM_GPU_LOCK_MODE": "exclusive"})
        assert not conftest.gpu_lock_held({k: v for k, v in good.items() if k != "TMLM_GPU_LOCK_MODE"})
        assert not conftest.gpu_lock_held({**good, "TMLM_GPU_LOCK_MODE": "yes"})
        assert not conftest.gpu_lock_held({**good, "TMLM_GPU_LOCK_FD": str(fd_other)})  # a reused number
        assert not conftest.gpu_lock_held({**good, "TMLM_GPU_LOCK_FD": ""})
        assert not conftest.gpu_lock_held({**good, "TMLM_LOCK_DIR": str(tmp_path / "absent")})
    finally:
        os.close(fd)
        os.close(fd_other)


@pytest.mark.skipif(shutil.which("make") is None, reason="no make")
def test_make_test_hides_the_gpu_and_make_test_gpu_takes_the_lock(repo_root):
    def recipe(target: str) -> str:
        proc = subprocess.run(["make", "-n", "-C", str(repo_root), target], capture_output=True, text=True,
                              timeout=60)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    cpu = recipe("test")
    assert "CUDA_VISIBLE_DEVICES= " in cpu
    assert '-m "not gpu and not slow"' in cpu
    assert "gpu_run.sh" not in cpu
    gpu = recipe("test-gpu")
    assert "scripts/gpu_run.sh shared --" in gpu
    assert "-m gpu" in gpu
    slow = recipe("test-slow")
    assert "CUDA_VISIBLE_DEVICES= " in slow
    assert '-m "slow and not gpu"' in slow
    assert "gpu_run.sh" not in slow


def _marked(node: ast.AST, source: str) -> bool:
    return any("mark.gpu" in seg or "gpu_when_visible" in seg
               for seg in (ast.get_source_segment(source, d) or "" for d in node.decorator_list))


def unmarked_gpu_tests(folder: Path) -> list[str]:
    """Test functions that ask for a CUDA device but carry no gpu marker (module, decorator or
    parameter). A module-level `pytestmark` that names the gpu marker covers the whole module."""
    found = []
    for path in sorted(folder.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_marked = any(
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
            and "mark.gpu" in (ast.get_source_segment(source, node) or "")
            for node in tree.body)
        if module_marked:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            body = ast.get_source_segment(source, node) or ""
            decorators = " ".join(ast.get_source_segment(source, d) or "" for d in node.decorator_list)
            uses_gpu = "needs_gpu" in decorators or any(s in body for s in GPU_USE)
            if uses_gpu and not _marked(node, source) and "BACKEND_PARAMS" not in decorators:
                found.append(f"{path.name}::{node.name}")
    return found


def test_every_test_that_asks_for_a_cuda_device_is_marked_gpu():
    assert unmarked_gpu_tests(TESTS) == []


def test_conftest_sees_the_lock_of_gpu_run_sh(scripts_dir, tmp_path):
    """End to end with the real scripts/gpu_run.sh on a scratch lock folder (no GPU is used: the device list
    stays empty): under the lock conftest reports it held and leaves CUDA_VISIBLE_DEVICES alone; outside the
    lock it hides the devices."""
    probe = ("import os, sys; sys.path.insert(0, sys.argv[1]); import conftest; "
             "print(conftest.GPU_LOCK_HELD, repr(os.environ.get('CUDA_VISIBLE_DEVICES')))")
    env = {**os.environ, "TMLM_LOCK_DIR": str(tmp_path / "locks"), "CUDA_VISIBLE_DEVICES": "keep"}
    for name in ("TMLM_GPU_LOCK_FD", "TMLM_GPU_LOCK_MODE"):
        env.pop(name, None)
    held = subprocess.run([str(scripts_dir / "gpu_run.sh"), "shared", "--", sys.executable, "-c", probe, str(TESTS)],
                          capture_output=True, text=True, timeout=60, env={**env, "CUDA_VISIBLE_DEVICES": ""})
    assert held.returncode == 0, held.stderr
    assert held.stdout.split() == ["True", "''"]
    free = subprocess.run([sys.executable, "-c", probe, str(TESTS)], capture_output=True, text=True, timeout=60, env=env)
    assert free.returncode == 0, free.stderr
    assert free.stdout.split() == ["False", "''"]
