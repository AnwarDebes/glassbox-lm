"""scripts/env_check.py in --no-gpu mode: no CUDA context is created by this test."""

from __future__ import annotations

import json
import subprocess
import sys

from tmlm.common import runlog


def run_env_check(scripts_dir, results_dir, *flags):
    proc = subprocess.run(
        [sys.executable, str(scripts_dir / "env_check.py"), "--no-gpu", "--results-dir",
         str(results_dir), *flags],
        capture_output=True, text=True, timeout=900)
    run_dirs = sorted(results_dir.glob("*_envcheck*"))
    return proc, run_dirs[-1]


def test_env_check_reports_failures_without_dying(scripts_dir, tmp_path, fake_nvidia_smi):
    proc, run_dir = run_env_check(scripts_dir, tmp_path / "res", "--extra-import",
                                  "no_such_module_xyz")
    assert proc.returncode == 0, proc.stderr  # a failed probe is reported, not fatal
    out = proc.stdout
    assert "cpu.max:" in out and "memory.max:" in out
    assert "Fake GPU 9000" in out
    assert "FAIL no_such_module_xyz: ModuleNotFoundError" in out
    assert "SKIP GraphTsetlinMachine.tm" in out
    assert "OK   numpy" in out  # numpy must import here, the test suite itself needs it

    report = json.loads((run_dir / "envcheck.json").read_text())
    assert set(report["cgroup"]) >= {"cpu.max", "memory.max", "cpu_quota_cpus", "os_cpu_count"}
    assert report["imports"]["numpy"]["ok"] is True
    assert report["imports"]["no_such_module_xyz"]["ok"] is False
    assert report["imports"]["GraphTsetlinMachine.tm"]["ok"] is None
    for module in ("numpy", "pycuda", "cupy", "tmu", "GraphTsetlinMachine"):
        assert module in report["imports"]

    # Protocol rule 10: signatures are printed from the installed package and saved.
    graphs = report["signatures"]["GraphTsetlinMachine.graphs"]
    if graphs["ok"]:
        signature = graphs["classes"]["Graphs"]["init"]
        assert signature.startswith("(") and "number_of_graphs" in signature
        assert f"class Graphs{signature}" in (run_dir / "log.txt").read_text()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["probes_failed"]["value"] >= 1
    assert all(entry["label"] == "measured" for entry in metrics.values())
    assert json.loads((run_dir / "config.json").read_text())["no_gpu"] is True
    assert runlog.verify_manifest(run_dir) == []


def test_env_check_strict_mode_fails_on_a_broken_import(scripts_dir, tmp_path, fake_nvidia_smi):
    proc, run_dir = run_env_check(scripts_dir, tmp_path / "res", "--strict", "--extra-import",
                                  "no_such_module_xyz")
    assert proc.returncode == 1
    assert json.loads((run_dir / "run.json").read_text())["exit_status"] == 1
