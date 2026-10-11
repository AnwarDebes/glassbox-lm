"""tmlm.common.runlog and scripts/run_logged.py, end to end in temp directories."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tmlm.common import hashing, runlog
from tmlm.common.runlog import RunLog

EXPECTED_FILES = {"config.json", "env.json", "run.json", "metrics.json", "log.txt", "MANIFEST.json"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fake_root(tmp_path) -> Path:
    """A tiny project tree with the five hashed source directories."""
    root = tmp_path / "proj"
    for sub in ("src/pkg", "experiments", "scripts", "tests", "env"):
        (root / sub).mkdir(parents=True)
    (root / "src/pkg/a.py").write_text("A = 1\n")
    (root / "src/pkg/b.py").write_text("B = 2\n")
    (root / "scripts/run.sh").write_text("echo hi\n")
    (root / "env/env.sh").write_text("export X=1\n")
    return root


# ------------------------------------------------------------------ run dir, end to end


def test_run_dir_is_created_with_all_files(tmp_path, fake_root, fake_nvidia_smi):
    config = {"lr": 0.1, "layers": (1, 2), "nested": {"b": 2, "a": 1}}
    digest = "ab" * 32
    with RunLog("unit_run", config, results_dir=tmp_path / "res", root=fake_root,
                dataset_hashes={"train": digest}, gpu_poll_s=0.05) as run:
        print("hello from the run")
        print("warning text", file=sys.stderr)
        run.add_metric("top1", 0.25, "measured", unit="fraction")
        run.add_metric("flops", 1e9, "derived")
        run.write_json("extra/curve.json", {"x": [1, 2, 3]})
        run_dir = run.run_dir

    assert run_dir.parent == tmp_path / "res"
    assert run_dir.name.endswith("_unit_run")
    stamp = run_dir.name.split("_")[0]
    assert len(stamp) == 16 and stamp[8] == "T" and stamp.endswith("Z")
    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES | {"extra"}

    assert load(run_dir / "config.json") == {"lr": 0.1, "layers": [1, 2], "nested": {"a": 1, "b": 2}}

    record = load(run_dir / "run.json")
    assert record["status"] == "finished"
    assert record["exit_status"] == 0
    assert record["error"] is None
    assert record["wall_time_s"] >= 0
    assert record["git_commit"] is None  # fake_root is not a git repository
    assert record["source_tree_hash"] == runlog.source_tree_hash(fake_root)
    assert record["prereg_sha256"] is None  # fake_root has no docs/13-preregistration.md
    assert record["config_sha256"] == hashing.config_hash(config)
    assert record["config_sha256"] == hashing.config_hash(load(run_dir / "config.json"))
    assert record["dataset_hashes"] == {"train": digest}
    assert record["peak_gpu_memory_mib"] == 1234  # from the fake nvidia-smi
    assert record["gpu_memory"]["status"] == "ok"
    assert record["gpu_memory"]["n_samples"] >= 2

    env = load(run_dir / "env.json")
    assert env["python_version"] == sys.version
    assert env["hostname"]
    assert set(env["cgroup"]) == {"cpu.max", "memory.max"}
    assert {"numpy", "pycuda", "cupy"} <= set(env["packages"])
    assert env["nvidia_smi"]["gpus"][0]["name"] == "Fake GPU 9000"
    assert env["nvidia_smi"]["gpus"][0]["driver_version"] == "1.2.3"

    metrics = load(run_dir / "metrics.json")
    assert metrics == {
        "top1": {"value": 0.25, "label": "measured", "unit": "fraction"},
        "flops": {"value": 1e9, "label": "derived"},
    }

    log = (run_dir / "log.txt").read_text()
    assert "hello from the run" in log
    assert "warning text" in log


def test_env_collection_imports_no_gpu_library(tmp_path, fake_root, no_nvidia_smi):
    before = set(sys.modules)
    with RunLog("noimport", {}, results_dir=tmp_path, root=fake_root):
        pass
    new = set(sys.modules) - before
    assert not {m for m in new if m.split(".")[0] in {"pycuda", "cupy", "numba", "torch"}}


def test_gpu_sampler_degrades_without_nvidia_smi(tmp_path, fake_root, no_nvidia_smi):
    with RunLog("nogpu", {"x": 1}, results_dir=tmp_path, root=fake_root) as run:
        pass
    record = load(run.run_dir / "run.json")
    assert record["exit_status"] == 0
    assert record["peak_gpu_memory_mib"] is None
    assert "not found" in record["gpu_memory"]["status"]
    assert "not found" in load(run.run_dir / "env.json")["nvidia_smi"]["status"]
    assert runlog.verify_manifest(run.run_dir) == []


def test_gpu_sampler_survives_a_broken_nvidia_smi(tmp_path, fake_root, fake_nvidia_smi):
    fake_nvidia_smi.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    with RunLog("brokengpu", {}, results_dir=tmp_path, root=fake_root, gpu_poll_s=0.05) as run:
        pass
    record = load(run.run_dir / "run.json")
    assert record["exit_status"] == 0
    assert record["peak_gpu_memory_mib"] is None
    assert record["gpu_memory"]["status"].startswith("error")
    assert record["gpu_memory"]["n_errors"] >= 1


# ------------------------------------------------------------------ never overwritten


def test_run_dir_is_never_overwritten(tmp_path, fake_root, monkeypatch):
    monkeypatch.setattr(runlog, "utc_stamp", lambda: "20260101T000000Z")
    dirs = []
    for index in range(3):
        with RunLog("same", {"i": index}, results_dir=tmp_path, root=fake_root,
                    sample_gpu=False) as run:
            print(f"run number {index}")
        dirs.append(run.run_dir)
    assert len(set(dirs)) == 3
    assert [d.name for d in dirs] == [
        "20260101T000000Z_same", "20260101T000000Z_same_01", "20260101T000000Z_same_02"]
    for index, run_dir in enumerate(dirs):
        assert load(run_dir / "config.json") == {"i": index}
        assert f"run number {index}" in (run_dir / "log.txt").read_text()
        assert runlog.verify_manifest(run_dir) == []


def test_existing_directory_is_left_untouched(tmp_path, fake_root, monkeypatch):
    monkeypatch.setattr(runlog, "utc_stamp", lambda: "20260101T000000Z")
    squatter = tmp_path / "20260101T000000Z_taken"
    squatter.mkdir()
    (squatter / "precious.txt").write_text("do not touch")
    with RunLog("taken", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False) as run:
        pass
    assert run.run_dir != squatter
    assert [p.name for p in squatter.iterdir()] == ["precious.txt"]
    assert (squatter / "precious.txt").read_text() == "do not touch"


def test_a_runlog_cannot_be_reused(tmp_path, fake_root):
    run = RunLog("once", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False)
    with run:
        pass
    with pytest.raises(RuntimeError):
        run.__enter__()


@pytest.mark.parametrize("bad", ["", "has space", "../escape", "a/b", ".hidden", "x" * 81])
def test_bad_run_names_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        RunLog(bad, {}, results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ manifest


def test_manifest_hashes_verify_and_detect_tampering(tmp_path, fake_root):
    with RunLog("manifest", {"k": "v"}, results_dir=tmp_path, root=fake_root,
                sample_gpu=False) as run:
        print("some output")
        run.write_json("artifact.json", [1, 2, 3])
    run_dir = run.run_dir
    manifest = load(run_dir / "MANIFEST.json")
    listed = set(manifest["files"])
    assert listed == {"config.json", "env.json", "run.json", "metrics.json", "log.txt",
                      "artifact.json"}
    for rel, entry in manifest["files"].items():
        data = (run_dir / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["size"]
    assert runlog.verify_manifest(run_dir) == []

    (run_dir / "metrics.json").write_text('{"forged": {"value": 99, "label": "measured"}}\n')
    assert runlog.verify_manifest(run_dir) == ["metrics.json: sha256 mismatch"]
    (run_dir / "log.txt").unlink()
    (run_dir / "new.txt").write_text("x")
    problems = runlog.verify_manifest(run_dir)
    assert "log.txt: listed in MANIFEST but missing" in problems
    assert "new.txt: present but not listed in MANIFEST" in problems


# ------------------------------------------------------------------ metric labels


def test_metric_labels_are_enforced(tmp_path, fake_root):
    with RunLog("labels", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False) as run:
        with pytest.raises(ValueError):
            run.add_metric("acc", 0.9, "verified")
        with pytest.raises(ValueError):
            run.add_metric("acc", 0.9, "")
        with pytest.raises(ValueError):
            run.set_metrics({"acc": 0.9})
        with pytest.raises(ValueError):
            run.set_metrics({"acc": {"value": 0.9}})
        with pytest.raises(ValueError):
            run.set_metrics({"acc": {"label": "measured"}})
        with pytest.raises(ValueError):
            run.set_metrics({"acc": {"value": object(), "label": "measured"}})
        run.set_metrics({"ok": {"value": 1, "label": "unverified"}})
        run.add_metric("diverged", float("nan"), "measured")
    metrics = load(run.run_dir / "metrics.json")
    assert set(metrics) == {"ok", "diverged"}
    assert metrics["ok"] == {"value": 1, "label": "unverified"}
    assert metrics["diverged"]["value"] != metrics["diverged"]["value"]  # NaN survives
    assert all(entry["label"] in runlog.METRIC_LABELS for entry in metrics.values())


def test_dataset_hashes_must_be_sha256(tmp_path, fake_root):
    with pytest.raises(ValueError):
        RunLog("d", {}, results_dir=tmp_path, root=fake_root, dataset_hashes={"train": "abc"})
    with RunLog("d", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False) as run:
        with pytest.raises(ValueError):
            run.add_dataset_hash("valid", "XYZ")
        run.add_dataset_hash("valid", "0" * 64)
    assert load(run.run_dir / "run.json")["dataset_hashes"] == {"valid": "0" * 64}


def test_config_must_be_json_like(tmp_path):
    with pytest.raises(TypeError):
        RunLog("cfg", {"bad": {1, 2}}, results_dir=tmp_path)
    with pytest.raises(TypeError):
        RunLog("cfg", {1: "int key"}, results_dir=tmp_path)


# ------------------------------------------------------------------ failures inside the run


def test_exception_is_recorded_and_propagates(tmp_path, fake_root):
    run = RunLog("boom", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False)
    with pytest.raises(ZeroDivisionError):
        with run:
            print("before the crash")
            1 / 0
    record = load(run.run_dir / "run.json")
    assert record["exit_status"] == 1
    assert record["error"].startswith("ZeroDivisionError")
    log = (run.run_dir / "log.txt").read_text()
    assert "before the crash" in log and "ZeroDivisionError" in log
    assert runlog.verify_manifest(run.run_dir) == []
    assert sys.stdout is not None and not isinstance(sys.stdout, runlog._Tee)


def test_system_exit_code_is_recorded(tmp_path, fake_root):
    run = RunLog("exit7", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False)
    with pytest.raises(SystemExit):
        with run:
            sys.exit(7)
    assert load(run.run_dir / "run.json")["exit_status"] == 7


def test_streams_are_restored(tmp_path, fake_root):
    out, err = sys.stdout, sys.stderr
    with RunLog("streams", {}, results_dir=tmp_path, root=fake_root, sample_gpu=False):
        assert isinstance(sys.stdout, runlog._Tee) and isinstance(sys.stderr, runlog._Tee)
    assert sys.stdout is out and sys.stderr is err


# ------------------------------------------------------------------ source tree hash


def test_source_tree_hash_is_stable_and_sensitive(fake_root):
    first = runlog.source_tree_hash(fake_root)
    assert first == runlog.source_tree_hash(fake_root)
    assert len(first) == 64

    # build products do not count
    (fake_root / "src/pkg/__pycache__").mkdir()
    (fake_root / "src/pkg/__pycache__/a.cpython-311.pyc").write_bytes(b"\x00\x01")
    (fake_root / "src/pkg/engine.so").write_bytes(b"\x7fELF")
    (fake_root / "src/tmlm.egg-info").mkdir()
    (fake_root / "src/tmlm.egg-info/PKG-INFO").write_text("x")
    assert runlog.source_tree_hash(fake_root) == first

    # directories outside the five hashed ones do not count
    (fake_root / "results").mkdir()
    (fake_root / "results/run.txt").write_text("x")
    assert runlog.source_tree_hash(fake_root) == first

    # a content change counts
    (fake_root / "src/pkg/a.py").write_text("A = 2\n")
    changed = runlog.source_tree_hash(fake_root)
    assert changed != first
    (fake_root / "src/pkg/a.py").write_text("A = 1\n")
    assert runlog.source_tree_hash(fake_root) == first

    # a rename counts, a new file counts, a new empty file counts
    (fake_root / "src/pkg/a.py").rename(fake_root / "src/pkg/a2.py")
    assert runlog.source_tree_hash(fake_root) != first
    (fake_root / "src/pkg/a2.py").rename(fake_root / "src/pkg/a.py")
    assert runlog.source_tree_hash(fake_root) == first
    (fake_root / "tests/test_new.py").write_text("")
    assert runlog.source_tree_hash(fake_root) != first


def test_source_tree_hash_does_not_mix_path_and_content(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    (one / "src").mkdir(parents=True)
    (two / "src").mkdir(parents=True)
    (one / "src/ab").write_text("c")
    (two / "src/a").write_text("bc")
    assert runlog.source_tree_hash(one) != runlog.source_tree_hash(two)


def test_source_tree_hash_of_the_real_tree_is_stable(repo_root):
    assert runlog.source_tree_hash(repo_root) == runlog.source_tree_hash(repo_root)


def test_git_commit_is_none_outside_a_repository(tmp_path):
    assert runlog.git_commit(tmp_path) is None


# ------------------------------------------------------------------ CLI wrapper


def run_cli(scripts_dir, tmp_path, name, command, config=None, extra=()):
    cfg = tmp_path / f"{name}_cfg.json"
    cfg.write_text(json.dumps(config if config is not None else {"name": name}))
    proc = subprocess.run(
        [sys.executable, str(scripts_dir / "run_logged.py"), "--name", name, "--config", str(cfg),
         "--results-dir", str(tmp_path / "cli"), "--no-gpu-sampler", *extra, "--", *command],
        capture_output=True, text=True, timeout=120)
    run_dirs = sorted((tmp_path / "cli").glob(f"*_{name}")) if (tmp_path / "cli").is_dir() else []
    return proc, (run_dirs[-1] if run_dirs else None)


def test_cli_passes_exit_code_and_tees_output(scripts_dir, tmp_path):
    code = "import sys; print('to stdout'); print('to stderr', file=sys.stderr); sys.exit(3)"
    proc, run_dir = run_cli(scripts_dir, tmp_path, "cli3", [sys.executable, "-c", code],
                            config={"alpha": 1})
    assert proc.returncode == 3
    assert "to stdout" in proc.stdout and "to stderr" in proc.stderr
    log = (run_dir / "log.txt").read_text()
    assert "to stdout" in log and "to stderr" in log
    record = load(run_dir / "run.json")
    assert record["exit_status"] == 3
    assert record["command"][0] == sys.executable
    assert load(run_dir / "config.json") == {"alpha": 1}
    assert runlog.verify_manifest(run_dir) == []


def test_cli_collects_metrics_and_dataset_hashes_from_the_command(scripts_dir, tmp_path, repo_root):
    data = tmp_path / "train.bin"
    data.write_bytes(b"tokens" * 100)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from tmlm.common import runlog\n"
        "runlog.write_metrics({'acc': {'value': 0.5, 'label': 'measured'}})\n"
        "runlog.write_dataset_hashes({'valid': 'c' * 64})\n"
    )
    proc, run_dir = run_cli(scripts_dir, tmp_path, "climetrics",
                            [sys.executable, "-c", code, str(repo_root / "src")],
                            extra=["--dataset", f"train={data}"])
    assert proc.returncode == 0, proc.stderr
    assert load(run_dir / "metrics.json") == {"acc": {"value": 0.5, "label": "measured"}}
    assert load(run_dir / "run.json")["dataset_hashes"] == {
        "train": hashlib.sha256(data.read_bytes()).hexdigest(), "valid": "c" * 64}
    assert runlog.verify_manifest(run_dir) == []


def test_cli_rejects_unlabelled_metrics_from_the_command(scripts_dir, tmp_path):
    code = ("import json, os\n"
            "open(os.path.join(os.environ['TMLM_RUN_DIR'], 'metrics.json'), 'w')"
            ".write(json.dumps({'acc': {'value': 0.99}}))\n")
    proc, run_dir = run_cli(scripts_dir, tmp_path, "clibad", [sys.executable, "-c", code])
    assert proc.returncode == 65
    assert load(run_dir / "metrics.json") == {}
    assert (run_dir / "metrics_rejected.json").is_file()
    record = load(run_dir / "run.json")
    assert record["exit_status"] == 65 and "rejected" in record["metrics_error"]
    assert runlog.verify_manifest(run_dir) == []


def test_cli_reports_death_by_signal(scripts_dir, tmp_path):
    proc, run_dir = run_cli(scripts_dir, tmp_path, "clisig", ["bash", "-c", "kill -KILL $$"])
    assert proc.returncode == 137
    record = load(run_dir / "run.json")
    assert record["exit_status"] == 137 and record["terminated_by_signal"] == 9


def test_cli_forwards_sigterm_and_still_finalizes_the_run_dir(scripts_dir, tmp_path):
    marker = tmp_path / "started"
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *args: sys.exit(21))\n"
            "print('child is up', flush=True)\n"
            "open(sys.argv[1], 'w').close()\n"
            "time.sleep(300)\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    proc = subprocess.Popen(
        [sys.executable, str(scripts_dir / "run_logged.py"), "--name", "clisigterm", "--config",
         str(cfg), "--results-dir", str(tmp_path / "cli"), "--no-gpu-sampler", "--",
         sys.executable, "-c", code, str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 30
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=60)
    assert proc.returncode == 21
    (run_dir,) = (tmp_path / "cli").glob("*_clisigterm")
    record = load(run_dir / "run.json")
    assert record["status"] == "finished" and record["exit_status"] == 21
    assert "child is up" in (run_dir / "log.txt").read_text()
    assert runlog.verify_manifest(run_dir) == []


def test_cli_does_not_wait_for_a_background_grandchild(scripts_dir, tmp_path):
    begin = time.monotonic()
    proc, run_dir = run_cli(scripts_dir, tmp_path, "cligrandchild",
                            ["bash", "-c", "sleep 10 & echo parent done"])
    assert proc.returncode == 0
    assert time.monotonic() - begin < 8, "the wrapper waited for the grandchild's pipe"
    assert "parent done" in (run_dir / "log.txt").read_text()
    assert runlog.verify_manifest(run_dir) == []


def test_cli_command_not_found(scripts_dir, tmp_path):
    proc, run_dir = run_cli(scripts_dir, tmp_path, "clinf", ["/no/such/program"])
    assert proc.returncode == 127
    assert load(run_dir / "run.json")["exit_status"] == 127


def test_cli_usage_errors(scripts_dir, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    base = [sys.executable, str(scripts_dir / "run_logged.py")]
    no_command = subprocess.run([*base, "--name", "x", "--config", str(cfg)],
                                capture_output=True, text=True)
    assert no_command.returncode == 2
    bad_name = subprocess.run([*base, "--name", "bad name", "--config", str(cfg), "--results-dir",
                               str(tmp_path / "u"), "--", "true"], capture_output=True, text=True)
    assert bad_name.returncode == 2
    assert not (tmp_path / "u").exists()


def test_prereg_sha256_is_recorded_when_the_file_exists(fake_root):
    """Protocol rule 16: run.json carries the sha256 of docs/13-preregistration.md so a
    number can be matched to the version of the pre-registration in force at run time."""
    assert runlog.prereg_sha256(fake_root) is None
    prereg = fake_root / "docs" / "13-preregistration.md"
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text("# 13 Pre-registration\n\nfrozen text\n", encoding="utf-8")
    expected = hashing.sha256_file(prereg)
    assert runlog.prereg_sha256(fake_root) == expected
    with runlog.RunLog("prereg_probe", {"seed": 1}, root=fake_root) as run:
        pass
    record = load(run.run_dir / "run.json")
    assert record["prereg_sha256"] == expected
    prereg.write_text("# 13 Pre-registration\n\namended\n", encoding="utf-8")
    assert runlog.prereg_sha256(fake_root) != expected

