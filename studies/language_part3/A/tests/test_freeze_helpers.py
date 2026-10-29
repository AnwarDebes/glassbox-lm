"""The frozen-record guard of real runs, the K7 clock and the bit-list LOG helper (each must be able to fail)."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from helpers import TEST_TMP

A = Path(__file__).resolve().parents[1]

from astruct import paths


def _record(root: Path, files: dict) -> Path:
    import shutil
    if root.exists():
        shutil.rmtree(root)
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    rec = {"enforced": {rel: paths.sha256_file(root / rel) for rel in files}, "recorded": {}}
    p = root / "frozen.json"
    p.write_text(json.dumps(rec))
    return p


def test_verify_frozen_passes_and_catches_changes():
    root = TEST_TMP / "freeze_case"
    rec = _record(root, {"code/a.py": "x = 1\n", "data/b.txt": "hello\n"})
    log = root / "LOG.md"
    log.write_text("- no R1 line yet\n")
    with pytest.raises(SystemExit, match="write the R1 line first"):
        paths.verify_frozen(rec, root, log)
    log.write_text(f"- R1 REGISTERED: frozen record sha256 {paths.sha256_file(rec)}\n")
    assert paths.verify_frozen(rec, root, log) == paths.sha256_file(rec)
    (root / "code/a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="code/a.py: changed"):
        paths.verify_frozen(rec, root, log)
    (root / "code/a.py").unlink()
    with pytest.raises(SystemExit, match="code/a.py: missing"):
        paths.verify_frozen(rec, root, log)
    with pytest.raises(SystemExit, match="no frozen record"):
        paths.verify_frozen(root / "absent.json", root, log)


def _held(lock: Path):
    """Hold an exclusive flock on lock (as chain/part3_cpu_job.sh does) until the returned fd is closed."""
    import fcntl
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_runner_problems_and_lock_probe():
    lock = TEST_TMP / "runner_case" / "chain.lock"
    if lock.exists():
        lock.unlink()
    assert not paths.lock_held(lock)  # no file: not held (and the probe does not create it)
    assert not lock.exists()
    env_ok = {"A_RUNNER": "1", "PART3_CHAIN_LOCK": str(lock)}
    fd = _held(lock)
    try:
        assert paths.lock_held(lock)
        assert paths.runner_problems(19, env_ok, lock) == []
        assert any("nice level is 0" in x for x in paths.runner_problems(0, env_ok, lock))
        assert any("A_RUNNER" in x for x in paths.runner_problems(19, {**env_ok, "A_RUNNER": "test"}, lock))
        assert any("PART3_CHAIN_LOCK" in x for x in paths.runner_problems(19, {"A_RUNNER": "1"}, lock))
    finally:
        os.close(fd)
    assert not paths.lock_held(lock)  # released
    assert any("is not held" in x for x in paths.runner_problems(19, env_ok, lock))
    assert paths.CHAIN_LOCK == paths.ROOT / "glassbox-lm/studies/language_part3/chain/part3_cpu.lock"


def test_real_outputs_need_the_frozen_record(monkeypatch):
    assert paths.is_real_output(paths.OUT / "s0") and paths.is_real_output(paths.OUT / "s2b_full_lr2")
    assert not paths.is_real_output(TEST_TMP / "e2e" / "out" / "s0")
    root = TEST_TMP / "freeze_case2"
    rec = _record(root, {"f.txt": "frozen\n"})
    (root / "LOG.md").write_text(f"- R1: frozen record {paths.sha256_file(rec)}\n")
    monkeypatch.setattr(paths, "OUT", root / "out")
    monkeypatch.setattr(paths, "FROZEN", rec)
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "LOGMD", root / "LOG.md")
    lock = root / "chain.lock"
    monkeypatch.setattr(paths, "CHAIN_LOCK", lock)
    # amendment 1: a real output needs the runner's mark and the held chain lock, before the frozen record is read
    monkeypatch.setenv("A_RUNNER", "1")
    monkeypatch.setenv("PART3_CHAIN_LOCK", str(lock))
    with pytest.raises(SystemExit, match="is not held"):
        paths.OutDir(root / "out" / "step")
    fd = _held(lock)
    monkeypatch.setenv("A_RUNNER", "test")
    with pytest.raises(SystemExit, match="A_RUNNER"):
        paths.OutDir(root / "out" / "step")
    monkeypatch.setenv("A_RUNNER", "1")
    assert not (root / "out" / "step.partial").exists()
    d = paths.OutDir(root / "out" / "step")  # verified: allowed
    assert d.frozen_sha256 == paths.sha256_file(rec)
    d.done({"x": 1})
    assert json.loads((root / "out" / "step" / "done.json").read_text())["frozen_record_sha256"] == d.frozen_sha256
    (root / "f.txt").write_text("edited after the freeze\n")
    with pytest.raises(SystemExit, match="frozen-record check failed"):
        paths.OutDir(root / "out" / "step2")
    assert not (root / "out" / "step2.partial").exists()
    e = paths.OutDir(root / "elsewhere" / "step3")  # not a real output: no check
    assert e.frozen_sha256 is None
    for p in (root / "out" / "step" / "done.json", root / "elsewhere" / "step3.partial"):
        p.unlink() if p.is_file() else p.rmdir()
    (root / "out" / "step").rmdir()
    os.close(fd)


def test_k7_clock_on_made_up_lines():
    import k7_clock
    lines = ["2026-10-21T10:00:00Z wait-lock s0_data ", "2026-10-21T10:00:05Z gate closed (x) s0_data ",
             "2026-10-21T10:02:05Z gate OPEN (x) s0_data ", "2026-10-21T10:02:10Z start s0_data ",
             "2026-10-21T11:02:10Z end s0_data  exit 0; listing: no change in 14 folders",
             "2026-10-21T11:05:00Z start s2b_arm --arm full --family tree",
             "2026-10-21T13:05:00Z end s2b_arm --arm full --family tree exit 0; listing: no change"]
    r = k7_clock.clock(lines, k7_clock.parse_utc("2026-10-21T14:00:00Z"))
    assert abs(r["gate_open_hours_lower_bound"] - 3.0) < 1e-9 and r["finished_jobs"] == 2 and r["closed_gate_checks"] == 1
    assert r["s0_start_utc"] == "2026-10-21T10:02:10Z" and not r["limit_reached"]
    r2 = k7_clock.clock(lines + ["2026-10-21T14:00:00Z start s1_floors "], k7_clock.parse_utc("2026-10-25T00:00:00Z"))
    assert r2["running_jobs"] == ["s1_floors"] and r2["gate_open_hours_lower_bound"] > 80 and r2["limit_reached"]
    r3 = k7_clock.clock(lines, k7_clock.parse_utc("2026-10-31T10:02:10Z"))
    assert r3["limit_reached"] and r3["days_since_s0_start"] == 10.0


def test_log_bitlist_helper():
    s0 = TEST_TMP / "bitlist_case" / "s0"
    s0.mkdir(parents=True, exist_ok=True)
    bl = json.dumps({"names": ["a", "b"], "count_derived": [False, True], "n_position_bits": 1})
    (s0 / "bit_list.json").write_text(bl)
    h = paths.sha256_file(s0 / "bit_list.json")
    counts = {"replies": 10, "fit_replies": 8, "tune_cal_replies": 1, "tune_hold_replies": 1, "fit_tokens_with_end": 99,
              "vocab": 7}
    (s0 / "done.json").write_text(json.dumps({"bit_list_sha256": h, "bits": 2, "count_derived_bits": 1, "bits_A_nf": 1,
                                              "counts": counts}))
    logmd = TEST_TMP / "bitlist_case" / "LOG.md"
    logmd.write_text("# test log\n")
    env = dict(os.environ, A_S0_DIR=str(s0))
    r = subprocess.run(["bash", str(A / "log_bitlist.sh"), str(logmd)], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    last = logmd.read_text().splitlines()[-1]
    assert h in last and "[A-Struct] S0 done" in last and last.startswith("- 20") and "Z: " in last
    # a bit list edited after S0 is refused and nothing is written
    (s0 / "bit_list.json").write_text(bl + " ")
    n = len(logmd.read_text().splitlines())
    r = subprocess.run(["bash", str(A / "log_bitlist.sh"), str(logmd)], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "changed after S0" in r.stdout and len(logmd.read_text().splitlines()) == n
