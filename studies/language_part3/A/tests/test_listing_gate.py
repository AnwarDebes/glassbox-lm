"""The before/after listing check and the CPU-gate parser (both must be able to fail)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import subprocess
import sys
import time
from pathlib import Path

from helpers import TEST_TMP

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
import listing


def test_listing_detects_a_new_file_and_ignores_parent_entries():
    root = TEST_TMP / "listing_case"
    sub = root / "watched"
    sub.mkdir(parents=True, exist_ok=True)
    for p in sub.iterdir():
        p.unlink()
    (sub / "kept.txt").write_text("x")
    before = listing.snapshot([sub])
    ok, ch, notes = listing.compare(before, listing.snapshot([sub]))
    assert ok and not ch
    time.sleep(0.01)
    (sub / "new.pyc").write_text("y")  # what an unwanted cache write would look like
    ok, ch, notes = listing.compare(before, listing.snapshot([sub]))
    assert not ok and any("new.pyc" in c for c in ch)
    (sub / "new.pyc").unlink()
    # a change in the parent folder only shows in the ".." line: a note, not a failure
    b2 = listing.snapshot([sub])
    (root / f"sibling_{time.time_ns()}").write_text("z")
    ok, ch, notes = listing.compare(b2, listing.snapshot([sub]))
    assert ok and notes


def test_listed_folders_cover_the_plan():
    names = {str(p) for p in listing.LISTED}
    for must in ("studies/tally/src/tally/floor", "studies/tally/src/tally/floor/__pycache__", "studies/decision_layer/src/dl",
                 "studies/decision_layer/src/dl/__pycache__", "src/tmlm/eval/baselines", "c4words_S1000_v1",
                 "c4words_S1000_v1/__pycache__", "tools/kenlm/bin"):
        assert any(n.endswith(must) for n in names), must


def _gate(line: str) -> int:
    return subprocess.run(["bash", str(A / "gate_check.sh")], input=line + "\n", text=True, check=False).returncode


def test_gate_check():
    assert _gate("2026-10-21T03:00:00Z cores 3.99 over 10.0 s; throttled 0 of 100 periods") == 0
    assert _gate("2026-10-21T03:00:00Z cores 4.00 over 10.0 s; throttled 0 of 100 periods") == 0
    assert _gate("2026-10-21T03:00:00Z cores 4.01 over 10.0 s; throttled 0 of 100 periods") != 0
    assert _gate("2026-10-21T03:00:00Z cores 2.00 over 10.0 s; throttled 3 of 100 periods") != 0
    assert _gate("garbage") != 0
    assert _gate("") != 0


def test_real_cpu_load_line_parses():
    """The real gate tool's line has the format gate_check.sh reads (open or closed, never a parse failure)."""
    import os
    import re
    py_dl = os.environ["PY_DL"]
    cpu = f"{_REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py"
    line = subprocess.run([py_dl, "-I", "-B", cpu, "2"], capture_output=True, text=True, check=True).stdout.strip()
    m = re.search(r"cores ([0-9.]+) over .*throttled ([0-9]+) of", line)
    assert m, line
    expect = 0 if (float(m.group(1)) <= 4.0 and m.group(2) == "0") else 1
    assert _gate(line) == expect


def test_run_job_wrapper_end_to_end():
    """run_job.sh: the gate loop (closed, then open; lines from a test file), the lock, nice, python -I -B, the
    listings and the exit code of the job; the test hook is refused for the real job log."""
    import os
    d = TEST_TMP / "wrapper"
    d.mkdir(parents=True, exist_ok=True)
    lines = d / "gate_lines.txt"
    lines.write_text("2026-10-21T03:00:00Z cores 4.80 over 10.0 s; throttled 0 of 100 periods\n"
                     "2026-10-21T03:00:10Z cores 2.00 over 10.0 s; throttled 0 of 100 periods\n")
    env = dict(os.environ, A_JOBLOG=str(d / "jobs.log"), PART3_CPU_LOCK=str(d / "lock"), A_TEST_GATE_LINES=str(lines),
               A_TEST_GATE_SLEEP="0")
    (d / "jobs.log").write_text("")
    r = subprocess.run(["bash", str(A / "run_job.sh"), "tm", "listing.py"], env=env, capture_output=True, text=True,
                       timeout=180, check=False)
    assert r.returncode == 2  # listing.py without arguments prints its usage and returns 2
    log = (d / "jobs.log").read_text()
    assert log.index("gate closed TEST (") < log.index("gate OPEN TEST (") < log.index("start listing")
    assert "exit 2" in log and "listing: no change" in log and lines.read_text() == ""
    r = subprocess.run(["bash", str(A / "run_job.sh"), "xx", "listing.py"], env=env, capture_output=True, text=True,
                       timeout=60, check=False)
    assert r.returncode == 2 and "tm or dl" in r.stdout
    env_real = dict(env)
    env_real.pop("A_JOBLOG")
    r = subprocess.run(["bash", str(A / "run_job.sh"), "tm", "listing.py"], env=env_real, capture_output=True, text=True,
                       timeout=60, check=False)
    assert r.returncode == 2 and "self-tests only" in r.stdout
    # the test mode marks the job A_RUNNER=test (real outputs refuse it) and runs it at nice 19
    lines.write_text("2026-10-21T03:00:20Z cores 1.00 over 10.0 s; throttled 0 of 100 periods\n")
    r = subprocess.run(["bash", str(A / "run_job.sh"), "tm", "tests/print_runner_env.py"], env=env, capture_output=True,
                       text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    import json
    got = json.loads([x for x in r.stdout.splitlines() if x.startswith("{")][0])
    assert got["A_RUNNER"] == "test" and got["nice"] == 19 and got["OMP_NUM_THREADS"] == "2"


def test_run_job_real_mode_needs_the_chain_lock():
    """Amendment 1: a real job (no test hook) starts only under the Part 3 chain lock; every refusal comes before
    anything is written (no A/out/jobs.log line, no wait for the CPU gate)."""
    import os
    sys.path.insert(0, str(A))
    from astruct import paths
    d = TEST_TMP / "wrapper_real"
    d.mkdir(parents=True, exist_ok=True)
    real_log = A / "out" / "jobs.log"
    size0 = real_log.stat().st_size if real_log.exists() else None
    base = {k: v for k, v in os.environ.items() if k not in ("A_TEST_GATE_LINES", "A_JOBLOG", "PART3_CHAIN_LOCK",
                                                              "PART3_CPU_LOCK", "A_RUNNER")}
    base.update(A_JOBLOG=str(d / "jobs.log"), PART3_CPU_LOCK=str(d / "lock"))
    chain = str(paths.CHAIN_LOCK)
    cases = [({}, "start only through chain/part3_cpu_job.sh"),
             ({"PART3_CHAIN_LOCK": str(d / "other.lock")}, "start only through chain/part3_cpu_job.sh"),
             ({"PART3_CHAIN_LOCK": chain, "PART3_CPU_LOCK": chain}, "must not be the chain lock")]
    if not paths.lock_held(paths.CHAIN_LOCK):  # never while a real chain job holds it
        cases.append(({"PART3_CHAIN_LOCK": chain}, "is not held"))
    for extra, msg in cases:
        r = subprocess.run(["bash", str(A / "run_job.sh"), "tm", "listing.py"], env={**base, **extra}, capture_output=True,
                           text=True, timeout=60, check=False)
        assert r.returncode == 2 and msg in r.stdout, (extra, r.stdout, r.stderr)
    assert not (d / "jobs.log").exists()
    assert (real_log.stat().st_size if real_log.exists() else None) == size0


def test_lock_held_probe():
    import fcntl
    import os
    d = TEST_TMP / "lock_probe"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "probe.lock"
    if f.exists():
        f.unlink()
    probe = lambda: subprocess.run(["bash", str(A / "lock_held.sh"), str(f)], check=False).returncode
    assert probe() == 1 and not f.exists()  # no file: not held, and the probe does not create it
    f.write_text("")
    assert probe() == 1  # unlocked
    fd = os.open(str(f), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert probe() == 0  # held by another open file description
    finally:
        os.close(fd)
    assert probe() == 1
