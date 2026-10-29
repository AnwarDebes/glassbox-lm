"""scripts/gpu_run.sh and scripts/cpu_slot.sh, with short sleep commands.

Every test uses its own lock directory (TMLM_LOCK_DIR, set by conftest.py), never the
real results/.locks. No GPU is touched: the "GPU jobs" are sleeps.

Timing: mutual exclusion is asserted on intervals that each job logs WHILE it holds the
lock, so a slow machine can never cause a false failure. Only the "these two overlap"
assertions depend on timing, and they use sleeps far longer than process start-up.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Logs "<tag> start <t>" on entry and "<tag> end <t>" on exit, inside the lock.
JOB = 'echo "$1 start $(date +%s.%N)" >>"$2"; sleep "$3"; echo "$1 end $(date +%s.%N)" >>"$2"'


# Logs the start, then BECOMES the sleep (exec): killing the process really ends the job.
# With JOB, a SIGKILL would orphan the inner `sleep`, which inherited the lock descriptor
# and would correctly keep the lock until it ends.
HOLDER = 'echo "$1 start $(date +%s.%N)" >>"$2"; exec sleep "$3"'


def job(tag: str, log: Path, seconds: float) -> list[str]:
    return ["bash", "-c", JOB, "job", tag, str(log), str(seconds)]


def holder_job(tag: str, log: Path, seconds: float) -> list[str]:
    return ["bash", "-c", HOLDER, "job", tag, str(log), str(seconds)]


def intervals(log: Path) -> dict[str, tuple[float, float]]:
    found: dict[str, dict[str, float]] = {}
    for line in log.read_text().splitlines():
        tag, kind, stamp = line.split()
        found.setdefault(tag, {})[kind] = float(stamp)
    return {tag: (times["start"], times["end"]) for tag, times in found.items()}


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def wait_for(predicate, timeout: float = 20.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def started(log: Path, tag: str) -> bool:
    return log.exists() and f"{tag} start" in log.read_text()


@pytest.fixture
def gpu_run(scripts_dir) -> str:
    return str(scripts_dir / "gpu_run.sh")


@pytest.fixture
def cpu_slot(scripts_dir) -> str:
    return str(scripts_dir / "cpu_slot.sh")


@pytest.fixture
def lock_dir() -> Path:
    return Path(os.environ["TMLM_LOCK_DIR"])


# ------------------------------------------------------------------ gpu_run.sh


@pytest.mark.parametrize("mode", ["shared", "exclusive"])
@pytest.mark.parametrize("code", [0, 7])
def test_gpu_run_passes_exit_code_through(gpu_run, mode, code):
    proc = subprocess.run([gpu_run, mode, "--", "bash", "-c", f"exit {code}"], timeout=60)
    assert proc.returncode == code


def test_gpu_run_exports_lock_info_and_passes_arguments(gpu_run, lock_dir):
    proc = subprocess.run(
        [gpu_run, "shared", "--", "bash", "-c",
         'echo "$TMLM_GPU_LOCK_MODE|$TMLM_LOCK_DIR|$1|$2"; readlink "/proc/$$/fd/$TMLM_GPU_LOCK_FD"',
         "job", "two words", "--flag"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    first, second = proc.stdout.splitlines()
    assert first == f"shared|{lock_dir.resolve()}|two words|--flag"
    assert second == str(lock_dir.resolve() / "gpu.lock")


@pytest.mark.parametrize("argv", [[], ["shared"], ["shared", "--"], ["bogus", "--", "true"],
                                  ["shared", "true"]])
def test_gpu_run_usage_errors(gpu_run, argv):
    proc = subprocess.run([gpu_run, *argv], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "usage" in proc.stderr


def test_gpu_run_command_not_found(gpu_run):
    proc = subprocess.run([gpu_run, "shared", "--", "/no/such/program"], capture_output=True,
                          timeout=60)
    assert proc.returncode == 127


def test_gpu_shared_jobs_overlap(gpu_run, tmp_path):
    log = tmp_path / "log"
    procs = [subprocess.Popen([gpu_run, "shared", "--", *job(tag, log, 1.5)]) for tag in "ab"]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    spans = intervals(log)
    assert overlap(spans["a"], spans["b"])


def test_gpu_exclusive_excludes_shared_and_exclusive(gpu_run, tmp_path):
    log = tmp_path / "log"
    first = subprocess.Popen([gpu_run, "exclusive", "--", *job("x1", log, 0.8)])
    assert wait_for(lambda: started(log, "x1"))
    others = [
        subprocess.Popen([gpu_run, "shared", "--", *job("s1", log, 0.3)]),
        subprocess.Popen([gpu_run, "exclusive", "--", *job("x2", log, 0.3)]),
        subprocess.Popen([gpu_run, "shared", "--", *job("s2", log, 0.3)]),
    ]
    assert first.wait(timeout=60) == 0
    assert [p.wait(timeout=60) for p in others] == [0, 0, 0]
    spans = intervals(log)
    assert set(spans) == {"x1", "x2", "s1", "s2"}
    for exclusive in ("x1", "x2"):
        for other in spans:
            if other != exclusive:
                assert not overlap(spans[exclusive], spans[other]), (exclusive, other, spans)


def test_gpu_exclusive_request_flag_and_writer_priority(gpu_run, tmp_path, lock_dir):
    log = tmp_path / "log"
    flag = lock_dir / "gpu_exclusive_request"
    holder = subprocess.Popen([gpu_run, "shared", "--", *job("holder", log, 1.5)])
    assert wait_for(lambda: started(log, "holder"))
    assert not flag.exists()

    waiter = subprocess.Popen([gpu_run, "exclusive", "--", *job("excl", log, 0.3)])
    assert wait_for(flag.exists), "no request flag while the exclusive job waits"
    assert not started(log, "excl")

    # A shared job arriving now must queue BEHIND the waiting exclusive job, although a
    # plain shared flock would be granted at once next to the running shared holder.
    late = subprocess.Popen([gpu_run, "shared", "--", *job("late", log, 0.1)])

    assert [p.wait(timeout=60) for p in (holder, waiter, late)] == [0, 0, 0]
    spans = intervals(log)
    assert spans["holder"][1] <= spans["excl"][0]
    assert spans["excl"][1] <= spans["late"][0]
    assert not flag.exists(), "request flag must be removed once the lock is acquired"
    assert list((lock_dir / "gpu_exclusive_waiters").iterdir()) == []


def test_gpu_request_flag_is_removed_when_the_waiter_is_killed(gpu_run, tmp_path, lock_dir):
    log = tmp_path / "log"
    flag = lock_dir / "gpu_exclusive_request"
    holder = subprocess.Popen([gpu_run, "shared", "--", *holder_job("holder", log, 30)])
    try:
        assert wait_for(lambda: started(log, "holder"))
        waiter = subprocess.Popen([gpu_run, "exclusive", "--", *job("never", log, 0.1)])
        assert wait_for(flag.exists)
        waiter.send_signal(signal.SIGTERM)
        assert waiter.wait(timeout=30) == 143
        assert not flag.exists()
        assert not started(log, "never")

        # A waiter that dies without cleanup (SIGKILL) leaves a stale marker. The next
        # shared job prunes it instead of queueing behind a ghost forever.
        ghost = subprocess.Popen([gpu_run, "exclusive", "--", *job("ghost", log, 0.1)])
        assert wait_for(flag.exists)
        ghost.kill()
        ghost.wait(timeout=30)
        assert flag.exists()
        fresh = subprocess.run([gpu_run, "shared", "--", *job("fresh", log, 0.1)], timeout=60)
        assert fresh.returncode == 0
        assert not flag.exists()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_gpu_lock_is_released_when_the_holder_is_killed(gpu_run, tmp_path):
    log = tmp_path / "log"
    holder = subprocess.Popen([gpu_run, "exclusive", "--", *holder_job("holder", log, 300)])
    assert wait_for(lambda: started(log, "holder"))
    holder.kill()  # after exec, this pid IS the job: no wrapper is left to clean up
    holder.wait(timeout=30)
    begin = time.monotonic()
    proc = subprocess.run([gpu_run, "exclusive", "--", *job("next", log, 0.1)], timeout=60)
    assert proc.returncode == 0
    assert time.monotonic() - begin < 30


def test_gpu_nested_calls_do_not_deadlock(gpu_run):
    inner_ok = subprocess.run(
        [gpu_run, "exclusive", "--", gpu_run, "shared", "--", gpu_run, "exclusive", "--",
         "bash", "-c", 'echo "$TMLM_GPU_LOCK_MODE"'],
        capture_output=True, text=True, timeout=60)
    assert inner_ok.returncode == 0, inner_ok.stderr
    assert inner_ok.stdout.strip() == "exclusive"

    refused = subprocess.run([gpu_run, "shared", "--", gpu_run, "exclusive", "--", "true"],
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2
    assert "deadlock" in refused.stderr


def test_gpu_shared_run_can_yield_to_an_exclusive_job(gpu_run, tmp_path, repo_root):
    """tmlm.common.locks.yield_gpu_if_requested: release, let the exclusive job run, resume."""
    log = tmp_path / "log"
    trainer = (
        "import sys, time; sys.path.insert(0, sys.argv[1])\n"
        "from tmlm.common import locks\n"
        "log = open(sys.argv[2], 'a', buffering=1)\n"
        "log.write('trainer start %.9f\\n' % time.time())\n"
        "yielded = 0\n"
        "for step in range(400):\n"
        "    time.sleep(0.02)\n"
        "    if locks.yield_gpu_if_requested(poll_s=0.02):\n"
        "        yielded += 1\n"
        "        log.write('resumed start %.9f\\n' % time.time())\n"
        "        break\n"
        "log.write('resumed end %.9f\\n' % time.time())\n"
        "log.write('trainer end %.9f\\n' % time.time())\n"
        "sys.exit(0 if yielded == 1 else 5)\n"
    )
    holder = subprocess.Popen([gpu_run, "shared", "--", sys.executable, "-c", trainer,
                               str(repo_root / "src"), str(log)])
    assert wait_for(lambda: started(log, "trainer"))
    excl = subprocess.run([gpu_run, "exclusive", "--", *job("excl", log, 0.5)], timeout=60)
    assert excl.returncode == 0
    assert holder.wait(timeout=60) == 0, "the trainer never saw the exclusive request"
    spans = intervals(log)
    assert spans["trainer"][0] < spans["excl"][0] and spans["excl"][1] < spans["trainer"][1]
    assert spans["excl"][1] <= spans["resumed"][0], "trainer resumed before the exclusive job ended"


def test_yield_is_a_no_op_outside_gpu_run():
    from tmlm.common import locks

    assert locks.held_gpu_lock_fd() is None
    assert locks.exclusive_requested() is False
    assert locks.yield_gpu_if_requested() is False


# ------------------------------------------------------------------ cpu_slot.sh


@pytest.mark.parametrize("code", [0, 9])
def test_cpu_slot_passes_exit_code_through(cpu_slot, code):
    proc = subprocess.run([cpu_slot, "1", "--", "bash", "-c", f"exit {code}"], timeout=60)
    assert proc.returncode == code


@pytest.mark.parametrize("slots", [1, 4, 6])
def test_cpu_slot_exports_thread_caps(cpu_slot, slots, repo_root):
    names = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "TMLM_CPU_SLOTS"]
    script = "; ".join(f'echo "${{{name}}}"' for name in names)
    script += '; echo "$TOKENIZERS_PARALLELISM"; echo "$TMLM_ROOT"; echo "$TMLM_CPU_SLOT_FDS" | wc -w'
    # Hermetic (2026-10-25): a caller that already caps its own thread pools below N (the min rule
    # of env/env.sh keeps such a value) must not decide this test; the jobs here are echo lines.
    env = {key: value for key, value in os.environ.items() if key not in names}
    proc = subprocess.run([cpu_slot, str(slots), "--", "bash", "-c", script],
                          capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.split()
    assert lines[:7] == [str(slots)] * 7
    assert lines[7:] == ["false", str(repo_root), str(slots)]


@pytest.mark.parametrize("argv", [[], ["2"], ["2", "true"], ["0", "--", "true"], ["7", "--", "true"],
                                  ["abc", "--", "true"], ["-1", "--", "true"], ["2.5", "--", "true"]])
def test_cpu_slot_usage_errors(cpu_slot, argv):
    proc = subprocess.run([cpu_slot, *argv], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "usage" in proc.stderr


def test_cpu_slot_never_exceeds_six_concurrent_slots(cpu_slot, tmp_path):
    log = tmp_path / "log"
    requests = {"a": 3, "b": 2, "c": 4, "d": 1, "e": 3, "f": 6}
    procs = [subprocess.Popen([cpu_slot, str(slots), "--", *job(tag, log, 1.0)])
             for tag, slots in requests.items()]
    assert [p.wait(timeout=120) for p in procs] == [0] * len(procs)
    spans = intervals(log)
    assert set(spans) == set(requests)

    events = []
    for tag, (begin, end) in spans.items():
        events.append((begin, 1, requests[tag]))
        events.append((end, 0, -requests[tag]))  # at equal times, releases sort first
    in_use = peak = 0
    for _, _, delta in sorted(events):
        in_use += delta
        peak = max(peak, in_use)
    assert peak <= 6, f"{peak} slots in use at once: {spans}"
    # The semaphore must also allow real concurrency, not just serialize everything:
    # "a" and "b" arrive together and fit together (3 + 2 slots).
    tags = sorted(spans)
    assert any(overlap(spans[x], spans[y]) for i, x in enumerate(tags) for y in tags[i + 1:])
    for other in tags:
        if other != "f":
            assert not overlap(spans["f"], spans[other]), "a 6-slot job must run alone"


def test_cpu_slots_are_released_when_the_holder_is_killed(cpu_slot, tmp_path):
    log = tmp_path / "log"
    holder = subprocess.Popen([cpu_slot, "6", "--", *holder_job("holder", log, 300)])
    assert wait_for(lambda: started(log, "holder"))
    holder.kill()
    holder.wait(timeout=30)
    proc = subprocess.run([cpu_slot, "6", "--", *job("next", log, 0.1)], timeout=60)
    assert proc.returncode == 0


def test_cpu_slot_waiter_can_be_cancelled(cpu_slot, tmp_path):
    log = tmp_path / "log"
    holder = subprocess.Popen([cpu_slot, "6", "--", *holder_job("holder", log, 30)])
    try:
        assert wait_for(lambda: started(log, "holder"))
        waiter = subprocess.Popen([cpu_slot, "1", "--", *job("never", log, 0.1)])
        time.sleep(0.5)
        waiter.send_signal(signal.SIGTERM)
        assert waiter.wait(timeout=30) == 143
        assert not started(log, "never")
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_cpu_slot_nested_calls_do_not_deadlock(cpu_slot):
    inner = subprocess.run([cpu_slot, "4", "--", cpu_slot, "2", "--", "bash", "-c",
                            'echo "$OMP_NUM_THREADS"'], capture_output=True, text=True, timeout=60)
    assert inner.returncode == 0, inner.stderr
    assert inner.stdout.strip() == "2"
    refused = subprocess.run([cpu_slot, "2", "--", cpu_slot, "4", "--", "true"],
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2
    assert "holds 2" in refused.stderr


# ------------------------------------------------------------------ env/env.sh


def test_env_sh_is_idempotent_and_sets_the_caps(repo_root):
    script = (
        "set -euo pipefail\n"
        f'source "{repo_root}/env/env.sh"\n'
        'first="$PATH|$LD_LIBRARY_PATH"\n'
        f'source "{repo_root}/env/env.sh"\n'
        '[ "$first" = "$PATH|$LD_LIBRARY_PATH" ] || { echo "not idempotent"; exit 1; }\n'
        'echo "$OMP_NUM_THREADS $MKL_NUM_THREADS $OPENBLAS_NUM_THREADS $NUMEXPR_NUM_THREADS '
        '$NUMBA_NUM_THREADS $RAYON_NUM_THREADS $TOKENIZERS_PARALLELISM"\n'
        'echo "$TMLM_ROOT|$CUDA_ROOT|$CUDA_HOME"\n'
        'echo "$PATH"\n'
        'echo "$LD_LIBRARY_PATH"\n'
        'declare -F _tmlm_prepend >/dev/null && { echo "helper leaked"; exit 1; }\n'
        "exit 0\n"
    )
    home = "/home/someone"
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env={"PATH": "/usr/bin:/bin", "HOME": home})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    caps, roots, path, ld_path = proc.stdout.splitlines()
    assert caps == "6 6 6 6 6 6 false"
    assert roots == f"{repo_root}|/usr/local/cuda|/usr/local/cuda"
    # Since 2026-10-25 (finish pass, SHR-1) the study interpreter's folder comes first when the
    # environment is built, so that `#!/usr/bin/env python3` finds a python3.
    venv_bin = repo_root / ".venv-tm" / "bin"
    first = f"{venv_bin}:" if os.access(venv_bin / "python3", os.X_OK) else ""
    assert path == f"{first}{home}/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin"
    assert ld_path == "/usr/local/cuda/lib64"


def test_env_sh_never_raises_the_caps_above_six(repo_root):
    for requested, expected in (("3", "3"), ("6", "6"), ("64", "6"), ("0", "6"), ("x", "6")):
        proc = subprocess.run(
            ["bash", "-c", f'source "{repo_root}/env/env.sh"; echo "$OMP_NUM_THREADS"'],
            capture_output=True, text=True, timeout=60,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TMLM_CPU_SLOTS": requested})
        assert proc.stdout.strip() == expected


# ------------------------------------------------------------------ python3 on PATH (SHR-1)
# The helper scripts start with `#!/usr/bin/env python3`, and the launch lines start
# scripts/run_logged.py (and, in docstrings, scripts/mem_guard.py) by name. The machine of the
# study has no python3 of its own on PATH, so env/env.sh must provide one: the interpreter of
# .venv-tm. Both tests skip in a clone where .venv-tm is not built.


def _venv_python3(repo_root: Path) -> Path:
    python3 = repo_root / ".venv-tm" / "bin" / "python3"
    if not os.access(python3, os.X_OK):
        pytest.skip(".venv-tm is not built")
    return python3


def test_env_sh_puts_the_study_python3_first_on_path(repo_root):
    python3 = _venv_python3(repo_root)
    script = (
        "set -euo pipefail\n"
        f'source "{repo_root}/env/env.sh"\n'
        "command -v python3\n"
        "command -v python\n"
        f'source "{repo_root}/env/env.sh"\n'
        "command -v python3\n"
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.split() == [str(python3), str(python3.with_name("python")), str(python3)]


def test_helpers_start_by_name_after_sourcing_env_sh(repo_root, tmp_path):
    """mem_guard.py and run_logged.py started by name, the command found as python3 on PATH:
    the composition of the recorded launch lines. Exit status 127 before the fix."""
    _venv_python3(repo_root)
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"name": "byname"}')
    script = (
        "set -uo pipefail\n"
        f'source "{repo_root}/env/env.sh"\n'
        f'cd "{repo_root}"\n'
        "scripts/mem_guard.py --scope group --limit-gib 4 -- scripts/run_logged.py --no-gpu-sampler "
        f'--name byname --config "{cfg}" --results-dir "{tmp_path}/res" -- '
        "python3 -c 'print(12345)'\n"
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = sorted((tmp_path / "res").glob("*_byname"))
    assert len(runs) == 1
    assert "12345" in (runs[0] / "log.txt").read_text()
    assert json.loads((runs[0] / "run.json").read_text())["exit_status"] == 0
