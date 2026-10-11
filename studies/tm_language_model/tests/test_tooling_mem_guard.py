"""scripts/mem_guard.py: exit code passthrough, and the kill path against a fake cgroup."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

GIB = 1024**3


@pytest.fixture
def guard(scripts_dir):
    def run(args, command, timeout=60):
        return subprocess.run([sys.executable, str(scripts_dir / "mem_guard.py"), *args, "--",
                               *command], capture_output=True, text=True, timeout=timeout)
    return run


def fake_cgroup(tmp_path: Path, current: int, inactive_file: int | None = 0, anon: int | None = None,
                active_file: int = 0) -> Path:
    """anon defaults to what is left after the file cache, so 'in use' means non-reclaimable."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir(exist_ok=True)
    (cgroup / "memory.current").write_text(f"{current}\n")
    if inactive_file is not None:
        a = current - inactive_file - active_file if anon is None else anon
        (cgroup / "memory.stat").write_text(f"anon {a}\nactive_file {active_file}\ninactive_file {inactive_file}\n"
                                            f"file {active_file + inactive_file}\n")
    return cgroup


@pytest.mark.parametrize("code", [0, 1, 5, 42])
def test_exit_code_passthrough_with_the_real_cgroup(guard, code):
    proc = guard(["--interval-s", "0.05"], ["bash", "-c", f"sleep 0.2; exit {code}"])
    assert proc.returncode == code
    assert "mem_guard" not in proc.stderr


def test_default_limit_is_85_gib(guard):
    proc = guard(["--help"], ["true"])
    assert proc.returncode == 0
    assert "kill above this usage (default: 85.0)" in " ".join(proc.stdout.split())


def test_output_and_arguments_pass_through(guard):
    proc = guard([], ["bash", "-c", 'echo "out:$1"; echo "err:$2" >&2', "job", "a b", "--limit-gib"])
    assert proc.returncode == 0
    assert proc.stdout == "out:a b\n"
    assert proc.stderr == "err:--limit-gib\n"


def test_death_by_signal_is_128_plus_n(guard):
    proc = guard(["--interval-s", "0.05"], ["bash", "-c", "kill -SEGV $$"])
    assert proc.returncode == 128 + signal.SIGSEGV


def test_command_not_found(guard):
    assert guard([], ["/no/such/program"]).returncode == 127


@pytest.mark.parametrize("args", [[], ["--limit-gib", "0"], ["--limit-gib", "x"], ["--interval-s", "0"]])
def test_usage_errors(scripts_dir, args):
    command = [] if not args else ["--", "true"]
    proc = subprocess.run([sys.executable, str(scripts_dir / "mem_guard.py"), *args, *command],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2


def test_unreadable_cgroup_fails_fast(guard, tmp_path):
    proc = guard(["--cgroup-dir", str(tmp_path / "missing")], ["bash", "-c", "exit 0"])
    assert proc.returncode == 2
    assert "cannot read" in proc.stderr


def test_guard_terminates_the_whole_process_group(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, current=10 * GIB)
    marker = tmp_path / "pids"
    # The command starts a child of its own: both must die.
    script = f'sleep 300 & echo "$$ $!" >"{marker}"; wait'
    begin = time.monotonic()
    proc = guard(["--limit-gib", "1", "--interval-s", "0.05", "--grace-s", "5",
                  "--cgroup-dir", str(cgroup)], ["bash", "-c", script])
    assert proc.returncode == 137
    assert time.monotonic() - begin < 30
    assert "exceeds the limit" in proc.stderr and "SIGTERM" in proc.stderr
    assert "SIGKILL" not in proc.stderr  # SIGTERM was enough, no escalation message
    for pid in marker.read_text().split():
        assert not Path(f"/proc/{pid}").exists(), f"process {pid} survived the guard"


def test_guard_escalates_to_sigkill(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, current=10 * GIB)
    begin = time.monotonic()
    proc = guard(["--limit-gib", "1", "--interval-s", "0.05", "--grace-s", "0.5",
                  "--cgroup-dir", str(cgroup)], ["bash", "-c", 'trap "" TERM; sleep 300'])
    assert proc.returncode == 137
    assert time.monotonic() - begin < 30
    assert "SIGKILL" in proc.stderr


def test_guard_reports_137_even_when_the_command_exits_cleanly_on_sigterm(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, current=10 * GIB)
    proc = guard(["--limit-gib", "1", "--interval-s", "0.05", "--cgroup-dir", str(cgroup)],
                 ["bash", "-c", 'trap "exit 0" TERM; sleep 300 & wait'])
    assert proc.returncode == 137


def test_guard_stays_quiet_below_the_limit(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, current=10 * GIB)
    proc = guard(["--limit-gib", "11", "--interval-s", "0.05", "--cgroup-dir", str(cgroup)],
                 ["bash", "-c", "sleep 0.5; exit 4"])
    assert proc.returncode == 4
    assert proc.stderr == ""


def test_working_set_ignores_reclaimable_page_cache(guard, tmp_path):
    # 10 GiB charged, 9.5 GiB of it is inactive file cache: working set 0.5 GiB.
    cgroup = fake_cgroup(tmp_path, current=10 * GIB, inactive_file=int(9.5 * GIB))
    base = ["--limit-gib", "1", "--interval-s", "0.05", "--grace-s", "1", "--cgroup-dir", str(cgroup)]
    calm = guard(base, ["bash", "-c", "sleep 0.5; exit 3"])
    assert calm.returncode == 3
    strict = guard([*base, "--metric", "current"], ["bash", "-c", "sleep 300"])
    assert strict.returncode == 137


def test_guard_reacts_when_memory_grows_during_the_run(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, current=1 * GIB)
    grow = f'sleep 0.5; echo {10 * GIB} >"{cgroup}/memory.current"; sleep 300'
    proc = guard(["--limit-gib", "2", "--interval-s", "0.05", "--grace-s", "1",
                  "--cgroup-dir", str(cgroup)], ["bash", "-c", grow])
    assert proc.returncode == 137


def test_sigterm_to_the_guard_is_forwarded(scripts_dir, tmp_path):
    marker = tmp_path / "started"
    proc = subprocess.Popen(
        [sys.executable, str(scripts_dir / "mem_guard.py"), "--interval-s", "0.05", "--",
         "bash", "-c", f'trap "exit 21" TERM; touch "{marker}"; sleep 300 & wait'])
    deadline = time.monotonic() + 20
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 21


# --scope group (added 2026-09-24, docs/12 F25): the limit applies to the command's own process
# group, so jobs sharing the container do not kill each other; the container cap stays as a last resort.

def test_group_scope_ignores_other_jobs_in_the_container(guard, tmp_path):
    # the fake cgroup says 50 GiB are in use, far above the 1 GiB limit, but this job is small
    cgroup = fake_cgroup(tmp_path, 50 * GIB)
    proc = guard(["--scope", "group", "--limit-gib", "1", "--interval-s", "0.05",
                  "--cgroup-dir", str(cgroup)], ["bash", "-c", "sleep 0.5; exit 3"])
    assert proc.returncode == 3
    assert "mem_guard" not in proc.stderr


def test_group_scope_fires_on_the_jobs_own_rss(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, 1 * GIB)
    grow = [sys.executable, "-c", "import time; b = bytearray(300 * 1024 * 1024); time.sleep(30)"]
    proc = guard(["--scope", "group", "--limit-gib", "0.1", "--interval-s", "0.05", "--grace-s", "1",
                  "--cgroup-dir", str(cgroup)], grow, timeout=60)
    assert proc.returncode == 137
    assert "process-group RSS" in proc.stderr


def test_group_scope_keeps_the_container_cap(guard, tmp_path):
    cgroup = fake_cgroup(tmp_path, 95 * GIB)
    proc = guard(["--scope", "group", "--limit-gib", "50", "--cgroup-cap-gib", "90", "--interval-s", "0.05",
                  "--grace-s", "1", "--cgroup-dir", str(cgroup)], ["sleep", "30"], timeout=60)
    assert proc.returncode == 137
    assert "exceeds the cap" in proc.stderr


def test_group_rss_counts_only_the_group(scripts_dir):
    import importlib
    sys.path.insert(0, str(scripts_dir))
    try:
        mg = importlib.import_module("mem_guard")
    finally:
        sys.path.pop(0)
    child = subprocess.Popen([sys.executable, "-c", "import time; b = bytearray(64 * 1024 * 1024); time.sleep(10)"],
                             process_group=0)
    try:
        time.sleep(1.0)
        rss = mg.group_rss(child.pid)
        assert 64 * 1024 * 1024 <= rss < 2 * GIB
        assert mg.group_rss(2**22 + 12345) == 0
    finally:
        child.kill()
        child.wait()


# 2026-09-25: the container cap of --scope group counts non-reclaimable memory by default (--cap-metric), and inherited
# GPU lock / CPU slot descriptors are passed to the command.

def test_group_cap_ignores_active_file_cache(guard, tmp_path):
    # 96 GiB charged: 5 GiB anon, 86 GiB active file cache, 5 GiB inactive: working set 91 GiB, non-reclaimable 5 GiB
    cgroup = fake_cgroup(tmp_path, 96 * GIB, inactive_file=5 * GIB, anon=5 * GIB, active_file=86 * GIB)
    base = ["--scope", "group", "--limit-gib", "50", "--cgroup-cap-gib", "90", "--interval-s", "0.05", "--grace-s", "1",
            "--cgroup-dir", str(cgroup)]
    calm = guard(base, ["bash", "-c", "sleep 0.5; exit 3"])
    assert calm.returncode == 3 and "mem_guard" not in calm.stderr
    strict = guard([*base, "--cap-metric", "working_set"], ["sleep", "30"], timeout=60)
    assert strict.returncode == 137 and "working_set" in strict.stderr


def test_nonreclaimable_needs_an_anon_line(guard, tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("100\n")
    (cgroup / "memory.stat").write_text("file 5\n")
    proc = guard(["--scope", "group", "--cgroup-dir", str(cgroup)], ["true"])
    assert proc.returncode == 2


def test_lock_descriptors_are_passed_and_others_dropped(scripts_dir, tmp_path):
    (tmp_path / "gpu.lock").write_text("")
    (tmp_path / "cpu_slot_3.lock").write_text("")
    (tmp_path / "other.txt").write_text("")
    probe = "for f in 7 8 9; do [ -e /proc/self/fd/$f ] && echo open$f || echo closed$f; done"
    script = (f'exec 7<"{tmp_path}/gpu.lock" 8<"{tmp_path}/cpu_slot_3.lock" 9<"{tmp_path}/other.txt"; '
              f'TMLM_GPU_LOCK_FD=7 TMLM_CPU_SLOT_FDS="8 9" "{sys.executable}" "{scripts_dir / "mem_guard.py"}" '
              f'--scope group --limit-gib 1 -- bash -c \'{probe}\'')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["open7", "open8", "closed9"]
