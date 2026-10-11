"""Python side of scripts/gpu_run.sh and scripts/cpu_slot.sh.

Protocol (the shell side is documented in scripts/gpu_run.sh):
  - gpu_run.sh holds a flock on results/.locks/gpu.lock through a file descriptor that
    the wrapped command inherits. It exports TMLM_GPU_LOCK_FD (the descriptor number),
    TMLM_GPU_LOCK_MODE (shared or exclusive) and TMLM_LOCK_DIR. cpu_slot.sh does the
    same with TMLM_CPU_SLOT_FDS for its slot locks.
  - A job waiting for the EXCLUSIVE lock (timing, energy, profiling) keeps a marker
    results/.locks/gpu_exclusive_waiters/<pid> in place until it has acquired the lock.
    The file results/.locks/gpu_exclusive_request mirrors "some marker exists" for the
    shell side. A marker counts only while its process is alive, so a waiter that was
    killed with SIGKILL never stalls anybody: see `exclusive_requested`.
  - A long shared run calls `yield_gpu_if_requested()` at a safe point (after a
    checkpoint, with no kernel in flight). If a live request is pending, it releases
    its shared lock, waits until no live waiter is left (the exclusive job now holds
    the lock, or it died), then blocks re-acquiring the shared lock until the
    exclusive job ends.

Waiting for the waiters to disappear before re-locking matters: Linux flock gives
writers no priority, so re-locking at once could win the lock back before the waiting
exclusive job is scheduled.

Descriptors and Python wrappers: `subprocess.Popen` closes every descriptor above 2 in
the child unless it is listed in `pass_fds`. A Python wrapper between gpu_run.sh or
cpu_slot.sh and the real job must therefore start the job with
`pass_fds=inherited_lock_fds()`, as RunLog.run_command and scripts/mem_guard.py do.
Otherwise the job cannot yield, and a nested gpu_run.sh or cpu_slot.sh call cannot see
its own lock (the shell side then falls back to looking at its ancestors).

A flock belongs to the open file description, which the wrappers and the job share.
So LOCK_UN in the job releases the lock although the wrappers keep their copies open.

This module uses no GPU library and no thread. Standard library only.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from pathlib import Path
from typing import Mapping

REQUEST_FILE = "gpu_exclusive_request"
WAITERS_DIR = "gpu_exclusive_waiters"
MUTEX_FILE = "gpu_request.mutex"
GPU_LOCK_FILE = "gpu.lock"
CPU_SLOT_GLOB = "cpu_slot_*.lock"

_warned_cannot_yield = False


def lock_dir(env: Mapping[str, str] | None = None) -> Path:
    """$TMLM_LOCK_DIR, else <project root>/results/.locks."""
    override = (os.environ if env is None else env).get("TMLM_LOCK_DIR")
    if override:
        return Path(override)
    from tmlm.common.runlog import project_root

    return project_root() / "results" / ".locks"


# --------------------------------------------------------------------------- descriptors


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _fd_identity(fd: int) -> tuple[int, int] | None:
    try:
        st = os.fstat(fd)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _ints(text: str | None) -> list[int]:
    return [int(part) for part in (text or "").split() if part.isdigit()]


def inherited_lock_fds(env: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Descriptors of THIS process that really are inherited GPU or CPU slot locks.

    Candidates are the numbers in TMLM_GPU_LOCK_FD and TMLM_CPU_SLOT_FDS of `env`
    (default os.environ). A candidate is kept only when it is open here and is the very
    file (same device and inode) <lock dir>/gpu.lock, respectively one of
    <lock dir>/cpu_slot_*.lock. A number that was reused for some other file is
    dropped. Pass the result as `pass_fds` when starting a job from Python.
    """
    source = os.environ if env is None else env
    directory = lock_dir(source)
    keep: list[int] = []
    gpu_identity = _identity(directory / GPU_LOCK_FILE)
    for fd in _ints(source.get("TMLM_GPU_LOCK_FD")):
        if fd > 2 and gpu_identity is not None and _fd_identity(fd) == gpu_identity:
            keep.append(fd)
    slot_fds = _ints(source.get("TMLM_CPU_SLOT_FDS"))
    if slot_fds:
        try:
            slots = {_identity(path) for path in directory.glob(CPU_SLOT_GLOB)} - {None}
        except OSError:
            slots = set()
        for fd in slot_fds:
            if fd > 2 and _fd_identity(fd) in slots and fd not in keep:
                keep.append(fd)
    return tuple(keep)


def held_gpu_lock_fd() -> int | None:
    """The inherited lock descriptor when running under `gpu_run.sh shared`, else None.

    The descriptor is checked for identity with <lock dir>/gpu.lock, not just for
    being open: when a wrapper closed the real one, the same number can belong to any
    file the job opened later. When the environment says "shared" but no valid
    descriptor is there, this run CANNOT yield: a warning is printed once.
    """
    global _warned_cannot_yield
    if os.environ.get("TMLM_GPU_LOCK_MODE") != "shared":
        return None
    reference = _identity(lock_dir() / GPU_LOCK_FILE)
    for fd in _ints(os.environ.get("TMLM_GPU_LOCK_FD")):
        if fd > 2 and reference is not None and _fd_identity(fd) == reference:
            return fd
    if not _warned_cannot_yield:
        _warned_cannot_yield = True
        print(
            "tmlm.common.locks: WARNING: TMLM_GPU_LOCK_MODE=shared but descriptor "
            f"TMLM_GPU_LOCK_FD={os.environ.get('TMLM_GPU_LOCK_FD')!r} is not the GPU lock "
            "(a wrapper closed it: start jobs with pass_fds=locks.inherited_lock_fds()). "
            "This run cannot yield the GPU to an exclusive job.",
            file=sys.stderr,
            flush=True,
        )
    return None


# --------------------------------------------------------------------------- waiters


def pid_alive(pid: int) -> bool:
    """True when /proc/<pid> is a process that still runs. A zombie (killed, not yet
    reaped by its parent) counts as dead: it will never take the lock. Same rule as
    tmlm_pid_alive in scripts/lib_locks.sh."""
    if pid <= 0:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    # "<pid> (<comm>) <state> ...": comm may contain spaces and parentheses.
    state = stat.rpartition(")")[2].split()
    return bool(state) and state[0] not in ("Z", "X", "x")


def live_waiters() -> list[int]:
    """Pids of the LIVE jobs that wait for the exclusive GPU lock."""
    try:
        names = os.listdir(lock_dir() / WAITERS_DIR)
    except OSError:
        return []
    return sorted(int(name) for name in names if name.isdigit() and pid_alive(int(name)))


def sync_requests(timeout_s: float = 1.0) -> bool:
    """Prune the markers of dead waiters and make the request flag match, under the
    same mutex as gpu_run.sh (function sync_requests there). Best effort: returns False
    when the lock directory is missing or the mutex stayed busy for `timeout_s`."""
    directory = lock_dir()
    waiters = directory / WAITERS_DIR
    if not waiters.is_dir():
        return False
    try:
        mutex = os.open(directory / MUTEX_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return False
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
        alive = False
        for name in os.listdir(waiters):
            if name.isdigit() and pid_alive(int(name)):
                alive = True
            else:
                try:
                    os.unlink(waiters / name)
                except OSError:
                    pass
        flag = directory / REQUEST_FILE
        try:
            if alive:
                flag.touch()
            else:
                flag.unlink(missing_ok=True)
        except OSError:
            pass
        return True
    except OSError:
        return False
    finally:
        os.close(mutex)  # closing drops the flock


def exclusive_requested() -> bool:
    """True while some LIVE job is waiting for the exclusive GPU lock.

    The bare request flag is not trusted: a waiter killed with SIGKILL leaves its marker
    and the flag behind. Leftovers of dead waiters are pruned on the way.
    """
    directory = lock_dir()
    if live_waiters():
        return True
    try:
        stale = (directory / REQUEST_FILE).exists() or bool(os.listdir(directory / WAITERS_DIR))
    except OSError:
        stale = False
    if stale:
        sync_requests()
        return bool(live_waiters())
    return False


def yield_gpu_if_requested(poll_s: float = 0.5, handover_timeout_s: float = 600.0) -> bool:
    """Give the GPU to a waiting exclusive job, then take the shared lock back.

    Call this only at a point where the GPU is idle for this process. Returns False at
    once when no live job is waiting or when this process does not hold the shared lock
    of `gpu_run.sh shared`. Otherwise blocks for as long as the exclusive job runs and
    returns True.

    Args:
        poll_s: polling period while waiting for the waiters to take the lock.
        handover_timeout_s: stop waiting for the handover after this long and re-lock
            anyway. A waiter that DIES during the handover ends the wait at the next
            poll, so this only guards against a live waiter that never gets scheduled.
    """
    fd = held_gpu_lock_fd()
    if fd is None or not exclusive_requested():
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    deadline = time.monotonic() + handover_timeout_s
    while exclusive_requested() and time.monotonic() < deadline:
        time.sleep(poll_s)
    fcntl.flock(fd, fcntl.LOCK_SH)
    return True
