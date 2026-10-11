#!/usr/bin/env python3
"""Abort a job before the cgroup OOM killer picks a victim for us.

    mem_guard.py [--limit-gib 85] [--interval-s 1] [--grace-s 10] -- <command...>

This container has memory.max = 96 GiB and no swap [stated: design notes]. The guard
runs the command in its own process group, polls the cgroup memory counters, and when
usage exceeds the limit sends SIGTERM to the whole group, then SIGKILL after the grace
period.

What is compared with the limit (--metric):
    working_set (default)  memory.current minus inactive_file from memory.stat. Page
                           cache that the kernel can drop at no cost is not counted, so
                           reading a big dataset does not kill the job.
    current                memory.current as is.
    nonreclaimable         anon + shmem + kernel + file_mapped from memory.stat: memory the kernel
                           cannot drop (added 2026-09-25: active file cache made the working set
                           overstate use by tens of GiB).
The counters are those of the whole cgroup (all processes in the container), because
that is what the OOM killer acts on.

What is guarded (--scope, added 2026-09-24, docs/12 F25):
    cgroup (default)       the limit applies to the whole container, as above. Unchanged
                           behaviour, so every earlier run replays the same way.
    group                  the limit applies to the resident memory (VmRSS summed over
                           /proc) of the command's own process group only, so several jobs
                           that share the container no longer kill each other; a second,
                           container-wide cap (--cgroup-cap-gib, default 90) still fires as a
                           last resort before the OOM killer, measured by --cap-metric (default
                           nonreclaimable since 2026-09-25).

Lock descriptors (added 2026-09-25): the GPU lock and CPU slot descriptors that the guard inherited
(TMLM_GPU_LOCK_FD, TMLM_CPU_SLOT_FDS; kept only when /proc/self/fd names gpu.lock or cpu_slot_*.lock)
are passed to the command, so a job can yield the GPU to an exclusive request.

Exit code: the command's own exit code, or 128 + N when it died from signal N. When the
guard fired: always 137, whatever the command returned, so callers can tell.
SIGINT, SIGTERM and SIGHUP sent to the guard are forwarded to the command's group.
Usage errors and unreadable cgroup files: exit 2.

Standard library only. No thread: one poll loop.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
GUARD_EXIT_CODE = 137


def inherited_lock_fds() -> tuple[int, ...]:
    """Inherited GPU lock and CPU slot descriptors that are really those lock files (standard library only)."""
    candidates = []
    for value in [os.environ.get("TMLM_GPU_LOCK_FD", "")] + os.environ.get("TMLM_CPU_SLOT_FDS", "").split():
        value = value.strip()
        if value.isdigit():
            candidates.append(int(value))
    keep = set()
    for fd in candidates:
        try:
            name = os.path.basename(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            continue
        if name == "gpu.lock" or (name.startswith("cpu_slot_") and name.endswith(".lock")):
            keep.add(fd)
    return tuple(sorted(keep))


def read_usage(cgroup_dir: Path, metric: str) -> int:
    """Bytes in use according to `metric`. Raises OSError or ValueError when unreadable."""
    current = int((cgroup_dir / "memory.current").read_text().strip())
    if metric == "current":
        return current
    if metric == "nonreclaimable":
        stat = {}
        for line in (cgroup_dir / "memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            stat[key] = value
        if "anon" not in stat:
            raise ValueError("memory.stat has no anon line")
        return sum(int(stat.get(k, 0)) for k in ("anon", "shmem", "kernel", "file_mapped"))
    inactive_file = 0
    for line in (cgroup_dir / "memory.stat").read_text().splitlines():
        key, _, value = line.partition(" ")
        if key == "inactive_file":
            inactive_file = int(value)
            break
    return max(0, current - inactive_file)


def group_rss(pgid: int, proc_dir: Path = Path("/proc")) -> int:
    """Bytes of VmRSS summed over the live processes whose process group is `pgid`."""
    total = 0
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # the comm field may contain spaces and parentheses: split after the last ')'
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[2]) != pgid:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1 :]


def signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def main(argv: list[str]) -> int:
    own, command = split_argv(argv)
    parser = argparse.ArgumentParser(
        prog="mem_guard.py",
        usage="mem_guard.py [options] -- <command...>",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit-gib", type=float, default=85.0, help="kill above this usage")
    parser.add_argument("--interval-s", type=float, default=1.0, help="polling period")
    parser.add_argument("--grace-s", type=float, default=10.0, help="SIGTERM to SIGKILL delay")
    parser.add_argument("--metric", choices=("working_set", "current", "nonreclaimable"), default="working_set",
                        help="what is compared with the limit")
    parser.add_argument("--cgroup-dir", type=Path, default=Path("/sys/fs/cgroup"),
                        help="directory with memory.current and memory.stat")
    parser.add_argument("--scope", choices=("cgroup", "group"), default="cgroup",
                        help="what the limit applies to: the whole container, or the command's process group")
    parser.add_argument("--cgroup-cap-gib", type=float, default=90.0,
                        help="with --scope group: container-wide last-resort cap")
    parser.add_argument("--cap-metric", choices=("working_set", "current", "nonreclaimable"), default="nonreclaimable",
                        help="with --scope group: what the container-wide cap is compared with")
    args = parser.parse_args(own)
    if not command:
        parser.error("no command given after --")
    if args.limit_gib <= 0 or args.interval_s <= 0 or args.grace_s < 0 or args.cgroup_cap_gib <= 0:
        parser.error("--limit-gib, --cgroup-cap-gib and --interval-s must be > 0, --grace-s >= 0")
    limit = int(args.limit_gib * GIB)
    cgroup_cap = int(args.cgroup_cap_gib * GIB)

    try:
        read_usage(args.cgroup_dir, args.metric)
        if args.scope == "group":
            read_usage(args.cgroup_dir, args.cap_metric)
    except (OSError, ValueError) as exc:
        print(f"mem_guard: cannot read cgroup memory counters: {exc}", file=sys.stderr)
        return 2

    # Handlers first, so no signal can kill the guard while the command lives on.
    group: list[int] = []
    pending: list[int] = []

    def forward(received: int, _frame: object) -> None:
        if group:
            signal_group(group[0], received)
        else:
            pending.append(received)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, forward)

    try:
        proc = subprocess.Popen(command, process_group=0, pass_fds=inherited_lock_fds())
    except OSError as exc:
        print(f"mem_guard: cannot start {command[0]!r}: {exc}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 126
    pgid = proc.pid
    group.append(pgid)
    for received in pending:
        signal_group(pgid, received)

    fired = False
    while True:
        try:
            proc.wait(timeout=args.interval_s)
            break
        except subprocess.TimeoutExpired:
            pass
        try:
            usage = read_usage(args.cgroup_dir, args.metric if args.scope == "cgroup" else args.cap_metric)
        except (OSError, ValueError) as exc:
            print(f"mem_guard: read failed, still guarding: {exc}", file=sys.stderr)
            continue
        if args.scope == "cgroup":
            if usage <= limit:
                continue
            reason = (f"{args.metric} {usage / GIB:.2f} GiB exceeds the limit "
                      f"{args.limit_gib:.2f} GiB")
        else:
            own_rss = group_rss(pgid)
            if own_rss <= limit and usage <= cgroup_cap:
                continue
            if own_rss > limit:
                reason = (f"process-group RSS {own_rss / GIB:.2f} GiB exceeds the limit "
                          f"{args.limit_gib:.2f} GiB")
            else:
                reason = (f"container {args.cap_metric} {usage / GIB:.2f} GiB exceeds the cap "
                          f"{args.cgroup_cap_gib:.2f} GiB")
        fired = True
        print(
            f"mem_guard: {reason}: SIGTERM to process group {pgid}",
            file=sys.stderr,
            flush=True,
        )
        signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + args.grace_s
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            print(
                f"mem_guard: grace period over: SIGKILL to process group {pgid}",
                file=sys.stderr,
                flush=True,
            )
        # Also reaps group members that outlived the command itself.
        signal_group(pgid, signal.SIGKILL)
        proc.wait()
        break

    if fired:
        return GUARD_EXIT_CODE
    code = proc.returncode
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
