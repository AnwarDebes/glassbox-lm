#!/usr/bin/env python
"""spin_check.py: CPU use of one process's threads over a window, read-only from /proc, printed as JSON.

    spin_check.py PID [--seconds 10] [--out F.json]
    spin_check.py --descendant-of RUNNER_PID [--comm-prefix python] [--seconds 10] [--out F.json]

Per thread (/proc/PID/task/TID/stat fields 14 + 15, utime + stime, at the start and at the end of the window):
cores = delta ticks / CLK_TCK / wall seconds; voluntary and non-voluntary context switches (task/TID/status); state;
thread name (task/TID/comm). Reported: the main thread (TID = PID), the sum over the threads alive at both reads, the
process total from /proc/PID/stat (the whole thread group, including threads that ended inside the window), and the
threads that started or ended inside the window. Resolution: 1 tick (1 / CLK_TCK s) per read.

--descendant-of: the target is the ONLY process in RUNNER_PID's subtree whose comm starts with --comm-prefix; the
subtree is walked through /proc/X/task/*/children (no process table is listed). None or several: exit 2, nothing
measured. For PROTOCOL BG2 the runner is review/bench/run_gpu_job.sh's PID: its python is a child or a grandchild
(run_gpu_job.sh runs it inside a subshell; gpu_run.sh then execs env, which execs python), so one `pgrep -P RUNNER`
can name the subshell instead.

Exit 0 measured; 2 target not found / ambiguous / ended during the window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

TCK = os.sysconf("SC_CLK_TCK")


def _stat_fields(path: str) -> list[str]:
    return open(path).read().rsplit(")", 1)[1].split()  # fields 3.. (state is field 3 = index 0)


def comm(pid: int) -> str:
    return open(f"/proc/{pid}/comm").read().strip()


def read_task(pid: int, tid: int) -> dict:
    f = _stat_fields(f"/proc/{pid}/task/{tid}/stat")
    st = open(f"/proc/{pid}/task/{tid}/status").read()
    return {"state": f[0], "ticks": int(f[11]) + int(f[12]),
            "vol": int(re.search(r"^voluntary_ctxt_switches:\s+(\d+)", st, re.M).group(1)),
            "nonvol": int(re.search(r"^nonvoluntary_ctxt_switches:\s+(\d+)", st, re.M).group(1)),
            "comm": open(f"/proc/{pid}/task/{tid}/comm").read().strip()}


def snapshot(pid: int) -> tuple[dict, int, float]:
    tasks = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            tasks[int(t)] = read_task(pid, int(t))
        except (FileNotFoundError, ProcessLookupError):
            pass  # the thread ended between the listing and the read
    f = _stat_fields(f"/proc/{pid}/stat")
    return tasks, int(f[11]) + int(f[12]), time.monotonic()


def children(pid: int) -> list[int]:
    kids: set[int] = set()
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            try:
                kids.update(int(x) for x in open(f"/proc/{pid}/task/{t}/children").read().split())
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    return sorted(kids)


def subtree(root: int) -> list[list]:
    """[[pid, parent, comm], ...] of every descendant of root, breadth first."""
    out, todo = [], [root]
    while todo:
        p = todo.pop(0)
        for c in children(p):
            try:
                out.append([c, p, comm(c)])
            except FileNotFoundError:
                continue
            todo.append(c)
    return out


def measure(pid: int, seconds: float) -> dict:
    utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    a, pa, t0 = snapshot(pid)
    time.sleep(seconds)
    b, pb, t1 = snapshot(pid)
    wall = t1 - t0
    threads = []
    for tid in sorted(set(a) & set(b)):
        threads.append({"tid": tid, "comm": b[tid]["comm"], "cores": round((b[tid]["ticks"] - a[tid]["ticks"]) / TCK / wall, 4),
                        "vol_switches": b[tid]["vol"] - a[tid]["vol"], "nonvol_switches": b[tid]["nonvol"] - a[tid]["nonvol"],
                        "state_end": b[tid]["state"]})
    threads.sort(key=lambda x: (-x["cores"], x["tid"]))
    main = next((x for x in threads if x["tid"] == pid), None)
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()[:200]
    except FileNotFoundError:
        cmd = ""
    return {"utc_start": utc0, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": pid,
            "comm": comm(pid), "seconds": round(wall, 3), "clk_tck": TCK, "main_thread": main,
            "main_thread_cores": None if main is None else main["cores"],
            "all_threads_cores": round(sum(x["cores"] for x in threads), 4),
            "process_cores": round((pb - pa) / TCK / wall, 4), "n_threads_start": len(a), "n_threads_end": len(b),
            "threads_started": sorted(set(b) - set(a)), "threads_ended": sorted(set(a) - set(b)),
            "threads": threads, "cmdline_head": cmd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", nargs="?", type=int)
    ap.add_argument("--descendant-of", type=int)
    ap.add_argument("--comm-prefix", default="python")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if (a.pid is None) == (a.descendant_of is None):
        raise SystemExit("give either PID or --descendant-of RUNNER_PID")
    found = None
    if a.descendant_of is not None:
        tree = subtree(a.descendant_of)
        hits = [x for x in tree if x[2].startswith(a.comm_prefix)]
        found = {"descendant_of": a.descendant_of, "comm_prefix": a.comm_prefix, "subtree": tree}
        if len(hits) != 1:
            print(json.dumps({"error": f"{len(hits)} descendants with comm prefix {a.comm_prefix!r}", **found}), flush=True)
            raise SystemExit(2)
        pid = hits[0][0]
    else:
        pid = a.pid
    try:
        res = measure(pid, a.seconds)
    except (FileNotFoundError, ProcessLookupError) as e:
        print(json.dumps({"error": f"process {pid} ended during the window: {e}", "found_via": found}), flush=True)
        raise SystemExit(2) from None
    res["found_via"] = found
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
