#!/usr/bin/env python
"""K7 time box of A-Struct (PLAN_part3.md section 3): no gate decision within 72 gate-open hours, or within 10 calendar
days of starting S0, whichever comes first -> stop and report. Reads run_job.sh's log (A/out/jobs.log); writes nothing.

    python -I -B k7_clock.py [--jobs A/out/jobs.log] [--now 2026-10-22T00:00:00Z]

Gate-open hours are counted as the wall time of the jobs the runner started (each started only with the CPU gate
open); the hours between jobs are not observed, so the count is a lower bound. Calendar days run from the first start
of S0 (s0_data). Prints one JSON line; exit 3 when a limit is reached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LIMIT_HOURS, LIMIT_DAYS = 72.0, 10.0


def parse_utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def clock(lines: list, now: float) -> dict:
    open_jobs, hours, s0_start, n_jobs, closed = {}, 0.0, None, 0, 0
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        t, kind = parse_utc(parts[0]), parts[1]
        if kind == "start":
            tag = " ".join(parts[2:])
            open_jobs[tag] = t
            if parts[2:3] == ["s0_data"] and s0_start is None:
                s0_start = t
        elif kind == "end":
            tag = " ".join(parts[2:]).split(" exit ")[0]
            if tag in open_jobs:
                hours += (t - open_jobs.pop(tag)) / 3600
                n_jobs += 1
        elif kind == "gate" and parts[2:3] == ["closed"]:
            closed += 1
    running = sum((now - t) / 3600 for t in open_jobs.values())
    days = (now - s0_start) / 86400 if s0_start is not None else 0.0
    gate_hours = hours + running
    return {"gate_open_hours_lower_bound": round(gate_hours, 3), "finished_jobs": n_jobs, "running_jobs": sorted(open_jobs),
            "closed_gate_checks": closed, "days_since_s0_start": round(days, 3),
            "s0_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s0_start)) if s0_start else None,
            "limit_reached": bool(gate_hours >= LIMIT_HOURS or days >= LIMIT_DAYS),
            "limits": {"gate_open_hours": LIMIT_HOURS, "calendar_days": LIMIT_DAYS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(Path(__file__).resolve().parent / "out" / "jobs.log"))
    ap.add_argument("--now", default=None)
    args = ap.parse_args()
    p = Path(args.jobs)
    lines = p.read_text().splitlines() if p.exists() else []
    now = parse_utc(args.now) if args.now else time.time()
    res = clock(lines, now)
    print(json.dumps(res))
    return 3 if res["limit_reached"] else 0


if __name__ == "__main__":
    sys.exit(main())
