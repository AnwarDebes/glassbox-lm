"""Put new job lines next in run_queue.sh's job file, without moving any line the runner has already read.

usage: python3.12 -I queue_reg_jobs.py --jobs jobs_ft.txt --log queue.log --new jobs_reg.txt [--dry]
run_queue.sh reads line i of the job file before each job (1-based, by position). The last job named in a "start:"
line of the log is the one running (or the last one run); its line and every line before it stay where they are. The
new lines go right after it; the lines that were pending after it follow the new lines. The file is replaced
atomically. Refuses when a new name is already in the file or the last started name is not found exactly once.
"""
import argparse
import os
import re
import sys


def plan(jobs_lines, log_text, new_lines):
    started = re.findall(r"^\S+ (\S+) start:", log_text, flags=re.M)
    names = [l.split()[0] if l.strip() else "" for l in jobs_lines]
    new_names = [l.split()[0] for l in new_lines if l.strip()]
    if any(n in names for n in new_names):
        raise SystemExit("refused: a new job name is already in the job file")
    if not started:
        return new_lines + jobs_lines, 0
    last = started[-1]
    if names.count(last) != 1:
        raise SystemExit(f"refused: the last started job {last} is not in the job file exactly once")
    k = names.index(last) + 1
    return jobs_lines[:k] + [l for l in new_lines if l.strip()] + jobs_lines[k:], k


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    jobs = open(a.jobs).read().splitlines()
    new = open(a.new).read().splitlines()
    out, k = plan(jobs, open(a.log).read(), new)
    if a.dry:
        print("\n".join(f"{i + 1:3d} {l[:110]}" for i, l in enumerate(out)))
        return 0
    tmp = a.jobs + ".tmp"
    with open(tmp, "w") as h:
        h.write("\n".join(out) + "\n")
    os.replace(tmp, a.jobs)
    print(f"inserted {len([l for l in new if l.strip()])} lines after line {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
