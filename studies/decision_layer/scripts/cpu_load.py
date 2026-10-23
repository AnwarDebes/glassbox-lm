"""Measure this cgroup's CPU use over a window: cores in use and throttled periods (read-only)."""
import sys
import time


def read():
    d = {}
    for line in open("/sys/fs/cgroup/cpu.stat"):
        k, v = line.split()
        d[k] = int(v)
    return d


secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
a, t0 = read(), time.time()
time.sleep(secs)
b, t1 = read(), time.time()
cores = (b["usage_usec"] - a["usage_usec"]) / 1e6 / (t1 - t0)
print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} cores {cores:.2f} over {t1 - t0:.1f} s; throttled {b['nr_throttled'] - a['nr_throttled']} of {b['nr_periods'] - a['nr_periods']} periods")
