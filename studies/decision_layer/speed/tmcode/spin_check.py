"""Read-only: CPU share of one process's main thread over a window (from /proc/<pid>/task/<pid>/stat), and its
voluntary context switches. Usage: spin_check.py PID [SECONDS]."""
import os
import sys
import time

pid, secs = int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
tck = os.sysconf("SC_CLK_TCK")


def read():
    f = open(f"/proc/{pid}/task/{pid}/stat").read().rsplit(")", 1)[1].split()
    vol = [l for l in open(f"/proc/{pid}/status") if l.startswith("voluntary_ctxt_switches")][0].split()[1]
    return int(f[11]) + int(f[12]), int(vol), f[0]  # utime + stime (fields 14, 15), vol switches, state


a, t0 = read(), time.time()
time.sleep(secs)
b, t1 = read(), time.time()
cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")[:160]
print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid {pid} main-thread CPU {(b[0] - a[0]) / tck / (t1 - t0):.3f} cores over {t1 - t0:.1f} s; "
      f"voluntary ctx switches +{b[1] - a[1]}; state {b[2]}; cmd: {cmd}")
