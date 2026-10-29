"""Run a command with NO network except a private loopback.

Use:  unshare --user --map-current-user --net --keep-caps  python -I netns_exec.py  <program> [args...]

Inside the fresh network namespace only a downed 'lo' exists. This script brings 'lo' up (needs the
CAP_NET_ADMIN that --keep-caps leaves in the new user namespace), clears the ambient capabilities so the
program runs as the plain user (uid 1000 mapped to itself, no capabilities), then execs the program.
So chromedriver <-> Python <-> Chrome can talk over 127.0.0.1 inside the namespace, but nothing can reach
the host's loopback or any outside network.
"""
import ctypes
import fcntl
import os
import socket
import struct
import sys

SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL = 47, 4

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ifr = struct.pack("16sH22x", b"lo", 0)          # struct ifreq is 40 bytes on x86_64
flags = struct.unpack("16sH22x", fcntl.ioctl(s, SIOCGIFFLAGS, ifr))[1]
fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH22x", b"lo", flags | IFF_UP))
s.close()
# /proc/net/dev is per network namespace (/sys/class/net would still show the host, since /sys is not remounted)
ifaces = sorted(l.split(":")[0].strip() for l in open("/proc/net/dev").read().splitlines()[2:])
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0:
    sys.exit(f"prctl(PR_CAP_AMBIENT_CLEAR_ALL) failed, errno {ctypes.get_errno()}")
print(f"netns_exec: interfaces {ifaces}, lo up, ambient caps cleared; exec {sys.argv[1]}", file=sys.stderr, flush=True)
os.execv(sys.argv[1], sys.argv[1:])
