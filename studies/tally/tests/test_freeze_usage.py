"""scripts/freeze.py gives a usage message instead of a traceback, and freezes nothing then.

Run: python tests/test_freeze_usage.py

The test runs freeze.py only in ways that stop before any record is written: no argument,
--help, and a registration path that does not exist (in a temporary folder).
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FREEZE = os.path.join(os.path.dirname(HERE), "scripts", "freeze.py")
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def run(*args):
    return subprocess.run([sys.executable, FREEZE, *args], capture_output=True, text=True)


def main():
    r = run()
    check("no argument: exit 2 with the usage line on stderr, no traceback",
          r.returncode == 2 and r.stderr.startswith("usage:") and "Traceback" not in r.stderr,
          f"exit {r.returncode}, stderr {r.stderr[-200:]!r}")
    r = run("--help")
    check("--help: exit 0 with the usage line on stdout", r.returncode == 0 and r.stdout.startswith("usage:"),
          f"exit {r.returncode}, stdout {r.stdout!r}, stderr {r.stderr[-200:]!r}")
    with tempfile.TemporaryDirectory() as tmp:
        reg = os.path.join(tmp, "missing.md")
        r = run(reg)
        check("missing registration: exit 2, message names it, no record written",
              r.returncode == 2 and "not a file" in r.stderr and not os.listdir(tmp),
              f"exit {r.returncode}, stderr {r.stderr[-200:]!r}, files {os.listdir(tmp)}")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
