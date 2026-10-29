"""Is Chromium's own sandbox really on? Start the headless shell (wrapper) through Playwright with
chromium_sandbox on or off, open a MiniWoB page, then read /proc for the browser's own child processes
(only processes this script started): seccomp mode, user/pid namespace vs this process, --no-sandbox flag.
Run with python -I."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import json
import os
import sys

from playwright.sync_api import sync_playwright

F = f"{_REPO_ROOT}/studies/agent_guards/miniwob"
WRAP = f"{F}/browser/chrome-headless-shell"
REAL = "chrome-headless-shell-linux64/chrome-headless-shell"
TASK_HTML = f"{F}/.venv/lib/python3.12/site-packages/miniwob/html/miniwob/click-test-2.html"


def proc_info(pid):
    d = {"pid": pid}
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        # Chrome rewrites its process title, so the arguments may be one space-separated string
        args = [a for c in cmd if c for a in c.decode(errors="replace").split()]
        d["type"] = next((a.split("=", 1)[1] for a in args if a.startswith("--type=")), "browser")
        d["no_sandbox_flag"] = "--no-sandbox" in args
        st = open(f"/proc/{pid}/status").read().splitlines()
        for line in st:
            if line.startswith(("Seccomp:", "NSpid:", "Uid:", "CapEff:")):
                k, v = line.split(":", 1)
                d[k] = v.strip()
        d["user_ns_same_as_me"] = os.readlink(f"/proc/{pid}/ns/user") == os.readlink("/proc/self/ns/user")
        d["pid_ns_same_as_me"] = os.readlink(f"/proc/{pid}/ns/pid") == os.readlink("/proc/self/ns/pid")
        d["net_ns_same_as_me"] = os.readlink(f"/proc/{pid}/ns/net") == os.readlink("/proc/self/ns/net")
    except Exception as e:
        d["error"] = repr(e)[:200]
    return d


def my_chrome_pids():
    """Descendants of this Python process whose command line is the headless shell."""
    me = os.getpid()
    parent = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                fields = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
                parent[int(p)] = int(fields[1])
            except Exception:
                pass
    out = []
    for pid in parent:
        q, depth = pid, 0
        while q in parent and q != me and depth < 50:
            q, depth = parent[q], depth + 1
        if q == me:
            try:
                if REAL in open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace"):
                    out.append(pid)
            except Exception:
                pass
    return sorted(out)


def main():
    sandbox = sys.argv[1] == "1"
    rep = {"chromium_sandbox": sandbox}
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=WRAP, headless=True, chromium_sandbox=sandbox, timeout=60000)
        except Exception as e:
            rep["launch_error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            print(json.dumps(rep, indent=1))
            return
        page = b.new_page()
        page.goto("file://" + TASK_HTML)
        page.wait_for_selector("#sync-task-cover", timeout=10000)
        rep["procs"] = [proc_info(pid) for pid in my_chrome_pids()]
        b.close()
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
