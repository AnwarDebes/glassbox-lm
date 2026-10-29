"""Run records for the tally study.

Every run gets results/<UTC>_<name>/ with config.json, metrics.json, log.txt (a copy of every
printed line), source_hashes.json (sha256 of every file under src/ and of the registration the
run belongs to), data_hashes.json (sha256 of every data file the run names, keyed by its path
relative to the study folder), and run.json (start, end, wall time, exit state).
A run that ends with an exception is marked "crashed" and keeps its partial log. A run that gets
SIGTERM or SIGHUP is marked "killed" (key "signal" names the signal) and then ends the way the
signal's default action ends it. A signal that was ignored when the run started (SIGHUP under
nohup) stays ignored. SIGKILL cannot be caught, so such a run stays "running". Python runs a
signal handler between bytecodes: a signal that arrives during a long compiled (numba) call is
handled when that call returns.

Changed 2026-10-25: the signal handler, and the keys of data_hashes.json. Before, the keys were
file names, so the PTB train and validation texts (both named text.utf8) kept one entry, the hash
of the validation text.
"""
import hashlib
import json
import os
import platform
import signal
import sys
import time
import traceback

STUDY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(STUDY, "results")
SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(root):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith((".py", ".sh", ".md", ".json", ".toml")):
                p = os.path.join(d, f)
                out[os.path.relpath(p, STUDY)] = sha256(p)
    return out


def data_hashes(paths):
    """sha256 of each data file, keyed by its path relative to the study folder."""
    return {os.path.relpath(os.path.abspath(p), STUDY): sha256(p) for p in paths}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Run:
    def __init__(self, name, config, data_files=(), registration=None):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.id = f"{stamp}_{name}"
        self.dir = os.path.join(RESULTS, self.id)
        os.makedirs(self.dir, exist_ok=False)
        self.t0 = time.time()
        self.meta = {"run_id": self.id, "start_utc": utc(), "argv": sys.argv,
                     "python": sys.version.split()[0], "host_cpus_quota": "6 (cgroup)",
                     "pid": os.getpid(), "nice": os.nice(0), "machine": platform.machine(),
                     "slots": os.environ.get("TMLM_CPU_SLOTS"), "state": "running"}
        self._log = open(os.path.join(self.dir, "log.txt"), "a", buffering=1)
        self._dump("config.json", config)
        src = tree_hashes(os.path.join(STUDY, "src"))
        if registration:
            src[os.path.relpath(registration, STUDY)] = sha256(registration)
        self._dump("source_hashes.json", src)
        self._dump("data_hashes.json", data_hashes(data_files))
        self._dump("run.json", self.meta)
        self._prev_hook = sys.excepthook
        sys.excepthook = self._crash
        self._prev = {}
        for sig in SIGNALS:
            try:
                prev = signal.getsignal(sig)
                if prev is signal.SIG_IGN:      # e.g. SIGHUP under nohup: keep ignoring it
                    continue
                signal.signal(sig, self._on_signal)
                self._prev[sig] = prev
            except ValueError:                  # not the main thread: no handler there
                pass
        self.print(f"run {self.id}")

    def _dump(self, name, obj):
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)

    def _crash(self, et, ev, tb):
        self.print("".join(traceback.format_exception(et, ev, tb)))
        self.meta.update(state="crashed", end_utc=utc(), wall_s=round(time.time() - self.t0, 1))
        self._dump("run.json", self.meta)
        sys.__excepthook__(et, ev, tb)

    def _on_signal(self, signum, frame):
        if self.meta.get("state") == "running":
            name = signal.Signals(signum).name
            try:
                self._log.write(f"{utc()} killed by {name}\n")
            except (OSError, ValueError):
                pass
            self.meta.update(state="killed", signal=name, end_utc=utc(),
                             wall_s=round(time.time() - self.t0, 1))
            self._dump("run.json", self.meta)
        prev = self._prev.get(signum)
        if callable(prev):
            prev(signum, frame)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def path(self, name):
        return os.path.join(self.dir, name)

    def print(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self._log.write(f"{utc()} {s}\n")

    def finish(self, metrics):
        self._dump("metrics.json", metrics)
        self.meta.update(state="finished", end_utc=utc(), wall_s=round(time.time() - self.t0, 1))
        self._dump("run.json", self.meta)
        self.print("METRICS " + json.dumps(metrics, sort_keys=True))
        self._log.close()
        if sys.excepthook == self._crash:        # a later exception is not this run's crash
            sys.excepthook = self._prev_hook
        for sig, prev in self._prev.items():     # hand the signals back
            try:
                if signal.getsignal(sig) == self._on_signal:
                    signal.signal(sig, prev if prev is not None else signal.SIG_DFL)
            except ValueError:
                pass
        return self.dir
