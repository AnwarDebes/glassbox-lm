"""Run records of the C-core arm: runs/<UTC>_<name>/ with config.json, log.txt (every printed line,
UTC-stamped), source_hashes.json (code and inputs at start), run.json (start, end, wall time, state),
metrics.json and outputs.json (sha256 of every output file). Nothing is written outside C/."""
import json
import os
import platform
import sys
import time
import traceback

from . import env, hashing, paths


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Run:
    def __init__(self, name, config, dry=False):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.name = name + ("_dry" if dry else "")
        self.id = f"{stamp}_{self.name}"
        self.dir = paths.RUNS / self.id
        os.makedirs(self.dir, exist_ok=False)
        self.t0 = time.time()
        self.outputs = []
        self.meta = {"run_id": self.id, "start_utc": utc(), "argv": sys.argv, "python": sys.version.split()[0],
                     "executable": sys.executable, "pid": os.getpid(), "nice": os.nice(0),
                     "machine": platform.machine(), "dry": dry, "state": "running",
                     "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
                                                                "RAYON_NUM_THREADS")}}
        self._log = open(self.dir / "log.txt", "a", buffering=1)
        self.dump("config.json", config)
        self.dump("source_hashes.json", hashing.snapshot())
        self.dump("run.json", self.meta)
        self.print("run", self.id)

    def print(self, *a):
        line = " ".join(str(x) for x in a)
        for part in line.split("\n"):
            s = f"{utc()} {part}"
            print(s, flush=True)
            self._log.write(s + "\n")

    def dump(self, name, obj):
        hashing.write_json(self.dir / name, obj)

    def add_output(self, path):
        self.outputs.append(str(path))

    def finish(self, metrics):
        loaded = env.neural_modules_loaded()
        if loaded:                       # every C stage of R1 runs with no neural network at any stage
            raise SystemExit(f"neural-network libraries were imported: {loaded}")
        self.meta["neural_modules_loaded"] = loaded
        self.dump("metrics.json", metrics)
        self.dump("outputs.json", hashing.manifest(self.outputs))
        self.meta.update(state="finished", end_utc=utc(), wall_s=round(time.time() - self.t0, 1))
        self.dump("run.json", self.meta)
        self.print("finished in", self.meta["wall_s"], "s")
        self._log.close()

    def crash(self):
        self.print("CRASHED\n" + traceback.format_exc())
        self.meta.update(state="crashed", end_utc=utc(), wall_s=round(time.time() - self.t0, 1))
        self.dump("run.json", self.meta)
        self._log.close()


def main_wrapper(run, fn):
    try:
        fn()
    except BaseException:
        run.crash()
        raise


def finished_runs(name, dry=False):
    """Finished run folders of a stage (real or dry)."""
    import glob
    suffix = name + ("_dry" if dry else "")
    out = []
    for d in sorted(glob.glob(str(paths.RUNS / f"*_{suffix}"))):
        try:
            if read_json(os.path.join(d, "run.json"))["state"] == "finished":
                out.append(d)
        except (OSError, ValueError, KeyError):
            pass
    return out


def refuse_second_real_run(name, dry):
    """A real stage runs once: a second look at valid needs a dated amendment first."""
    if not dry:
        done = finished_runs(name)
        if done:
            raise SystemExit(f"a finished real {name} run exists ({done}); a second run needs a dated amendment")


def check_inputs(stages, dry):
    """The outputs of earlier stages are read only if they equal what those runs wrote (outputs.json),
    and, for real runs, only if each stage ran exactly once. Returns {stage: run folder}."""
    out = {}
    for st in stages:
        runs = finished_runs(st, dry)
        if not runs or (not dry and len(runs) != 1):
            raise SystemExit(f"expected {'exactly one' if not dry else 'a'} finished {st} run, found {runs}")
        d = runs[-1]
        for p, h in read_json(os.path.join(d, "outputs.json")).items():
            if hashing.sha256(paths.ROOT / p) != h:
                raise SystemExit(f"{p} differs from what run {d} wrote")
        out[st] = d
    return out


def out_dir(dry):
    d = paths.DATA_DRY if dry else paths.DATA
    os.makedirs(d, exist_ok=True)
    return d


def read_json(p):
    with open(p) as f:
        return json.load(f)
