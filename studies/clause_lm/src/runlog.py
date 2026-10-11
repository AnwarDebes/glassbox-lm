"""Minimal run logging for clause_lm (study rule 4).

Every run gets results/<UTC>_<name>/ holding config.json, metrics.json, log.txt (tee of
prints), a copy of the calling script and runlog.py, and sha256 hashes of the data files used.
"""
import hashlib, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Run:
    def __init__(self, name, config, data_files=(), script=None):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.dir = os.path.join(ROOT, "results", f"{stamp}_{name}")
        os.makedirs(self.dir, exist_ok=False)
        self.t0 = time.time()
        self._log = open(os.path.join(self.dir, "log.txt"), "a", buffering=1)
        json.dump(config, open(os.path.join(self.dir, "config.json"), "w"), indent=1)
        json.dump({os.path.relpath(p, os.path.dirname(ROOT)): sha256(p) for p in data_files},
                  open(os.path.join(self.dir, "data_hashes.json"), "w"), indent=1)
        for f in filter(None, [script or sys.argv[0], __file__]):
            if os.path.exists(f):
                shutil.copy(f, self.dir)
        self.print(f"run dir: {self.dir}")

    def print(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self._log.write(s + "\n")

    def finish(self, metrics):
        metrics = {**metrics, "wall_s": round(time.time() - self.t0, 1)}
        json.dump(metrics, open(os.path.join(self.dir, "metrics.json"), "w"), indent=1)
        self.print("METRICS " + json.dumps(metrics))
        self._log.close()
        return self.dir
