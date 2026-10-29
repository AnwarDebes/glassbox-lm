"""Fixed paths, the import set-up for the read-only study code, and small run helpers.

Every job runs as ``python -I -B`` (no PYTHONPATH, no .pyc next to imported code), so the study ``src``
folders are put on sys.path here. ``check_env`` refuses to run when a cache or temporary directory points
outside A/tmp/ or when the thread variables are not 2 (PLAN_part3.md section 6, "Files and run rules").
``check_runner`` (amendment 1): a real run (outputs under A/out/) must also run at nice 19, be started by run_job.sh
(A_RUNNER=1) and hold the Part 3 chain lock through chain/part3_cpu_job.sh (one Part 3 CPU job at a time, A and C).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(f"{_WS_ROOT}")
A_DIR = ROOT / "glassbox-lm" / "studies" / "language_part3" / "A"
TMP = A_DIR / "tmp"
OUT = A_DIR / "out"
MWOZ = ROOT / "glassbox-lm" / "studies" / "decision_layer" / "data" / "untrusted" / "multiwoz22"
C4 = ROOT / "glassbox-lm" / "studies" / "decision_layer" / "data" / "tables" / "c4words_S1000_v1"
TALLY_SRC = ROOT / "glassbox-lm" / "studies" / "tally" / "src"
DL_SRC = ROOT / "glassbox-lm" / "studies" / "decision_layer" / "src"
TMLM_BASELINES = ROOT / "glassbox-lm" / "studies" / "tm_language_model" / "src" / "tmlm" / "eval" / "baselines"
KENLM_BIN = ROOT / "tools" / "kenlm" / "bin"
CPU_LOAD = ROOT / "glassbox-lm" / "studies" / "decision_layer" / "scripts" / "cpu_load.py"
# the Part 3 chain lock: held by chain/part3_cpu_job.sh for every Part 3 CPU job of A and C (amendment 1)
CHAIN_LOCK = ROOT / "glassbox-lm" / "studies" / "language_part3" / "chain" / "part3_cpu.lock"

# folders listed before and after every job (section 6); never searched recursively
LISTED = [
    TALLY_SRC / "tally" / "floor", TALLY_SRC / "tally" / "floor" / "__pycache__",
    TALLY_SRC / "tally", TALLY_SRC / "tally" / "__pycache__",
    DL_SRC / "dl", DL_SRC / "dl" / "__pycache__",
    TMLM_BASELINES,
    C4, C4 / "__pycache__",
    KENLM_BIN,
    MWOZ, MWOZ / "train", MWOZ / "dev", MWOZ / "test",
]

CACHE_VARS = ("NUMBA_CACHE_DIR", "TMPDIR", "JOBLIB_TEMP_FOLDER", "MPLCONFIGDIR", "XDG_CACHE_HOME")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")


def check_env(threads: int = 2) -> dict:
    """The run rules as assertions: caches inside A/tmp/, thread variables pinned, no GPU, no bytecode."""
    bad = []
    for v in CACHE_VARS:
        p = os.environ.get(v)
        if not p or not Path(p).resolve().is_relative_to(TMP.resolve()):
            bad.append(f"{v}={p!r} (must be inside {TMP})")
    for v in THREAD_VARS:
        if os.environ.get(v) != str(threads):
            bad.append(f"{v}={os.environ.get(v)!r} (must be {threads})")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        bad.append("CUDA_VISIBLE_DEVICES must be set and empty (no GPU in A's CPU steps)")
    if not sys.dont_write_bytecode:
        bad.append("run with python -B")
    if not sys.flags.isolated:
        bad.append("run with python -I")
    if bad:
        raise SystemExit("run rules not met:\n  " + "\n  ".join(bad))
    return {v: os.environ.get(v) for v in CACHE_VARS + THREAD_VARS + ("CUDA_VISIBLE_DEVICES",)}


def lock_held(lock: Path) -> bool:
    """True when another open file description holds an exclusive flock on the file (the chain wrapper's lock).
    The probe opens the file anew and asks for a shared lock without waiting; it never creates the file."""
    try:
        fd = os.open(str(lock), os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def runner_problems(nice: int | None = None, environ: dict | None = None, chain_lock: Path | None = None) -> list:
    """What a real run lacks (amendment 1): nice 19, the runner's mark A_RUNNER=1 (run_job.sh sets it, its test mode
    sets "test"), and the Part 3 chain lock named in PART3_CHAIN_LOCK and held (chain/part3_cpu_job.sh)."""
    nice = os.nice(0) if nice is None else nice
    environ = os.environ if environ is None else environ
    chain_lock = Path(CHAIN_LOCK if chain_lock is None else chain_lock)
    bad = []
    if nice != 19:
        bad.append(f"nice level is {nice}, not 19")
    if environ.get("A_RUNNER") != "1":
        bad.append(f"A_RUNNER is {environ.get('A_RUNNER')!r}: real runs start only through run_job.sh (real mode)")
    if environ.get("PART3_CHAIN_LOCK") != str(chain_lock):
        bad.append(f"PART3_CHAIN_LOCK is {environ.get('PART3_CHAIN_LOCK')!r}, not {chain_lock}: start real jobs through "
                   "chain/part3_cpu_job.sh")
    elif not lock_held(chain_lock):
        bad.append(f"the Part 3 chain lock {chain_lock} is not held")
    return bad


def check_runner() -> None:
    bad = runner_problems()
    if bad:
        raise SystemExit("real run refused (run rules, amendment 1):\n  " + "\n  ".join(bad))


def add_study_paths(tally: bool = False, dl: bool = False) -> None:
    for flag, p in ((tally, TALLY_SRC), (dl, DL_SRC)):
        if flag and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_tokenize_ref():
    """The C4 table's reference tokenizer, loaded from its file (read-only; -B keeps its folder unchanged)."""
    name = "astruct_tokenize_ref"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, C4 / "tokenize_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes() -> dict:
    """Full SHA-256 of A's code and of every imported study file this code can load."""
    files = sorted(list((A_DIR / "astruct").glob("*.py")) + list(A_DIR.glob("*.py")) + list(A_DIR.glob("*.sh")))
    files += [TALLY_SRC / "tally" / "floor" / f for f in ("kn.py", "kenlm.py", "mix.py", "stats.py", "aa.py", "__init__.py")]
    files += [TALLY_SRC / "tally" / "__init__.py", DL_SRC / "dl" / "tm.py", DL_SRC / "dl" / "tm_init.py",
              DL_SRC / "dl" / "cpu_tm_fast.py", C4 / "tokenize_ref.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files if p.exists()}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    """Timestamped lines to stdout and to <out>/log.txt."""

    def __init__(self, path: Path | None):
        self.path = path
        self.t0 = time.time()

    def __call__(self, *a) -> None:
        msg = f"[{utc()} +{time.time() - self.t0:8.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(msg + "\n")


FROZEN = A_DIR / "R1_A-Struct.frozen.json"
LOGMD = A_DIR.parent / "LOG.md"


def verify_frozen(record: Path | None = None, root: Path | None = None, logmd: Path | None = None) -> str:
    """The frozen record of R1 (freeze.py): every enforced file must still have its frozen SHA-256, and LOG.md must
    hold the record's own SHA-256 (the R1 line, written before any real run). Returns that SHA-256. A real run
    (outputs under A/out/) calls this before it writes anything."""
    record = Path(record if record is not None else FROZEN)
    root = Path(root if root is not None else ROOT)
    logmd = Path(logmd if logmd is not None else LOGMD)
    if not record.exists():
        raise SystemExit(f"no frozen record {record}: real runs need R1 frozen first (freeze.py)")
    rec = json.loads(record.read_text())
    bad = []
    for rel, h in sorted(rec["enforced"].items()):
        p = root / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
        elif sha256_file(p) != h:
            bad.append(f"{rel}: changed since the freeze")
    if bad:
        raise SystemExit("frozen-record check failed (a dated amendment line in LOG.md and a new freeze are needed):\n  "
                         + "\n  ".join(bad))
    h = sha256_file(record)
    if not logmd.exists() or h not in logmd.read_text():
        raise SystemExit(f"{logmd} has no line with the frozen record's SHA-256 {h}: write the R1 line first")
    return h


def is_real_output(p: Path) -> bool:
    return Path(p).resolve().is_relative_to(OUT.resolve())


class OutDir:
    """Outputs go to <name>.partial and are renamed to <name> at the end; an existing <name> is never overwritten.
    Outputs under A/out/ are real runs: the run rules of a real run (check_runner) and the frozen record
    (verify_frozen) are checked first."""

    def __init__(self, final: Path):
        self.final = Path(final)
        self.part = self.final.with_name(self.final.name + ".partial")
        real = is_real_output(self.final)
        if real:
            check_runner()
        self.frozen_sha256 = verify_frozen() if real else None
        if self.final.exists():
            raise SystemExit(f"{self.final} exists; outputs are never overwritten")
        if self.part.exists():
            for p in sorted(self.part.rglob("*"), reverse=True):
                p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
            self.part.rmdir()
        self.part.mkdir(parents=True)

    def __truediv__(self, name: str) -> Path:
        return self.part / name

    def done(self, summary: dict) -> None:
        summary = {**summary, "frozen_record_sha256": self.frozen_sha256}
        (self.part / "done.json").write_text(json.dumps(summary, indent=1, default=str))
        self.part.rename(self.final)
