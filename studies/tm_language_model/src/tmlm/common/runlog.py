"""Run governance (protocol rules 2, 3, 5): every run gets its own directory under results/.

Usage inside Python:

    from tmlm.common.runlog import RunLog
    with RunLog("e1_sweep", config, dataset_hashes={"train": sha}) as run:
        ...                                   # stdout and stderr are teed to log.txt
        run.add_metric("top1", 0.123, "measured")
        run.write_json("curve.json", curve)   # any extra artifact, covered by MANIFEST

Usage for an arbitrary command: scripts/run_logged.py --name N --config cfg.json -- cmd...

Layout of results/<UTC timestamp>_<name>/ :
    config.json    the full config, normalized to plain JSON
    env.json       python, package versions, hostname, cgroup limits, GPU name and driver
    run.json       name, UTC start and end, wall_time_s, exit_status, git_commit,
                   source_tree_hash, config_sha256, dataset_hashes, gpu_memory summary,
                   peak_gpu_memory_mib, command, thread_caps, stream_errors,
                   finalize_error. Written once at start with status "running" and
                   rewritten at the end with status "finished", so a run that was killed
                   hard is recognizable: status "running", no MANIFEST.
    metrics.json   {metric name: {"value": ..., "label": measured|derived|unverified}}
    log.txt        tee of stdout and stderr
    MANIFEST.json  sha256 and size of every other file in the run dir, written last

Guarantees:
  - A run dir is never overwritten: it is created with an exclusive mkdir, and a name
    collision within the same second gets a numeric suffix.
  - metrics.json never contains a value without a valid label.
  - Metrics reach the disk when they are reported, not at the end: `RunLog.add_metric`
    and `set_metrics` merge into metrics.json at once, `add_dataset_hash` rewrites
    run.json at once. A run that is killed hard keeps what it had reported.
  - `write_metrics` and `write_dataset_hashes` MERGE into the existing file under a
    lock (<run dir>/.runlog.lock, removed at the end), so several calls, also from
    several processes, never drop each other's entries.
  - log.txt is written before the terminal, and a terminal stream that breaks (the
    reader of a pipe went away) is dropped without touching the run: the command keeps
    running, log.txt stays complete, run.json names the broken stream.
  - Finalization always ends with MANIFEST.json. A step that fails on the way is
    recorded as finalize_error in run.json, turns exit status 0 into 70, and raises
    RunLogFinalizeError when the block itself had not failed.
  - Commands started by `run_command` get the thread caps and PYTHONHASHSEED of
    tmlm.common.threadcaps and inherit the GPU and CPU slot lock descriptors.
  - This module imports neither numpy nor any GPU library at import time. Package
    versions come from importlib.metadata, which does not import the packages.

Limits:
  - In-process tee is Python level: text written straight to file descriptors 1 or 2 by
    C extensions or child processes is not captured. `RunLog.run_command` and
    scripts/run_logged.py pipe the child's output and capture all of it.
  - Peak GPU memory is the device-wide `memory.used` reported by nvidia-smi, polled
    every `gpu_poll_s` seconds. It includes other processes on the GPU and can miss
    spikes shorter than the polling period. The first sample is kept as a baseline.
"""

from __future__ import annotations

import codecs
import contextlib
import datetime as _dt
import fcntl
import importlib.metadata
import io
import json
import os
import platform
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from tmlm.common import hashing, locks, threadcaps

METRIC_LABELS = ("measured", "derived", "unverified")

# Exit status of a run whose command succeeded but whose run dir could not be finalized
# (EX_SOFTWARE of sysexits.h). 65 (EX_DATAERR) is used for rejected metrics files.
FINALIZE_ERROR_EXIT = 70
METRICS_ERROR_EXIT = 65

# Lock file for read-merge-write cycles on metrics.json and dataset_hashes.json.
RUN_LOCK_FILE = ".runlog.lock"


class CommandStartError(OSError):
    """`RunLog.run_command` could not start the command (not found, not executable).
    Anything that goes wrong AFTER the start is never reported through this class.
    `exit_status` is what a shell would report: 127 for not found, else 126."""

    def __init__(self, command: Sequence[str], cause: OSError) -> None:
        program = command[0] if command else ""
        super().__init__(cause.errno, f"cannot start {program!r}: {cause.strerror or cause}")
        self.not_found = isinstance(cause, FileNotFoundError)
        self.exit_status = 127 if self.not_found else 126


class RunLogFinalizeError(RuntimeError):
    """The run dir could not be finalized completely. run.json names the failed steps
    in finalize_error (when run.json itself could still be written)."""


# source_tree_hash covers these directories of the project root.
SOURCE_SUBDIRS = ("src", "experiments", "scripts", "tests", "env")
SOURCE_IGNORE_DIRS = ("__pycache__", ".pytest_cache")
SOURCE_IGNORE_DIR_SUFFIXES = (".egg-info",)
SOURCE_IGNORE_FILE_SUFFIXES = (".so", ".pyc", ".o", ".cubin")

# The pre-registration file (docs/13). Its sha256 is recorded in run.json as
# prereg_sha256 so that every number can be matched to the version of the file that was in
# force when it was produced (protocol rule 16). None when the file is absent (test roots).
PREREG_RELPATH = os.path.join("docs", "13-preregistration.md")

# Distribution names looked up with importlib.metadata (no import happens).
PACKAGE_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "pycuda": ("pycuda",),
    "cupy": ("cupy-cuda12x", "cupy-cuda11x", "cupy"),
    "numba": ("numba",),
    "tmu": ("tmu",),
    "GraphTsetlinMachine": ("graphtsetlinmachine", "GraphTsetlinMachine"),
    "PyHierarchicalTsetlinMachineCUDA": (
        "pyhierarchicaltsetlinmachinecuda",
        "PyHierarchicalTsetlinMachineCUDA",
    ),
    "torch": ("torch",),
}

RECORDED_ENV_VARS = (
    "TMLM_ROOT",
    "TMLM_CPU_SLOTS",
    "TMLM_CPU_SLOT_FDS",
    "TMLM_GPU_LOCK_MODE",
    "TMLM_GPU_LOCK_FD",
    "TMLM_LOCK_DIR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
)

CGROUP_DIR = Path("/sys/fs/cgroup")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- helpers


def project_root() -> Path:
    """The project root: $TMLM_ROOT when it names a directory, else derived from this file."""
    env_root = os.environ.get("TMLM_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


def default_results_dir(root: Path | None = None) -> Path:
    """$TMLM_RESULTS_DIR when set, else <root>/results."""
    override = os.environ.get("TMLM_RESULTS_DIR")
    if override:
        return Path(override)
    return (root or project_root()) / "results"


def utc_stamp() -> str:
    """UTC timestamp used in run dir names, for example 20260919T165501Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: Any, *, allow_nan: bool = False) -> None:
    """Write `obj` as pretty, key-sorted JSON through a temp file and an atomic rename."""
    text = json.dumps(
        hashing.normalize_json(obj, allow_nan=allow_nan),
        indent=2,
        sort_keys=True,
        allow_nan=allow_nan,
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_tree_hash(root: str | os.PathLike[str] | None = None) -> str:
    """sha256 over sorted relative paths and contents of src/, experiments/, scripts/,
    tests/ and env/ under `root`. Build products are ignored: __pycache__, .pytest_cache,
    *.egg-info, *.so, *.pyc, *.o, *.cubin. Format: see `hashing.sha256_tree`."""
    return hashing.sha256_tree(
        Path(root) if root is not None else project_root(),
        SOURCE_SUBDIRS,
        ignore_dirs=SOURCE_IGNORE_DIRS,
        ignore_dir_suffixes=SOURCE_IGNORE_DIR_SUFFIXES,
        ignore_file_suffixes=SOURCE_IGNORE_FILE_SUFFIXES,
    )


def prereg_sha256(root: str | os.PathLike[str] | None = None) -> str | None:
    """sha256 of docs/13-preregistration.md under `root`, or None if it does not exist."""
    path = (Path(root) if root is not None else project_root()) / PREREG_RELPATH
    if not path.is_file():
        return None
    return hashing.sha256_file(path)


def git_commit(root: str | os.PathLike[str] | None = None) -> str | None:
    """`git rev-parse HEAD` of the repository AT `root`, or None.

    Read-only. Returns None when root has no .git entry (so a parent repository is
    never reported by accident), when git is missing, or when there is no commit yet.
    """
    root_path = Path(root) if root is not None else project_root()
    if not (root_path / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root_path), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return None
    return commit


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, candidates in PACKAGE_DISTRIBUTIONS.items():
        versions[name] = None
        for dist in candidates:
            try:
                versions[name] = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return versions


def _run_tool(argv: Sequence[str], timeout: float = 20.0) -> tuple[str | None, str]:
    """Run a read-only tool. Returns (stdout or None, status text)."""
    if shutil.which(argv[0]) is None:
        return None, f"{argv[0]} not found"
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"error: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return None, f"error: exit {proc.returncode}: {proc.stderr.strip()[:200]}"
    return proc.stdout, "ok"


def nvidia_smi_info() -> dict[str, Any]:
    """GPU name, driver version and total memory from nvidia-smi, or a status saying why not."""
    out, status = _run_tool(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    gpus = []
    if out is not None:
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}
                )
    return {"status": status, "gpus": gpus}


def nvcc_version() -> str | None:
    """The `release X.Y, VX.Y.Z` text of `nvcc --version`, or None."""
    out, _ = _run_tool(["nvcc", "--version"])
    if out is None:
        return None
    match = re.search(r"release\s+[^\n]+", out)
    return match.group(0).strip() if match else out.strip()


def collect_env() -> dict[str, Any]:
    """Everything env.json records. Imports no third-party package."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "os_cpu_count": os.cpu_count(),
        "hash_randomization": sys.flags.hash_randomization,
        "thread_caps": {
            "cap": threadcaps.thread_cap(),
            "violations": threadcaps.violations(),
        },
        "packages": _package_versions(),
        "cgroup": {
            "cpu.max": _read_text(CGROUP_DIR / "cpu.max"),
            "memory.max": _read_text(CGROUP_DIR / "memory.max"),
        },
        "nvidia_smi": nvidia_smi_info(),
        "nvcc": nvcc_version(),
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }


# --------------------------------------------------------------------------- metrics


def validate_metrics(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Check and normalize a metrics mapping.

    Every entry must look like {"value": <JSON value>, "label": <one of METRIC_LABELS>}
    and may carry extra keys (unit, source, note, ...). Raises ValueError otherwise.
    NaN and infinite values are allowed: a diverged run must still be logged honestly.
    """
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping of name to {value, label}")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"metric name {name!r} is not a non-empty str")
        if not isinstance(entry, Mapping) or "value" not in entry or "label" not in entry:
            raise ValueError(f"metric {name!r} must be a mapping with 'value' and 'label'")
        if entry["label"] not in METRIC_LABELS:
            raise ValueError(
                f"metric {name!r} has label {entry['label']!r}, allowed: {METRIC_LABELS}"
            )
        try:
            out[name] = hashing.normalize_json(dict(entry), allow_nan=True)
        except TypeError as exc:
            raise ValueError(f"metric {name!r} is not JSON serializable: {exc}") from exc
    return out


def validate_dataset_hashes(hashes: Mapping[str, Any]) -> dict[str, str]:
    """Check a mapping of dataset name to sha256 hex digest (64 lowercase hex chars)."""
    if not isinstance(hashes, Mapping):
        raise ValueError("dataset_hashes must be a mapping of name to sha256 hex digest")
    out: dict[str, str] = {}
    for name, digest in hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"dataset name {name!r} is not a non-empty str")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise ValueError(f"dataset {name!r}: {digest!r} is not a sha256 hex digest")
        out[name] = digest
    return out


def current_run_dir() -> Path | None:
    """The run dir of the enclosing scripts/run_logged.py wrapper ($TMLM_RUN_DIR), if any."""
    value = os.environ.get("TMLM_RUN_DIR")
    return Path(value) if value else None


@contextlib.contextmanager
def _run_dir_lock(run_dir: Path) -> Iterator[None]:
    """Exclusive flock on <run_dir>/.runlog.lock for one read-merge-write cycle."""
    fd = os.open(run_dir / RUN_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing drops the flock


def _merge_into_file(
    path: Path,
    new: Mapping[str, Any],
    validate: Callable[[Mapping[str, Any]], dict[str, Any]],
    on_conflict: str,
    allow_nan: bool,
) -> None:
    """Merge validated `new` entries into the JSON mapping at `path`, under the run dir
    lock. A conflict is an existing name whose entry differs from the new one."""
    if on_conflict not in ("replace", "warn", "error"):
        raise ValueError(f"on_conflict must be replace, warn or error, got {on_conflict!r}")
    with _run_dir_lock(path.parent):
        merged: dict[str, Any] = {}
        if path.is_file():
            try:
                merged = validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"{path} exists but is not valid, it is left untouched: {exc}"
                ) from exc
        conflicts = sorted(
            name
            for name, entry in new.items()
            if name in merged
            and json.dumps(merged[name], sort_keys=True) != json.dumps(entry, sort_keys=True)
        )
        if conflicts and on_conflict == "error":
            raise ValueError(
                f"{path.name}: conflicting redefinition of {conflicts}, the file is left untouched"
            )
        if conflicts and on_conflict == "warn":
            print(
                f"[runlog] WARNING: {path.name}: {conflicts} redefined, the new value replaces "
                "the old one",
                file=sys.stderr,
                flush=True,
            )
        merged.update(new)
        write_json(path, merged, allow_nan=allow_nan)


def _target_run_dir(run_dir: str | os.PathLike[str] | None) -> Path:
    target = Path(run_dir) if run_dir is not None else current_run_dir()
    if target is None:
        raise RuntimeError("no run dir: pass run_dir or run under scripts/run_logged.py")
    return target


def write_metrics(
    metrics: Mapping[str, Any],
    run_dir: str | os.PathLike[str] | None = None,
    *,
    on_conflict: str = "warn",
) -> Path:
    """For a command running under scripts/run_logged.py: MERGE validated metrics into
    metrics.json of the wrapper's run dir (or of `run_dir`). Entries written by earlier
    calls are kept, so validation metrics and test metrics can be reported one after
    the other. The wrapper validates the file again at the end.

    on_conflict says what happens when a name is reported again with a DIFFERENT entry:
    "warn" (default) replaces it and prints one warning line to stderr, which lands in
    the run's log.txt. "replace" does so quietly. "error" raises ValueError and leaves
    the file untouched. Raises ValueError as well when the existing file is invalid.
    """
    path = _target_run_dir(run_dir) / "metrics.json"
    _merge_into_file(path, validate_metrics(metrics), validate_metrics, on_conflict, True)
    return path


def write_dataset_hashes(
    hashes: Mapping[str, Any],
    run_dir: str | os.PathLike[str] | None = None,
    *,
    on_conflict: str = "error",
) -> Path:
    """Like `write_metrics`, for dataset_hashes.json (merged into run.json by the
    wrapper). The default on_conflict is "error": one dataset name with two different
    digests inside one run is an integrity problem, not an update."""
    path = _target_run_dir(run_dir) / "dataset_hashes.json"
    _merge_into_file(
        path, validate_dataset_hashes(hashes), validate_dataset_hashes, on_conflict, False
    )
    return path


# --------------------------------------------------------------------------- manifest


def build_manifest(run_dir: Path) -> dict[str, Any]:
    """sha256 and size of every file under run_dir except MANIFEST.json itself."""
    files: dict[str, dict[str, Any]] = {}
    for full in sorted(run_dir.rglob("*")):
        if not full.is_file():
            continue
        if full.parent == run_dir and full.name in ("MANIFEST.json", "MANIFEST.json.tmp"):
            continue
        rel = full.relative_to(run_dir).as_posix()
        files[rel] = {"sha256": hashing.sha256_file(full), "size": full.stat().st_size}
    return {"format": "tmlm.manifest.v1", "files": files}


def verify_manifest(run_dir: str | os.PathLike[str]) -> list[str]:
    """Problems found when checking run_dir against its MANIFEST.json. Empty means intact."""
    run_path = Path(run_dir)
    manifest_path = run_path / "MANIFEST.json"
    if not manifest_path.is_file():
        return ["MANIFEST.json is missing"]
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (ValueError, KeyError, TypeError) as exc:
        return [f"MANIFEST.json is unreadable: {exc}"]
    actual = build_manifest(run_path)["files"]
    problems = []
    for rel in sorted(set(recorded) | set(actual)):
        if rel not in actual:
            problems.append(f"{rel}: listed in MANIFEST but missing")
        elif rel not in recorded:
            problems.append(f"{rel}: present but not listed in MANIFEST")
        elif recorded[rel].get("sha256") != actual[rel]["sha256"]:
            problems.append(f"{rel}: sha256 mismatch")
    return problems


# --------------------------------------------------------------------------- GPU sampler


class GpuMemorySampler:
    """Background thread polling `nvidia-smi --query-gpu=memory.used` for the peak value.

    One thread, mostly asleep. When nvidia-smi is missing the sampler never starts and
    reports that in its summary. After 5 consecutive failures it stops polling.
    """

    QUERY = ("--query-gpu=memory.used", "--format=csv,noheader,nounits")
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self, poll_s: float = 1.0) -> None:
        self.poll_s = float(poll_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executable = shutil.which("nvidia-smi")
        self.status = "ok" if self._executable else "nvidia-smi not found"
        self.peak_mib: int | None = None
        self.first_mib: int | None = None
        self.n_samples = 0
        self.n_errors = 0
        self._consecutive_errors = 0

    def _sample(self) -> None:
        assert self._executable is not None
        try:
            proc = subprocess.run(
                [self._executable, *self.QUERY], capture_output=True, text=True, timeout=10
            )
            values = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
            if proc.returncode != 0 or not values:
                raise ValueError(f"exit {proc.returncode}, output {proc.stdout.strip()[:80]!r}")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.n_errors += 1
            self._consecutive_errors += 1
            self.status = f"error: {type(exc).__name__}: {exc}"
            return
        self._consecutive_errors = 0
        self.status = "ok"
        used = max(values)  # one GPU is visible here; with several, the fullest one
        self.n_samples += 1
        if self.first_mib is None:
            self.first_mib = used
        if self.peak_mib is None or used > self.peak_mib:
            self.peak_mib = used

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                return
            self._sample()

    def start(self) -> None:
        if self._executable is None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="tmlm-gpu-mem", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            if self._consecutive_errors < self.MAX_CONSECUTIVE_ERRORS:
                self._sample()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "peak_mib": self.peak_mib,
            "first_sample_mib": self.first_mib,
            "n_samples": self.n_samples,
            "n_errors": self.n_errors,
            "poll_s": self.poll_s,
            "scope": "device-wide memory.used from nvidia-smi, includes other processes",
        }


# --------------------------------------------------------------------------- tee


class _Tee(io.TextIOBase):
    """Text stream that writes to the run's log file and to the original stream.

    The log comes first and is the record. The original stream (a terminal, or a pipe
    whose reader may go away: `| head`, a closed pipe) is best effort: its first
    OSError or ValueError is remembered in `stream_error` and the stream is not used
    again. A failing log file is remembered in `log_error` the same way and reported
    when the run is finalized. Nothing is raised, so neither a broken terminal nor a
    full disk ends the command that is being logged.
    """

    def __init__(self, stream: Any, log: Any, lock: threading.Lock) -> None:
        super().__init__()
        self._stream = stream
        self._log = log
        self._lock = lock
        self.stream_error: str | None = None
        self.log_error: str | None = None

    def _to_log(self, method: str, *args: Any) -> None:
        if self.log_error is not None or self._log.closed:
            return
        try:
            getattr(self._log, method)(*args)
        except (OSError, ValueError) as exc:
            self.log_error = f"{type(exc).__name__}: {exc}"

    def _to_stream(self, method: str, *args: Any) -> None:
        if self.stream_error is not None:
            return
        try:
            getattr(self._stream, method)(*args)
        except (OSError, ValueError) as exc:
            self.stream_error = f"{type(exc).__name__}: {exc}"

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._stream, "encoding", None) or "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._lock:
            self._to_log("write", text)
            self._to_stream("write", text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._to_log("flush")
            self._to_stream("flush")

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._stream.fileno()


def _emit(sink: Any, text: str) -> None:
    """Write and flush, never raise: a broken sink must not end the logged command.
    (A `_Tee` does not raise anyway. This covers RunLog(tee=False) and odd sinks.)"""
    try:
        sink.write(text)
        sink.flush()
    except (OSError, ValueError):
        pass


def _stop_process(proc: "subprocess.Popen[bytes]", grace_s: float = 10.0) -> None:
    """SIGTERM, then SIGKILL after `grace_s`, so that the command can still checkpoint."""
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except OSError:
        pass


# --------------------------------------------------------------------------- RunLog


class RunLog:
    """Context manager that owns one run directory. See the module docstring.

    Args:
        name: run name, 1 to 80 chars of [A-Za-z0-9._-], starting with a letter or digit.
        config: the FULL config of the run, JSON-like (see `hashing.normalize_json`).
        results_dir: parent of the run dir. Default: $TMLM_RESULTS_DIR or <root>/results.
        root: project root used for source_tree_hash and git_commit.
        dataset_hashes: {dataset name: sha256 hex}. More can be added during the run.
        command: recorded in run.json (the CLI wrapper passes the wrapped argv).
        sample_gpu: poll nvidia-smi for peak GPU memory.
        gpu_poll_s: polling period in seconds.
        tee: tee sys.stdout and sys.stderr into log.txt while the context is open.

    Set when the block has ended: exit_status, metrics_error, finalize_error,
    stream_errors ({"stdout" or "stderr": error text} for terminal streams that broke).

    Entering the block applies tmlm.common.threadcaps to os.environ (thread caps,
    TOKENIZERS_PARALLELISM, PYTHONHASHSEED for children) and records what it changed in
    run.json. Libraries that were imported before may have sized their pools already:
    that case is recorded too and a warning is printed.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        *,
        results_dir: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
        dataset_hashes: Mapping[str, str] | None = None,
        command: Sequence[str] | None = None,
        sample_gpu: bool = True,
        gpu_poll_s: float = 1.0,
        tee: bool = True,
    ) -> None:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"bad run name {name!r}: use 1 to 80 chars of [A-Za-z0-9._-]")
        self.name = name
        self.root = Path(root).resolve() if root is not None else project_root()
        self.results_dir = (
            Path(results_dir) if results_dir is not None else default_results_dir(self.root)
        )
        self._config = hashing.normalize_json(config)
        self._dataset_hashes = validate_dataset_hashes(dataset_hashes or {})
        self._command = [str(part) for part in command] if command is not None else None
        self._sample_gpu = sample_gpu
        self._gpu_poll_s = gpu_poll_s
        self._tee = tee
        self._tees: dict[str, _Tee] = {}
        self._metrics: dict[str, dict[str, Any]] = {}
        self._exit_status: int | None = None
        self._signal: int | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._entered = False
        self._finished = False
        self._thread_caps: dict[str, Any] = {"process": None, "command": None}
        self.run_dir: Path | None = None
        self.exit_status: int | None = None
        self.metrics_error: str | None = None
        self.finalize_error: str | None = None
        self.stream_errors: dict[str, str] = {}

    # ---- caller API

    def _is_open(self) -> bool:
        return self.run_dir is not None and not self._finished

    def _flush_metrics(self, new: Mapping[str, Any]) -> None:
        """Merge `new` into metrics.json now, so a hard kill cannot lose it."""
        if not self._is_open() or not new:
            return
        assert self.run_dir is not None
        try:
            _merge_into_file(self.run_dir / "metrics.json", new, validate_metrics, "replace", True)
        except (OSError, ValueError) as exc:
            print(
                f"[runlog] WARNING: metrics.json not updated now ({exc}). The metrics stay in "
                "memory and are written when the run ends.",
                file=sys.stderr,
                flush=True,
            )

    def add_metric(self, name: str, value: Any, label: str, **extra: Any) -> None:
        """Record one metric. `label` must be measured, derived, or unverified. The
        metric is written to metrics.json at once. The same name again replaces it."""
        new = validate_metrics({name: {"value": value, "label": label, **extra}})
        self._metrics.update(new)
        self._flush_metrics(new)

    def set_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Record several metrics at once: {name: {"value": ..., "label": ...}}. They
        are written to metrics.json at once."""
        new = validate_metrics(metrics)
        self._metrics.update(new)
        self._flush_metrics(new)

    def add_dataset_hash(self, name: str, sha256_hex: str) -> None:
        """Record one dataset digest and rewrite run.json at once. A name that already
        has a DIFFERENT digest raises ValueError: data that changes inside one run is
        an integrity problem, not an update."""
        new = validate_dataset_hashes({name: sha256_hex})
        known = self._dataset_hashes.get(name)
        if known is not None and known != new[name]:
            raise ValueError(f"dataset {name!r} already has digest {known}, not {new[name]}")
        self._dataset_hashes.update(new)
        if self._is_open():
            assert self.run_dir is not None
            try:
                write_json(self.run_dir / "run.json", self._run_record("running"))
            except OSError as exc:
                print(f"[runlog] WARNING: run.json not updated now ({exc})", file=sys.stderr,
                      flush=True)

    def path(self, relname: str) -> Path:
        """Path of an extra artifact inside the run dir (parents are created)."""
        if self.run_dir is None:
            raise RuntimeError("RunLog is not open")
        target = (self.run_dir / relname).resolve()
        if self.run_dir.resolve() not in target.parents:
            raise ValueError(f"{relname!r} escapes the run dir")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relname: str, obj: Any) -> Path:
        """Write an extra JSON artifact into the run dir. It is covered by MANIFEST.json."""
        target = self.path(relname)
        write_json(target, obj, allow_nan=True)
        return target

    def set_exit_status(self, code: int) -> None:
        """Override the exit status recorded when the block ends without an exception."""
        self._exit_status = int(code)

    def forward_signal(self, signum: int) -> None:
        """Send `signum` to the command started by `run_command`, if it is running."""
        child = self._child
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except OSError:
                pass

    def run_command(
        self,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> int:
        """Run `command`, tee its stdout and stderr, record and return its exit code.

        The child stays in the caller's process group, so a group-wide kill (for
        example by scripts/mem_guard.py) reaches it. It sees TMLM_RUN_DIR and
        TMLM_RUN_NAME, and PYTHONUNBUFFERED=1 unless the caller set that variable.
        Its environment is passed through `threadcaps.capped_env` (thread caps,
        PYTHONHASHSEED default 0), and the report is recorded in run.json. It inherits
        the GPU and CPU slot lock descriptors of scripts/gpu_run.sh and
        scripts/cpu_slot.sh (`locks.inherited_lock_fds`), so it can yield the GPU and
        nested lock calls recognize their own lock.

        A death by signal N is reported as 128 + N, like a shell does. No thread is
        used: both pipes are multiplexed with `selectors`.

        Raises CommandStartError when the command cannot be started, and nothing else
        for the command's sake: a terminal that breaks while the command runs is
        dropped (see `_Tee`), the command runs on and log.txt stays complete.
        """
        if self.run_dir is None:
            raise RuntimeError("RunLog is not open")
        child_env, caps = threadcaps.capped_env(os.environ if env is None else env)
        child_env["TMLM_RUN_DIR"] = str(self.run_dir)
        child_env["TMLM_RUN_NAME"] = self.name
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        lock_fds = locks.inherited_lock_fds(child_env)
        self._thread_caps["command"] = {**caps, "inherited_lock_fds": list(lock_fds)}
        self._command = [str(part) for part in command]
        _emit(sys.stdout, "")
        _emit(sys.stderr, "")
        try:
            proc = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                cwd=cwd,
                pass_fds=lock_fds,
            )
        except OSError as exc:
            failure = CommandStartError(self._command, exc)
            self._exit_status = failure.exit_status
            raise failure from exc
        self._child = proc
        assert proc.stdout is not None and proc.stderr is not None
        selector = selectors.DefaultSelector()
        for pipe, sink in ((proc.stdout, sys.stdout), (proc.stderr, sys.stderr)):
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            selector.register(pipe, selectors.EVENT_READ, (sink, decoder))
        try:
            while selector.get_map():
                events = selector.select(timeout=1.0)
                if not events and proc.poll() is not None:
                    # The command is gone and its pipes are quiet, but still open: a
                    # background grandchild inherited them. Do not wait for that one.
                    break
                for key, _ in events:
                    sink, decoder = key.data
                    data = os.read(key.fd, 65536)
                    if data:
                        _emit(sink, decoder.decode(data))
                    else:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            _emit(sink, tail)
                        selector.unregister(key.fileobj)
            returncode = proc.wait()
        finally:
            selector.close()
            if proc.poll() is None:
                # Only reached when this wrapper itself fails. The command must not run
                # on unlogged, but it gets the chance to checkpoint.
                _stop_process(proc)
            proc.stdout.close()
            proc.stderr.close()
            self._child = None
        if returncode < 0:
            self._signal = -returncode
            returncode = 128 - returncode
        self._exit_status = returncode
        return returncode

    # ---- context manager

    def _create_run_dir(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_stamp()
        for attempt in range(1000):
            suffix = "" if attempt == 0 else f"_{attempt:02d}"
            candidate = self.results_dir / f"{stamp}_{self.name}{suffix}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate
        raise RuntimeError(f"could not create a fresh run dir for {self.name!r}")

    def _run_record(self, status: str) -> dict[str, Any]:
        assert self.run_dir is not None
        return {
            "format": "tmlm.run.v1",
            "name": self.name,
            "run_dir": self.run_dir.name,
            "status": status,
            "pid": os.getpid(),
            "started_utc": self._started_utc,
            "git_commit": self._git_commit,
            "source_tree_hash": self._source_tree_hash,
            "prereg_sha256": self._prereg_sha256,
            "config_sha256": hashing.config_hash(self._config),
            "dataset_hashes": dict(self._dataset_hashes),
            "command": self._command,
            "thread_caps": self._thread_caps,
        }

    def __enter__(self) -> "RunLog":
        if self._entered:
            raise RuntimeError("a RunLog can be entered only once")
        self._entered = True
        self.run_dir = self._create_run_dir()
        self._started_utc = _utc_iso()
        self._t0 = time.monotonic()
        self._thread_caps["process"] = threadcaps.apply_to_process()
        write_json(self.run_dir / "config.json", self._config)
        write_json(self.run_dir / "env.json", collect_env())
        self._git_commit = git_commit(self.root)
        self._source_tree_hash = source_tree_hash(self.root)
        self._prereg_sha256 = prereg_sha256(self.root)
        write_json(self.run_dir / "run.json", self._run_record("running"))

        self._sampler = GpuMemorySampler(self._gpu_poll_s) if self._sample_gpu else None
        if self._sampler is not None:
            self._sampler.start()

        self._log = open(
            self.run_dir / "log.txt", "w", encoding="utf-8", errors="replace", buffering=1
        )
        self._saved_streams = (sys.stdout, sys.stderr)
        if self._tee:
            lock = threading.Lock()
            self._tees = {
                "stdout": _Tee(self._saved_streams[0], self._log, lock),
                "stderr": _Tee(self._saved_streams[1], self._log, lock),
            }
            sys.stdout, sys.stderr = self._tees["stdout"], self._tees["stderr"]
        caps = self._thread_caps["process"]
        late = [name for name in caps["changed"] if name in threadcaps.THREAD_VARS]
        if late and caps["already_imported"]:
            _emit(
                sys.stderr,
                f"[runlog] WARNING: thread caps {late} were missing or above {caps['cap']} and "
                f"{caps['already_imported']} were imported before: their pools may be "
                "uncapped. Source env/env.sh, or call "
                "tmlm.common.threadcaps.apply_to_process() before those imports.\n",
            )
        if self._metrics:
            self._flush_metrics(self._metrics)
        return self

    def _merge_child_files(self) -> str | None:
        """Pick up metrics.json and dataset_hashes.json written by a wrapped command.
        Entries recorded in this process win over entries of the same name in the file.
        A dataset digest in the file that contradicts a recorded one is an error."""
        assert self.run_dir is not None
        errors = []
        for filename, validate, target in (
            ("metrics.json", validate_metrics, self._metrics),
            ("dataset_hashes.json", validate_dataset_hashes, self._dataset_hashes),
        ):
            path = self.run_dir / filename
            if not path.is_file():
                continue
            try:
                loaded = validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                rejected = path.with_name(path.stem + "_rejected.json")
                os.replace(path, rejected)
                errors.append(f"{filename} rejected (kept as {rejected.name}): {exc}")
                continue
            for key, value in loaded.items():
                if filename == "dataset_hashes.json" and target.get(key, value) != value:
                    errors.append(
                        f"dataset {key!r}: the command reported digest {value}, the run "
                        f"recorded {target[key]}"
                    )
                target.setdefault(key, value)
        return "; ".join(errors) if errors else None

    def _write_final_metrics(self) -> None:
        assert self.run_dir is not None
        with _run_dir_lock(self.run_dir):
            write_json(self.run_dir / "metrics.json", self._metrics, allow_nan=True)
        (self.run_dir / RUN_LOCK_FILE).unlink(missing_ok=True)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self.run_dir is not None
        run_dir = self.run_dir
        self._finished = True
        error = None
        if exc_type is None:
            status = self._exit_status if self._exit_status is not None else 0
        elif issubclass(exc_type, SystemExit):
            code = exc.code
            status = 0 if code is None else code if isinstance(code, int) else 1
        elif issubclass(exc_type, KeyboardInterrupt):
            status, error = 130, "KeyboardInterrupt"
        else:
            status, error = 1, f"{exc_type.__name__}: {exc}"
            # Into log.txt only: the interpreter prints it on the terminal anyway.
            try:
                traceback.print_exception(exc_type, exc, tb, file=self._log)
            except (OSError, ValueError):
                pass

        # Every step is guarded: one failing step must not skip the ones after it, and
        # MANIFEST.json is always attempted last.
        problems: list[str] = []

        def step(what: str, action: Callable[[], Any]) -> tuple[bool, Any]:
            try:
                return True, action()
            except Exception as failure:  # noqa: BLE001 - reported in finalize_error
                problems.append(f"{what}: {type(failure).__name__}: {failure}")
                return False, None

        gpu = step("gpu sampler", self._sampler.stop)[1] if self._sampler is not None else None
        merged, metrics_error = step("merge of the command's files", self._merge_child_files)
        if metrics_error and status == 0:
            status = METRICS_ERROR_EXIT
        if merged:  # else: do not overwrite a metrics.json that could not even be read
            step("metrics.json", self._write_final_metrics)

        for name, tee in self._tees.items():
            tee.flush()
            if tee.stream_error is not None:
                self.stream_errors[name] = tee.stream_error
            if tee.log_error is not None and f"log.txt: {tee.log_error}" not in problems:
                problems.append(f"log.txt: {tee.log_error}")
        if self._tee:
            sys.stdout, sys.stderr = self._saved_streams
        step("log.txt", self._log.close)

        if problems and status == 0:
            status = FINALIZE_ERROR_EXIT
        self.exit_status = status
        self.metrics_error = metrics_error
        record = self._run_record("finished")
        record.update(
            {
                "finished_utc": _utc_iso(),
                "wall_time_s": round(time.monotonic() - self._t0, 3),
                "exit_status": status,
                "terminated_by_signal": self._signal,
                "error": error,
                "metrics_error": metrics_error,
                "finalize_error": "; ".join(problems) or None,
                "stream_errors": dict(self.stream_errors) or None,
                "gpu_memory": gpu,
                "peak_gpu_memory_mib": gpu["peak_mib"] if gpu else None,
            }
        )
        step("run.json", lambda: write_json(run_dir / "run.json", record))
        manifest_ok, _ = step(
            "MANIFEST.json", lambda: write_json(run_dir / "MANIFEST.json", build_manifest(run_dir))
        )
        if not manifest_ok:
            # No MANIFEST covers run.json, so it can still say why there is none.
            record["finalize_error"] = "; ".join(problems)
            step("run.json (second write)", lambda: write_json(run_dir / "run.json", record))

        if problems:
            self.finalize_error = "; ".join(problems)
            _emit(
                self._saved_streams[1],
                f"[runlog] ERROR: run dir {run_dir} is not finalized completely: "
                f"{self.finalize_error}\n",
            )
            if exc_type is None:
                raise RunLogFinalizeError(self.finalize_error)
        return False


def install_signal_forwarding(
    run: RunLog, signums: Sequence[int] = (signal.SIGTERM, signal.SIGHUP)
) -> None:
    """Forward SIGTERM and SIGHUP to the wrapped command, and survive SIGINT.

    SIGINT is not forwarded: Ctrl-C already reaches the command through the terminal's
    foreground process group, and a second copy could interrupt its cleanup. The
    wrapper only swallows its own copy so that it lives on to finalize the run dir.
    A handler is used, not SIG_IGN, because an ignored signal stays ignored across
    exec and the command would then never see Ctrl-C.
    """
    for signum in signums:
        signal.signal(signum, lambda received, _frame: run.forward_signal(received))
    signal.signal(signal.SIGINT, lambda _received, _frame: None)
