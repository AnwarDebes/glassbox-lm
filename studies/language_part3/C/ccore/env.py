"""Run-rule guard (PLAN_part3.md section 6 and the task's hard rules). A job that is not started the
registered way stops before it reads any data."""
import os
import sys
import tempfile

from . import paths

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
               "RAYON_NUM_THREADS")
TMP_VARS = ("NUMBA_CACHE_DIR", "TMPDIR", "JOBLIB_TEMP_FOLDER", "MPLCONFIGDIR", "XDG_CACHE_HOME")
THREADS = 2
NICE = 19


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def problems():
    out = []
    if not sys.flags.isolated:
        out.append("python was not started with -I")
    if not sys.flags.dont_write_bytecode:
        out.append("python was not started with -B")
    for k in THREAD_VARS:
        if os.environ.get(k) != str(THREADS):
            out.append(f"{k} is {os.environ.get(k)!r}, not '{THREADS}'")
    for k in TMP_VARS:
        v = os.environ.get(k)
        if not v or not _inside(v, paths.TMP):
            out.append(f"{k} is {v!r}, not inside {paths.TMP}")
    if not _inside(tempfile.gettempdir(), paths.TMP):
        out.append(f"tempfile.gettempdir() is {tempfile.gettempdir()}, not inside {paths.TMP}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        out.append("CUDA_VISIBLE_DEVICES is not set to the empty string (no GPU)")
    if os.nice(0) != NICE:
        out.append(f"nice level is {os.nice(0)}, not {NICE}")
    return out


def guard():
    """Stop unless every run rule holds; then cap the thread pools of numpy's BLAS at 2."""
    p = problems()
    if p:
        raise SystemExit("run rules not met:\n  " + "\n  ".join(p))
    global _LIMITS
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(limits=THREADS)  # kept referenced for the life of the job
    except ImportError:  # the environment variables already cap the pools
        _LIMITS = None
    paths.setup()


_LIMITS = None

NEURAL_MODULES = ("torch", "tensorflow", "jax", "transformers", "sentence_transformers", "model2vec")


def neural_modules_loaded():
    """Names of neural-network libraries imported in this process (the CPU stages must have none)."""
    return sorted(m for m in NEURAL_MODULES if m in sys.modules)
