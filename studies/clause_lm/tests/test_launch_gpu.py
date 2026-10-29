"""src/launch_gpu.sh: argument handling and the launch chain, with a stub in place of the GPU lock.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_launch_gpu.py [LAUNCHER]

Run from studies/clause_lm. LAUNCHER defaults to src/launch_gpu.sh; give another copy (for example the version before
the fix of 2026-10-25) to test it the same way. No GPU, no GPU lock, nothing written in the repository: every case runs
in a temporary copy of the folder layout, where studies/tm_language_model holds copies of env/env.sh and
scripts/mem_guard.py and a stub scripts/gpu_run.sh that takes no lock and runs its command, and studies/decision_layer
is a link to the real folder (its .venv-htm python only runs a dummy job that prints what it sees; nothing is written
into that environment). A second group of cases runs the launcher in place, in dry mode (CLM_DRY_RUN=1) and with the
stub (CLM_GPU_RUN), so the real gpu_run.sh is never started.

Asserts:
- the dry mode prints nice -n 19, timeout, the guard interpreter with -I -B and the absolute path of mem_guard.py, the
  stub, mode exclusive, the job interpreter with -u, the absolute job path and the job arguments unchanged;
- by default both interpreters are ../decision_layer/.venv-htm/bin/python; CLM_JOB_PYTHON and CLM_GUARD_PYTHON
  replace them;
- a relative job path is read from the caller's working directory;
- the real chain (timeout -> mem_guard.py -> stub -> job) starts with no python3 on PATH: the job runs in the
  decision_layer environment, in its own process group (set by mem_guard.py), as a child of mem_guard.py, in the
  tm_language_model folder, at nice 19, with the OpenMP, OpenBLAS, MKL, numexpr, numba and rayon pools at 2 threads
  (also when the caller set them higher), unbuffered, with its arguments intact, and its exit code comes back;
- usage errors exit 2 with a message: too few arguments, MINUTES not a whole number (1h, -5, empty) or 0, a missing job
  script, a missing interpreter, a missing wrapper.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
STUDIES = STUDY.parent
TMLM = STUDIES / "tm_language_model"
DL = STUDIES / "decision_layer"
DL_PY = DL / ".venv-htm" / "bin" / "python"

STUB = """#!/usr/bin/env bash
# test stub of gpu_run.sh: takes no lock, checks its arguments, runs the command
[ "$1" = exclusive ] && [ "$2" = -- ] || { echo "STUB bad arguments: $*" >&2; exit 9; }
shift 2
echo "STUB_GPU_RUN exclusive" >&2
exec "$@"
"""

JOB = """import json, os, sys
with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
    parent = [a.decode() for a in f.read().split(b"\\0") if a]
print("JOB " + json.dumps({"exe": sys.executable, "prefix": sys.prefix, "argv": sys.argv[1:], "cwd": os.getcwd(),
                           "omp": os.environ.get("OMP_NUM_THREADS"), "numba": os.environ.get("NUMBA_NUM_THREADS"),
                           "mkl": os.environ.get("MKL_NUM_THREADS"), "numexpr": os.environ.get("NUMEXPR_NUM_THREADS"),
                           "rayon": os.environ.get("RAYON_NUM_THREADS"), "nice": os.nice(0),
                           "openblas": os.environ.get("OPENBLAS_NUM_THREADS"), "own_group": os.getpgrp() == os.getpid(),
                           "write_through": sys.stdout.write_through, "parent": parent}), flush=True)
sys.exit(int(os.environ.get("JOB_EXIT", "0")))
"""

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def no_python3_path():
    """PATH without any directory that holds a python3 (the machine's state: none is on PATH)."""
    keep = [d for d in os.environ.get("PATH", "").split(":") if d and not os.path.exists(os.path.join(d, "python3"))]
    return ":".join(keep)


def run(launcher, args, cwd, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLM_")}
    env.update({"PATH": no_python3_path(), "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(extra or {})
    p = subprocess.run(["bash", str(launcher), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def fake_tree(root, launcher):
    """root/studies/{clause_lm/src/launch_gpu.sh, tm_language_model (copies + stub), decision_layer -> real}."""
    st = root / "studies"
    (st / "clause_lm" / "src").mkdir(parents=True)
    shutil.copy(launcher, st / "clause_lm" / "src" / "launch_gpu.sh")
    (st / "tm_language_model" / "env").mkdir(parents=True)
    (st / "tm_language_model" / "scripts").mkdir()
    shutil.copy(TMLM / "env" / "env.sh", st / "tm_language_model" / "env" / "env.sh")
    shutil.copy(TMLM / "scripts" / "mem_guard.py", st / "tm_language_model" / "scripts" / "mem_guard.py")
    stub = st / "tm_language_model" / "scripts" / "gpu_run.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    if os.access(DL_PY, os.X_OK):
        (st / "decision_layer").symlink_to(DL, target_is_directory=True)
    else:                                   # a clone without decision_layer/.venv-htm: a stand-in interpreter
        print(f"note: {DL_PY} is missing; the fake tree links its default interpreter to {sys.executable}")
        (st / "decision_layer" / ".venv-htm" / "bin").mkdir(parents=True)
        (st / "decision_layer" / ".venv-htm" / "bin" / "python").symlink_to(sys.executable)
    job = root / "work" / "job.py"
    job.parent.mkdir()
    job.write_text(JOB)
    return st / "clause_lm" / "src" / "launch_gpu.sh", st, job


def guard_py(lines):
    """The guard interpreter in a printed chain: the argument after timeout's duration."""
    return lines[lines.index("--kill-after=60") + 2] if "--kill-after=60" in lines else None


def job_py(lines):
    """The job interpreter in a printed chain: the argument after 'exclusive --'."""
    return lines[lines.index("exclusive") + 2] if "exclusive" in lines else None


def job_record(out):
    lines = [l for l in out.splitlines() if l.startswith("JOB ")]
    return json.loads(lines[-1][4:]) if lines else None


def cases_fake(launcher):
    with tempfile.TemporaryDirectory(prefix="clm_launch_") as tmp:
        root = Path(os.path.realpath(tmp))
        lg, st, job = fake_tree(root, launcher)
        tm = st / "tm_language_model"
        dl_py = str(st / "clause_lm" / ".." / "decision_layer" / ".venv-htm" / "bin" / "python")
        stub = str(tm / "scripts" / "gpu_run.sh")

        # 1. dry mode, default interpreters, relative job path from the caller's folder
        rc, out, err = run(lg, ["7", "job.py", "--seed", "3", "two words"], cwd=job.parent, extra={"CLM_DRY_RUN": "1"})
        want = ["nice", "-n", "19", "timeout", "--kill-after=60", "7m", dl_py, "-I", "-B", str(tm / "scripts" / "mem_guard.py"), "--scope",
                "group", "--limit-gib", "12", "--", stub, "exclusive", "--", dl_py, "-u", str(job.resolve()),
                "--seed", "3", "two words"]
        check(rc == 0 and out.splitlines() == want, f"dry mode prints the chain (rc {rc}; stderr {err.strip()[-200:]!r})")

        # 2. overrides of both interpreters
        alt = sys.executable
        rc, out, err = run(lg, ["5", str(job)], cwd=root,
                           extra={"CLM_DRY_RUN": "1", "CLM_JOB_PYTHON": alt, "CLM_GUARD_PYTHON": alt})
        lines = out.splitlines()
        check(rc == 0 and guard_py(lines) == alt and job_py(lines) == alt,
              "CLM_JOB_PYTHON and CLM_GUARD_PYTHON replace the interpreters")
        rc, out, err = run(lg, ["5", str(job)], cwd=root, extra={"CLM_DRY_RUN": "1", "CLM_JOB_PYTHON": alt})
        lines = out.splitlines()
        check(rc == 0 and guard_py(lines) == alt and job_py(lines) == alt,
              "the guard interpreter defaults to the job interpreter")

        # 3. the real chain with the stub, no python3 on PATH
        rc, out, err = run(lg, ["2", "job.py", "a", "b c", "--x=1"], cwd=job.parent)
        r = job_record(out)
        check(rc == 0 and r is not None, f"the chain runs the job (rc {rc}; stderr {err.strip()[-300:]!r})")
        if r:
            check(r["argv"] == ["a", "b c", "--x=1"], "job arguments arrive unchanged")
            if os.access(DL_PY, os.X_OK):
                check(os.path.realpath(r["prefix"]) == os.path.realpath(DL / ".venv-htm"),
                      f"job runs in decision_layer/.venv-htm (prefix {r['prefix']})")
            check(r["own_group"], "job runs in its own process group (mem_guard.py started it)")
            check(any(a.endswith("mem_guard.py") for a in r["parent"]) and "-I" in r["parent"] and "-B" in r["parent"],
                  f"job's parent is mem_guard.py run with -I -B ({r['parent'][:4]})")
            check(os.path.realpath(r["cwd"]) == os.path.realpath(tm), "job runs in the tm_language_model folder")
            check((r["omp"], r["numba"], r["openblas"], r["mkl"], r["numexpr"], r["rayon"]) == ("2",) * 6,
                  f"all six thread pools are capped at 2 ({r['omp']}, {r['numba']}, {r['openblas']}, {r['mkl']}, "
                  f"{r['numexpr']}, {r['rayon']})")
            check(r["nice"] == 19, f"the job runs at nice 19 ({r['nice']})")
            check(r["write_through"] is True, "job output is unbuffered (-u)")
        check("STUB_GPU_RUN exclusive" in err, "the lock wrapper was called with mode exclusive")
        rc, out, err = run(lg, ["2", "job.py"], cwd=job.parent, extra={"MKL_NUM_THREADS": "6", "NUMEXPR_NUM_THREADS": "6",
                                                                  "RAYON_NUM_THREADS": "6", "OMP_NUM_THREADS": "5"})
        r = job_record(out)
        check(rc == 0 and r is not None and (r["omp"], r["mkl"], r["numexpr"], r["rayon"]) == ("2",) * 4,
              "a caller's higher thread settings are brought down to 2")
        rc, out, err = run(lg, ["2", "job.py"], cwd=job.parent, extra={"JOB_EXIT": "3"})
        check(rc == 3, f"the job's exit code comes back (rc {rc})")

        # 4. usage errors
        for args, what in ((["7"], "too few arguments"), (["1h", "job.py"], "MINUTES 1h"),
                           (["-5", "job.py"], "MINUTES -5"), (["", "job.py"], "MINUTES empty"),
                           (["0", "job.py"], "MINUTES 0"), (["5", "missing.py"], "missing job script")):
            rc, out, err = run(lg, args, cwd=job.parent, extra={"CLM_DRY_RUN": "1"})
            check(rc == 2 and "launch_gpu.sh:" in err and not out.strip(), f"usage error exits 2: {what} (rc {rc})")
        rc, out, err = run(lg, ["5", "job.py"], cwd=job.parent,
                           extra={"CLM_DRY_RUN": "1", "CLM_JOB_PYTHON": str(root / "no" / "python")})
        check(rc == 2 and "CLM_JOB_PYTHON" in err, f"missing job interpreter exits 2 (rc {rc})")
        rc, out, err = run(lg, ["5", "job.py"], cwd=job.parent,
                           extra={"CLM_DRY_RUN": "1", "CLM_GUARD_PYTHON": str(root / "no" / "python")})
        check(rc == 2 and "CLM_GUARD_PYTHON" in err, f"missing guard interpreter exits 2 (rc {rc})")
        rc, out, err = run(lg, ["5", "job.py"], cwd=job.parent,
                           extra={"CLM_DRY_RUN": "1", "CLM_GPU_RUN": str(root / "no" / "gpu_run.sh")})
        check(rc == 2 and "wrapper" in err, f"missing lock wrapper exits 2 (rc {rc})")


def cases_in_place(launcher):
    """The launcher in its real place: dry mode, and the chain with the stub (never the real gpu_run.sh)."""
    with tempfile.TemporaryDirectory(prefix="clm_launch_real_") as tmp:
        root = Path(os.path.realpath(tmp))
        job = root / "job.py"
        job.write_text(JOB)
        stub = root / "gpu_run_stub.sh"
        stub.write_text(STUB)
        stub.chmod(0o755)
        rc, out, err = run(launcher, ["9", str(job), "z"], cwd=root, extra={"CLM_DRY_RUN": "1"})
        lines = out.splitlines()
        check(rc == 0 and lines[:3] == ["nice", "-n", "19"]
              and guard_py(lines) == str(STUDY / ".." / "decision_layer" / ".venv-htm" / "bin" / "python")
              and lines[lines.index("-B") + 1] == str(TMLM.resolve() / "scripts" / "mem_guard.py")
              and lines[lines.index("exclusive") - 1] == str(TMLM.resolve() / "scripts" / "gpu_run.sh")
              and lines[-1] == "z",
              f"in place, dry mode names the real interpreter, guard and lock wrapper (rc {rc})")
        check(os.access(DL_PY, os.X_OK), "decision_layer/.venv-htm/bin/python exists")
        rc, out, err = run(launcher, ["2", str(job), "z"], cwd=root, extra={"CLM_GPU_RUN": str(stub)})
        r = job_record(out)
        check(rc == 0 and r is not None and r["argv"] == ["z"] and r["own_group"],
              f"in place, the chain runs with the stub wrapper (rc {rc}; stderr {err.strip()[-300:]!r})")


def main():
    launcher = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else STUDY / "src" / "launch_gpu.sh"
    print(f"launcher under test: {launcher}")
    cases_fake(launcher)
    if launcher == (STUDY / "src" / "launch_gpu.sh").resolve() and os.access(DL_PY, os.X_OK):
        cases_in_place(launcher)
    elif launcher == (STUDY / "src" / "launch_gpu.sh").resolve():
        print(f"skip the in-place cases: {DL_PY} is missing")
    else:
        print("skip the in-place cases: they run only for src/launch_gpu.sh itself")
    print(f"{len(FAILS)} failed" if FAILS else "PASS")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
