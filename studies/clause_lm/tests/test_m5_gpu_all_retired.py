"""src/m5_gpu_all.sh is retired: it exits 3 at once with a message and starts nothing.

    CUDA_VISIBLE_DEVICES= .venv-cdllm/bin/python tests/test_m5_gpu_all_retired.py [SCRIPT]

Run from studies/clause_lm. SCRIPT defaults to src/m5_gpu_all.sh; give another copy (for example the version before
2026-10-25) to test it the same way. No GPU, nothing written in the repository: the script runs from a temporary copy of
the folder layout in which studies/tm_language_model/scripts/gpu_run.sh is a stub that only records that it was called,
the study's interpreters are stubs that record their calls, and studies/hierarchical_tm/queue does not exist (as in the
published tree). MAX_WAIT_MIN=0, so a version that still waits gives up at its first check.

Asserts: exit code 3; the message names the retirement; nothing was called (no stub record); the queue files were not
read (no error about them); it ends within 10 s.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]

RECORDER = """#!/usr/bin/env bash
echo "$0 $*" >> "{log}"
exit 0
"""


def main():
    script = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else STUDY / "src" / "m5_gpu_all.sh"
    print(f"script under test: {script}")
    fails = []
    with tempfile.TemporaryDirectory(prefix="clm_gpuall_") as tmp:
        root = Path(os.path.realpath(tmp))
        st = root / "studies"
        (st / "clause_lm" / "src").mkdir(parents=True)
        shutil.copy(script, st / "clause_lm" / "src" / "m5_gpu_all.sh")
        log = root / "calls.log"
        for p in (st / "tm_language_model" / "scripts" / "gpu_run.sh", st / "clause_lm" / ".venv-cdllm" / "bin" / "python",
                  st / "clause_lm" / ".venv-cdllm-gpu" / "bin" / "python"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(RECORDER.format(log=log))
            p.chmod(0o755)
        env = {**os.environ, "MAX_WAIT_MIN": "0", "CUDA_VISIBLE_DEVICES": ""}
        t0 = time.time()
        p = subprocess.run(["bash", str(st / "clause_lm" / "src" / "m5_gpu_all.sh")], cwd=root, env=env,
                           capture_output=True, text=True, timeout=120)
        dt = time.time() - t0
        out = p.stdout + p.stderr
        print(out.strip())
        for cond, what in ((p.returncode == 3, f"exit code 3 (got {p.returncode})"),
                           ("retired" in out, "the message names the retirement"),
                           (not log.exists(), "nothing was called"),
                           ("queue" not in p.stderr, "the queue files were not read"),
                           (dt < 10, f"ends within 10 s ({dt:.1f} s)")):
            print(("ok   " if cond else "FAIL ") + what)
            if not cond:
                fails.append(what)
    print(f"{len(fails)} failed" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
