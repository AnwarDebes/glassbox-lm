"""src/runlog.py: the run folder and its files (study rule 4), in a temporary folder.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_runlog.py

Run from studies/clause_lm. CPU only, under a second. runlog.ROOT is pointed at a temporary folder, so nothing is
written into results/. runlog.py itself is not changed by this test (the run records of 2026-10-14 hold copies of it).

Asserts:
- Run() makes results/<UTC time>_<name>/ with the time of the call (format %Y%m%dT%H%M%SZ);
- config.json holds the config; data_hashes.json holds, for each data file, the path relative to the folder above
  ROOT and the SHA-256 computed independently with hashlib;
- the calling script and runlog.py are copied into the folder byte for byte;
- print() writes to standard output and to log.txt; finish() writes metrics.json with wall_s and returns the folder;
- a second Run with the same name in the same second fails (no folder is reused);
- sha256() equals hashlib on a file larger than one read chunk.
"""

import calendar
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()


def main():
    with tempfile.TemporaryDirectory(prefix="clm_runlog_") as tmp:
        root = Path(os.path.realpath(tmp)) / "studyroot"
        root.mkdir()
        _, runlog = toyutil.use_src(root)
        data = root / "data" / "toy.bin"
        data.parent.mkdir()
        data.write_bytes(os.urandom(3 * (1 << 20) + 17))
        script = root / "myscript.py"
        script.write_text("print('hello')\n")
        before = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run = runlog.Run("toy_run", {"a": 1, "b": [1, 2]}, data_files=[str(data)], script=str(script))
            run.print("line", 42)
        after = time.time()
        d = Path(run.dir)
        stamp = d.name.split("_toy_run")[0]
        t = calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))
        check(d.parent == root / "results" and d.name.endswith("_toy_run") and int(before) - 1 <= t <= after + 1,
              f"folder results/<UTC>_toy_run made at the call ({d.name})")
        check(json.loads((d / "config.json").read_text()) == {"a": 1, "b": [1, 2]}, "config.json holds the config")
        hashes = json.loads((d / "data_hashes.json").read_text())
        want_key = os.path.relpath(str(data), os.path.dirname(str(root)))
        check(hashes == {want_key: hashlib.sha256(data.read_bytes()).hexdigest()},
              f"data_hashes.json: key {want_key!r} and the hashlib SHA-256")
        check((d / "myscript.py").read_bytes() == script.read_bytes(), "the calling script is copied")
        check((d / "runlog.py").read_bytes() == (toyutil.SRC / "runlog.py").read_bytes(), "runlog.py is copied")
        check("line 42" in buf.getvalue(), "print() writes to standard output")
        out_dir = run.finish({"x": 0.5})
        log = (d / "log.txt").read_text()
        check(log.startswith("run dir: ") and "line 42\n" in log and "METRICS " in log, "log.txt holds every printed line")
        m = json.loads((d / "metrics.json").read_text())
        check(out_dir == run.dir and m["x"] == 0.5 and isinstance(m["wall_s"], float), "finish() writes metrics.json")

        fixed = time.gmtime()
        real_gmtime = runlog.time.gmtime
        runlog.time.gmtime = lambda *a: fixed
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r1 = runlog.Run("same", {})
                try:
                    runlog.Run("same", {})
                    second = "made"
                except FileExistsError:
                    second = "refused"
            r1._log.close()
        finally:
            runlog.time.gmtime = real_gmtime
        check(second == "refused", "a second Run with the same name in the same second fails")
        check(runlog.sha256(str(data)) == hashlib.sha256(data.read_bytes()).hexdigest(), "sha256() equals hashlib")
    check.done()


if __name__ == "__main__":
    main()
