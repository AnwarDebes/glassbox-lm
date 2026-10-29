"""scripts/freeze.py in a temporary copy of the study layout: the record, the pin check and every refusal.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_freeze.py

Run from studies/clause_lm. CPU only, under a second. The tool is copied into a temporary study folder with a toy
src/, scripts/ and tests/, a toy plan and a file outside the study; nothing of the real study is frozen or changed.

Asserts:
- a plan whose hash block pins a relative and an absolute path is frozen: exit 0, a record with the time of the call,
  the plan's SHA-256, the two pins, the study files (src/, scripts/, tests/; no __pycache__) and the tool's hash; the
  plan and the record become read-only; a sha256sum-like line outside a '## Hashes' block is not a pin;
- a second freeze of the same plan is refused (exit 1) and the record is unchanged;
- refused, with no record written (exit 1): a pinned file that differs, a pinned file that is missing, a plan without
  a hash block;
- --check prints the result and writes nothing: exit 0 when every pin holds, 1 when one differs;
- usage errors exit 2: no argument, two arguments, a path that is not a .md file, a missing plan.
"""

import calendar
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
TOOL = toyutil.STUDY / "scripts" / "freeze.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(study, *args):
    p = subprocess.run([sys.executable, "-I", str(study / "scripts" / "freeze.py"), *args], cwd=study,
                       capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr


def plan_text(pins, extra=""):
    lines = "\n".join(f"{h}  {p}" for h, p in pins)
    return (f"# Toy plan\n\nSome text.\n\n```\n{'0' * 64}  src/not_a_pin.py\n```\n\n## Hashes (toy)\n\n```\n{lines}\n```\n"
            f"\n{extra}## After\n\ntext\n")


def main():
    with tempfile.TemporaryDirectory(prefix="clm_freeze_") as tmp:
        root = Path(os.path.realpath(tmp))
        study = root / "studies" / "clause_lm"
        for d in ("scripts", "src/sub/__pycache__", "tests", "docs"):
            (study / d).mkdir(parents=True)
        shutil.copy(TOOL, study / "scripts" / "freeze.py")
        (study / "src" / "a.py").write_text("x = 1\n")
        (study / "src" / "sub" / "b.sh").write_text("echo b\n")
        (study / "src" / "sub" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
        (study / "tests" / "t.py").write_text("pass\n")
        outside = root / "records" / "out.bin"
        outside.parent.mkdir()
        outside.write_bytes(os.urandom(1000))
        good = [(sha(study / "src" / "a.py"), "src/a.py"), (sha(outside), str(outside))]
        plan = study / "docs" / "plan_toy.md"
        plan.write_text(plan_text(good))

        rc, out, err = run(study, "--check", "docs/plan_toy.md")
        check(rc == 0 and "every pin holds" in out and not (study / "docs" / "plan_toy.frozen.json").exists()
              and os.stat(plan).st_mode & stat.S_IWUSR, f"--check: every pin holds, nothing written (rc {rc})")
        before = int(time.time())
        rc, out, err = run(study, "docs/plan_toy.md")
        after = int(time.time())
        rec_p = study / "docs" / "plan_toy.frozen.json"
        check(rc == 0 and rec_p.exists(), f"a plan with true pins is frozen (rc {rc}; {err.strip()})")
        rec = json.loads(rec_p.read_text())
        t = calendar.timegm(time.strptime(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        check(before - 1 <= t <= after + 1, f"the record holds the time of the call ({rec['frozen_utc']})")
        check(rec["plan"] == {"docs/plan_toy.md": sha(plan)}, "the record holds the plan's SHA-256")
        check(rec["pinned"] == {p: h for h, p in good}, f"the record holds the two pins only ({sorted(rec['pinned'])})")
        check(sorted(rec["study_files"]) == ["scripts/freeze.py", "src/a.py", "src/sub/b.sh", "tests/t.py"]
              and rec["study_files"]["src/a.py"] == good[0][0], "study files of src/, scripts/, tests/ (no __pycache__)")
        check(rec["tool"] == {"scripts/freeze.py": sha(TOOL)}, "the record holds the tool's hash")
        ro = all(not (os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) for p in (plan, rec_p))
        check(ro, "the plan and the record are read-only")
        snapshot = rec_p.read_bytes()
        rc, out, err = run(study, "docs/plan_toy.md")
        check(rc == 1 and "already frozen" in err and rec_p.read_bytes() == snapshot,
              f"a second freeze is refused and the record unchanged (rc {rc})")

        cases = [("plan_wrong.md", plan_text([(good[0][0], "src/a.py"), ("f" * 64, str(outside))]), "differ"),
                 ("plan_missing.md", plan_text([(good[0][0], "src/a.py"), (good[0][0], "src/gone.py")]), "missing"),
                 ("plan_nohash.md", "# Toy plan\n\n## Hashes\n\nnone here\n", "no hash block")]
        (study / "docs" / "plan_wrong.md").write_text(cases[0][1])
        rc, out, err = run(study, "--check", "docs/plan_wrong.md")
        check(rc == 1 and "differ" in err and not (study / "docs" / "plan_wrong.frozen.json").exists(),
              f"--check: a differing pin exits 1, nothing written (rc {rc})")
        for name, text, what in cases:
            (study / "docs" / name).write_text(text)
            rc, out, err = run(study, f"docs/{name}")
            check(rc == 1 and not (study / "docs" / (name[:-3] + ".frozen.json")).exists() and what in err,
                  f"refused without a record: {what} (rc {rc}; {err.strip().splitlines()[0] if err.strip() else ''})")
        for args, what in (([], "no argument"), (["docs/plan_wrong.md", "x"], "two arguments"),
                           (["src/a.py"], "not a .md file"), (["docs/none.md"], "missing plan")):
            rc, out, err = run(study, *args)
            check(rc == 2 and "usage" in err, f"usage error exits 2: {what} (rc {rc})")
    check.done()


if __name__ == "__main__":
    main()
