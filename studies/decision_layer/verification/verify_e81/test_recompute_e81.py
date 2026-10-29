"""Test of recompute_e81.py on the made-up rows of the study's tests/test_e81_pick.py (private; temporary folders)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import subprocess, sys, tempfile, shutil
from pathlib import Path
S = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(S / "tests"))
import test_e81_pick as T  # noqa: E402
HERE = Path(__file__).resolve().parent
for b2_met in (True, False):
    tmp = Path(tempfile.mkdtemp())
    T.write(tmp, {**T.stage_a_rows(), **T.stage_b_rows(b2_met)})
    q = subprocess.run([sys.executable, "-I", str(tmp / "scripts" / "e81_pick.py"), "--stage", "final"], cwd=tmp, capture_output=True, text=True)
    assert q.returncode == 0, q.stderr
    (tmp / "final.out").write_text(q.stdout)
    p = subprocess.run([sys.executable, "-I", str(HERE / "recompute_e81.py"), str(tmp), str(tmp / "final.out")], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip().endswith("MATCH"), (p.stdout, p.stderr)
    bad = q.stdout.replace("mean of seeds 1-2: +", "mean of seeds 1-2: +9", 1)
    (tmp / "bad.out").write_text(bad)
    p2 = subprocess.run([sys.executable, "-I", str(HERE / "recompute_e81.py"), str(tmp), str(tmp / "bad.out")], capture_output=True, text=True)
    assert "MISMATCH" in p2.stdout, p2.stdout
    print(f"ok (B2 {'met' if b2_met else 'not met'}): recompute MATCHes e81_pick's final output; a tampered output gives MISMATCH")
    shutil.rmtree(tmp)
print("PASSED")
