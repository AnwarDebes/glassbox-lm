"""scripts/e83_pick_noL8.py on made-up rows (dated note logs/plan_e83_note_2026-10-21.md; added 2026-10-21).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e83_pick_noL8.py

CPU only; nothing written in the study. It reuses tests/test_e83_pick.py's made-up folders, with no b8 row, and copies
scripts/e83_pick_noL8.py beside the copy of scripts/e83_pick.py. It asserts that the screen runs on the four arms (no
b8 line), that the pick rule and P1 work as before, that the mid and final stages run, and that the frozen
e83_pick.py itself still refuses for the missing b8 row.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_e83_pick as T  # noqa: E402


def case(gains: dict, pairs: dict | None = None):
    c = T.screen_case(gains, pairs)
    c.rows = [d for d in c.rows if not d["tag"].endswith("_b8")]
    shutil.copy(ROOT / "scripts" / "e83_pick_noL8.py", c.tmp / "scripts" / "e83_pick_noL8.py")
    return c


def run(c, stage: str, script: str = "e83_pick_noL8.py"):
    (c.tmp / "results/explore/abcd.jsonl").write_text("".join(T.json.dumps(d) + "\n" for d in c.rows))
    return T.subprocess.run([sys.executable, str(c.tmp / "scripts" / script), "--stage", stage], capture_output=True, text=True, cwd=c.tmp)


def main() -> None:
    c = case({"drop50": (1, 1)}, {"ref": 20, "nrand": 22})
    out = T.expect(run(c, "screen"), 0, "pick: drop50", "P1 (nrand)", "MET", "wrote logs/jobs_e83b.txt (2 jobs)")
    assert "  b8 " not in out and "b8" not in out.split("per gold label")[0], out
    T.expect(run(c, "screen", "e83_pick.py"), 1, "e83_multiwoz_a3B_c4000_s8_b8 seed 2: 0 rows (need exactly 1)")
    c.done()
    print("ok screen: four arms, no b8 line; pick drop50; P1 met at +2.0; the frozen e83_pick.py still refuses for the missing b8 row")
    c = case({})
    T.expect(run(c, "screen"), 0, "pick: none", "wrote logs/jobs_e83b.txt (0 jobs)")
    T.expect(run(c, "mid"), 0, "C1: not tested (no pick)", "wrote logs/jobs_e83c.txt (0 jobs)")
    for sd, acc in ((1, 0.3122706889523033), (2, 0.3092811523304797)):
        c.ref(T.P.REFS["multiwoz"], sd, "multiwoz", 16000, 8, acc, [0.28] * 20)
    out = T.expect(run(c, "final"), 0, "none of the four knobs lifts the reader at 4k (one seed)")
    assert "     b8" not in out.split("per gold label")[1].split("\n")[1], out
    c.done()
    c = case({"nrand": (1, 1)})
    T.full_rows(c, "nrand", 68, 57, c3={"star": 0, "abcd": 0})
    T.expect(run(c, "mid"), 0, "C1 (nrand at 16k", "wrote logs/jobs_e83c.txt (4 jobs)")
    T.expect(run(c, "final"), 0, "C2 (nrand, MultiWOZ 16k)", "C3 (nrand on abcd")
    c.done()
    print("ok mid and final: no pick (not tested, closing line, no b8 column) and a pick with C1, C2 and C3")
    print("PASSED")


if __name__ == "__main__":
    main()
