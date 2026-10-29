"""scripts/e80_report.py on made-up rows (dated note logs/plan_e80_note_2026-10-25.md; added 2026-10-25, before any e80
number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e80_report.py

CPU only; nothing is written in the study. It asserts the three readings (best-dev, final epoch, last-10 mean; means
of seeds 1 and 2; candidate minus reference in points) for STAR and ABCD, the MultiWOZ line called "view B +
xcrecent", the caveats, and the refusals: no freeze record, a changed note, a missing row, two rows, a candidate row
written before the note's freeze.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e80_report as Rp  # noqa: E402

FROZEN, LATE, EARLY = "2026-10-25T14:00:00Z", "2026-10-25T15:00:00Z", "2026-10-19T16:00:00Z"


def row(tag, seed, best, last, early_curve, when):
    """20 evaluations: 10 at early_curve, then 10 at last; the best-dev acc is best."""
    hist = [{"epoch": 2 * (k + 1), "val_acc": early_curve if k < 10 else last, "t": 1.0} for k in range(20)]
    return {"tag": tag, "seed": seed, "arm": "tm", "acc": best, "history": hist, "time": when}


class Case:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("scripts", "logs", "results/explore"):
            (self.tmp / d).mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "e80_report.py", self.tmp / "scripts" / "e80_report.py")
        (self.tmp / "logs" / f"{Rp.NOTE}.md").write_text("note stub\n")
        (self.tmp / "logs" / f"{Rp.NOTE}.frozen.json").write_text(json.dumps({"sha256": hashlib.sha256(b"note stub\n").hexdigest(), "frozen_utc": FROZEN}))
        self.rows = []
        (s_r1, s_r2, s_c), (a_r1, a_r2, a_c) = Rp.SETS["star"], Rp.SETS["abcd"]
        self.rows += [row(s_r1, 1, 0.6500, 0.6400, 0.60, EARLY), row(s_r2, 2, 0.6600, 0.6500, 0.60, EARLY),
                      row(s_c, 1, 0.6600, 0.6450, 0.60, LATE), row(s_c, 2, 0.6600, 0.6550, 0.60, LATE)]
        self.rows += [row(a_r1, 1, 0.8400, 0.8300, 0.80, EARLY), row(a_r2, 2, 0.8400, 0.8300, 0.80, EARLY),
                      row(a_c, 1, 0.8350, 0.8300, 0.80, LATE), row(a_c, 2, 0.8350, 0.8300, 0.80, LATE)]
        self.rows += [row(Rp.MWOZ[0], 1, 0.3000, 0.2900, 0.28, EARLY), row(Rp.MWOZ[1], 1, 0.3050, 0.3000, 0.28, LATE)]

    def run(self):
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e80_report.py")], capture_output=True, text=True, cwd=self.tmp)

    def done(self):
        shutil.rmtree(self.tmp)


def main() -> None:
    c = Case()
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    # STAR: best-dev 0.6550 -> 0.6600 (+0.50); final 0.6450 -> 0.6500 (+0.50); last-10 = final here (10 equal values)
    assert ("star: best-dev: view B 0.6550, view B + xcrecent 0.6600, +0.50 points; final epoch: view B 0.6450, view B + xcrecent 0.6500,"
            " +0.50 points; last-10 mean: view B 0.6450, view B + xcrecent 0.6500, +0.50 points (the best-dev line is the rule's statistic)") in p.stdout, p.stdout
    assert "abcd: best-dev: view B 0.8400, view B + xcrecent 0.8350, -0.50 points; final epoch: view B 0.8300, view B + xcrecent 0.8300, +0.00 points" in p.stdout, p.stdout
    assert "multiwoz (seed 1 only, a reading with no rule): best-dev: view B 0.3000, view B + xcrecent 0.3050, +0.50 points" in p.stdout, p.stdout
    assert "selection is optimistic" in p.stdout and "no multiplicity correction" in p.stdout and "xcustn" not in p.stdout, p.stdout
    c.done()
    print("ok readings: best-dev, final epoch and last-10 mean for STAR and ABCD (means of seeds 1 and 2), MultiWOZ as view B + xcrecent, the caveats")
    cases = [("is missing", lambda c: (c.tmp / "logs" / f"{Rp.NOTE}.frozen.json").unlink()),
             ("does not hash to its freeze record", lambda c: (c.tmp / "logs" / f"{Rp.NOTE}.md").write_text("edited\n")),
             ("e80_mwoz_BX_c16000_s8 seed 1: 0 rows (need exactly 1)", lambda c: c.rows.pop()),
             ("e78_abcd_B_c4000_s16 seed 2: 2 rows (need exactly 1)", lambda c: c.rows.append(dict(c.rows[5]))),
             ("e80_star_BX_c16000_s16 seed 2: written before the note's freeze", lambda c: c.rows[3].update(time="2026-10-25T13:00:00Z"))]
    for why, change in cases:
        c = Case()
        change(c)
        p = c.run()
        assert p.returncode == 1 and why in p.stderr, (why, p.stdout, p.stderr)
        c.done()
    print("ok refusals: no freeze record, a changed note, a missing row, two rows, a candidate row before the note's freeze")
    print("PASSED")


if __name__ == "__main__":
    main()
