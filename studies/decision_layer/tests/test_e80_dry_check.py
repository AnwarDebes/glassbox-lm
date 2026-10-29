"""scripts/e80_dry_check.py on made-up dry rows (dated note logs/plan_e80_note_2026-10-25.md; added 2026-10-25, before any
e80 number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e80_dry_check.py

CPU only; nothing is written in the study. Each case runs a copy of the check in a temporary folder whose
results/explore/abcd.jsonl holds filler rows, the five real reference lines copied byte for byte from the study's file
at their real line numbers (read, never printed), and made-up dry rows after them. It asserts:
- the pass: five dry rows, the shared bits of the two seeds, the reference rows, "all dry checks passed"; no accuracy
  printed;
- refusals: no freeze record, a changed note, a missing dry row, two rows, a row before the note's freeze, 40 epochs
  in place of 2, another bit count, another xcrecent block size, xcrecent off the C4 table, the two seeds on different
  bits, a reference line edited, a reference row added twice.
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
import e80_dry_check as D  # noqa: E402

FROZEN, LATE = "2026-10-25T14:00:00Z", "2026-10-25T15:00:00Z"
REAL = (ROOT / "results/explore/abcd.jsonl").read_text().splitlines()


def dry_row(ds, sd, n_bits=None, xc=None, table=D.TABLE, epochs=2, when=LATE, bits=None):
    g = D.DRY[ds]
    return {"tag": g["tag"], "seed": sd, "arm": "tm", "dataset": ds, "encoder": "none", "parts": D.BN, "lex_k": 1000, "acc": 0.5,
            "cfg": {"n_clauses": 200, "T": 200, "epochs": epochs, "s": g["s"], "seed": sd}, "time": when,
            "n_bits": g["n_bits"] if n_bits is None else n_bits, "bits_sha256": bits or {"train": f"{ds}t", "dev": f"{ds}d"},
            "static_info": {"xcrecent": {"n_bits": g["xcrecent"] if xc is None else xc, "table_file_sha256": table}}}


class Case:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "scripts").mkdir()
        (self.tmp / "logs").mkdir()
        (self.tmp / "results/explore").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "e80_dry_check.py", self.tmp / "scripts" / "e80_dry_check.py")
        (self.tmp / "logs" / f"{D.NOTE}.md").write_text("note stub\n")
        (self.tmp / "logs" / f"{D.NOTE}.frozen.json").write_text(json.dumps({"sha256": hashlib.sha256(b"note stub\n").hexdigest(), "frozen_utc": FROZEN}))
        want = {n: k for k, (n, _) in D.REFS.items()}
        self.lines = [REAL[n - 1] if n in want else json.dumps({"tag": "filler", "seed": n, "arm": "lr"}) for n in range(1, max(want) + 1)]
        self.dry = [dry_row(ds, sd) for ds, g in D.DRY.items() for sd in g["seeds"]]

    def run(self):
        text = "\n".join(self.lines + [json.dumps(r) for r in self.dry]) + "\n"
        (self.tmp / "results/explore/abcd.jsonl").write_text(text)
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e80_dry_check.py")], capture_output=True, text=True, cwd=self.tmp)

    def done(self):
        shutil.rmtree(self.tmp)


def main() -> None:
    c = Case()
    p = c.run()
    assert p.returncode == 0 and p.stdout.rstrip().endswith("all dry checks passed"), p.stdout + p.stderr
    assert p.stdout.count("\nok e80dry_") + p.stdout.startswith("ok e80dry_") == 7 and "ok the 5 reference rows: one row each, at their pinned lines, unchanged" in p.stdout, p.stdout
    assert "0.5" not in p.stdout, p.stdout  # no accuracy printed
    c.done()
    print("ok pass: five dry rows (settings, 10,361 / 9,657 / 9,315 bits, the xcrecent block on the C4 table), shared bits, the five reference rows")

    def edit_line(c, n):
        d = json.loads(c.lines[n - 1])
        d["time"] = d["time"][:-1] + "Z "  # one byte more
        c.lines[n - 1] = json.dumps(d)

    cases = [
        ("is missing: the dry run is checked only after", lambda c: (c.tmp / "logs" / f"{D.NOTE}.frozen.json").unlink()),
        ("does not hash to its freeze record", lambda c: (c.tmp / "logs" / f"{D.NOTE}.md").write_text("note stub, edited\n")),
        ("e80dry_mwoz_BX seed 1: 0 rows (need exactly 1)", lambda c: c.dry.pop()),
        ("e80dry_abcd_BX seed 1: 2 rows (need exactly 1)", lambda c: c.dry.append(dry_row("abcd", 1))),
        ("e80dry_star_BX seed 2: written before the note's freeze", lambda c: c.dry.__setitem__(3, dry_row("star", 2, when="2026-10-25T13:00:00Z"))),
        ("e80dry_star_BX seed 1: settings differ", lambda c: c.dry.__setitem__(2, dry_row("star", 1, epochs=40))),
        ("e80dry_abcd_BX seed 2: bits 9658 with xcrecent 2973", lambda c: c.dry.__setitem__(1, dry_row("abcd", 2, n_bits=9658))),
        ("e80dry_mwoz_BX seed 1: bits 9315 with xcrecent 2734", lambda c: c.dry.__setitem__(4, dry_row("multiwoz", 1, xc=2734))),
        ("e80dry_star_BX seed 1: bits 10361 with xcrecent 3134 (table 00000000)", lambda c: c.dry.__setitem__(2, dry_row("star", 1, table="0" * 64))),
        ("e80dry_abcd_BX: the two seeds' train and dev bits differ", lambda c: c.dry.__setitem__(1, dry_row("abcd", 2, bits={"train": "x", "dev": "y"}))),
        ("reference e78_abcd_B_c4000_s16 seed 2: not exactly one row at line 109", lambda c: edit_line(c, 109)),
        ("reference e74c_star_Bc4_c16000_s16 seed 1: not exactly one row at line 86", lambda c: c.lines.append(REAL[85])),
    ]
    for why, change in cases:
        c = Case()
        change(c)
        p = c.run()
        assert p.returncode == 1 and why in p.stderr and "all dry checks passed" not in p.stdout, (why, p.stdout, p.stderr)
        c.done()
    print(f"ok {len(cases)} refusals: no freeze record, a changed note, a missing row, two rows, a row before the freeze, 40 epochs, another"
          " bit count, another xcrecent size, xcrecent off the C4 table, seeds on different bits, a reference line edited, a reference row added twice")
    print("PASSED")


if __name__ == "__main__":
    main()
