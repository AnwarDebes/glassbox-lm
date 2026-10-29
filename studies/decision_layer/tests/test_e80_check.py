"""scripts/e80_check.py (plan logs/plan_e80_2d_recent.md, frozen 2026-10-20T20:53:40Z) on made-up rows. Added 2026-10-25
with e80's dated note (logs/plan_e80_note_2026-10-25.md), before any e80 number; the checker itself is unchanged.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e80_check.py

CPU only; nothing is written in the study (each case runs a copy of the checker in a temporary folder, with a made-up
freeze record and results/explore/abcd.jsonl). It asserts:
- the rule at its bounds: STAR candidate minus reference exactly +0.5 and ABCD exactly -0.5 (means of seeds 1 and 2)
  is TAKEN; STAR +0.4 or ABCD -0.6 is NOT taken;
- the MultiWOZ line (--mwoz-s 8) is printed with no rule;
- refusals ("FAIL", exit 1): a second candidate row, a candidate row written before the plan's freeze, a candidate
  without xcrecent in its parts, another s, a reference seed 2 on other bits than seed 1, candidate seeds on
  different bits, a candidate whose bit count is not view B's plus xcrecent's, an xcrecent block not on the C4 table.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN, LATE, EARLY = "2026-10-20T20:53:40Z", "2026-10-25T14:00:00Z", "2026-10-19T16:00:00Z"
C4 = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
B = ["mem", "words", "xcust", "xrecent"]
BN = B + ["xcrecent"]
CELLS = {"star": (16000, 16.0, 7227, 3134, "e74c_star_Bc4_c16000_s16", "e78_star_B_c16000_s16", "e80_star_BX_c16000_s16"),
         "abcd": (4000, 16.0, 6684, 2973, "e74c_abcd_Bc4_c4000_s16", "e78_abcd_B_c4000_s16", "e80_abcd_BX_c4000_s16")}


def row(tag, seed, ds, parts, clauses, s, acc, bits, n_bits, when, xc=None):
    r = {"tag": tag, "seed": seed, "arm": "tm", "dataset": ds, "encoder": "none", "parts": parts, "lex_k": 1000, "acc": acc,
         "cfg": {"n_clauses": clauses, "T": clauses, "s": s, "seed": seed}, "bits_sha256": {"train": bits + "t", "dev": bits + "d"},
         "n_bits": n_bits, "time": when, "static_info": {}}
    if xc is not None:
        r["static_info"]["xcrecent"] = {"n_bits": xc, "table_file_sha256": C4}
    return r


class Case:
    def __init__(self, star=(0.600, 0.620, 0.610, 0.620), abcd=(0.840, 0.840, 0.835, 0.835), mwoz=(0.300, 0.305)):
        """per dataset: reference seed 1, reference seed 2, candidate seed 1, candidate seed 2 (dev accuracies)."""
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "scripts").mkdir()
        (self.tmp / "logs").mkdir()
        (self.tmp / "results/explore").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "e80_check.py", self.tmp / "scripts" / "e80_check.py")
        (self.tmp / "logs/plan_e80_2d_recent.frozen.json").write_text(json.dumps({"frozen_utc": FROZEN}))
        self.rows = []
        for ds, accs in (("star", star), ("abcd", abcd)):
            clauses, s, nb, nx, ref1, ref2, cand = CELLS[ds]
            self.rows += [row(ref1, 1, ds, B, clauses, s, accs[0], f"{ds}B", nb, EARLY),
                          row(ref2, 2, ds, B, clauses, s, accs[1], f"{ds}B", nb, EARLY),
                          row(cand, 1, ds, BN, clauses, s, accs[2], f"{ds}BN", nb + nx, LATE, nx),
                          row(cand, 2, ds, BN, clauses, s, accs[3], f"{ds}BN", nb + nx, LATE, nx)]
        self.rows += [row("e77_mwoz_Bc4_c16000_s8", 1, "multiwoz", B, 16000, 8.0, mwoz[0], "mB", 6580, EARLY),
                      row("e80_mwoz_BX_c16000_s8", 1, "multiwoz", BN, 16000, 8.0, mwoz[1], "mBN", 9315, LATE, 2735)]

    def pick(self, tag, seed):
        return [r for r in self.rows if r["tag"] == tag and r["seed"] == seed][0]

    def run(self, *args):
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e80_check.py"), *args], capture_output=True, text=True, cwd=self.tmp)

    def done(self):
        shutil.rmtree(self.tmp)


def main() -> None:
    c = Case()
    p = c.run("--mwoz-s", "8")
    assert p.returncode == 0 and "FAIL" not in p.stdout, p.stdout + p.stderr
    assert "RULE: STAR +0.50 (needs >= +0.5), ABCD -0.50 (needs >= -0.5) -> TAKEN into the agent (then R3 and R5)" in p.stdout, p.stdout
    assert "multiwoz (reported, no rule): view B 0.3000, view B + xcustn 0.3050: +0.50 points (seed 1)" in p.stdout, p.stdout
    c.done()
    print("ok the rule at its bounds: STAR exactly +0.5 and ABCD exactly -0.5 are TAKEN; the MultiWOZ line with no rule")
    for star, abcd, why in (((0.600, 0.620, 0.610, 0.618), (0.840, 0.840, 0.835, 0.835), "STAR +0.40"),
                            ((0.600, 0.620, 0.610, 0.620), (0.840, 0.840, 0.834, 0.834), "ABCD -0.60")):
        c = Case(star=star, abcd=abcd)
        p = c.run()
        assert p.returncode == 0 and p.stdout.rstrip().endswith("NOT taken"), (why, p.stdout + p.stderr)
        c.done()
    print("ok STAR +0.40 or ABCD -0.60: NOT taken")
    cases = [
        ("e80_star_BX_c16000_s16 seed 1: exactly one row", lambda c: c.rows.append(dict(c.pick("e80_star_BX_c16000_s16", 1)))),
        ("e80_abcd_BX_c4000_s16 seed 2: written 2026-10-20T20:00:00Z, after the freeze", lambda c: c.pick("e80_abcd_BX_c4000_s16", 2).update(time="2026-10-20T20:00:00Z")),
        ("e80_star_BX_c16000_s16 seed 1: settings", lambda c: c.pick("e80_star_BX_c16000_s16", 1).update(parts=B)),
        ("e80_abcd_BX_c4000_s16 seed 2: settings", lambda c: c.pick("e80_abcd_BX_c4000_s16", 2)["cfg"].update(s=8.0)),
        ("star: the reference seed 2 has seed 1's train and dev bits", lambda c: c.pick("e78_star_B_c16000_s16", 2)["bits_sha256"].update(dev="x")),
        ("abcd: the two candidate seeds share their bits", lambda c: c.pick("e80_abcd_BX_c4000_s16", 2)["bits_sha256"].update(train="y")),
        ("star: candidate bits", lambda c: c.pick("e80_star_BX_c16000_s16", 1).update(n_bits=10000)),
        ("abcd: candidate bits", lambda c: c.pick("e80_abcd_BX_c4000_s16", 1)["static_info"]["xcrecent"].update(table_file_sha256="0" * 64)),
    ]
    for why, change in cases:
        c = Case()
        change(c)
        p = c.run()
        fails = [ln for ln in p.stdout.splitlines() if ln.startswith("FAIL ")]
        assert p.returncode == 1 and len(fails) == 1 and why in fails[0] and "RULE" not in p.stdout, (why, p.stdout + p.stderr)
        c.done()
    print(f"ok {len(cases)} refusals: a second candidate row, a candidate before the freeze, a candidate without xcrecent, another s, a reference"
          " seed 2 on other bits, candidate seeds on different bits, a wrong bit count, xcrecent not on the C4 table")
    print("PASSED")


if __name__ == "__main__":
    main()
