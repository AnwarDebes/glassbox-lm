"""e81x's report on made-up rows (dated note logs/plan_e81_note2_2026-10-21.md; added 2026-10-21).

    OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e81x_report.py

CPU only; reads no data and writes nothing in the study. Each case builds a temporary folder like
tests/test_e81_pick.py does, adds the note's frozen record, the old edge LR rows (C 0.001) and the six e81x LR rows,
and runs a copy of scripts/e81x_report.py there. It asserts:
- the per-C lines, the best C with ties to the larger C, and "inside the extended grid" against "again at the grid's
  edge";
- the reproduction line for the C 1e-3 rerun (reproduced and not reproduced);
- beside B2: the refined TM at e81's pick, the same value e81_pick.py's B2 line uses (checked here from the rows),
  minus the best LR and minus the old LR; for C4, the TM at T = clauses minus the best LR;
- the refusals: an e81x row written before the note's freeze, and an e81x row on other bits.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(1, str(ROOT / "scripts"))
import e81_pick as P  # noqa: E402
import test_e81_pick as T  # noqa: E402

CS = [("C1e-3", 0.001), ("C3e-4", 0.0003), ("C1e-4", 0.0001)]
OLD = {"a3B": ("dry_e82mwoz_lr_mem_a3B", P.B2_LR), "C4": ("dry_e77mwoz_lr_mem_c4", T.LR_ACC["multiwoz"])}


def build(acc: dict, time: str = "2026-10-21T08:00:00Z", bits_bad: str = "") -> tuple:
    rows = {**T.stage_a_rows(), **T.stage_b_rows(True)}
    tmp = Path(tempfile.mkdtemp())
    T.write(tmp, rows)
    shutil.copy(ROOT / "scripts" / "e81x_report.py", tmp / "scripts" / "e81x_report.py")
    (tmp / "logs" / "plan_e81_note2_2026-10-21.frozen.json").write_text(json.dumps({"frozen_utc": "2026-10-21T06:30:00Z"}))
    for agent, (name, a) in OLD.items():
        p = tmp / "results" / "explore" / "dryrun" / name / "seed0.json"
        p.write_text(json.dumps({"arm": "lr", "dev_acc": a, "C": 0.001, "bits_sha256": {"dev": f"bits-{agent}-multiwoz"}, "grid_c": P.GRID_C}))
    for agent in OLD:
        for cname, c in CS:
            p = tmp / "results" / "explore" / "dryrun" / f"dry_e81x_mwoz_lr_{agent}_{cname}"
            p.mkdir(parents=True)
            bits = "bits-other" if (agent, cname) == (bits_bad, "C3e-4") else f"bits-{agent}-multiwoz"
            (p / "seed0.json").write_text(json.dumps({"arm": "lr", "dev_acc": acc[agent][c], "C": c, "grid_c": [c], "time": time,
                                                      "bits_sha256": {"dev": bits}}))
    p = subprocess.run([sys.executable, "-I", str(tmp / "scripts" / "e81x_report.py")], cwd=tmp, capture_output=True, text=True)
    return rows, tmp, p


def main() -> None:
    acc = {"a3B": {0.001: P.B2_LR, 0.0003: 0.3700, 0.0001: 0.3690}, "C4": {0.001: T.LR_ACC["multiwoz"], 0.0003: T.LR_ACC["multiwoz"], 0.0001: 0.355}}
    rows, tmp, p = build(acc)
    assert p.returncode == 0, p.stderr
    out = p.stdout.splitlines()
    tm = (rows[(P.tag("multiwoz", "a3B", 16000, 8, 8000), 1)]["acc"] + rows[(P.tag("multiwoz", "a3B", 16000, 8, 8000), 2)]["acc"]) / 2
    c4 = [rows[k]["acc"] for k in P.AGENTS["C4"]["reffull"]["multiwoz"]]
    tm4 = (c4[0] + c4[1]) / 2
    want = [f"a3B  LR on MultiWOZ dev: C 1e-3 {P.B2_LR:.4f}, C 3e-4 0.3700, C 1e-4 0.3690 (C 1e-3 reproduces the old row); best C 3e-4 0.3700 (inside the extended grid)",
            f"B2 beside it (extended LR, no rule): TM at e81's pick, 16k, mean of seeds 1-2 {tm:.4f} minus the best LR {100 * (tm - 0.37):+.2f} "
            f"(against the old LR at C 1e-3: {100 * (tm - P.B2_LR):+.2f})",
            f"C4   LR on MultiWOZ dev: C 1e-3 0.3600, C 3e-4 0.3600, C 1e-4 0.3550 (C 1e-3 reproduces the old row); best C 1e-3 0.3600 (inside the extended grid)",
            f"C4 beside it (extended LR, no rule): TM at T = clauses, 16k, mean of seeds 1-2 {tm4:.4f} minus the best LR {100 * (tm4 - 0.36):+.2f} "
            f"(against the old LR at C 1e-3: {100 * (tm4 - 0.36):+.2f})"]
    assert out == want, (out, want)
    # the same refined TM value as e81_pick.py's B2 line
    q = subprocess.run([sys.executable, "-I", str(tmp / "scripts" / "e81_pick.py"), "--stage", "final"], cwd=tmp, capture_output=True, text=True)
    b2 = [ln for ln in q.stdout.splitlines() if ln.startswith("B2 (refined agent")][0]
    assert f"TM minus LR {100 * (tm - P.B2_LR):+.2f}" in b2, (b2, tm)
    print("ok: per-C lines, best with ties to the larger C, reproduction, B2 and C4 lines; the refined TM equals e81_pick's B2 value")
    shutil.rmtree(tmp)
    acc2 = {"a3B": {0.001: P.B2_LR + 0.0001, 0.0003: 0.3700, 0.0001: 0.3710}, "C4": acc["C4"]}
    rows, tmp, p = build(acc2)
    assert p.returncode == 0, p.stderr
    a = p.stdout.splitlines()[0]
    assert f"(C 1e-3 does NOT reproduce the old row ({P.B2_LR:.4f})); best C 1e-4 0.3710 (again at the grid's edge)" in a, a
    print("ok: a C 1e-3 rerun that differs is reported; a best at 1e-4 is 'again at the grid's edge'")
    shutil.rmtree(tmp)
    for name, kw, msg in (("an e81x row written before the note's freeze", {"time": "2026-10-21T06:29:59Z"}, "written after the note's freeze"),
                          ("an e81x row on other bits", {"bits_bad": "a3B"}, "on the TM rows' dev bits")):
        rows, tmp, p = build(acc, **kw)
        assert p.returncode != 0 and msg in p.stderr, (name, p.stdout, p.stderr)
        print(f"ok refusal: {name}")
        shutil.rmtree(tmp)
    print("PASSED")


if __name__ == "__main__":
    main()
