"""e81's driver on made-up rows with a stub queue (plan logs/plan_e81_T_screen.md; added 2026-10-21).

    OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e81_driver.py

CPU only; no GPU, no GPU lock, no data read, nothing written in the study. Each case builds a temporary folder like
tests/test_e81_pick.py does (a copy of scripts/e81_pick.py, a frozen record, the reference rows, the LR rows) and runs a
copy of logs/e81_driver.sh there. E81_QUEUE points it to a stub queue that, for each job file, appends the made-up rows
that file's jobs would write, and logs "queue done". It asserts:
- the full path: smoke, its check, the screens, the pick, the follow-ups, the final stage, in that order, with each
  checker output saved and the follow-up queue fed by the job file the pick wrote;
- no follow-ups: when no pick moves, the follow-up queue is skipped and the final stage still runs;
- a failed smoke check (the --T flag lost): the driver stops with exit 1 before any screen job runs.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_e81_pick as T  # noqa: E402

STUB = r'''#!/usr/bin/env bash
# stub queue: JOBFILE LOGFILE -> append the made-up rows of that job file, log like seq_queue.sh
jf="$1"; lf="$2"
echo "$(date -u +%FT%TZ) queue start: stub for $jf" >> "$lf"
cat "fake/$(basename "$jf").rows" >> results/explore/abcd.jsonl
echo "$(date -u +%FT%TZ) queue done" >> "$lf"
'''


def case(name: str, screen: dict, smoke_T: int) -> tuple:
    A = T.stage_a_rows(screen)
    refs = {k: v for k, v in A.items() if not k[0].startswith("e81_")}
    screens = [v for k, v in A.items() if k[0].startswith("e81_")]
    td = tempfile.mkdtemp()
    tmp = Path(td)
    T.write(tmp, refs)
    (tmp / "logs" / "e81_driver.sh").write_text((ROOT / "logs" / "e81_driver.sh").read_text())
    for f in ("jobs_e81_smoke.txt", "jobs_e81a.txt"):
        (tmp / "logs" / f).write_text("# stub job file\nnot run\n")
    (tmp / "fake").mkdir()
    (tmp / "fake" / "jobs_e81_smoke.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in T.smoke_rows(T=smoke_T).values()))
    (tmp / "fake" / "jobs_e81a.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in screens))
    (tmp / "fake" / "jobs_e81b.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in T.stage_b_rows(True).values()))
    (tmp / "stub_queue.sh").write_text(STUB)
    env = dict(os.environ, E81_ROOT=str(tmp), E81_QUEUE=f"bash {tmp / 'stub_queue.sh'}", E81_PY=sys.executable)
    p = subprocess.run(["bash", str(tmp / "logs" / "e81_driver.sh")], env=env, capture_output=True, text=True)
    return tmp, p


def main() -> None:
    tmp, p = case("full path", T.SCREEN, 50)
    out = p.stdout
    order = ["e81 smoke (3 tiny jobs)", "e81_pick --stage smoke: exit 0", "e81a: 18 screens and 5 LR runs", "e81_pick --stage screen: exit 0",
             "e81b: the follow-ups", "e81_pick --stage final: exit 0", "e81 GPU stages done"]
    pos = [out.find(s) for s in order]
    assert p.returncode == 0 and -1 not in pos and pos == sorted(pos), (pos, out, p.stderr)
    for f in ("e81_smoke_check.out", "e81_pick_screen.out", "e81_pick_final.out", "jobs_e81b.txt", "e81smoke_queue.log", "e81a_queue.log", "e81b_queue.log"):
        assert (tmp / "logs" / f).exists(), f
    final = (tmp / "logs" / "e81_pick_final.out").read_text()
    assert "B2 (refined agent, MultiWOZ)" in final and "B2 beside it" in final, final
    print("ok full path: smoke, check, screens, pick, follow-ups, final, in order; every checker output saved")
    shutil.rmtree(tmp)
    flat = {k: (0.0, 0.0) for k in T.SCREEN}
    tmp, p = case("no follow-ups", flat, 50)
    out = p.stdout
    assert p.returncode == 0 and "e81b: no follow-ups (no pick moved)" in out and "e81b: the follow-ups" not in out, (out, p.stderr)
    assert not (tmp / "logs" / "e81b_queue.log").exists() and "e81 GPU stages done" in out
    assert "B2 (refined agent, MultiWOZ)" in (tmp / "logs" / "e81_pick_final.out").read_text()
    print("ok no follow-ups: the follow-up queue is skipped and the final stage still runs")
    shutil.rmtree(tmp)
    tmp, p = case("failed smoke", T.SCREEN, 200)
    out = p.stdout
    rows = (tmp / "results" / "explore" / "abcd.jsonl").read_text()
    assert p.returncode == 1 and "smoke failed: e81 stops" in out and "e81a:" not in out, (out, p.stderr)
    assert not (tmp / "logs" / "e81a_queue.log").exists() and '"tag": "e81_' not in rows and not (tmp / "logs" / "e81_pick_screen.out").exists()
    print("ok failed smoke: the driver stops with exit 1 before any screen job")
    shutil.rmtree(tmp)
    print("PASSED")


if __name__ == "__main__":
    main()
