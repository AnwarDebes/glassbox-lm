"""logs/e83_driver.sh on made-up rows with a stub queue (plan logs/plan_e83_learning_dynamics.md; added 2026-10-21).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e83_driver.py

CPU only; no GPU, no GPU lock, no data read, nothing written in the study. Each case builds a temporary folder with
test_e83_pick.py's made-up data, a copy of scripts/e83_pick.py and of logs/e83_driver.sh, stub job files, and e82's
reference rows. E83_QUEUE points the driver to a stub queue that, for each job file, appends that file's made-up rows.
The preds files are in place from the start. It asserts, each time with the end line
"e83 end: <branch> <UTC> exit <code>" as the last line of logs/e83_end.log:
- the done path (a pick, C1 met): smoke, e83a, the screen, e83b, mid, e83c, final, in order; end line done, exit 0;
- a failed smoke check: the driver stops, e83a does not start (end line: smoke failed, exit 1);
- no pick: e83b and e83c are skipped, and final closes with "none of the four knobs" (end line: done);
- C1 missed: e83b runs, e83c is skipped, and final says "C3: not tested" (end line: done);
- a SIGTERM while the smoke queue runs: the driver stops that stage's process group (no process left) and writes the
  end line (signal TERM, exit 143);
- an existing logs/e83_end.log: the driver refuses and runs nothing (end line: refused).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(1, str(ROOT / "scripts"))
import e83_pick as P  # noqa: E402
import test_e83_pick as T  # noqa: E402

STUB = r'''#!/usr/bin/env bash
jf="$1"; lf="$2"; b="$(basename "$jf")"
echo "$(date -u +%FT%TZ) queue start: stub for $jf" >> "$lf"
if [ -n "${STUB_SLEEP:-}" ] && [ "$b" = "jobs_e83_smoke.txt" ]; then
  sleep "$STUB_SLEEP" &
  echo $! > fake/sleep.pid
  touch fake/started
  wait $!
fi
if [ -f "fake/$b.rows" ]; then cat "fake/$b.rows" >> results/explore/abcd.jsonl; fi
echo "$(date -u +%FT%TZ) queue done" >> "$lf"
'''
END = re.compile(r"^e83 end: (.+) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) exit (\d+)$")


def build(gains: dict, full: tuple | None, smoke_bad: dict | None = None) -> Path:
    """gains: the screen; full: (pick, best, last, last2, c3) for the 16k rows (None: no 16k rows)."""
    c = T.smoke_case(**(smoke_bad or {}))
    smoke = list(c.rows)
    c.rows = []
    s = T.screen_case(gains)
    for d in s.rows:  # the screen rows and their preds files, moved into this case's folder
        c.rows.append(d)
    for f in (s.tmp / "results/explore/e83_preds").iterdir():
        shutil.copy(f, c.tmp / "results/explore/e83_preds" / f.name)
    s.done()
    screen = list(c.rows)
    c.rows = []
    if full:
        pick, best, last, last2, c3 = full
        T.full_rows(c, pick, best, last, c3=c3, last2=last2)
    else:
        for sd, acc in ((1, 0.3122706889523033), (2, 0.3092811523304797)):
            c.ref(P.REFS["multiwoz"], sd, "multiwoz", 16000, 8, acc, [0.28] * 20)
    refs = [d for d in c.rows if not d["tag"].startswith("e83")]
    b = [d for d in c.rows if d["tag"].startswith("e83_multiwoz")]
    cc = [d for d in c.rows if d["tag"].startswith(("e83_star", "e83_abcd"))]
    tmp = c.tmp
    (tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in refs))
    (tmp / "fake").mkdir()
    for name, rows in (("jobs_e83_smoke.txt", smoke), ("jobs_e83a.txt", screen), ("jobs_e83b.txt", b), ("jobs_e83c.txt", cc)):
        (tmp / "fake" / f"{name}.rows").write_text("".join(json.dumps(d) + "\n" for d in rows))
    for name in ("jobs_e83_smoke.txt", "jobs_e83a.txt"):
        (tmp / "logs" / name).write_text("# stub job file\nnot run\n")
    shutil.copy(ROOT / "logs" / "e83_driver.sh", tmp / "logs" / "e83_driver.sh")
    (tmp / "stub_queue.sh").write_text(STUB)
    return tmp


def env(tmp: Path, **extra) -> dict:
    return dict(os.environ, E83_ROOT=str(tmp), E83_QUEUE=f"bash {tmp / 'stub_queue.sh'}", E83_PY=sys.executable, **extra)


def run(tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(tmp / "logs" / "e83_driver.sh")], env=env(tmp), capture_output=True, text=True, cwd=tmp)


def end_line(tmp: Path) -> tuple:
    lines = (tmp / "logs" / "e83_end.log").read_text().splitlines()
    m = END.match(lines[-1])
    assert m, lines
    return m.group(1), int(m.group(3)), len(lines)


def main() -> None:
    tmp = build({"nrand": (1, 1)}, ("nrand", 68, 57, None, {"star": 0, "abcd": 0}))
    p = run(tmp)
    order = ["e83 start", "e83_pick --stage smoke: exit 0", "e83a: the screen", "e83_pick --stage screen: exit 0", "e83b: the pick at 16k",
             "e83_pick --stage mid: exit 0", "e83c: the pick on STAR and ABCD", "e83_pick --stage final: exit 0", "e83 GPU stages done"]
    pos = [p.stdout.find(s) for s in order]
    assert p.returncode == 0 and -1 not in pos and pos == sorted(pos), (pos, p.stdout, p.stderr)
    assert "C3 (nrand on abcd" in (tmp / "logs" / "e83_pick_final.out").read_text()
    assert end_line(tmp) == ("done", 0, 1)
    print("ok done path: smoke, e83a, screen, e83b, mid, e83c, final in order; end line 'done', exit 0")
    shutil.rmtree(tmp)

    tmp = build({"nrand": (1, 1)}, ("nrand", 68, 57, None, {"star": 0, "abcd": 0}), smoke_bad={"b8_long": True})
    p = run(tmp)
    assert p.returncode == 1 and "smoke failed: e83 stops" in p.stdout and not (tmp / "logs" / "e83a_queue.log").exists(), (p.stdout, p.stderr)
    assert end_line(tmp) == ("smoke failed", 1, 1)
    print("ok failed smoke: e83a not started; end line 'smoke failed', exit 1")
    shutil.rmtree(tmp)

    tmp = build({}, None)
    p = run(tmp)
    assert p.returncode == 0 and "e83b: no job (no pick)" in p.stdout and "e83c: no job" in p.stdout, (p.stdout, p.stderr)
    assert not (tmp / "logs" / "e83b_queue.log").exists() and not (tmp / "logs" / "e83c_queue.log").exists()
    assert "none of the four knobs lifts the reader" in (tmp / "logs" / "e83_pick_final.out").read_text()
    assert end_line(tmp) == ("done", 0, 1)
    print("ok no pick: e83b and e83c skipped; final closes with 'none of the four knobs'; end line 'done'")
    shutil.rmtree(tmp)

    tmp = build({"nrand": (1, 1)}, ("nrand", 64, 57, 56, None))
    p = run(tmp)
    assert p.returncode == 0 and (tmp / "logs" / "e83b_queue.log").exists() and not (tmp / "logs" / "e83c_queue.log").exists(), (p.stdout, p.stderr)
    assert "C3: not tested (C1 not met)" in (tmp / "logs" / "e83_pick_final.out").read_text()
    assert end_line(tmp) == ("done", 0, 1)
    print("ok C1 missed: e83b ran, e83c skipped, final says 'C3: not tested'; end line 'done'")
    shutil.rmtree(tmp)

    tmp = build({"nrand": (1, 1)}, ("nrand", 68, 57, None, {"star": 0, "abcd": 0}))
    q = subprocess.Popen(["bash", str(tmp / "logs" / "e83_driver.sh")], env=env(tmp, STUB_SLEEP="60"), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=tmp)
    t0 = time.time()
    while not (tmp / "fake" / "started").exists():
        assert time.time() - t0 < 20, "the stub job did not start"
        time.sleep(0.1)
    sleep_pid = int((tmp / "fake" / "sleep.pid").read_text())
    q.send_signal(signal.SIGTERM)
    q.wait(timeout=20)
    assert q.returncode == 143, q.returncode
    assert end_line(tmp) == ("signal TERM", 143, 1)
    t0 = time.time()
    while True:
        try:
            os.kill(sleep_pid, 0)
        except ProcessLookupError:
            break
        assert time.time() - t0 < 10, "the stub job's process is still alive"
        time.sleep(0.1)
    assert not (tmp / "logs" / "e83a_queue.log").exists()
    print("ok SIGTERM: the driver stopped its stage's process group (no process left) and wrote the end line 'signal TERM', exit 143")
    shutil.rmtree(tmp)

    tmp = build({}, None)
    (tmp / "logs" / "e83_end.log").write_text("e83 end: done 2026-10-21T00:00:00Z exit 0\n")
    p = run(tmp)
    assert p.returncode == 1 and "exists: not overwriting; e83 stops" in p.stdout and not (tmp / "logs" / "e83smoke_queue.log").exists(), p.stdout
    assert end_line(tmp) == ("refused (logs/e83_end.log exists)", 1, 2)
    print("ok existing end log: the driver refuses and runs nothing; end line 'refused (logs/e83_end.log exists)', exit 1")
    shutil.rmtree(tmp)
    print("PASSED")


if __name__ == "__main__":
    main()
