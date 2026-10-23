"""e81's chain (driver, then e81x) on made-up rows with a stub queue (notes logs/plan_e81_note_2026-10-21.md and
logs/plan_e81_note2_2026-10-21.md; added 2026-10-21).

    OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e81_chain.py

CPU only; no GPU, no GPU lock, no data read, nothing written in the study. Each case builds a temporary folder with
copies of scripts/e81_pick.py, scripts/e81x_report.py, logs/e81_driver.sh and logs/e81_chain.sh, the frozen records,
the reference rows and the old LR rows. E81_QUEUE points the chain and the driver to a stub queue that, for each job
file, appends that file's made-up TM rows or writes its made-up LR rows. It asserts, each time with the one end line
"e81 chain end: <branch> <UTC> exit <code>" in logs/e81_chain.log:
- the done path: the driver's stages, then the e81x jobs, then the report, then "e81x done", with the report's lines
  in logs/e81x_report.out (end line: e81x done, exit 0);
- a failed smoke check: the driver stops, e81x does not start, and the chain exits 1 (end line: driver stopped);
- a SIGTERM to the chain while the driver's first job runs: the chain stops that job's process group, writes the end
  line (signal TERM, exit 143) and leaves no process behind;
- an existing logs/e81_driver.log: the chain refuses to start and nothing runs (end line: refused).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(1, str(ROOT / "scripts"))
import e81_pick as P  # noqa: E402
import test_e81_pick as T  # noqa: E402

STUB = r'''#!/usr/bin/env bash
jf="$1"; lf="$2"; b="$(basename "$jf")"
echo "$(date -u +%FT%TZ) queue start: stub for $jf" >> "$lf"
if [ -n "${STUB_SLEEP:-}" ] && [ "$b" = "jobs_e81_smoke.txt" ]; then
  sleep "$STUB_SLEEP" &
  echo $! > fake/sleep.pid
  touch fake/started
  wait $!
fi
if [ -f "fake/$b.rows" ]; then cat "fake/$b.rows" >> results/explore/abcd.jsonl; fi
if [ -f "fake/$b.py" ]; then "$PYBIN" -I "fake/$b.py"; fi
echo "$(date -u +%FT%TZ) queue done" >> "$lf"
'''
LR_WRITER = r'''import json, pathlib
for agent, accs in {"a3B": {"C1e-3": %(a1)r, "C3e-4": 0.37, "C1e-4": 0.369}, "C4": {"C1e-3": 0.36, "C3e-4": 0.361, "C1e-4": 0.3605}}.items():
    for cname, a in accs.items():
        c = {"C1e-3": 0.001, "C3e-4": 0.0003, "C1e-4": 0.0001}[cname]
        p = pathlib.Path("results/explore/dryrun") / f"dry_e81x_mwoz_lr_{agent}_{cname}"
        p.mkdir(parents=True, exist_ok=True)
        (p / "seed0.json").write_text(json.dumps({"arm": "lr", "dev_acc": a, "C": c, "grid_c": [c], "time": "2026-10-21T09:00:00Z",
                                                  "bits_sha256": {"dev": f"bits-{agent}-multiwoz"}}))
'''
END = re.compile(r"^e81 chain end: (.+) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) exit (\d+)$")


def build(smoke_T: int = 50) -> Path:
    A = T.stage_a_rows()
    refs = {k: v for k, v in A.items() if not k[0].startswith("e81_")}
    screens = [v for k, v in A.items() if k[0].startswith("e81_")]
    tmp = Path(tempfile.mkdtemp())
    T.write(tmp, refs)
    shutil.copy(ROOT / "scripts" / "e81x_report.py", tmp / "scripts" / "e81x_report.py")
    for f in ("e81_driver.sh", "e81_chain.sh"):
        shutil.copy(ROOT / "logs" / f, tmp / "logs" / f)
    (tmp / "logs" / "plan_e81_note2_2026-10-21.frozen.json").write_text(json.dumps({"frozen_utc": "2026-10-21T06:30:00Z"}))
    for agent, name, acc in (("a3B", "dry_e82mwoz_lr_mem_a3B", P.B2_LR), ("C4", "dry_e77mwoz_lr_mem_c4", 0.36)):
        (tmp / "results" / "explore" / "dryrun" / name / "seed0.json").write_text(
            json.dumps({"arm": "lr", "dev_acc": acc, "C": 0.001, "bits_sha256": {"dev": f"bits-{agent}-multiwoz"}, "grid_c": P.GRID_C}))
    for f in ("jobs_e81_smoke.txt", "jobs_e81a.txt", "jobs_e81x_lr_lowC.txt"):
        (tmp / "logs" / f).write_text("# stub job file\nnot run\n")
    (tmp / "fake").mkdir()
    (tmp / "fake" / "jobs_e81_smoke.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in T.smoke_rows(T=smoke_T).values()))
    (tmp / "fake" / "jobs_e81a.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in screens))
    (tmp / "fake" / "jobs_e81b.txt.rows").write_text("".join(json.dumps(d) + "\n" for d in T.stage_b_rows(True).values()))
    (tmp / "fake" / "jobs_e81x_lr_lowC.txt.py").write_text(LR_WRITER % {"a1": P.B2_LR})
    (tmp / "stub_queue.sh").write_text(STUB)
    return tmp


def env(tmp: Path, **extra) -> dict:
    return dict(os.environ, E81_ROOT=str(tmp), E81_QUEUE=f"bash {tmp / 'stub_queue.sh'}", E81_PY=sys.executable, PYBIN=sys.executable, **extra)


def end_line(tmp: Path) -> tuple:
    lines = (tmp / "logs" / "e81_chain.log").read_text().splitlines()
    assert len(lines) == 1, lines
    m = END.match(lines[0])
    assert m, lines[0]
    return m.group(1), int(m.group(3))


def main() -> None:
    tmp = build()
    p = subprocess.run(["bash", str(tmp / "logs" / "e81_chain.sh")], env=env(tmp), capture_output=True, text=True, cwd=tmp)
    out = p.stdout
    order = ["chain start", "driver exit 0; its last line: e81 GPU stages done", "e81x: the 6 LR jobs", "e81x_report exit 0", "e81x done"]
    pos = [out.find(s) for s in order]
    assert p.returncode == 0 and -1 not in pos and pos == sorted(pos), (pos, out, p.stderr)
    rep = (tmp / "logs" / "e81x_report.out").read_text()
    assert "best C 3e-4 0.3700 (inside the extended grid)" in rep and "B2 beside it (extended LR" in rep, rep
    assert end_line(tmp) == ("e81x done (report exit 0)", 0)
    print("ok done path: driver, e81x jobs, report, 'e81x done'; end line 'e81x done (report exit 0)', exit 0")
    shutil.rmtree(tmp)

    tmp = build(smoke_T=200)
    p = subprocess.run(["bash", str(tmp / "logs" / "e81_chain.sh")], env=env(tmp), capture_output=True, text=True, cwd=tmp)
    assert p.returncode == 1 and "e81 did not finish: e81x not started" in p.stdout, (p.stdout, p.stderr)
    assert not (tmp / "logs" / "e81x_queue.log").exists() and not (tmp / "logs" / "e81x_report.out").exists()
    assert end_line(tmp) == ("driver stopped (e81x not started)", 1)
    print("ok failed smoke: e81x not started; end line 'driver stopped (e81x not started)', exit 1")
    shutil.rmtree(tmp)

    tmp = build()
    q = subprocess.Popen(["bash", str(tmp / "logs" / "e81_chain.sh")], env=env(tmp, STUB_SLEEP="60"), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=tmp)
    t0 = time.time()
    while not (tmp / "fake" / "started").exists():
        assert time.time() - t0 < 20, "the stub job did not start"
        time.sleep(0.1)
    sleep_pid = int((tmp / "fake" / "sleep.pid").read_text())
    q.send_signal(signal.SIGTERM)
    q.wait(timeout=20)
    assert q.returncode == 143, q.returncode
    assert end_line(tmp) == ("signal TERM", 143)
    t0 = time.time()
    while True:
        try:
            os.kill(sleep_pid, 0)
        except ProcessLookupError:
            break
        assert time.time() - t0 < 10, "the stub job's process is still alive"
        time.sleep(0.1)
    assert not (tmp / "logs" / "e81x_queue.log").exists()
    print(f"ok SIGTERM: the chain stopped its stage's process group (no process left) and wrote the end line 'signal TERM', exit 143 "
          f"(after {time.time() - t0:.1f} s)")
    shutil.rmtree(tmp)

    tmp = build()
    (tmp / "logs" / "e81_driver.log").write_text("an earlier run\n")
    p = subprocess.run(["bash", str(tmp / "logs" / "e81_chain.sh")], env=env(tmp), capture_output=True, text=True, cwd=tmp)
    assert p.returncode == 1 and "exists: not overwriting; the chain stops" in p.stdout, p.stdout
    assert (tmp / "logs" / "e81_driver.log").read_text() == "an earlier run\n" and not (tmp / "logs" / "e81smoke_queue.log").exists()
    assert end_line(tmp) == ("refused (logs/e81_driver.log exists)", 1)
    print("ok existing driver log: the chain refuses and nothing runs; end line 'refused (logs/e81_driver.log exists)', exit 1")
    shutil.rmtree(tmp)
    print("PASSED")


if __name__ == "__main__":
    main()
