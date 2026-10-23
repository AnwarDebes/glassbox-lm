"""R24's network-free arms on their exact run path, under an audit hook (registrations/R24_multiwoz_random_negatives.md;
added 2026-10-22, before the freeze).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_r24_network_free.py --selftest   # CPU, nothing written in the study
    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 \
        .venv-dl/bin/python tests/test_r24_network_free.py --run   # GPU: step 0b of logs/R24_chain.sh, before any test file

--run takes the refined and C4 TM lines of logs/jobs_R24_dry.txt (scripts/knob_run.py -> newtable_run.py / table_run.py
-> abcd_confirm.py --encoder none ... --dry-run), renames each output (--name dry_... -> nf_dry_...), and runs each in a
child process of its own with runpy. The child records every module imported (audit event "import" and sys.modules) and
every file opened ("open"). Per arm it asserts: exit 0; no sentence_transformers, model2vec or transformers module; the
only file opened under data/emb/ is multiwoz__none.npz. The files opened under data/tables/ are printed (the checker
compares the arm's table hash). --selftest checks the hook on two probe scripts: a planted 'transformers' import and a
data/emb/ open are caught, and a clean probe passes.
"""

from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("sentence_transformers", "model2vec", "transformers")


def child(report: str, argv: list[str]) -> None:
    mods, opened = set(), set()

    def hook(ev, args):
        if ev == "import":
            mods.add(args[0])
        elif ev == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.add(os.path.relpath(os.path.abspath(os.fsdecode(args[0]))))

    sys.addaudithook(hook)
    sys.argv, code = argv, 0
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    Path(report).write_text(json.dumps({"exit": code, "modules": sorted(mods | set(sys.modules)), "opened": sorted(opened)}))


def run_child(argv: list[str], cwd: Path, env: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        rep = Path(d) / "report.json"
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(rep), *argv], cwd=cwd, env=env or os.environ.copy())
        return json.loads(rep.read_text()) if rep.exists() else {"exit": p.returncode or 1, "modules": [], "opened": []}


def problems(r: dict) -> list[str]:
    out = [] if r["exit"] == 0 else [f"exit {r['exit']}"]
    out += [f"imported {m}" for m in r["modules"] if any(m == b or m.startswith(b + ".") for b in BANNED)]
    emb = sorted({p for p in r["opened"] if p.startswith("data/emb/")})
    out += [] if emb == ["data/emb/multiwoz__none.npz"] else [f"data/emb opens {emb}"]
    return out


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        (t / "fake" / "transformers").mkdir(parents=True)
        (t / "fake" / "transformers" / "__init__.py").write_text("")
        (t / "data" / "emb").mkdir(parents=True)
        (t / "data" / "emb" / "multiwoz__none.npz").write_text("x")
        (t / "data" / "emb" / "multiwoz__bge-large.npz").write_text("x")
        (t / "bad.py").write_text("import transformers\nopen('data/emb/multiwoz__bge-large.npz').close()\n")
        (t / "good.py").write_text("import json\nopen('data/emb/multiwoz__none.npz').close()\n")
        env = {**os.environ, "PYTHONPATH": str(t / "fake")}
        bad, good = problems(run_child(["bad.py"], t, env)), problems(run_child(["good.py"], t, env))
    assert "imported transformers" in bad and any("multiwoz__bge-large.npz" in p for p in bad), bad
    assert good == [], good
    print("ok the hook catches a planted transformers import and a bge-large vectors open; a clean probe passes")
    print("PASSED")


def run() -> None:
    fails = 0
    for line in (ROOT / "logs" / "jobs_R24_dry.txt").read_text().splitlines():
        if line.startswith("#") or "abcd_confirm.py" not in line or "--arm tm" not in line or "--encoder none" not in line:
            continue
        tok = shlex.split(line.split(">>")[0])
        argv = tok[[i for i, x in enumerate(tok) if x.endswith("bin/python")][0] + 1:]
        k = max(i for i, x in enumerate(argv) if x == "--name") + 1
        name, argv[k] = argv[k], "nf_" + argv[k]
        r = run_child(argv, ROOT)
        bad = problems(r)
        tables = sorted({str(Path(p).parent) for p in r["opened"] if p.startswith("data/tables/")})
        print(f"{name}: " + ("ok, no " + ", ".join(BANNED) + "; data/emb opens only multiwoz__none.npz" if not bad else "FAIL " + "; ".join(bad))
              + f"; data/tables folders opened: {tables}", flush=True)
        fails += bool(bad)
    if fails:
        sys.exit(1)
    print("network-free path checks passed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3:])
    elif sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        raise SystemExit(__doc__)
