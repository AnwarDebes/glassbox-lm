"""scripts/e85_pick.py on made-up rows (plan logs/plan_e85_s_under_random.md; added 2026-10-22).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e85_pick.py

CPU only; nothing written in the study. Each case gets a temporary folder with copies of scripts/e85_pick.py and
scripts/e84_check.py (its AG and BASE_CFG), made-up MultiWOZ dev labels, the freeze record, and made-up rows with preds
files. It asserts:
- screen: the refined agent's s 16 at exactly +0.5 is picked; bge-large's s 24 is picked; C4's best at +0.0 stays s 8;
  an exact tie between s 12 and s 16 goes to s 12; jobs_e85b.txt holds the 2 moved picks at 16k;
- final: a 16k pick at or above its 16k s 8 is kept (s* 16); one below it falls back (s* 8); jobs_R24.txt holds the 18
  jobs, each TM arm at its s*;
- refusals: an e85 row before the freeze, a missing preds file, other dev bits, two rows for one tag and seed, and an
  existing jobs file.
"""

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT / "tests"))
import e85_pick as E  # noqa: E402
import test_e83_pick as T  # noqa: E402

N = 200


class Case:
    def __init__(self, screen: dict, full: dict) -> None:
        """screen[agent][s] = 4k seed-1 count (of 200); full[agent] = (16k s 8 count, 16k pick count or None)."""
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("scripts", "logs", "data/multiwoz", "results/explore/e85_preds"):
            (self.tmp / d).mkdir(parents=True)
        for f in ("e85_pick.py", "e84_check.py"):
            shutil.copy(ROOT / "scripts" / f, self.tmp / "scripts" / f)
        self.y = T.labels("multiwoz")
        with gzip.open(self.tmp / "data/multiwoz/examples_dev.json.gz", "wt") as f:
            json.dump({"rows": [{"y": int(v)} for v in self.y], "buttons": T.NAMES["multiwoz"]}, f)
        (self.tmp / "logs" / "plan_e85_s_under_random.frozen.json").write_text(json.dumps({"frozen_utc": "2026-10-22T19:00:00Z"}))
        self.rows = []
        for a, by_s in screen.items():
            for s, c in by_s.items():
                self.e85(a, 4000, s, c)
            c8, cp = full[a]
            self.rows.append(self.row(a, E.AG[a]["tag"], 16000, 8, [0.3] * 20, c8 / N, "2026-10-22T12:00:00Z"))
            pick = max(E.S, key=lambda s: (by_s[s], -E.S.index(s)))
            if cp is not None:
                self.e85(a, 16000, pick, cp)

    def row(self, a, t, clauses, s, hist, acc, when):
        g = E.AG[a]
        d = {"tag": t, "dataset": "multiwoz", "encoder": g["enc"], "parts": g["parts"], "lex_k": 1000, "init": "half", "seed": 1, "arm": "tm",
             "bits_sha256": {"dev": g["dev"]}, "acc": acc, "history": [{"epoch": 2 * (k + 1), "val_acc": v} for k, v in enumerate(hist)],
             "cfg": {"n_clauses": clauses, "T": clauses, "s": float(s), "seed": 1, **E.BASE_CFG, "neg_sampling": "random"}, "time": when}
        if g["table"]:
            d["static_info"] = {k: {"table_file_sha256": g["table"]} for k in ("xcust", "xrecent")}
        return d

    def e85(self, a, clauses, s, c):
        counts = T.hist(c, c - 4)
        preds = [T.pred_with(self.y, "multiwoz", x) for x in counts]
        preds.append(preds[4].copy())
        t = E.tag(a, clauses, s)
        np.savez_compressed(self.tmp / "results/explore/e85_preds" / f"{t}_seed1.npz", preds=np.stack(preds).astype(np.int16), y_val=self.y.astype(np.int16))
        self.rows.append(self.row(a, t, clauses, s, [x / N for x in counts], c / N, "2026-10-22T20:00:00Z"))

    def run(self, stage):
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e85_pick.py"), "--stage", stage], capture_output=True, text=True, cwd=self.tmp)


SCREEN = {"refined": {8: 60, 12: 60, 16: 61, 24: 59}, "bge-large": {8: 60, 12: 61, 16: 61, 24: 62}, "C4": {8: 60, 12: 60, 16: 60, 24: 59}}


def main() -> None:
    c = Case(SCREEN, {"refined": (64, 65), "bge-large": (66, 65), "C4": (63, None)})
    p = c.run("screen")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "refined" in out and "pick s 16" in out.split("refined")[1].split("\n")[0], out
    assert "pick s 24" in out.split("bge-large")[1].split("\n")[0] and "pick s 8" in out.split("C4 ")[1].split("\n")[0], out
    jobs = (c.tmp / "logs" / "jobs_e85b.txt").read_text().splitlines()[1:]
    assert jobs == [E.job("refined", 16000, 16), E.job("bge-large", 16000, 24)], jobs
    p = c.run("screen")
    assert p.returncode == 1 and "exists; not overwriting" in p.stdout + p.stderr
    p = c.run("final")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "s*: refined 16, bge-large 8, C4 8" in out, out
    jobs = (c.tmp / "logs" / "jobs_R24.txt").read_text().splitlines()[1:]
    assert jobs == E.r24_jobs({"refined": 16, "bge-large": 8, "C4": 8}) and len(jobs) == 18, jobs
    assert sum("--s 16 --name tm_mem_ref_nrand" in j for j in jobs) == 5 and sum("--s 8 --name tm_mem_bge_nrand" in j for j in jobs) == 5
    shutil.rmtree(c.tmp)
    print("ok screen and final: +0.5 picked, +0.0 stays s 8; a pick at or above its 16k s 8 kept, one below falls back; "
          "jobs_e85b.txt (2) and jobs_R24.txt (18) written; no overwrite")
    tie = {**SCREEN, "C4": {8: 60, 12: 61, 16: 61, 24: 59}}
    c = Case(tie, {"refined": (64, 65), "bge-large": (66, 65), "C4": (63, 64)})
    p = c.run("screen")
    assert "pick s 12" in (p.stdout + p.stderr).split("C4 ")[1].split("\n")[0], p.stdout + p.stderr
    shutil.rmtree(c.tmp)
    print("ok an exact tie between s 12 and s 16 goes to s 12")
    for k, why in ((0, "written before the freeze"), (1, "no preds file"), (2, "dev bits"), (3, "2 rows (need exactly 1)")):
        c = Case(SCREEN, {"refined": (64, 65), "bge-large": (66, 65), "C4": (63, None)})
        r = [d for d in c.rows if d["tag"] == E.tag("C4", 4000, 12)][0]
        if k == 0:
            r["time"] = "2026-10-22T18:00:00Z"
        elif k == 1:
            (c.tmp / "results/explore/e85_preds" / f"{E.tag('C4', 4000, 12)}_seed1.npz").unlink()
        elif k == 2:
            r["bits_sha256"]["dev"] = "0" * 64
        else:
            c.rows.append(dict(r))
        p = c.run("screen")
        assert p.returncode == 1 and why in p.stdout + p.stderr, (k, p.stdout, p.stderr)
        shutil.rmtree(c.tmp)
    print("ok refusals: before the freeze, no preds file, other dev bits, two rows")
    print("PASSED")


if __name__ == "__main__":
    main()
