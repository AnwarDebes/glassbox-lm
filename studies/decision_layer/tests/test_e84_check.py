"""scripts/e84_check.py on made-up rows (plan logs/plan_e84_negatives_fair.md; added 2026-10-22).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e84_check.py

CPU only; nothing written in the study. Each case gets a temporary folder with a copy of scripts/e84_check.py,
made-up MultiWOZ dev labels, both freeze records, the three LR rows, the hardest-negative reference rows and the random
rows with preds files. It asserts:
- both gates met and steady;
- G-C4 not met;
- G-ref met at best-dev only;
- refusals: an e84 row written before e84's freeze, an e84 row with hardest negatives, other dev bits, a second row
  for one tag and seed, and a preds file off the row's acc.
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
import e84_check as E  # noqa: E402
import test_e83_pick as T  # noqa: E402

N = 200


class Case:
    def __init__(self, best: dict, last: dict) -> None:
        """best / last: correct counts (of 200) per agent for the random rows (seed 1 and 2 alike)."""
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("scripts", "logs", "data/multiwoz", "results/explore/e83_preds", "results/explore/e84_preds"):
            (self.tmp / d).mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "e84_check.py", self.tmp / "scripts" / "e84_check.py")
        self.y = T.labels("multiwoz")
        with gzip.open(self.tmp / "data/multiwoz/examples_dev.json.gz", "wt") as f:
            json.dump({"rows": [{"y": int(v)} for v in self.y], "buttons": T.NAMES["multiwoz"]}, f)
        for rec, t in (("plan_e83_learning_dynamics", "2026-10-21T14:36:33Z"), ("plan_e84_negatives_fair", "2026-10-22T11:00:00Z")):
            (self.tmp / "logs" / f"{rec}.frozen.json").write_text(json.dumps({"frozen_utc": t}))
        self.rows = []
        for a, g in E.AG.items():
            (self.tmp / "results/explore/dryrun" / g["lr"]).mkdir(parents=True)
            (self.tmp / "results/explore/dryrun" / g["lr"] / "seed0.json").write_text(json.dumps({"arm": "lr", "dev_acc": 0.36, "bits_sha256": {"dev": g["dev"]}}))
            for t, sd in g["hard"]:
                self.rows.append(self.row(a, t, sd, "hardest", [0.30] * 20, 0.31, "2026-10-20T00:00:00Z"))
            for sd in (1, 2):
                counts = T.hist(best[a], last[a])
                preds = [T.pred_with(self.y, "multiwoz", c) for c in counts]
                preds.append(preds[4].copy())
                np.savez_compressed(self.tmp / "results/explore" / g["preds"] / f"{g['tag']}_seed{sd}.npz", preds=np.stack(preds).astype(np.int16),
                                    y_val=self.y.astype(np.int16))
                when = "2026-10-21T15:30:00Z" if a == "refined" else "2026-10-22T12:00:00Z"
                self.rows.append(self.row(a, g["tag"], sd, "random", [c / N for c in counts], best[a] / N, when))

    def row(self, a: str, t: str, sd: int, neg: str, hist: list, acc: float, when: str) -> dict:
        g = E.AG[a]
        d = {"tag": t, "dataset": "multiwoz", "encoder": g["enc"], "parts": g["parts"], "lex_k": 1000, "init": "half", "seed": sd, "arm": "tm",
             "bits_sha256": {"dev": g["dev"]}, "acc": acc, "history": [{"epoch": 2 * (k + 1), "val_acc": v} for k, v in enumerate(hist)],
             "cfg": {"n_clauses": 16000, "T": 16000, "s": 8.0, "seed": sd, **E.BASE_CFG, "neg_sampling": neg}, "time": when}
        if g["table"]:
            d["static_info"] = {k: {"table_file_sha256": g["table"]} for k in ("xcust", "xrecent")}
        return d

    def run(self) -> subprocess.CompletedProcess:
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e84_check.py")], capture_output=True, text=True, cwd=self.tmp)


def line(out: str, name: str) -> str:
    return [ln for ln in out.splitlines() if ln.startswith(name + " ")][0]


def main() -> None:
    c = Case({"refined": 70, "bge-large": 71, "C4": 69}, {"refined": 66, "bge-large": 67, "C4": 65})  # -0.5 and -1.0 exactly
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "MET, steady" in line(p.stdout, "G-ref") and "MET, steady" in line(p.stdout, "G-C4"), p.stdout
    assert "an R24 draft may follow" in line(p.stdout, "G-C4") and "TM minus LR" in p.stdout, p.stdout
    shutil.rmtree(c.tmp)
    print("ok both gates met and steady (G-C4 at exactly -1.0)")
    c = Case({"refined": 70, "bge-large": 71, "C4": 68}, {"refined": 66, "bge-large": 67, "C4": 66})
    p = c.run()
    assert "not met; no R24 draft" in line(p.stdout, "G-C4") and "MET, steady" in line(p.stdout, "G-ref"), p.stdout
    shutil.rmtree(c.tmp)
    print("ok G-C4 not met at -1.5")
    c = Case({"refined": 70, "bge-large": 71, "C4": 71}, {"refined": 60, "bge-large": 67, "C4": 67})
    p = c.run()
    assert "met on best-dev only; no R24 draft" in line(p.stdout, "G-ref"), p.stdout
    shutil.rmtree(c.tmp)
    print("ok G-ref met at best-dev only (last-10 -3.5): no R24 draft")
    good = ({"refined": 70, "bge-large": 71, "C4": 69}, {"refined": 66, "bge-large": 67, "C4": 65})
    for k, why in ((0, "written before its plan's freeze"), (1, "settings differ"), (2, "are not the C4 agent's"), (3, "2 rows (need exactly 1)"),
                   (4, "does not give the history or the row's acc")):
        c = Case(*good)
        r = [d for d in c.rows if d["tag"] == E.AG["C4"]["tag"]][0]
        if k == 0:
            r["time"] = "2026-10-22T10:00:00Z"
        elif k == 1:
            r["cfg"]["neg_sampling"] = "hardest"
        elif k == 2:
            r["bits_sha256"]["dev"] = "0" * 64
        elif k == 3:
            c.rows.append(dict(r))
        else:
            r["acc"] += 0.005
        p = c.run()
        assert p.returncode == 1 and why in p.stdout + p.stderr, (k, p.stdout, p.stderr)
        shutil.rmtree(c.tmp)
    print("ok refusals: before the freeze, hardest negatives on an e84 row, other dev bits, two rows, a preds file off the acc")
    print("PASSED")


if __name__ == "__main__":
    main()
