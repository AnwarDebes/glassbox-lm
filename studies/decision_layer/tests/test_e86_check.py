"""scripts/e86_check.py on made-up rows (plan logs/plan_e86_clause_drop.md; added 2026-10-23, before any e86 number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e86_check.py

CPU only; nothing written in the study. Each case gets a temporary folder with a copy of scripts/e86_check.py,
made-up dev labels for the three datasets (200 rows each, so one row is 0.5 points), both freeze records, the three
LR rows, R22's and R23's seed-1 rows, and every cell's rows with preds files. It asserts:
- dropout kept on all three datasets at exactly +0.5 (E86.1 met) and the one setting at exactly -0.5 against STAR's
  kept dropout cell (E86.3 met);
- the kept dropout cell as E86.3's comparator: +0.5 over STAR's reference but -1.5 under its kept cell fails;
- dropout kept on MultiWOZ only, and E86.3 failing on STAR;
- a gain at best-dev only (last-10 +0.0) is not kept;
- the reference's seed 1 beside R22's seed-1 row, and TM minus LR;
- the dry mode on tiny rows, and its refusal of a drop cell without dropout;
- refusals: an e86 row before e86's freeze, a drop cell without dropout, an e84 reference row with dropout, other dev
  bits, a second row for one tag and seed, a preds file off the row's acc, an LR row on other bits.
Added 2026-10-25 with the amendment (dated note logs/plan_e86_note_2026-10-25.md), before the STAR jobs ran again:
- a STAR row written before the note's freeze (the stopped queue's) is set aside and printed, and the row written
  after the note is used;
- a STAR tag and seed with only a row from before the note, or with two rows after it, is refused;
- the full check refuses when the note's freeze record is missing.
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
import e86_check as E  # noqa: E402

N, K = 200, 5
FROZEN = {E.E86: "2026-10-23T02:00:00Z", E.E84: "2026-10-22T10:37:12Z", E.NOTE: "2026-10-25T11:00:00Z"}
WHEN = {E.E86: "2026-10-23T03:00:00Z", E.E84: "2026-10-22T12:30:00Z"}
WHEN_STAR = "2026-10-25T12:00:00Z"  # the STAR jobs ran again after the note's freeze


def labels(ds: str) -> np.ndarray:
    return np.random.default_rng({"multiwoz": 1, "star": 2, "abcd": 3}[ds]).integers(0, K, N)


def pred_with(y: np.ndarray, c: int) -> np.ndarray:
    out = (y + 1) % K
    out[:c] = y[:c]
    return out


def hist(best: int, last: int, n_eval: int = 20) -> list[int]:
    h = [last - 3] * n_eval
    h[4] = best
    h[10:] = [last] * 10
    return h[:n_eval]


class Case:
    def __init__(self, counts: dict, dry: bool = False) -> None:
        """counts[(ds, cell)] = (best, last-10) correct rows of 200, the same for both seeds."""
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("scripts", "logs", "results/explore/e84_preds", "results/explore/e86_preds"):
            (self.tmp / d).mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "e86_check.py", self.tmp / "scripts" / "e86_check.py")
        for r, t in FROZEN.items():
            (self.tmp / "logs" / f"{r}.frozen.json").write_text(json.dumps({"frozen_utc": t}))
        self.rows, self.y = [], {}
        for ds, g in E.DS.items():
            self.y[ds] = y = labels(ds)
            (self.tmp / "data" / ds).mkdir(parents=True)
            with gzip.open(self.tmp / "data" / ds / "examples_dev.json.gz", "wt") as f:
                json.dump({"rows": [{"y": int(v)} for v in y]}, f)
            (self.tmp / "results/explore/dryrun" / g["lr"]).mkdir(parents=True)
            (self.tmp / "results/explore/dryrun" / g["lr"] / "seed0.json").write_text(
                json.dumps({"arm": "lr", "dev_acc": 0.31, "bits_sha256": {"dev": g["dev"]}}))
            if g["earlier"]:
                p = self.tmp / g["earlier"][1]
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps({"dev_acc": 0.295, "bits_sha256": {"dev": g["dev"]}, "cfg": {"s": g["s"], "n_clauses": g["clauses"], "seed": 1}}))
            for cell, tag, neg, drop, preds, rec in E.CELLS[ds]:
                if dry:
                    if rec == E.E86:
                        self.add(ds, f"e86dry_{ds}_{cell}", 1, neg, drop, preds, WHEN[rec], 60, 60, dry=True)
                    continue
                best, last = counts[ds, cell]
                for sd in E.SEEDS:
                    self.add(ds, tag, sd, neg, drop, preds, WHEN_STAR if (ds == "star" and rec == E.E86) else WHEN[rec], best, last)

    def add(self, ds: str, tag: str, sd: int, neg: str, drop: float, preds: str, when: str, best: int, last: int, dry: bool = False) -> None:
        g, y = E.DS[ds], self.y[ds]
        counts = [best] if dry else hist(best, last)
        P = [pred_with(y, c) for c in counts]
        P.append(P[min(4, len(P) - 1)].copy())
        np.savez_compressed(self.tmp / "results/explore" / preds / f"{tag}_seed{sd}.npz", preds=np.stack(P).astype(np.int16), y_val=y.astype(np.int16))
        n_c, T, ep = (200, 50, 2) if dry else (g["clauses"], g["clauses"], 40)
        self.rows.append({"tag": tag, "dataset": ds, "encoder": "none", "parts": E.VB, "lex_k": 1000, "init": "half", "seed": sd, "arm": "tm",
                          "bits_sha256": {"dev": g["dev"]}, "static_info": {k: {"table_file_sha256": E.TABLE} for k in ("xcust", "xrecent")},
                          "acc": best / N, "history": [{"epoch": 2 * (k + 1), "val_acc": c / N} for k, c in enumerate(counts)],
                          "cfg": {"n_clauses": n_c, "T": T, "s": g["s"], "seed": sd, **E.BASE_CFG, "epochs": ep, "neg_sampling": neg,
                                  "clause_drop": drop}, "time": when})

    def run(self, *args: str) -> subprocess.CompletedProcess:
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in self.rows))
        p = subprocess.run([sys.executable, str(self.tmp / "scripts" / "e86_check.py"), *args], capture_output=True, text=True, cwd=self.tmp)
        return p

    def done(self) -> None:
        shutil.rmtree(self.tmp)


def counts(drop: dict, one: dict) -> dict:
    """drop / one: per dataset, (best, last) of the dropout cell / the random + drop cell; references at (60, 56)."""
    out = {}
    for ds in E.DS:
        out[ds, "ref"] = (60, 56)
        out[ds, E.DROP_CELL[ds]] = drop[ds]
        if ds != "multiwoz":
            out[ds, "nrand_drop50"] = one[ds]
    return out


def line(out: str, start: str) -> str:
    got = [ln for ln in out.splitlines() if ln.strip().startswith(start)]
    assert got, (start, out)
    return got[0]


def main() -> None:
    c = Case(counts({"multiwoz": (61, 57), "star": (62, 58), "abcd": (62, 58)}, {"star": (61, 57), "abcd": (62, 58)}))
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    for ds in E.DS:
        assert line(p.stdout, f"{ds}: ").endswith("kept") and "not kept" not in line(p.stdout, f"{ds}: "), p.stdout
    assert "MET" in line(p.stdout, "E86.1") and "kept on: multiwoz, star, abcd" in p.stdout, p.stdout
    assert "(hardest + drop 0.5) best-dev -0.50, last-10 -0.50" in line(p.stdout, "E86.3 star") and "holds" in line(p.stdout, "E86.3 star"), p.stdout
    assert "(random + drop 0.5) best-dev +0.00" in line(p.stdout, "E86.3 multiwoz") and line(p.stdout, "E86.3 (").endswith("MET"), p.stdout
    assert "minus R22's seed-1 dev row (0.2950): +0.50" in p.stdout and "TM minus LR -1.00" in p.stdout, p.stdout
    c.done()
    print("ok dropout kept on all three at exactly +0.5 (E86.1 met); random + drop at exactly -0.5 against STAR's kept dropout cell (E86.3 met);"
          " R22 seed-1 reading; TM minus LR")
    c = Case(counts({"multiwoz": (61, 57), "star": (64, 60), "abcd": (60, 56)}, {"star": (61, 57), "abcd": (60, 56)}))
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert "(hardest + drop 0.5) best-dev -1.50, last-10 -1.50" in line(p.stdout, "E86.3 star") and "fails" in line(p.stdout, "E86.3 star"), p.stdout
    assert "(reference) best-dev +0.00" in line(p.stdout, "E86.3 abcd") and "not met" in line(p.stdout, "E86.3 ("), p.stdout
    c.done()
    print("ok the kept dropout cell is E86.3's comparator: random + drop +0.5 over STAR's reference but -1.5 under its kept cell fails")
    c = Case(counts({"multiwoz": (63, 59), "star": (60, 56), "abcd": (60, 56)}, {"star": (38, 34), "abcd": (59, 56)}))
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert line(p.stdout, "multiwoz: ").endswith(": kept") and line(p.stdout, "star: ").endswith("not kept"), p.stdout
    assert "not met (kept on: multiwoz)" in line(p.stdout, "E86.1"), p.stdout
    assert "fails" in line(p.stdout, "E86.3 star") and "holds" in line(p.stdout, "E86.3 abcd") and "not met" in line(p.stdout, "E86.3 ("), p.stdout
    c.done()
    print("ok kept on MultiWOZ only (E86.1 not met); E86.3 fails on STAR at -11.0")
    c = Case(counts({"multiwoz": (62, 56), "star": (62, 58), "abcd": (62, 58)}, {"star": (60, 56), "abcd": (60, 56)}))
    p = c.run()
    assert line(p.stdout, "multiwoz: ").endswith("not kept") and "not met (kept on: star, abcd)" in line(p.stdout, "E86.1"), p.stdout
    c.done()
    print("ok a gain at best-dev only (+1.0, last-10 +0.0) is not kept")
    c = Case({}, dry=True)
    p = c.run("--dry")
    assert p.returncode == 0 and p.stdout.rstrip().endswith("all dry checks passed") and p.stdout.count("\nok ") + p.stdout.startswith("ok ") == 7, p.stdout + p.stderr
    [r for r in c.rows if r["tag"] == "e86dry_star_drop50"][0]["cfg"]["clause_drop"] = 0.0
    p = c.run("--dry")
    assert p.returncode == 1 and "e86dry_star_drop50 seed 1: settings differ" in p.stderr, p.stdout + p.stderr
    c.done()
    print("ok dry mode: 7 tiny cells pass; a drop cell without dropout is refused")
    good = counts({"multiwoz": (61, 57), "star": (62, 58), "abcd": (62, 58)}, {"star": (59, 55), "abcd": (60, 56)})
    cases = [("written before its plan's freeze", lambda c: c.pick("e86_abcd_C4_c4000_s16_drop50").update(time="2026-10-23T01:00:00Z")),
             ("settings differ", lambda c: c.pick("e86_abcd_C4_c4000_s16_drop50")["cfg"].update(clause_drop=0.0)),
             ("settings differ", lambda c: c.pick("e84_multiwoz_C4_c16000_s8_nrand")["cfg"].update(clause_drop=0.5)),
             ("are not the C4 agent's on star", lambda c: c.pick("e86_star_C4_c16000_s16_ref")["bits_sha256"].update(dev="0" * 64)),
             ("2 rows (need exactly 1)", lambda c: c.rows.append(dict(c.pick("e86_abcd_C4_c4000_s16_ref")))),
             ("does not give the history or the row's acc", lambda c: c.pick("e86_multiwoz_C4_c16000_s8_nrand_drop50").update(acc=0.3)),
             ("not LR on the C4 agent's abcd dev bits", lambda c: (c.tmp / "results/explore/dryrun/dry_R21abcd_lr_mem_c4/seed0.json").write_text(
                 json.dumps({"arm": "lr", "dev_acc": 0.31, "bits_sha256": {"dev": "1" * 64}})))]
    Case.pick = lambda self, t: [r for r in self.rows if r["tag"] == t and r["seed"] == 1][0]
    for why, change in cases:
        c = Case(good)
        change(c)
        p = c.run()
        assert p.returncode == 1 and why in p.stderr, (why, p.stdout, p.stderr)
        c.done()
    print("ok refusals: before the freeze, a drop cell without dropout, an e84 reference with dropout, other dev bits, two rows, a preds file off"
          " the acc, an LR row on other bits")
    # the amendment of 2026-10-25: the stopped queue's STAR row is set aside, the row after the note is used
    c = Case(good)
    early = dict(c.pick("e86_star_C4_c16000_s16_ref"), time="2026-10-23T02:26:20Z", acc=0.29)
    c.rows.append(early)
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert ("Set aside (no rule; the queue stopped on 2026-10-23): e86_star_C4_c16000_s16_ref seed 1, written 2026-10-23T02:26:20Z, acc 0.2900;"
            " the row written after the note: 0.3000, difference +1.00") in p.stdout, p.stdout
    assert "minus R22's seed-1 dev row (0.2950): +0.50" in p.stdout, p.stdout
    c.done()
    c = Case(good)
    c.pick("e86_star_C4_c16000_s16_drop50").update(time="2026-10-23T02:26:20Z")
    p = c.run()
    assert p.returncode == 1 and "e86_star_C4_c16000_s16_drop50 seed 1: 0 rows (need exactly 1)" in p.stderr, p.stdout + p.stderr
    c.done()
    c = Case(good)
    c.rows.append(dict(c.pick("e86_star_C4_c16000_s16_nrand_drop50")))
    p = c.run()
    assert p.returncode == 1 and "e86_star_C4_c16000_s16_nrand_drop50 seed 1: 2 rows (need exactly 1)" in p.stderr, p.stdout + p.stderr
    c.done()
    c = Case(good)
    (c.tmp / "logs" / f"{E.NOTE}.frozen.json").unlink()
    p = c.run()
    assert p.returncode == 1 and f"{E.NOTE}.frozen.json is missing" in p.stderr, p.stdout + p.stderr
    c.done()
    print("ok amendment of 2026-10-25: the stopped queue's STAR row is set aside and printed; only-early and two-late STAR rows refused;"
          " the full check refuses without the note's freeze record")
    print("PASSED")


if __name__ == "__main__":
    main()
