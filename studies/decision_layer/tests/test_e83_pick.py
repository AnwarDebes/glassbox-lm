"""scripts/e83_pick.py on made-up rows (plan logs/plan_e83_learning_dynamics.md; added 2026-10-21).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e83_pick.py

CPU only; reads no data and writes nothing in the study. Each case gets a temporary folder with a copy of
scripts/e83_pick.py, a frozen record (12:00 UTC), made-up example files (data/<ds>/examples_{train,dev}.json.gz:
labels only), made-up TM rows, preds files of the knob_run.py format and e82's LR row. The copy runs there as a child
process. It asserts:
- smoke: passes, with another clock reading 't' in the knob-off row's evaluations; fails when the knob-off row differs
  in one field, when the knob row equals the plain row (but for 't'), when the budget row has a clause of 9 literals,
  or when a preds file's labels are not ABCD dev's;
- screen: the pick at exactly +0.5 on both statistics; no pick at +0.0 on either one (or -0.5); the larger last-10
  gain between two qualifiers, and an exact tie to the earlier arm; no qualifier writes jobs_e83b.txt with no job; P1
  met at exactly +2.0 and not met at +1.0;
- mid: C1 met (best-dev +0.92, last-10 exactly +0.5) writes the 4 jobs of jobs_e83c.txt; C1 not met (last-10 +0.25)
  writes none; no pick: "not tested";
- final: C2 met (TM 0.34 against the bar 0.33755) and not met (0.335); C3 met at 0 and not met at -1.67; with no
  pick, the closing line;
- refusals: an e83 row written before the freeze, two rows for one tag and seed, other dev bits, not the a3B table, a
  preds file whose final prediction does not give the row's acc, an LR row with another C, and an existing jobs file.
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
import e83_pick as P  # noqa: E402

N = {"multiwoz": 200, "abcd": 50, "star": 60}
NAMES = {"multiwoz": [lb for pair in P.PAIRS for lb in pair] + ["Attraction-Inform", "Hotel-Inform", "Taxi-Request", "OTHER"],
         "abcd": ["a", "b", "c", "d", "e"], "star": ["s1", "s2", "s3", "s4"]}
FROZEN = "2026-10-21T12:00:00Z"
AFTER, BEFORE = "2026-10-21T15:00:00Z", "2026-10-21T05:00:00Z"


def labels(ds: str) -> np.ndarray:
    rng = np.random.default_rng({"multiwoz": 1, "abcd": 2, "star": 3}[ds])
    if ds == "multiwoz":  # exactly 100 rows on the 6 pair labels (the first 6 names) and 100 on the other 4
        return rng.permutation(np.concatenate([rng.integers(0, 6, 100), rng.integers(6, 10, 100)]))
    return rng.integers(0, len(NAMES[ds]), N[ds])


def pair_rows(y: np.ndarray, ds: str) -> np.ndarray:
    ids = [NAMES[ds].index(lb) for pair in P.PAIRS for lb in pair] if ds == "multiwoz" else []
    return np.isin(y, ids)


def pred_with(y: np.ndarray, ds: str, c: int, p: int | None = None) -> np.ndarray:
    """A prediction with exactly c rows right; with p, exactly p of them on the 6 pair labels."""
    m = pair_rows(y, ds)
    pi, oi = np.flatnonzero(m), np.flatnonzero(~m)
    right = np.concatenate([oi, pi])[:c] if p is None else np.concatenate([pi[:p], oi[: c - p]])
    assert len(right) == c, (c, p)
    out = (y + 1) % len(NAMES[ds])
    out[right] = y[right]
    return out


def cfg(clauses: int, T: int, s: int, epochs: int, seed: int, more: dict) -> dict:
    return {"n_clauses": clauses, "T": T, "s": float(s), "seed": seed, **{**P.BASE_CFG, "epochs": epochs}, **more}


def tm_row(t: str, seed: int, ds: str, clauses: int, s: int, epochs: int, more: dict, init: str, hist: list[float], acc: float,
           time: str = AFTER, T: int | None = None) -> dict:
    return {"tag": t, "dataset": ds, "encoder": "none", "parts": P.PARTS, "lex_k": 1000, "seed": seed, "arm": "tm",
            "bits_sha256": {"train": "t", "dev": P.DEV_BITS[ds]},
            "static_info": {k: {"table_file_sha256": P.A3B_SHA} for k in ("xcust", "xrecent")},
            "acc": acc, "history": [{"epoch": 2 * (k + 1), "val_acc": v, "t": 1.0} for k, v in enumerate(hist)],
            "cfg": cfg(clauses, T if T is not None else clauses, s, epochs, seed, more), "init": init, "best_epoch": 2,
            "stats": {"max_literals": 5}, "time": time, "train_s": 1.0, "peak_gpu_mb": 1.0}


class Case:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("scripts", "logs", "results/explore/dryrun/" + P.LR_NAME, "results/explore/e83_preds"):
            (self.tmp / d).mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "scripts" / "e83_pick.py", self.tmp / "scripts" / "e83_pick.py")
        (self.tmp / "logs" / "plan_e83_learning_dynamics.frozen.json").write_text(json.dumps({"frozen_utc": FROZEN}))
        self.y = {}
        for ds in N:
            (self.tmp / "data" / ds).mkdir(parents=True)
            self.y[ds] = labels(ds)
            for split, rows in (("train", [{"y": 0}]), ("dev", [{"y": int(v)} for v in self.y[ds]])):
                with gzip.open(self.tmp / "data" / ds / f"examples_{split}.json.gz", "wt") as f:
                    json.dump({"rows": rows, "buttons": NAMES[ds]}, f)
        self.rows: list[dict] = []
        self.lr(0.001, P.LR_ACC)

    def lr(self, C: float, acc: float) -> None:
        (self.tmp / "results/explore/dryrun" / P.LR_NAME / "seed0.json").write_text(json.dumps(
            {"arm": "lr", "dev_acc": acc, "C": C, "grid_c": P.GRID_C, "time": BEFORE, "bits_sha256": {"dev": P.DEV_BITS["multiwoz"]}}))

    def e83(self, t: str, seed: int, ds: str, clauses: int, s: int, epochs: int, arm_more: dict, init: str, counts: list[int],
            p: int | None = None, time: str = AFTER, final_acc: float | None = None, y: np.ndarray | None = None) -> dict:
        """An e83 row with its preds file: counts per evaluation; the best one is the final prediction (p of it on pairs)."""
        yy, n = self.y[ds], N[ds]
        best = int(np.argmax(counts))
        preds = [pred_with(yy, ds, c, p if k == best else None) for k, c in enumerate(counts)]
        preds.append(preds[best].copy())
        np.savez_compressed(self.tmp / P.preds_path(t, seed), preds=np.stack(preds).astype(np.int16),
                            y_val=(yy if y is None else y).astype(np.int16), knobs=np.array("{}"), argv=np.array("[]"))
        d = tm_row(t, seed, ds, clauses, s, epochs, arm_more, init, [c / n for c in counts], counts[best] / n if final_acc is None else final_acc, time)
        self.rows.append(d)
        return d

    def ref(self, t: str, seed: int, ds: str, clauses: int, s: int, acc: float, hist: list[float]) -> None:
        self.rows.append(tm_row(t, seed, ds, clauses, s, 40, {}, "half", hist, acc, BEFORE))

    def run(self, stage: str) -> subprocess.CompletedProcess:
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e83_pick.py"), "--stage", stage], capture_output=True, text=True, cwd=self.tmp)

    def done(self) -> None:
        shutil.rmtree(self.tmp)


def hist(best: int, last: int, n_eval: int = 20) -> list[int]:
    """20 evaluation counts: the best at evaluation 5, the last 10 at `last` (all below `best`)."""
    h = [last - 3] * n_eval
    h[4] = best
    h[10:] = [last] * 10
    return h


def smoke_case(**bad) -> Case:
    c = Case()
    base = dict(ds="abcd", clauses=200, s=16, epochs=2)
    plain = tm_row("e83smoke_plain", 1, "abcd", 200, 16, 2, {}, "half", [30 / 50], 30 / 50, T=50)
    c.rows.append(plain)
    off = c.e83("e83smoke_off", 1, init="half", arm_more={}, counts=[30], **base)
    off["cfg"]["T"] = 50
    off["history"] = [{**e, "t": e["t"] + 0.7} for e in plain["history"]]  # another clock reading: must not count
    off["stats"], off["best_epoch"] = dict(plain["stats"]), plain["best_epoch"]
    if bad.get("off_stats"):
        off["stats"]["max_literals"] = 6
    on = c.e83("e83smoke_on", 1, init="half", arm_more={"neg_sampling": "random", "clause_drop": 0.5}, counts=[30 if bad.get("on_same") else 28], **base)
    on["cfg"]["T"] = 50
    if bad.get("on_same"):
        on["history"], on["stats"] = [{**e, "t": e["t"] + 0.3} for e in plain["history"]], dict(plain["stats"])
    b8 = c.e83("e83smoke_b8", 1, init="trim", arm_more={"max_literals": 8}, counts=[27], **base,
               y=(c.y["abcd"] + 1) % 5 if bad.get("b8_labels") else None)
    b8["cfg"]["T"] = 50
    b8["stats"] = {"max_literals": 9 if bad.get("b8_long") else 8}
    return c


def screen_case(gains: dict, p_pairs: dict | None = None) -> Case:
    """The 5 screen rows: REF best 60/200 and last-10 58/200; each arm (+best counts, +last-10 counts) over REF."""
    c = Case()
    p_pairs = p_pairs or {}
    for a in P.ARMS:
        gb, gl = gains.get(a, (0, 0))
        _, _, more, init = P.ARMS[a]
        c.e83(P.tag("multiwoz", a, 4000), 2, "multiwoz", 4000, 8, 40, more, init, hist(60 + gb, 58 + gl), p=p_pairs.get(a, 20))
    return c


def full_rows(c: Case, pick: str, best: int, last: int, c3: dict | None = None, last2: int | None = None) -> None:
    """e82's 16k references (the real accs, last-10 0.28) and the pick's 16k rows; with c3, STAR and ABCD too."""
    for sd, acc in ((1, 0.3122706889523033), (2, 0.3092811523304797)):
        c.ref(P.REFS["multiwoz"], sd, "multiwoz", 16000, 8, acc, [0.25] * 10 + [0.28] * 10)
    _, _, more, init = P.ARMS[pick]
    for sd in (1, 2):
        c.e83(P.tag("multiwoz", pick, 16000), sd, "multiwoz", 16000, 8, 40, more, init, hist(best, last2 if sd == 2 and last2 else last))
    for ds, gb in (c3 or {}).items():
        n = N[ds]
        for sd in (1, 2):
            c.ref(P.REFS[ds], sd, ds, P.FULL[ds], 16, 30 / n, [20 / n] * 10 + [25 / n] * 10)
            c.e83(P.tag(ds, pick, P.FULL[ds]), sd, ds, P.FULL[ds], 16, 40, more, init, hist(30 + gb, 25))


def expect(p: subprocess.CompletedProcess, code: int, *texts: str) -> str:
    out = p.stdout + p.stderr
    assert p.returncode == code and all(t in out for t in texts), (p.returncode, texts, out)
    return out


def main() -> None:
    c = smoke_case()
    expect(c.run("smoke"), 0, "equals the plain row", "differs from the plain row", "longest clause 8 literals", "smoke: ok")
    c.done()
    for bad, why in (("off_stats", "DIFFERS FROM"), ("on_same", "SAME AS"), ("b8_long", "longest clause 9"), ("b8_labels", "are not abcd dev's")):
        c = smoke_case(**{bad: True})
        expect(c.run("smoke"), 1, why)
        c.done()
    print("ok smoke: passes; fails on a knob-off row with one other field, a knob row equal to the plain row, a 9-literal clause, wrong preds labels")

    c = screen_case({"nrand": (1, 1)})  # +0.5 exactly on both (1 of 200)
    out = expect(c.run("screen"), 0, "qualifiers (+0.5 at best-dev and on the last-10 mean): nrand; pick: nrand", "wrote logs/jobs_e83b.txt (2 jobs)")
    jobs = (c.tmp / "logs" / "jobs_e83b.txt").read_text().splitlines()[1:]
    assert jobs == [P.job("multiwoz", "nrand", 16000, sd) for sd in (1, 2)] and "--neg-sampling random" in jobs[0] and "--clauses 16000 --T 16000" in jobs[0], jobs
    expect(c.run("screen"), 1, "exists; not overwriting")
    c.done()
    for g in ((1, 0), (0, 1), (2, -1)):
        c = screen_case({"drop50": g})
        expect(c.run("screen"), 0, "qualifiers (+0.5 at best-dev and on the last-10 mean): none; pick: none", "wrote logs/jobs_e83b.txt (0 jobs)")
        c.done()
    c = screen_case({"nrand": (2, 1), "b8": (1, 2)})
    expect(c.run("screen"), 0, "qualifiers (+0.5 at best-dev and on the last-10 mean): nrand, b8; pick: b8")
    c.done()
    c = screen_case({"nmix50": (1, 2), "drop50": (3, 2)})
    expect(c.run("screen"), 0, "pick: nmix50")
    c.done()
    print("ok screen: +0.5 on both qualifies; +0.0 on either one does not; the larger last-10 gain wins; an exact tie goes to the earlier arm; "
          "no qualifier writes no job; no overwrite")

    n_pair = int(pair_rows(labels("multiwoz"), "multiwoz").sum())
    assert n_pair == 100, n_pair
    c = screen_case({}, {"ref": 20, "nrand": 22, "nmix50": 21})  # +2.0 exactly and +1.0
    out = expect(c.run("screen"), 0)
    assert "P1 (nrand)" in out and "(needs >= +2.0): MET" in out.split("P1 (nrand)")[1].split("\n")[0], out
    assert "(needs >= +2.0): not met" in out.split("P1 (nmix50)")[1].split("\n")[0], out
    c.done()
    print(f"ok P1: met at exactly +2.0 (2 more of {n_pair} pair rows), not met at +1.0")

    c = screen_case({"nrand": (1, 1)})
    full_rows(c, "nrand", 64, 57)  # best 0.32 against 0.310777: +0.92; last-10 0.285 against 0.28: +0.5
    out = expect(c.run("mid"), 0, "C1 (nrand at 16k", "MET", "wrote logs/jobs_e83c.txt (4 jobs)")
    jobs = (c.tmp / "logs" / "jobs_e83c.txt").read_text().splitlines()[1:]
    assert jobs == [P.job(ds, "nrand", P.FULL[ds], sd) for ds in ("star", "abcd") for sd in (1, 2)], jobs
    c.done()
    c = screen_case({"nrand": (1, 1)})
    full_rows(c, "nrand", 64, 57, last2=56)  # last-10 0.285 and 0.28: +0.25 on the mean of 2 seeds
    out = c.run("mid")
    assert "not met" in out.stdout and "wrote logs/jobs_e83c.txt (0 jobs)" in out.stdout, out.stdout + out.stderr
    c.done()
    c = screen_case({})
    expect(c.run("mid"), 0, "C1: not tested (no pick)", "wrote logs/jobs_e83c.txt (0 jobs)")
    c.done()
    print("ok mid: C1 met writes the 4 STAR / ABCD jobs; C1 missed on the last-10 mean writes none; no pick: not tested")

    c = screen_case({"nrand": (1, 1)})
    full_rows(c, "nrand", 68, 57, c3={"star": 0, "abcd": 0})
    c.rows = [d for d in c.rows if not (d["tag"].startswith("e83_star") or d["tag"].startswith("e83_abcd"))]
    for ds, cnt in (("star", 29), ("abcd", 30)):  # STAR -1/60 = -1.67 (not met); ABCD 0 (met)
        _, _, more, init = P.ARMS["nrand"]
        for sd in (1, 2):
            c.e83(P.tag(ds, "nrand", P.FULL[ds]), sd, ds, P.FULL[ds], 16, 40, more, init, hist(cnt, 25))
    out = expect(c.run("final"), 0, "C2 (nrand, MultiWOZ 16k): TM minus LR -2.43", "MET", "per gold label")
    assert "C3 (nrand on star" in out and "(needs >= -0.3): not met" in out.split("C3 (nrand on star")[1].split("\n")[0], out
    assert "(needs >= -0.3): MET" in out.split("C3 (nrand on abcd")[1].split("\n")[0], out
    c.done()
    c = screen_case({"nrand": (1, 1)})
    full_rows(c, "nrand", 67, 57, c3={"star": 0, "abcd": 0})
    out = expect(c.run("final"), 0, "C2 (nrand, MultiWOZ 16k): TM minus LR -2.93")
    assert "not met" in out.split("C2 (nrand")[1].split("\n")[0], out
    c.done()
    c = screen_case({})
    for sd, acc in ((1, 0.3122706889523033), (2, 0.3092811523304797)):
        c.ref(P.REFS["multiwoz"], sd, "multiwoz", 16000, 8, acc, [0.28] * 20)
    expect(c.run("final"), 0, "C1, C2, C3: not tested (no pick)", "none of the four knobs lifts the reader at 4k (one seed)")
    c.done()
    print("ok final: C2 met at 0.34 and not met at 0.335 against the bar 0.33755; C3 met at 0 and not met at -1.67; no pick: the closing line")

    refusals = []
    c = screen_case({})
    c.rows[0]["time"] = BEFORE
    refusals.append((c, "written before the freeze"))
    c = screen_case({})
    c.rows.append(dict(c.rows[1]))
    refusals.append((c, "2 rows (need exactly 1)"))
    c = screen_case({})
    c.rows[2]["bits_sha256"] = {"train": "t", "dev": "f" * 64}
    refusals.append((c, "are not the refined agent's multiwoz bits"))
    c = screen_case({})
    c.rows[3]["static_info"]["xcust"]["table_file_sha256"] = "0" * 64
    refusals.append((c, "not on the a3B table"))
    c = screen_case({})
    c.rows[4]["acc"] = c.rows[4]["acc"] + 0.005
    refusals.append((c, "the final prediction does not give the row's acc"))
    for c, why in refusals:
        expect(c.run("screen"), 1, why)
        c.done()
    c = screen_case({})
    for sd, acc in ((1, 0.3122706889523033), (2, 0.3092811523304797)):
        c.ref(P.REFS["multiwoz"], sd, "multiwoz", 16000, 8, acc, [0.28] * 20)
    c.lr(0.003, P.LR_ACC)
    expect(c.run("final"), 1, "not e82's LR on the same dev bits")
    c.done()
    print("ok refusals: a row before the freeze, two rows for one tag and seed, other dev bits, another table, a final prediction off "
          "the row's acc, an LR row with another C (and an existing jobs file, above)")
    print("PASSED")


if __name__ == "__main__":
    main()
