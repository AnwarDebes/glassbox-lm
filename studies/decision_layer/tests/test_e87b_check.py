"""scripts/e87b_check.py on made-up files (dated note logs/plan_e87_note_2026-10-25.md; added 2026-10-25, before any Part B
number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-dl/bin/python tests/test_e87b_check.py

CPU only; nothing is written in the study. Each case gets a temporary folder with copies of scripts/e87b_check.py and
scripts/e87_readability.py, a stub note with its freeze record, and, per dataset, a made-up A0 file (4 bits, 300 train
and 200 dev rows, 5 classes), the registered arm's seed1.json, the LR dev row, Part A's grid, and three seeds' rows,
run records and model files at the cell's clause count. A model's decisions come from 3-literal clauses, one per
class pattern of the first 3 bits, plus one 4-literal clause that pruning removes. A "good" seed has pruned dev
accuracy 0.90 and S1 point exactly 0.90; a "bad" seed votes wrong on one pattern (0.675); a "nocov" seed lacks one
pattern's clause (S1 point 0.675). It asserts:
- pass: every seed exactly at the bound (LR minus 2.0 points) and at 0.90 coverage: the bar met (3 of 3); the kill
  rule's comparator is the most accurate grid C at 0.90 coverage, below the pruned mean minus 0.5: "met and not killed";
  the pruned scores differ from the unpruned ones (the 4-literal clause), and the literal-count distribution;
- the bar at 2 of 3 (met) and at 1 of 3 (not met: the FAILURE_MAP line), and one row below the bound (not met);
- killed at exactly -0.5; the extension with a real L1-LR fit (C 0.003) when no grid C reaches 0.90; "not testable"
  and the extension's accuracy stop (in process, with a fake fit); the --bar-only mode (no Part A file needed);
- the dry mode (three tiny jobs, the tiny ABCD job's plain-select twin, equal array by array, and the full-size STAR
  job, the GPU-minute projection; no accuracy or coverage printed) and its refusals of a clause over 3 literals and of
  a twin that differs;
- two passes: a refusal on the last dataset leaves no number of the others printed; the warning for a clause over 3
  literals in a Part B model; which of the bar's conditions failed (the accuracy, or the receipts' length alone); a job
  that the queue started twice is noted;
- --bar-only (this pass): the kill rule "not evaluated in this pass (Part A not run)", never "not killed"; a met bar
  reaches no FAILURE_MAP line and no registered test; the cells and caveats named; B's STAR seed 1 against the dry
  run's full-size job (not available, the same, different);
- the dry mode's check of the code on e87's path since the plan's record (the note's list; a new src/dl file refused);
- refusals: no freeze record, a changed note, a row before the freeze, two rows, other settings (start half, the tmu
  budget, another negative rule), other bits, A0 off the registered arm, A0's LR off its row, no run record, a wrong
  step count, another CHUNK_ELEMS or the plain select in a B job's record, a job started before the freeze, a model file changed after its run, a row's acc off the recomputed one,
  no Part A file, an incomplete grid, an existing output file;
- CHUNK_ELEMS equals tm.py's; logs/jobs_e87b.txt (9 jobs) and logs/jobs_e87b_dry.txt (5 jobs) agree with the
  checker's cells, run records and model paths, and scripts/e87b_run.py accepts every line.
"""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT / "src"))
import e87_readability as R  # noqa: E402
import e87b_check as E  # noqa: E402

NS, F, C, NTR, NDV = 256, 4, 5, 300, 200
FROZEN, LATE, START = "2026-10-25T12:00:00Z", "2026-10-25T13:00:00Z", "2026-10-25T12:30:00Z"
CODES = {1: (0, 0, 1), 2: (0, 1, 0), 3: (0, 1, 1), 4: (1, 0, 0)}  # class -> bits 0-2; class 0 has no clause
TRAIN_S = {"multiwoz": 1000.0, "star": 1200.0, "abcd": 150.0}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dev_data():
    """200 dev rows: 20 of pattern 0 labelled 1 (always wrong, never covered), 45 of each class pattern 1-4; 10 rows
    of pattern 1 have bit 3 on (the 4-literal clause turns them to class 2 before pruning)."""
    X, y = [], []
    for _ in range(20):
        X.append((0, 0, 0, 0)), y.append(1)
    for c, code in CODES.items():
        for i in range(45):
            X.append((*code, 1 if (c == 1 and i < 10) else 0)), y.append(c)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def model(n_clauses: int, kind: str, long: bool = True):
    """ta [n_clauses, 2F], w [n_clauses, C]: clause c-1 is pattern c (3 literals), clause 4 the 4-literal clause (not in the
    dry models, which must keep the budget)."""
    ta = np.ones((n_clauses, 2 * F), dtype=np.int16)  # every literal excluded: empty clauses never fire
    w = np.zeros((n_clauses, C), dtype=np.int32)
    for c, code in CODES.items():
        if kind == "nocov" and c == 3:
            continue
        for j, bit in enumerate(code):
            ta[c - 1, j if bit else F + j] = NS + 1
        w[c - 1, 3 if (kind == "bad" and c == 4) else c] = 2
    if long:
        for j, bit in enumerate(CODES[1]):
            ta[4, j if bit else F + j] = NS + 1
        ta[4, 3] = NS + 1  # and bit 3: 4 literals
        w[4, 2] = 5
    return ta, w


def pred_with(y: np.ndarray, n_right: int) -> np.ndarray:
    out = (y + 1) % C
    out[:n_right] = y[:n_right]
    return out


class Case:
    def __init__(self, kinds=("good", "good", "good"), lr_right=184, grid=None, part_a=True, dry=False, over_dry=False, twin_differs=False,
                 dry_size=None, path_extra=False):
        self.tmp = t = Path(tempfile.mkdtemp())
        (t / "scripts").mkdir()
        (t / "logs").mkdir()
        for f in ("e87b_check.py", "e87_readability.py", "e87b_run.py"):
            shutil.copy(ROOT / "scripts" / f, t / "scripts" / f)
        # the plan's record: the dated note's 5 changed files on e87's path (stubs here), the 2 new scripts not in it,
        # and one changed file off the path; path_extra adds a new file on the path (src/dl/tm.py), which the dry check refuses
        (t / "src/dl").mkdir(parents=True)
        for f in ("abcd.py", "count_bits.py", "count_text.py", "static_text.py") + (("tm.py",) if path_extra else ()):
            (t / "src/dl" / f).write_text("# stub\n")
        (t / "scripts/zz_off_path.py").write_text("# stub\n")
        rec = {f: "0" * 64 for f in ("src/dl/abcd.py", "src/dl/count_bits.py", "src/dl/count_text.py", "src/dl/static_text.py",
                                      "scripts/e87_readability.py", "scripts/zz_off_path.py")}
        (t / "logs/plan_e87_receipts.frozen.json").write_text(json.dumps({"frozen_utc": "2026-10-23T01:55:38Z", "sources": rec}))
        self.dry_size = dry_size
        (t / "logs" / f"{E.NOTE}.md").write_text("note stub\n")
        (t / "logs" / f"{E.NOTE}.frozen.json").write_text(json.dumps({"sha256": sha(b"note stub\n"), "frozen_utc": FROZEN}))
        self.rows = []
        Xdv, ydv = dev_data()
        rng = np.random.default_rng(0)
        for ds, g in E.DS.items():
            Xtr = rng.integers(0, 2, (NTR, F)).astype(np.uint8)
            ytr = rng.integers(0, C, NTR)
            h = {"train": sha(Xtr.tobytes()), "dev": sha(Xdv.tobytes())}
            lp = pred_with(ydv, lr_right)
            (t / "results/explore/e87").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(t / f"results/explore/e87/{ds}_c4.npz", X_train=Xtr, y_train=ytr, X_dev=Xdv, y_dev=ydv,
                                names=np.array([f"b{i}" for i in range(F)]), buttons=np.array([f"k{i}" for i in range(C)]),
                                lr_W=np.zeros((F, C)), lr_b=np.zeros(C), lr_C=np.float64(0.01), lr_pred_dev=lp,
                                bits_sha256=json.dumps(h), table="{}")
            p = t / E.REG[ds] / "seed1.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"bits_sha256": h, "train_s": TRAIN_S[ds]}))
            p = t / "results/explore/dryrun" / E.LR_ROW[ds] / "seed0.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"arm": "lr", "dev_acc": float((lp == ydv).mean()), "bits_sha256": {"dev": h["dev"]}}))
            if part_a:
                gr = grid or {"0.01": (0.70, 0.95), "0.03": (0.80, 0.92), "0.1": (0.89, 0.85), "0.3": (0.90, 0.50), "1": (0.91, 0.30)}
                (t / f"results/explore/e87/readability_{ds}.json").write_text(json.dumps(
                    {"dataset": ds, "n_dev": NDV, "n_bits": F, "time": LATE,
                     "l1": {k: {"dev_acc": v[0], "receipts": {"coverage_at_k_literals": {"9": v[1]}}} for k, v in gr.items()}}))
            if dry:
                jobs = [(f"e87bdry_{ds}_L3", 1, "good", "tiny", "skip0")]
                jobs += [(E.PLAIN["tag"], 1, "bad" if twin_differs else "good", "tiny", "plain")] if ds == E.PLAIN["ds"] else []
                jobs += [(E.SIZE["tag"], 1, "good", "size", "skip0")] if ds == E.SIZE["ds"] else []
            else:
                jobs = [(g["tag"], sd, k, None, "skip0") for sd, k in zip(E.SEEDS, kinds)]
            for tag, sd, kind, dk, sel in jobs:
                self.add(ds, tag, sd, kind, dk, Xdv, ydv, h, over=over_dry and tag == f"e87bdry_{ds}_L3" and ds == "abcd", select=sel)
            if dry_size and ds == E.SIZE["ds"]:  # the dry run's full-size STAR row beside the B rows: its first evaluation
                b1 = self.pick(g["tag"], 1)
                first = dict(b1["history"][0], val_acc=b1["history"][0]["val_acc"] + (0 if dry_size == "same" else 0.01), t=9.9)
                self.rows.append({**b1, "tag": E.SIZE["tag"], "history": [first], "cfg": E.want_cfg(ds, 1, "size")})

    def add(self, ds, tag, sd, kind, dk, Xdv, ydv, h, over=False, select="skip0"):
        t = self.tmp
        cfg = E.want_cfg(ds, sd, dk)
        ta, w = model(cfg["n_clauses"], kind, long=dk is None)
        if over:
            ta[0, 3] = NS + 1  # a fourth literal in a clause of the dry model
        mp = t / "results/explore/abcd_models" / f"{tag}_seed{sd}.npz"
        mp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mp, ta=ta, w=w, cfg=json.dumps(cfg), names=np.array([f"b{i}" for i in range(F)]),
                            buttons=np.array([f"k{i}" for i in range(C)]))
        acc = R.tm_part(Xdv, ydv, ta, w, NS)["dev_acc"]
        ce = E.DRY["chunk_elems"] if dk == "tiny" else E.CHUNK_ELEMS
        info = {"e87b_run": LATE, "started": START, "knobs": {"neg_sampling": E.DS[ds]["neg"], "chunk_budget": "dl.tm_chunkbudget.enable", "select": select},
                "chunk_elems": ce, "machines": [{"n_clauses": cfg["n_clauses"], "literals": 2 * F, "chunked": True, "max_literals": 3,
                                                 "budget_mode": "exact", "budget_select": "rank",
                                                 "chunked_budget_steps": cfg["epochs"] * math.ceil(NTR / 128)}],
                "model": f"results/explore/abcd_models/{tag}_seed{sd}.npz", "model_sha256": sha(mp.read_bytes())}
        ip = t / "results/explore/e87/b_runs" / f"{tag}_seed{sd}.json"
        ip.parent.mkdir(parents=True, exist_ok=True)
        ip.write_text(json.dumps(info))
        self.rows.append({"tag": tag, "dataset": ds, "encoder": "none", "parts": E.VB, "lex_k": 1000, "seed": sd, "arm": "tm", "init": "trim",
                          "bits_sha256": dict(h), "static_info": {k: {"table_file_sha256": E.TABLE} for k in ("xcust", "xrecent")},
                          "acc": acc, "cfg": cfg, "n_bits": F, "time": LATE, "best_epoch": 2, "peak_gpu_mb": 4000.0,
                          "history": [{"epoch": 2, "val_acc": acc, "t": 1.0 if select == "skip0" else 1.5}],
                          "train_s": 120.0 if dk == "size" else 5.0})

    def pick(self, tag, sd=1):
        return [r for r in self.rows if r["tag"] == tag and r["seed"] == sd][0]

    def run(self, *args):
        (self.tmp / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in self.rows))
        return subprocess.run([sys.executable, str(self.tmp / "scripts" / "e87b_check.py"), *args], capture_output=True, text=True, cwd=self.tmp)

    def done(self):
        shutil.rmtree(self.tmp)


def lines(out: str, start: str) -> list:
    got = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(start)]
    assert got, (start, out)
    return got


ENV = ["HF_HUB_OFFLINE=1", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "RAYON_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false",
       "DL_GPU_FRAC=0.25", ".venv-dl/bin/python"]


def job_files() -> None:
    """logs/jobs_e87b.txt and logs/jobs_e87b_dry.txt agree with the checker's cells, and the wrapper accepts every line."""
    import shlex

    import e87b_run as W

    def parse(line: str) -> tuple[dict, dict, str, Path]:
        tok = shlex.split(line)
        assert tok[: len(ENV)] == ENV and tok[len(ENV)] == "scripts/e87b_run.py" and tok[-3] == ">>" and tok[-1] == "2>&1", line
        argv = tok[len(ENV) + 1 : -3]
        wargs, rest, model = W.parse(argv)
        assert rest[:4] == ["scripts/table_run.py", "--table", "c4words-S1000-v1", "scripts/abcd_explore.py"], rest
        t = rest[4:]
        flags = {t[i]: (t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else True) for i in range(len(t)) if t[i].startswith("--")}
        assert len([x for x in t if x.startswith("--")]) == len(flags), t  # no flag twice
        return vars(wargs), flags, tok[-2], model

    def expect(ds: str, tag: str, seed: int, kind):
        cfg, g = E.want_cfg(ds, seed, kind), E.DS[ds]
        return {"--encoder": "none", "--arms": "tm", "--parts": ",".join(E.VB), "--save": True, "--dataset": ds,
                "--clauses": str(cfg["n_clauses"]), "--T": str(cfg["T"]), "--epochs": str(cfg["epochs"]), "--s": f"{g['s']:g}",
                "--max-literals": str(E.BUDGET), "--init": "trim", "--seed": str(seed), "--tag": tag}

    full = [ln for ln in (ROOT / "logs/jobs_e87b.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    seen = set()
    for ln in full:
        w, f, logp, model = parse(ln)
        ds = f["--dataset"]
        tag, sd = E.DS[ds]["tag"], int(f["--seed"])
        assert f == expect(ds, tag, sd, None) and w["neg_sampling"] == E.DS[ds]["neg"] and w["chunk_elems"] is None and not w["plain_select"], (ln, f, w)
        assert str(w["run_info"]) == f"results/explore/e87/b_runs/{tag}_seed{sd}.json" and logp == f"logs/{tag}_seed{sd}.log", ln
        assert model == ROOT / "results/explore/abcd_models" / f"{tag}_seed{sd}.npz"
        seen.add((ds, sd))
    assert len(full) == 9 and seen == {(ds, sd) for ds in E.DS for sd in E.SEEDS}, seen
    dry = [ln for ln in (ROOT / "logs/jobs_e87b_dry.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    want = [(ds, f"e87bdry_{ds}_L3", "tiny") for ds in E.DS]
    want.insert([x[1] for x in want].index(E.PLAIN["twin"]) + 1, (E.PLAIN["ds"], E.PLAIN["tag"], "tiny"))
    want.append((E.SIZE["ds"], E.SIZE["tag"], "size"))
    assert len(dry) == len(want) == 5, dry
    for ln, (ds, tag, kind) in zip(dry, want):
        w, f, logp, model = parse(ln)
        assert f == expect(ds, tag, 1, kind) and w["neg_sampling"] == E.DS[ds]["neg"] and w["plain_select"] == (tag == E.PLAIN["tag"]), (ln, f, w)
        assert w["chunk_elems"] == (E.DRY["chunk_elems"] if kind == "tiny" else None), w
        assert str(w["run_info"]) == f"results/explore/e87/b_runs/{tag}_seed1.json" and logp == f"logs/{tag}_seed1.log", ln
    print("ok job files: the 9 jobs (3 datasets x seeds 1-3) and the 5 dry jobs (with the plain-select twin) agree with the checker's"
          " cells and run records; the wrapper accepts every line")


def main() -> None:
    import dl.tm as tm_mod

    assert E.CHUNK_ELEMS == tm_mod.CHUNK_ELEMS
    job_files()
    Xdv, ydv = dev_data()
    for kind, acc, s1 in (("good", 0.90, 0.90), ("bad", 0.675, 0.90), ("nocov", 0.675, 0.675)):
        ta, w = model(4000, kind)
        lens, dist = E.lit_dist(ta, NS)
        wp = w.copy()
        wp[lens > 3] = 0
        p = R.tm_part(Xdv, ydv, ta, wp, NS)
        assert (p["dev_acc"], p["receipts"]["s1_point"]) == (acc, s1), (kind, p["dev_acc"], p["receipts"]["s1_point"])
        assert dist == {"0": 3995 + (kind == "nocov"), "1": 0, "2": 0, "3": 4 - (kind == "nocov"), "over_3": 1, "max": 4}, dist
    print("ok made-up models: pruned accuracy and S1 point as designed (good 0.90 / 0.90, bad 0.675 / 0.90, nocov 0.675 / 0.675)")

    # pass: every seed exactly at the bound and at 0.90 coverage
    c = Case()
    p = c.run()
    assert p.returncode == 0, p.stdout + p.stderr
    for ds in E.DS:
        res = json.loads((c.tmp / "results/explore/e87/partb.json").read_text())["datasets"][ds]
        assert res["bar"]["met"] and res["bar"]["seeds_meeting"] == 3 and res["kill"]["verdict"] == "not killed" and res["kill"]["C"] == 0.03, res
        s = res["seeds"]["1"]
        assert s["pruned"]["dev_acc"] == 0.90 and s["unpruned"]["dev_acc"] == 0.85 and s["literals"]["over_3"] == 1, s
    assert len(lines(p.stdout, "the bar: MET (3 of 3 seeds meet both)")) == 3, p.stdout
    assert len(lines(p.stdout, "the kill rule: not killed (L1-LR at the budget: C 0.03, dev 0.8000")) == 3, p.stdout
    assert len(lines(p.stdout, "what follows: met and not killed on")) == 3 and "wrote results/explore/e87/partb.json" in p.stdout, p.stdout
    assert len(lines(p.stdout, "WARNING seed 1: 1 clauses over 3 literals (the exact budget should give none)")) == 3, p.stdout
    assert "was started twice" not in p.stdout, p.stdout
    p = c.run()
    assert p.returncode == 1 and "partb.json exists; not overwriting" in p.stderr, p.stdout + p.stderr
    c.done()
    print("ok pass: exactly at LR minus 2.0 and at 0.90 coverage on 3 of 3 seeds; comparator C 0.03 (most accurate at 0.90); met and not"
          " killed; pruned 0.90 against unpruned 0.85; the warning for a clause over 3 literals; no overwrite")

    c = Case(kinds=("good", "bad", "good"))
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "the bar: MET (2 of 3 seeds meet both)")) == 3, p.stdout + p.stderr
    c.done()
    c = Case(kinds=("good", "bad", "nocov"))
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "the bar: not met (1 of 3 seeds meet both)")) == 3, p.stdout + p.stderr
    assert ('what follows: not met: the plan\'s FAILURE_MAP line "short receipts cost the agent its accuracy on star", with the numbers;'
            " as run: accuracy held on 1 of 3 seeds; S1 coverage at least 0.90 on 2 of 3, so the cost is in the accuracy (the exact budget 3"
            " in the chunked update with the trim start") in p.stdout, p.stdout
    assert "seed 3: pruned dev 0.6750 (< bound), S1 point 0.675 (< 0.90): does not meet both" in p.stdout, p.stdout
    c.done()
    c = Case(lr_right=185)  # LR 0.925: the bound 0.905 is half a point above the pruned 0.90
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "the bar: not met (0 of 3 seeds meet both)")) == 3, p.stdout + p.stderr
    assert "as run: accuracy held on 0 of 3 seeds; S1 coverage at least 0.90 on 3 of 3, so the cost is in the accuracy" in p.stdout, p.stdout
    c.done()
    c = Case(kinds=("nocov", "nocov", "good"), lr_right=139)  # LR 0.695: the bound 0.675 holds for every seed; coverage fails on 2
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "the bar: not met (1 of 3 seeds meet both)")) == 3, p.stdout + p.stderr
    assert ("as run: accuracy held on 3 of 3 seeds; S1 coverage at least 0.90 on 1 of 3, so the cost is in the receipts' length, not the"
            " accuracy") in p.stdout, p.stdout
    c.done()
    print("ok the bar: 2 of 3 met; 1 of 3 not met (the plan's FAILURE_MAP line with the failed condition: the accuracy); one dev row below"
          " the bound on every seed: not met; coverage failing alone: the cost is in the receipts' length")

    pm = float(np.mean([0.90, 0.90, 0.675]))
    c = Case(kinds=("good", "good", "bad"), grid={"0.01": (0.70, 0.95), "0.03": (pm - 0.005, 0.90), "0.1": (0.89, 0.85), "0.3": (0.9, 0.5), "1": (0.91, 0.3)})
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "the kill rule: killed (L1-LR at the budget: C 0.03")) == 3, p.stdout + p.stderr
    assert 'what follows: killed: FAILURE_MAP gets "readable receipts are not specific to the TM on abcd"' in p.stdout, p.stdout
    c.done()
    c = Case(grid={k: (0.9, 0.5) for k in E.GRID})
    p = c.run()
    assert p.returncode == 0 and len(lines(p.stdout, "extension: L1-LR C 0.003: dev")) == 3, p.stdout + p.stderr
    res = json.loads((c.tmp / "results/explore/e87/partb.json").read_text())["datasets"]["multiwoz"]["kill"]
    assert res["added"] == [0.003] and res["C"] == 0.003 and res["verdict"] == "not killed" and "0.003" in res["extension_fits"], res
    c.done()
    print("ok the kill rule: killed at exactly -0.5; the extension fits L1-LR at C 0.003 for real when no grid C reaches 0.90 (comparator)")

    a = {"X_train": np.zeros((4, 2), np.uint8), "y_train": np.array([0, 1, 0, 1]), "X_dev": np.zeros((2, 2), np.uint8),
         "y_dev": np.array([0, 1]), "buttons": ["k0", "k1"]}
    old = (R.l1_fit, R.lr_part)
    try:
        # fake fits: (dev accuracy, coverage at 9 bits) per added C; the fake "W" is the C itself
        table = {0.003: (0.84, 0.70), 0.001: (0.83, 0.80), 0.0003: (0.82, 0.85), 0.0001: (0.81, 0.89)}
        calls = []
        R.l1_fit = lambda X, y, n, Cv: (Cv, None, {"C": Cv, "seconds": 0.0, "n_iter": 1, "converged": True})
        R.lr_part = lambda X, y, W, b: (calls.append(W), {"dev_acc": table[W][0], "bits_with_a_weight": 0,
                                                          "receipts": {"coverage_at_k_literals": {"9": table[W][1]}}})[1]
        grid = {Cv: {"dev_acc": 0.9, "cov9": 0.5} for Cv in (0.01, 0.03, 0.1, 0.3, 1.0)}
        v = E.kill(dict(grid), 0.85, a)
        assert v["verdict"] == "not testable" and v["added"] == [0.003, 0.001, 0.0003, 0.0001] and calls == v["added"], v
        calls.clear()
        v = E.kill(dict(grid), 0.90, a)  # 0.84 < 0.90 - 0.05: the extension stops after C 0.003
        assert v["verdict"] == "not testable" and v["added"] == [0.003] and calls == [0.003], v
        assert "the plan names no consequence for a met bar with a kill rule that is not testable" in E.follows("x", {"met": True}, v)
        assert E.follows("x", {"met": True}, dict(E.NOT_EVALUATED))[0].startswith('met; the plan\'s "met and not killed" is not reached in this pass')
    finally:
        R.l1_fit, R.lr_part = old
    print("ok the kill rule's extension in process: 'not testable' after all four added C; the accuracy stop after C 0.003; what follows")

    c = Case()
    c.pick(E.DS["abcd"]["tag"], 2)["cfg"].update(s=8.0)  # ABCD, the last dataset, is refused
    p = c.run()
    assert p.returncode == 1 and "e87b_abcd_C4_c4000_s16_L3 seed 2: settings differ" in p.stderr and "the bar" not in p.stdout and "dev" not in p.stdout, p.stdout + p.stderr
    c.done()
    c = Case()
    (c.tmp / "logs/e87b_queue.log").write_text("2026-10-25T13:00:00Z start (try 1): x --seed 2 --tag e87b_star_C4_c16000_s16_L3 >> y\n"
                                                "2026-10-25T13:20:00Z exit 1 (try 1)\n"
                                                "2026-10-25T13:20:01Z start (try 2): x --seed 2 --tag e87b_star_C4_c16000_s16_L3 >> y\n")
    p = c.run()
    assert p.returncode == 0 and "note (no rule): e87b_star_C4_c16000_s16_L3 seed 2 was started twice by the queue (try 2 in logs/e87b_queue.log)" in p.stdout, p.stdout
    assert json.loads((c.tmp / "results/explore/e87/partb.json").read_text())["retried"] == ["e87b_star_C4_c16000_s16_L3 seed 2"]
    c.done()
    print("ok two passes: a refusal on ABCD (the last dataset) leaves no number of MultiWOZ or STAR printed; a job the queue started"
          " twice is noted (no rule)")

    c = Case(part_a=False)
    p = c.run("--bar-only")
    assert p.returncode == 0 and "wrote results/explore/e87/partb_bar.json" in p.stdout, p.stdout + p.stderr
    assert len(lines(p.stdout, 'the kill rule (L1-LR at the same budget): not evaluated in this pass (Part A not run); never "not killed"')) == 3, p.stdout
    assert len(lines(p.stdout, "what follows: met; the plan's \"met and not killed\" is not reached in this pass: no FAILURE_MAP line and no"
                     " registered test follow")) == 3 and "not killed:" not in p.stdout and "not testable" not in p.stdout, p.stdout
    assert "open in this pass: the plan's second question" in p.stdout and "e87 Part B; the cells: the exact budget 3 in the chunked update" in p.stdout, p.stdout
    res = json.loads((c.tmp / "results/explore/e87/partb_bar.json").read_text())
    assert all(v["kill"] == {"verdict": "not evaluated in this pass", "why": "Part A not run"} for v in res["datasets"].values()), res
    assert res["cells"] == E.CELLS and res["caveats"] == E.CAVEATS, res
    p = c.run()
    assert p.returncode == 1 and "Part A (scripts/e87_readability.py) has not run for multiwoz" in p.stderr, p.stdout + p.stderr
    c.done()
    c = Case(kinds=("good", "bad", "nocov"), part_a=False)
    p = c.run("--bar-only")
    assert p.returncode == 0 and len(lines(p.stdout, "what follows: not met: the plan's FAILURE_MAP line")) == 3 and "what follows: met;" not in p.stdout, p.stdout
    c.done()
    print("ok --bar-only needs no Part A file: the kill rule 'not evaluated in this pass (Part A not run)', a met bar reaches no FAILURE_MAP"
          " line and no registered test, the cells and caveats named; a bar not met gives the plan's line; the full check refuses without Part A")
    for kind, want in ((None, "cannot be compared with the dry run's full-size job"), ("same", "repeats the dry run's full-size job at its first"
                       " evaluation (epoch 2, the same dev accuracy)"), ("diff", "WARNING: B's STAR seed 1 differs from the dry run's full-size job")):
        c = Case(part_a=False, dry_size=kind)
        p = c.run("--bar-only")
        assert p.returncode == 0 and want in p.stdout, (kind, p.stdout + p.stderr)
        c.done()
    print("ok determinism (no rule): B's STAR seed 1 against the dry run's full-size job: not available, the same, different (a warning)")

    c = Case(dry=True)
    p = c.run("--dry")
    assert p.returncode == 0 and p.stdout.rstrip().endswith("all dry checks passed"), p.stdout + p.stderr
    assert len(lines(p.stdout, "ok e87bdry_")) == 5 and "ok e87bdry_star_size_L3 seed 1 (size)" in p.stdout, p.stdout
    assert "ok e87bdry_abcd_plain_L3 seed 1: equal to e87bdry_abcd_L3 array by array (ta, w), and in acc and history" in p.stdout, p.stdout
    assert "is 2.00 times R22's seed-1 run; the 9 jobs then need about 246 GPU-minutes" in p.stdout, p.stdout
    assert "dev" not in p.stdout.replace("dev bits", "") and "S1" not in p.stdout, p.stdout  # the dry print shows no accuracy, no coverage
    assert "ok code since the plan's record: on e87's path exactly the note's 7 files; 1 files off the path changed or new (no rule)" in p.stdout, p.stdout
    assert not (c.tmp / "results/explore/e87/partb.json").exists() and not (c.tmp / "results/explore/e87/partb_bar.json").exists()
    c.done()
    c = Case(dry=True, over_dry=True)
    p = c.run("--dry")
    assert p.returncode == 1 and "e87bdry_abcd_L3 seed 1: 1 clauses over 3 literals: the exact budget did not hold" in p.stderr, p.stdout + p.stderr
    c.done()
    c = Case(dry=True, twin_differs=True)
    p = c.run("--dry")
    assert p.returncode == 1 and "the plain-select twin differs from e87bdry_abcd_L3" in p.stderr, p.stdout + p.stderr
    c.done()
    c = Case(dry=True, path_extra=True)
    p = c.run("--dry")
    assert p.returncode == 1 and "the code on e87's path since the plan's record is not the dated note's list: ['NEW src/dl/tm.py']" in p.stderr, p.stdout + p.stderr
    c.done()
    print("ok dry mode: the code on e87's path since the plan's record is the note's list (and a new src/dl file is refused); 3 tiny jobs,"
          " the plain-select twin (equal) and the full-size STAR job pass, no accuracy or coverage printed, the projection (2.00 x, 246"
          " GPU-minutes), nothing written; refused: a dry clause over 3 literals, a twin that differs")

    tag = E.DS["star"]["tag"]

    def edit_model(c):
        mp = c.tmp / "results/explore/abcd_models" / f"{tag}_seed1.npz"
        with np.load(mp) as z:
            d = {k: z[k] for k in z.files}
        d["w"] = d["w"] + 1
        np.savez_compressed(mp, **d)

    def edit_info(c, **kw):
        ip = c.tmp / "results/explore/e87/b_runs" / f"{tag}_seed1.json"
        info = json.loads(ip.read_text())
        for k, v in kw.items():
            if k == "steps":
                info["machines"][0]["chunked_budget_steps"] = v
            else:
                info[k] = v
        ip.write_text(json.dumps(info))

    def edit_json(c, rel, **kw):
        p = c.tmp / rel
        d = json.loads(p.read_text())
        d.update(kw)
        p.write_text(json.dumps(d))

    cases = [
        ("is missing: Part B is checked only after", lambda c: (c.tmp / "logs" / f"{E.NOTE}.frozen.json").unlink()),
        ("does not hash to its freeze record", lambda c: (c.tmp / "logs" / f"{E.NOTE}.md").write_text("note stub, edited\n")),
        ("written before the note's freeze", lambda c: c.pick(tag).update(time="2026-10-25T11:59:59Z")),
        ("2 rows (need exactly 1)", lambda c: c.rows.append(dict(c.pick(tag)))),
        ("settings differ from the note's", lambda c: c.pick(tag).update(init="half")),
        ("settings differ from the note's", lambda c: c.pick(tag)["cfg"].update(budget_mode="tmu")),
        ("settings differ from the note's", lambda c: c.pick(tag)["cfg"].update(neg_sampling="random")),
        ("bits differ from A0's", lambda c: c.pick(tag)["bits_sha256"].update(dev="0" * 64)),
        ("A0's bits are not the registered C4 arm's", lambda c: edit_json(c, E.REG["abcd"] + "/seed1.json", bits_sha256={"train": "1" * 64, "dev": "2" * 64})),
        ("is not its LR row's", lambda c: edit_json(c, "results/explore/dryrun/" + E.LR_ROW["star"] + "/seed0.json", dev_acc=0.5)),
        ("no run record", lambda c: (c.tmp / "results/explore/e87/b_runs" / f"{tag}_seed1.json").unlink()),
        ("0 chunked budget updates, not one per training step (120)", lambda c: edit_info(c, steps=0)),
        ("the run record is not the note's", lambda c: edit_info(c, chunk_elems=1_000_000)),
        ("the run record is not the note's (negative rule, select skip0", lambda c: edit_info(
            c, knobs={"neg_sampling": "hardest", "chunk_budget": "dl.tm_chunkbudget.enable", "select": "plain"})),
        ("the job started before the note's freeze", lambda c: edit_info(c, started="2026-10-25T11:00:00Z")),
        ("the model file is missing or not the one the run recorded", edit_model),
        ("is not the row's", lambda c: c.pick(tag).update(acc=0.5)),
        ("is not the plan's grid on A0's dev bits", lambda c: edit_json(c, "results/explore/e87/readability_abcd.json", l1={})),
        ("Part A's file was written before the note's freeze", lambda c: edit_json(c, "results/explore/e87/readability_abcd.json", time="2026-10-24T00:00:00Z")),
    ]
    for why, change in cases:
        c = Case()
        change(c)
        p = c.run()
        assert p.returncode == 1 and why in p.stderr and not (c.tmp / "results/explore/e87/partb.json").exists(), (why, p.stdout, p.stderr)
        c.done()
    print(f"ok {len(cases)} refusals: no freeze record, a changed note, a row before the freeze, two rows, start half, the tmu budget, another"
          " negative rule, other bits, A0 off the registered arm, A0's LR off its row, no run record, a wrong step count, another CHUNK_ELEMS,"
          " the plain select in a B job,"
          " a job started before the freeze, a model changed after its run, a row's acc off the recomputed one, an incomplete grid, an old Part A file")
    print("PASSED")


if __name__ == "__main__":
    main()
