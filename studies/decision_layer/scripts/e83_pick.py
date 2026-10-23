#!/usr/bin/env python
"""e83 (plan logs/plan_e83_learning_dynamics.md): the negative-class rule, clause dropout and a literal budget on the
refined agent's TM reader, MultiWOZ dev. Checks, the pick, follow-up jobs and bars. Read only, except that it writes
job files, refusing to overwrite them. Written before any e83 number.

    e83_pick.py --stage smoke     # after jobs_e83_smoke.txt: knob-off identity, knobs on, the budget, preds files
    e83_pick.py --stage screen    # after jobs_e83a.txt: the pick and P1; writes logs/jobs_e83b.txt (no job without a pick)
    e83_pick.py --stage mid       # after jobs_e83b.txt: C1; writes logs/jobs_e83c.txt (no job unless C1 is met)
    e83_pick.py --stage final     # after jobs_e83c.txt: the screen, P1, C1, C2, C3 and the per-label table

Rules (fixed in the plan):
- the screen: the five arms of ARMS at 4,000 clauses, T 4,000, s 8, seed 2. An arm qualifies with at least +0.5 points
  over REF at best-dev AND at least +0.5 on the last-10 mean. The pick is the qualifier with the largest last-10 gain,
  and ties go to the earlier arm;
- C1: the pick at 16k (T 16,000), seeds 1-2, against e82_multiwoz_Bt_c16000_s8 seeds 1-2: the 2-seed mean is at least
  +0.5 at best-dev AND at least +0.5 on the last-10 mean;
- C2: the pick's 16k best-dev mean minus LR (0.3643158037776872) is at least half of today's gap (-2.677);
- C3 (only if C1 is met): on STAR (16k, s 16) and ABCD (4k, s 16), seeds 1-2, the pick's 2-seed best-dev mean minus
  the e82 Stage A reference's is at least -0.3 on each;
- P1: under nrand or nmix50 (each alone), the screen's pooled dev accuracy on the 6 labels of PAIRS is at least +2.0
  points over REF.
Checks on every row it reads:
- the settings above, and the other TM settings (BASE_CFG plus the arm's knob);
- lex_k 1000, the start state, the a3B table, and the dev bits of DEV_BITS;
- e83 rows written after the plan's freeze, exactly one row per tag and seed;
- each e83 row's preds file (scripts/knob_run.py): its dev labels are the dataset's, its evaluation counts equal the
  history, and its final prediction gives the row's acc exactly;
- the LR row: same bits, the wide grid, C 0.001, written before the freeze.
Tested by tests/test_e83_pick.py (made-up rows in a temporary folder).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A3B_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"
DEV_BITS = {"star": "5d3cd86eb18788b28165baaaaeb9c87e4323eae90b56be62820725ebe4566254",
            "abcd": "3222f78c87af48579d5e214eb2ebf75b1a177e633851b8417985ed6bfb44d110",
            "multiwoz": "612b8f0b75e32a30b50493eaedc1eec9d6bac6893d48075144a1f48fad2e23a3"}
S = {"star": 16, "abcd": 16, "multiwoz": 8}
FULL = {"star": 16000, "abcd": 4000, "multiwoz": 16000}
PARTS = ["mem", "words", "xcust", "xrecent"]
TH = "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 .venv-dl/bin/python"
TAIL = ("scripts/newtable_run.py --table-dir data/tables/c4words_a3B_v1 --name c4words-a3B-v1 scripts/abcd_explore.py --encoder none "
        "--arms tm --parts mem,words,xcust,xrecent --save")
# name -> (knob_run.py flags, abcd_explore.py flags, cfg changes from BASE_CFG, start state)
ARMS = {"ref": ("", "", {}, "half"),
        "nrand": ("--neg-sampling random", "", {"neg_sampling": "random"}, "half"),
        "nmix50": ("--neg-sampling mix:0.5", "", {"neg_sampling": "mix:0.5"}, "half"),
        "drop50": ("--clause-drop 0.5", "", {"clause_drop": 0.5}, "half"),
        "b8": ("", "--max-literals 8 --init trim", {"max_literals": 8}, "trim")}
# every other TM setting, as in the e82 reference rows
BASE_CFG = {"n_states": 256, "epochs": 40, "batch_size": 128, "neg_sampling": "hardest", "negation": True, "max_literals": 0, "budget_mode": "exact",
            "max_negated": 0, "budget_select": "rank", "clause_drop": 0.0, "half": True, "eval_every": 2}
REFS = {"multiwoz": "e82_multiwoz_Bt_c16000_s8", "star": "e82_star_Bt_c16000_s16", "abcd": "e82_abcd_Bt_c4000_s16"}
LR_NAME, LR_ACC = "dry_e82mwoz_lr_mem_a3B", 0.3643158037776872
TODAY = (0.3122706889523033 + 0.3092811523304797) / 2  # e82_multiwoz_Bt_c16000_s8, seeds 1-2 (checked against the rows)
GRID_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
PAIRS = [("general-bye", "general-bye+general-welcome"), ("Train-Inform", "Train-Inform+Train-OfferBook"),
         ("Booking-Book", "Booking-Book+general-reqmore")]
SMOKE = {"plain": ("", "", {}, "half"), "off": ("", "", {}, "half"), "on": ("--neg-sampling random --clause-drop 0.5", "", {"neg_sampling": "random", "clause_drop": 0.5}, "half"),
         "b8": ("", "--max-literals 8 --init trim", {"max_literals": 8}, "trim")}


def tag(ds: str, arm: str, clauses: int) -> str:
    return f"e83_{ds}_a3B_c{clauses}_s{S[ds]}_{arm}"


def preds_path(t: str, seed: int) -> str:
    return f"results/explore/e83_preds/{t}_seed{seed}.npz"


def job(ds: str, arm: str, clauses: int, seed: int) -> str:
    kflags, xflags, _, _ = ARMS[arm]
    t = tag(ds, arm, clauses)
    parts = [TH, "scripts/knob_run.py", kflags, "--preds-out", preds_path(t, seed), "--", TAIL, f"--dataset {ds} --clauses {clauses} --T {clauses}",
             f"--s {S[ds]} --seed {seed}", xflags, f"--tag {t}", f">> logs/{t}_seed{seed}.log 2>&1"]
    return " ".join(p for p in parts if p)


def epochs(d: dict) -> tuple[float, float, float]:
    v = [x["val_acc"] for x in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / len(v[-10:])


def dev_labels(ds: str) -> tuple[np.ndarray, list[str]]:
    with gzip.open(ROOT / "data" / ds / "examples_dev.json.gz", "rt") as f:
        d = json.load(f)
    with gzip.open(ROOT / "data" / ds / "examples_train.json.gz", "rt") as f:
        names = json.load(f)["buttons"]
    if d["buttons"] != names:
        raise SystemExit(f"{ds}: the dev and train label lists differ")
    return np.array([r["y"] for r in d["rows"]]), names


class Rows:
    def __init__(self) -> None:
        self.frozen = json.loads((ROOT / "logs" / "plan_e83_learning_dynamics.frozen.json").read_text())["frozen_utc"]
        self.r: dict = {}
        for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)
        self.y: dict = {}

    def one(self, t: str, seed: int) -> dict:
        got = self.r.get((t, seed), [])
        if len(got) != 1:
            raise SystemExit(f"{t} seed {seed}: {len(got)} rows (need exactly 1)")
        return got[0]

    def common(self, d: dict, t: str, seed: int, ds: str, cfg_more: dict, init: str) -> None:
        if not (d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == PARTS and d["cfg"]["seed"] == seed
                and d.get("lex_k") == 1000 and d.get("init") == init):
            raise SystemExit(f"{t} seed {seed}: settings differ")
        want = {**BASE_CFG, **cfg_more}
        if {k: v for k, v in d["cfg"].items() if k not in ("n_clauses", "T", "s", "seed", "epochs")} != {k: v for k, v in want.items() if k != "epochs"}:
            raise SystemExit(f"{t} seed {seed}: TM settings differ from the plan's")
        if d["bits_sha256"]["dev"] != DEV_BITS[ds]:
            raise SystemExit(f"{t} seed {seed}: dev bits {d['bits_sha256']['dev'][:8]} are not the refined agent's {ds} bits")
        if not all(d["static_info"][k]["table_file_sha256"] == A3B_SHA for k in ("xcust", "xrecent")):
            raise SystemExit(f"{t} seed {seed}: not on the a3B table")
        if t.startswith("e83") and d["time"] <= self.frozen:
            raise SystemExit(f"{t} seed {seed}: written before the freeze")

    def get(self, t: str, seed: int, ds: str, clauses: int, arm: str | None) -> dict:
        """An e83 row (arm given) or a reference row (arm None) at full settings (40 epochs, T = clauses, s of S)."""
        d = self.one(t, seed)
        _, _, more, init = ARMS[arm] if arm else ("", "", {}, "half")
        if not (d["cfg"]["n_clauses"] == clauses and d["cfg"]["T"] == clauses and d["cfg"]["s"] == float(S[ds]) and d["cfg"]["epochs"] == 40):
            raise SystemExit(f"{t} seed {seed}: clauses, T, s or epochs differ")
        self.common(d, t, seed, ds, more, init)
        if arm:
            self.preds(d, t, seed, ds)
        return d

    def preds(self, d: dict, t: str, seed: int, ds: str) -> np.ndarray:
        """The row's final dev predictions, after checking its preds file against the dataset and the row."""
        p = ROOT / preds_path(t, seed)
        if not p.exists():
            raise SystemExit(f"{t} seed {seed}: no preds file {p.relative_to(ROOT)}")
        if ds not in self.y:
            self.y[ds] = dev_labels(ds)
        y = self.y[ds][0]
        with np.load(p) as z:
            P, yv = z["preds"].astype(np.int64), z["y_val"].astype(np.int64)
        if yv.shape != y.shape or not np.array_equal(yv, y):
            raise SystemExit(f"{t} seed {seed}: the preds file's dev labels are not {ds} dev's")
        h, n = d["history"], len(y)
        if P.shape != (len(h) + 1, n):
            raise SystemExit(f"{t} seed {seed}: preds file holds {P.shape[0]} predictions, the row {len(h)} evaluations + 1")
        for k, e in enumerate(h):
            c = int((P[k] == y).sum())
            if c != round(e["val_acc"] * n) or abs(c / n - e["val_acc"]) > 1e-6:
                raise SystemExit(f"{t} seed {seed}: evaluation {k + 1} of the preds file ({c} right) differs from the history")
        if float((P[-1] == y).mean()) != d["acc"]:
            raise SystemExit(f"{t} seed {seed}: the final prediction does not give the row's acc")
        return P[-1]

    def lr(self) -> float:
        j = json.loads((ROOT / "results" / "explore" / "dryrun" / LR_NAME / "seed0.json").read_text())
        if (j["arm"] != "lr" or j["bits_sha256"]["dev"] != DEV_BITS["multiwoz"] or j["grid_c"] != GRID_C or j["C"] != 0.001
                or j["dev_acc"] != LR_ACC or j["time"] >= self.frozen):
            raise SystemExit(f"{LR_NAME}: not e82's LR on the same dev bits (wide grid, C 0.001, before the freeze)")
        return j["dev_acc"]


def smoke(R: Rows) -> None:
    rows = {}
    for name, (_, _, more, init) in SMOKE.items():
        t = f"e83smoke_{name}"
        d = R.one(t, 1)
        if not ((d["cfg"]["n_clauses"], d["cfg"]["T"], d["cfg"]["epochs"], d["cfg"]["s"]) == (200, 50, 2, 16.0)):
            raise SystemExit(f"{t}: needs 200 clauses, T 50, 2 epochs, s 16")
        R.common(d, t, 1, "abcd", more, init)
        if name != "plain":
            R.preds(d, t, 1, "abcd")
        rows[name] = d
    skip = ("tag", "time", "train_s", "peak_gpu_mb")

    def strip(d: dict) -> dict:  # the fields that must not move: all but the run's name and clock readings (also each evaluation's "t")
        return {k: ([{a: b for a, b in e.items() if a != "t"} for e in v] if k == "history" else v) for k, v in d.items() if k not in skip}

    same = strip(rows["plain"]) == strip(rows["off"])
    print(f"smoke knob-off identity: the row through knob_run.py with no knob {'equals' if same else 'DIFFERS FROM'} the plain row "
          f"(every field but {', '.join(skip)} and the evaluations' clock 't')")
    moved = any(strip(rows["on"])[k] != strip(rows["plain"])[k] for k in ("acc", "history", "stats"))
    print(f"smoke knobs on: cfg neg_sampling {rows['on']['cfg']['neg_sampling']}, clause_drop {rows['on']['cfg']['clause_drop']}; "
          f"{'differs from' if moved else 'SAME AS'} the plain row")
    longest = rows["b8"]["stats"]["max_literals"]
    print(f"smoke budget: cfg max_literals {rows['b8']['cfg']['max_literals']}, start {rows['b8']['init']}; longest clause {longest} literals")
    print("smoke preds files: off, on and b8 agree with ABCD dev's labels, their history and their acc")
    if not (same and moved and longest <= 8):
        raise SystemExit("smoke: FAIL")
    print("smoke: ok")


def pooled(pred: np.ndarray, y: np.ndarray, names: list[str], labels: list[str]) -> float:
    ids = [names.index(lb) for lb in labels]
    m = np.isin(y, ids)
    return float((pred[m] == y[m]).mean())


def screen(R: Rows) -> tuple[dict, str | None, dict]:
    rows = {a: R.get(tag("multiwoz", a, 4000), 2, "multiwoz", 4000, a) for a in ARMS}
    e = {a: epochs(d) for a, d in rows.items()}
    print("screen (MultiWOZ dev, 4k, T 4000, s 8, seed 2): best-dev / final epoch / last-10 mean")
    gains = {}
    for a in ARMS:
        print(f"  {a:7} {e[a][0]:.4f} / {e[a][1]:.4f} / {e[a][2]:.4f}" + ("" if a == "ref" else
              f"   against REF: best-dev {100 * (e[a][0] - e['ref'][0]):+.2f}, last-10 {100 * (e[a][2] - e['ref'][2]):+.2f}"))
        if a != "ref":
            gains[a] = (100 * (e[a][0] - e["ref"][0]), 100 * (e[a][2] - e["ref"][2]))
    qual = [a for a, (gb, gl) in gains.items() if gb >= 0.5 - 1e-9 and gl >= 0.5 - 1e-9]
    pick = max(qual, key=lambda a: (gains[a][1], -list(ARMS).index(a))) if qual else None
    print(f"qualifiers (+0.5 at best-dev and on the last-10 mean): {', '.join(qual) if qual else 'none'}; pick: {pick or 'none'}")
    y, names = R.y["multiwoz"]
    preds = {a: R.preds(d, tag("multiwoz", a, 4000), 2, "multiwoz") for a, d in rows.items()}
    six = [lb for pair in PAIRS for lb in pair]
    pool = {a: pooled(preds[a], y, names, six) for a in ARMS}
    p1 = {}
    for a in ("nrand", "nmix50"):
        g = 100 * (pool[a] - pool["ref"])
        p1[a] = g >= 2.0 - 1e-9
        print(f"P1 ({a}): pooled accuracy on the 6 pair labels {100 * pool[a]:.2f} against REF {100 * pool['ref']:.2f}: {g:+.2f} "
              f"(needs >= +2.0): {'MET' if p1[a] else 'not met'}")
    plus = [b for _, b in PAIRS]
    print("beside P1 (no rule): pooled on the three '+' labels: " + ", ".join(f"{a} {100 * pooled(preds[a], y, names, plus):.2f}" for a in ARMS))
    return {"rows": rows, "e": e, "preds": preds}, pick, p1


def label_table(S_: dict, R: Rows) -> None:
    y, names = R.y["multiwoz"]
    top = [i for i, _ in Counter(y.tolist()).most_common(20)]
    six = {names.index(lb) for pair in PAIRS for lb in pair}
    print("per gold label, screen rows (dev accuracy %; n = dev turns; * = a P1 label):")
    print("  " + " ".join(f"{a:>7}" for a in ARMS) + "      n  label")
    for i in sorted(set(top) | six, key=lambda i: -int((y == i).sum())):
        m = y == i
        print("  " + " ".join(f"{100 * float((S_['preds'][a][m] == i).mean()):7.1f}" for a in ARMS) + f" {int(m.sum()):6d}  {names[i]}" + (" *" if i in six else ""))


def follow(R: Rows, ds: str, pick: str) -> tuple[list, list]:
    c = FULL[ds]
    new = [R.get(tag(ds, pick, c), sd, ds, c, pick) for sd in (1, 2)]
    ref = [R.get(REFS[ds], sd, ds, c, None) for sd in (1, 2)]
    return new, ref


def diff(new: list, ref: list) -> list[float]:
    en, er = [epochs(d) for d in new], [epochs(d) for d in ref]
    return [100 * ((en[0][i] + en[1][i]) / 2 - (er[0][i] + er[1][i]) / 2) for i in range(3)]


def c1(R: Rows, pick: str) -> bool:
    new, ref = follow(R, "multiwoz", pick)
    d = diff(new, ref)
    met = d[0] >= 0.5 - 1e-9 and d[2] >= 0.5 - 1e-9
    print(f"C1 ({pick} at 16k on MultiWOZ, mean of seeds 1-2, against e82's 16k rows): best-dev {d[0]:+.2f}, last-10 mean {d[2]:+.2f} "
          f"(each needs >= +0.5): {'MET' if met else 'not met'}; beside it: final epoch {d[1]:+.2f}")
    return met


def write_jobs(name: str, lines: list[str], what: str) -> None:
    out = ROOT / "logs" / name
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text("\n".join([f"# {name[:-4]} (plan logs/plan_e83_learning_dynamics.md): {what}. Written by scripts/e83_pick.py. "
                              "One at a time through seq_queue.sh."] + lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(lines)} jobs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "screen", "mid", "final"])
    args = ap.parse_args()
    R = Rows()
    if args.stage == "smoke":
        smoke(R)
        return
    S_, pick, _ = screen(R)
    if args.stage == "screen":
        write_jobs("jobs_e83b.txt", [job("multiwoz", pick, 16000, sd) for sd in (1, 2)] if pick else [], "the pick at 16k on MultiWOZ")
        return
    if args.stage == "mid":
        met = c1(R, pick) if pick else False
        if not pick:
            print("C1: not tested (no pick)")
        lines = [job(ds, pick, FULL[ds], sd) for ds in ("star", "abcd") for sd in (1, 2)] if met else []
        write_jobs("jobs_e83c.txt", lines, "the pick on STAR and ABCD (C3)")
        return
    label_table(S_, R)
    lr = R.lr()
    old = [R.get(REFS["multiwoz"], sd, "multiwoz", 16000, None) for sd in (1, 2)]
    if (old[0]["acc"] + old[1]["acc"]) / 2 != TODAY:
        raise SystemExit("C2's constants differ from the rows")
    bar = 100 * (TODAY - lr) / 2
    if not pick:
        print(f"C1, C2, C3: not tested (no pick). TM minus LR at REF's 16k reference: {100 * (TODAY - lr):+.2f}")
        print("e83: none of the four knobs lifts the reader at 4k (one seed)")
        return
    met = c1(R, pick)
    new, _ = follow(R, "multiwoz", pick)
    tm = (new[0]["acc"] + new[1]["acc"]) / 2
    l10 = (epochs(new[0])[2] + epochs(new[1])[2]) / 2
    print(f"C2 ({pick}, MultiWOZ 16k): TM minus LR {100 * (tm - lr):+.2f} (today {100 * (TODAY - lr):+.2f}; needs >= {bar:+.2f}): "
          f"{'MET' if 100 * (tm - lr) >= bar - 1e-9 else 'not met'}; beside it (no rule): last-10 mean minus LR {100 * (l10 - lr):+.2f}")
    if not met:
        print("C3: not tested (C1 not met)")
        return
    for ds in ("star", "abcd"):
        n, r = follow(R, ds, pick)
        d = diff(n, r)
        print(f"C3 ({pick} on {ds}, mean of seeds 1-2): best-dev {d[0]:+.2f} (needs >= -0.3): {'MET' if d[0] >= -0.3 - 1e-9 else 'not met'}; "
              f"beside it: final epoch {d[1]:+.2f}, last-10 mean {d[2]:+.2f}")


if __name__ == "__main__":
    main()
