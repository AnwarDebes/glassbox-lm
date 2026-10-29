#!/usr/bin/env python
"""e87 Part B (plan logs/plan_e87_receipts.md, frozen 2026-10-23T01:55:38Z; dated note
logs/plan_e87_note_2026-10-25.md): the C4 agent trained with clauses of at most 3 literals, scored on dev with every
clause over 3 literals pruned. Read only, CPU only. Written 2026-10-25, before any Part B number. No test file is read.

    e87b_check.py --dry        # after logs/jobs_e87b_dry.txt: settings, bits, the chunked budget, model files and the
                               # scoring path on the dry jobs; no rule; writes nothing
    e87b_check.py --bar-only   # after logs/jobs_e87b.txt: the bar per dataset (A0's files, not Part A's);
                               # writes results/explore/e87/partb_bar.json; the kill rule is "not evaluated in
                               # this pass (Part A not run)"
    e87b_check.py              # after Part A as well: the bar and the kill rule per dataset;
                               # writes results/explore/e87/partb.json (not run in the finish pass of 2026-10-25,
                               # which runs A0 and Part B's bar only; kept unchanged for a later pass)

The cells (the note's): the C4 agent (table c4words_S1000_v1; view mem, words, xcust, xrecent; lex_k 1000; encoder
none) at each dataset's size, s and negative rule: MultiWOZ 16k, T 16,000, s 8, random negatives; STAR 16k, T 16,000,
s 16, hardest; ABCD 4k, T 4,000, s 16, hardest. All three: max_literals 3, budget_mode "exact" with the rank
selection, kept inside the chunked update by scripts/e87b_run.py (dl.tm_chunkbudget.enable, with select_k_skip0, the
same selection without sorting rows that keep nothing), start trim, 40 epochs, seeds 1-3, the best-dev snapshot, the
model saved.
The rules are the plan's, unchanged. Per dataset:
- the bar: on at least 2 of 3 seeds, the pruned agent's dev accuracy is at least the L2 LR's dev accuracy (A0) minus
  2.0 points, and its coverage at S1's point (a receipt of at most 3 clauses of at most 3 literals) is at least 0.90;
- the kill rule: "the TM adds nothing at this budget on <dataset>" if L1-LR at the same budget comes within 0.5
  points of the pruned agent's mean over seeds 1-3: scripts/e87_readability.py's l1_at_budget on Part A's grid,
  with its extension C 0.003, 0.001, 0.0003, 0.0001 fitted here with e87_readability.l1_fit (on A0's train bits) only
  when no grid C reaches 0.90 at 9 bits. No comparator: "not testable", never "not killed".
- For the bar's two bounds, values within 1e-12 of a bound count as meeting it (as l1_at_budget does at -0.5; its
  0.90 coverage test and its -5 stop have no tolerance).
Pruned: every clause with more than 3 literals gets weight 0 (with the exact budget there should be none; a loud
warning is printed if there is any). For the statistics a pruned clause is also emptied, so it neither fires nor
counts in clause precision; decisions and receipts are the same either way. Scored with e87_readability.tm_part:
decisions, receipts, coverage, S1's point. Reported with no rule: the unpruned scores, the coverage curve, the
literal-count distribution of the final clauses, which of the bar's two conditions failed, and any job that the queue
started twice ("start (try 2)" in its queue log).
Two passes: every check of every job of all three datasets (and Part A's files), including the recomputed accuracy,
runs first; no number is printed except in a refusal message. Then come the scores, the rules and the report.
Every outcome line names the cells as run (the dated note's amendment: the exact budget 3 in the chunked update
with the trim start, not the plan's tmu with the half start; the unbudgeted agent's s, T and negative rule, not
re-tuned for the budget), says that selection is optimistic (dev only), and is per dataset with no combined line and
no multiplicity correction. A "not met" line keeps the plan's FAILURE_MAP sentence as its label and says which
condition failed. With --bar-only, B's STAR seed 1 is also compared with the dry run's full-size STAR job, which is
its first 2 epochs: the first dev evaluation must repeat (reported with no rule).
Checks on every job: exactly one row per tag and seed, written after the note's freeze; the row's
dataset, encoder, parts, lex_k, init, cfg and table as the cell's; its train and dev bits equal A0's, and A0's equal
the registered C4 arm's (bits_sha256 in its seed1.json) and A0's own arrays; the run record of e87b_run.py (the
negative rule, the selection, CHUNK_ELEMS, the chunked budget update at every training step, started after the
freeze, the model file's SHA-256 now); the model's cfg, names and buttons equal the row's and A0's; the model's dev
accuracy, recomputed on A0's dev bits, equals the row's acc exactly. A0's L2 LR dev accuracy must equal its LR dev
row's.
The dry mode also checks the code since the plan's record (as scripts/diff_frozen.py): on e87's path it must be
exactly the dated note's list (5 changed files, 2 new); files off the path are counted, with no rule. It compares
the tiny ABCD job with its twin run with --plain-select (dl.tm.select_k unchanged): the
two models must be equal array by array, and so must the two rows' accuracy and history (no wall-clock field). It
prints no accuracy and no coverage of any dry model (the full-size STAR job is 2 epochs of a Part B cell): only the
checks, the literal counts, the training time, the peak GPU memory, and a projection of the nine jobs' GPU minutes
from the STAR job's training time and R22's seed-1 train_s (read from its seed1.json). A refusal message may show
the one number it refuses on.
Tested by tests/test_e87b_check.py (made-up A0 files, rows, run records, models and Part A grids).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e87_readability as R  # noqa: E402

NOTE = "plan_e87_note_2026-10-25"
VB = ["mem", "words", "xcust", "xrecent"]
TABLE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"  # data/tables/c4words_S1000_v1/table.npz
BUDGET, BAR_POINTS, COVER, SEEDS, EPS = 3, 2.0, 0.90, (1, 2, 3), 1e-12
CHUNK_ELEMS = 50_000_000  # dl.tm.CHUNK_ELEMS (tests/test_e87b_check.py checks the two are equal)
REG = {"multiwoz": "results/confirm/R24mwoz/tm_mem_c4_nrand", "star": "results/confirm/R22star/tm_mem_c4_s16",
       "abcd": "results/confirm/R23abcd/tm_mem_c4_s16"}  # as scripts/e87_bits.py
LR_ROW = {"multiwoz": "dry_e77mwoz_lr_mem_c4", "star": "dry_R21star_lr_mem_c4", "abcd": "dry_R21abcd_lr_mem_c4"}
DS = {"multiwoz": {"clauses": 16000, "s": 8.0, "neg": "random", "tag": "e87b_multiwoz_C4_c16000_s8_nrand_L3"},
      "star": {"clauses": 16000, "s": 16.0, "neg": "hardest", "tag": "e87b_star_C4_c16000_s16_L3"},
      "abcd": {"clauses": 4000, "s": 16.0, "neg": "hardest", "tag": "e87b_abcd_C4_c4000_s16_L3"}}
BASE_CFG = {"n_states": 256, "epochs": 40, "batch_size": 128, "negation": True, "max_literals": BUDGET, "budget_mode": "exact",
            "max_negated": 0, "budget_select": "rank", "clause_drop": 0.0, "half": True, "eval_every": 2}
DRY = {"clauses": 200, "T": 50, "epochs": 2, "chunk_elems": 1_000_000}
SIZE = {"ds": "star", "tag": "e87bdry_star_size_L3", "epochs": 2}  # the dry run's STAR job at full size (time, memory)
PLAIN = {"ds": "abcd", "tag": "e87bdry_abcd_plain_L3", "twin": "e87bdry_abcd_L3"}  # dry: the same job with --plain-select
GRID = {"0.01", "0.03", "0.1", "0.3", "1"}  # Part A's L1-LR grid, as f"{C:g}" keys
OVERHEAD = {"multiwoz": 90, "star": 90, "abcd": 40}  # s per job beyond training (gaps between registered rows: 89.3, 94.1, 41.5)
PLAN_RECORD = "logs/plan_e87_receipts.frozen.json"
PATH_RE = re.compile(r"^(src/dl/|scripts/(e87|abcd_explore|table_run|seq_queue|cpu_load|diff_frozen))")
PATH_DIFF = {"CHANGED scripts/e87_readability.py", "CHANGED src/dl/abcd.py", "CHANGED src/dl/count_bits.py", "CHANGED src/dl/count_text.py",
             "CHANGED src/dl/static_text.py", "NEW scripts/e87b_check.py", "NEW scripts/e87b_run.py"}  # the dated note's list
CELLS = ("the exact budget 3 in the chunked update with the trim start (the dated note's amendment, not the plan's tmu with"
         " the half start); the unbudgeted agent's s, T and negative rule, not re-tuned for the budget")
CAVEATS = ("dev only; selection is optimistic: each model is the best of 20 dev evaluations on the dev split the bar reads,"
           " and LR's C was picked on dev; per dataset, no combined line, no multiplicity correction: each dataset's bar"
           " stands alone")
NOT_EVALUATED = {"verdict": "not evaluated in this pass", "why": "Part A not run"}


def refuse(msg: str) -> None:
    raise SystemExit(f"e87b_check: {msg}")


def sha_arr(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def note_frozen() -> str:
    """The note's freeze time; refuses unless its record holds the SHA-256 of the note's text."""
    rec, txt = ROOT / "logs" / f"{NOTE}.frozen.json", ROOT / "logs" / f"{NOTE}.md"
    if not rec.exists():
        refuse(f"{rec.name} is missing: Part B is checked only after the dated note of 2026-10-25 is frozen")
    r = json.loads(rec.read_text())
    if not txt.exists() or sha_file(txt) != r["sha256"]:
        refuse(f"logs/{NOTE}.md does not hash to its freeze record")
    return r["frozen_utc"]


def load_rows() -> dict:
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)
    return rows


def load_a0(ds: str) -> dict:
    """A0's file for one dataset (scripts/e87_bits.py), checked against its arrays, the registered arm and LR's row."""
    p = ROOT / "results" / "explore" / "e87" / f"{ds}_c4.npz"
    if not p.exists():
        refuse(f"{p.relative_to(ROOT)} is missing: A0 (scripts/e87_bits.py) has not run for {ds}")
    with np.load(p) as z:
        a = {k: z[k] for k in ("X_train", "y_train", "X_dev", "y_dev", "names", "buttons", "lr_pred_dev")}
        a["hashes"] = json.loads(str(z["bits_sha256"]))
    if {"train": sha_arr(a["X_train"]), "dev": sha_arr(a["X_dev"])} != {"train": a["hashes"]["train"], "dev": a["hashes"]["dev"]}:
        refuse(f"{ds}: A0's bits do not hash to its own record")
    reg = json.loads((ROOT / REG[ds] / "seed1.json").read_text())["bits_sha256"]
    if (reg["train"], reg["dev"]) != (a["hashes"]["train"], a["hashes"]["dev"]):
        refuse(f"{ds}: A0's bits are not the registered C4 arm's ({REG[ds]})")
    row = json.loads((ROOT / "results" / "explore" / "dryrun" / LR_ROW[ds] / "seed0.json").read_text())
    acc = float((a["lr_pred_dev"] == a["y_dev"]).mean())
    if row["arm"] != "lr" or row["bits_sha256"]["dev"] != a["hashes"]["dev"] or acc != row["dev_acc"]:
        refuse(f"{ds}: A0's L2 LR dev accuracy {acc!r} is not its LR row's ({LR_ROW[ds]}: {row.get('dev_acc')!r} on the same dev bits)")
    a["lr_acc"], a["lr_row"] = acc, LR_ROW[ds]
    a["names"], a["buttons"] = [str(x) for x in a["names"]], [str(x) for x in a["buttons"]]
    return a


def want_cfg(ds: str, seed: int, dry: str | None) -> dict:
    g = DS[ds]
    n_c, T, ep = (g["clauses"], g["clauses"], BASE_CFG["epochs"])
    if dry == "tiny":
        n_c, T, ep = DRY["clauses"], DRY["T"], DRY["epochs"]
    elif dry == "size":
        ep = SIZE["epochs"]
    return {**BASE_CFG, "n_clauses": n_c, "T": T, "s": g["s"], "neg_sampling": g["neg"], "seed": seed, "epochs": ep}


def check_job(rows: dict, frozen: str, ds: str, tag: str, seed: int, a: dict, dry: str | None = None, select: str = "skip0") -> dict:
    """Every check of one job; returns the row, the model's ta, w and n_states, and the run record."""
    got = rows.get((tag, seed), [])
    if len(got) != 1:
        refuse(f"{tag} seed {seed}: {len(got)} rows (need exactly 1)")
    d = got[0]
    if d["time"] <= frozen:
        refuse(f"{tag} seed {seed}: written before the note's freeze")
    cfg = want_cfg(ds, seed, dry)
    if not (d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == VB and d.get("lex_k") == 1000
            and d.get("init") == "trim" and d["cfg"] == cfg):
        refuse(f"{tag} seed {seed}: settings differ from the note's")
    if not all(d["static_info"][k]["table_file_sha256"] == TABLE for k in ("xcust", "xrecent")):
        refuse(f"{tag} seed {seed}: not on the C4 table")
    if d["bits_sha256"] != {"train": a["hashes"]["train"], "dev": a["hashes"]["dev"]} or d.get("n_bits") != a["X_dev"].shape[1]:
        refuse(f"{tag} seed {seed}: bits differ from A0's")
    ip = ROOT / "results" / "explore" / "e87" / "b_runs" / f"{tag}_seed{seed}.json"
    if not ip.exists():
        refuse(f"{tag} seed {seed}: no run record {ip.relative_to(ROOT)}")
    info = json.loads(ip.read_text())
    mp = ROOT / "results" / "explore" / "abcd_models" / f"{tag}_seed{seed}.npz"
    ce = DRY["chunk_elems"] if dry == "tiny" else CHUNK_ELEMS
    steps = cfg["epochs"] * math.ceil(len(a["y_train"]) / cfg["batch_size"])
    m = info.get("machines", [])
    if not (info.get("knobs") == {"neg_sampling": DS[ds]["neg"], "chunk_budget": "dl.tm_chunkbudget.enable", "select": select} and info.get("chunk_elems") == ce
            and len(m) == 1 and m[0]["chunked"] and m[0]["n_clauses"] == cfg["n_clauses"] and m[0]["literals"] == 2 * a["X_dev"].shape[1]
            and m[0]["max_literals"] == BUDGET and m[0]["budget_mode"] == "exact" and m[0]["budget_select"] == "rank"):
        refuse(f"{tag} seed {seed}: the run record is not the note's (negative rule, select {select}, CHUNK_ELEMS {ce}, one chunked machine with the exact budget)")
    if m[0]["chunked_budget_steps"] != steps:
        refuse(f"{tag} seed {seed}: {m[0]['chunked_budget_steps']} chunked budget updates, not one per training step ({steps})")
    if info.get("started", "") <= frozen:
        refuse(f"{tag} seed {seed}: the job started before the note's freeze")
    if info.get("model") != str(mp.relative_to(ROOT)) or not mp.exists() or sha_file(mp) != info.get("model_sha256"):
        refuse(f"{tag} seed {seed}: the model file is missing or not the one the run recorded")
    with np.load(mp) as z:
        ta, w, mcfg = z["ta"], z["w"], json.loads(str(z["cfg"]))
        names, buttons = [str(x) for x in z["names"]], [str(x) for x in z["buttons"]]
    if mcfg != d["cfg"] or names != a["names"] or buttons != a["buttons"]:
        refuse(f"{tag} seed {seed}: the model's cfg, names or buttons differ from the row's and A0's")
    if ta.shape != (cfg["n_clauses"], 2 * a["X_dev"].shape[1]) or w.shape != (cfg["n_clauses"], len(a["buttons"])):
        refuse(f"{tag} seed {seed}: the model's arrays do not fit its cfg and A0's bits")
    return {"row": d, "ta": ta, "w": w, "n_states": int(mcfg["n_states"]), "info": info}


def lit_dist(ta: np.ndarray, n_states: int) -> tuple[np.ndarray, dict]:
    lens = (ta > n_states).sum(1)
    dist = {str(k): int((lens == k).sum()) for k in range(BUDGET + 1)}
    dist["over_3"], dist["max"] = int((lens > BUDGET).sum()), int(lens.max())
    return lens, dist


def brief(t: dict) -> dict:
    return {"dev_acc": t["dev_acc"], "s1_point": t["receipts"]["s1_point"], "coverage_at_k_literals": t["receipts"]["coverage_at_k_literals"],
            "share_no_receipt_within_20": t["receipts"]["share_no_receipt_within_20"], "clauses": t["clauses"], "firing": t["firing"],
            "precision": t["precision"], "accuracy_short_clauses_only": t["accuracy_short_clauses_only"]}


def score(job: dict, a: dict, tag: str, seed: int) -> dict:
    """Recompute the row's acc (refuse if it differs), then the pruned scores (the registered reading)."""
    ta, w, ns, d = job["ta"], job["w"], job["n_states"], job["row"]
    u = R.tm_part(a["X_dev"], a["y_dev"], ta, w, ns)
    if u["dev_acc"] != d["acc"]:
        refuse(f"{tag} seed {seed}: recomputed dev accuracy {u['dev_acc']!r} is not the row's {d['acc']!r}")
    u.pop("pred")
    lens, dist = lit_dist(ta, ns)
    if dist["over_3"]:
        wp, tp = w.copy(), ta.copy()
        wp[lens > BUDGET] = 0
        tp[lens > BUDGET] = 1  # emptied as well: no effect on decisions or receipts, but no firing or precision counts
        p = R.tm_part(a["X_dev"], a["y_dev"], tp, wp, ns)
        p.pop("pred")
    else:
        p = u  # no clause over the budget: the pruned machine is the machine itself
    return {"pruned": brief(p), "unpruned": brief(u), "literals": dist, "row_time": d["time"], "best_epoch": d.get("best_epoch"),
            "train_s": d.get("train_s"), "peak_gpu_mb": d.get("peak_gpu_mb"), "chunked_budget_steps": job["info"]["machines"][0]["chunked_budget_steps"]}


def bar(per_seed: dict, lr_acc: float) -> dict:
    ok = {}
    for sd, r in per_seed.items():
        acc_ok = r["pruned"]["dev_acc"] >= lr_acc - BAR_POINTS / 100 - EPS
        cov_ok = r["pruned"]["s1_point"] >= COVER - EPS
        ok[sd] = {"accuracy": acc_ok, "coverage": cov_ok, "both": acc_ok and cov_ok}
    n = sum(v["both"] for v in ok.values())
    return {"lr_acc": lr_acc, "bound": lr_acc - BAR_POINTS / 100, "per_seed": ok, "seeds_meeting": n, "met": n >= 2}


def grid_from_part_a(ds: str, a: dict, frozen: str) -> dict:
    p = ROOT / "results" / "explore" / "e87" / f"readability_{ds}.json"
    if not p.exists():
        refuse(f"{p.relative_to(ROOT)} is missing: Part A (scripts/e87_readability.py) has not run for {ds}")
    res = json.loads(p.read_text())
    if res.get("dataset") != ds or res.get("n_dev") != len(a["y_dev"]) or res.get("n_bits") != a["X_dev"].shape[1] or set(res.get("l1", {})) != GRID:
        refuse(f"{ds}: Part A's file is not the plan's grid on A0's dev bits")
    if res.get("time", "") <= frozen:
        refuse(f"{ds}: Part A's file was written before the note's freeze")
    return {float(k): {"dev_acc": v["dev_acc"], "cov9": v["receipts"]["coverage_at_k_literals"]["9"]} for k, v in res["l1"].items()}


def kill(l1: dict, pruned_mean: float, a: dict) -> dict:
    """The plan's kill rule (e87_readability.l1_at_budget) with its extension fitted by e87_readability.l1_fit."""
    fitted: dict = {}

    def fit(C: float) -> dict:
        W, b, info = R.l1_fit(a["X_train"], a["y_train"], len(a["buttons"]), C)
        part = R.lr_part(a["X_dev"], a["y_dev"], W, b)
        out = {"dev_acc": part["dev_acc"], "cov9": part["receipts"]["coverage_at_k_literals"]["9"]}
        fitted[f"{C:g}"] = {**info, **out, "bits_with_a_weight": part["bits_with_a_weight"]}
        print(f"  extension: L1-LR C {C:g}: dev {out['dev_acc']:.4f}, coverage at 9 bits {out['cov9']:.3f}, {info['seconds']} s, "
              f"converged {info['converged']}", flush=True)
        return out

    v = R.l1_at_budget(dict(l1), pruned_mean, fit)
    return {**v, "pruned_mean": pruned_mean, "extension_fits": fitted}


def failed_conditions(b: dict) -> str:
    """Which of the bar's two conditions failed, in words (reporting only; the rule is the bar itself)."""
    na = sum(v["accuracy"] for v in b["per_seed"].values())
    nc = sum(v["coverage"] for v in b["per_seed"].values())
    held = f"accuracy held on {na} of 3 seeds; S1 coverage at least 0.90 on {nc} of 3"
    if na >= 2 and nc < 2:
        return held + ", so the cost is in the receipts' length, not the accuracy"
    if nc >= 2 and na < 2:
        return held + ", so the cost is in the accuracy"
    if na < 2 and nc < 2:
        return held + ", so both conditions fail"
    return held + ", but no 2 seeds meet both on the same seed"


def follows(ds: str, b: dict, k: dict | None) -> list[str]:
    out = []
    if not b["met"]:
        out.append(f'not met: the plan\'s FAILURE_MAP line "short receipts cost the agent its accuracy on {ds}", with the numbers; as run: '
                   f"{failed_conditions(b)} ({CELLS})")
    if k is None or k.get("verdict") == NOT_EVALUATED["verdict"]:
        if b["met"]:
            out.append('met; the plan\'s "met and not killed" is not reached in this pass: no FAILURE_MAP line and no registered test'
                       f" follow ({CELLS})")
        return out
    if k["verdict"] == "killed":
        out.append(f'killed: FAILURE_MAP gets "readable receipts are not specific to the TM on {ds}"')
    elif k["verdict"] == "not testable":
        out.append(f'"L1-LR does not reach the budget" on {ds}: the kill rule is not testable (never "not killed")')
        if b["met"]:
            out.append("the plan names no consequence for a met bar with a kill rule that is not testable")
    elif b["met"]:
        out.append(f"met and not killed on {ds}: a registered test may follow; it needs its own registration and a declared test use")
    return out


def retried(queue_log: Path) -> list[str]:
    """Jobs that scripts/seq_queue.sh started a second time ("start (try 2)" lines): reported with no rule."""
    if not queue_log.exists():
        return []
    out = []
    for line in queue_log.read_text().splitlines():
        if " start (try 2): " in line:
            t, sd = re.search(r"--tag (\S+)", line), re.search(r"--seed (\d+)", line)
            out.append(f"{t.group(1) if t else '?'} seed {sd.group(1) if sd else '?'}")
    return out


def same_run(x: dict, y: dict) -> bool:
    h = lambda d: [(e["epoch"], e["val_acc"]) for e in d["row"]["history"]]  # noqa: E731 (the field "t" is wall-clock time)
    return (np.array_equal(x["ta"], y["ta"]) and np.array_equal(x["w"], y["w"]) and x["row"]["acc"] == y["row"]["acc"]
            and h(x) == h(y) and x["row"].get("best_epoch") == y["row"].get("best_epoch"))


def plan_diff() -> list[str]:
    """scripts/diff_frozen.py's lines against the plan's freeze record (CHANGED, NEW, GONE)."""
    rec = json.loads((ROOT / PLAN_RECORD).read_text())["sources"]
    now = {str(p.relative_to(ROOT)): sha_file(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
    out = []
    for f in sorted(set(now) | set(rec)):
        a, b = rec.get(f), now.get(f)
        if a != b:
            out.append(("NEW " if a is None else "GONE " if b is None else "CHANGED ") + f)
    return out


def dry_mode(rows: dict, frozen: str) -> None:
    d = plan_diff()
    on_path = {x for x in d if PATH_RE.match(x.split(" ", 1)[1])}
    if on_path != PATH_DIFF:
        refuse(f"the code on e87's path since the plan's record is not the dated note's list: {sorted(on_path ^ PATH_DIFF)}")
    print(f"ok code since the plan's record: on e87's path exactly the note's {len(PATH_DIFF)} files; {len(d) - len(on_path)} files off the"
          " path changed or new (no rule)", flush=True)
    ref, done = {}, {}
    for ds in DS:
        a = load_a0(ds)
        for tag, kind in ((f"e87bdry_{ds}_L3", "tiny"),) + (((SIZE["tag"], "size"),) if ds == SIZE["ds"] else ()):
            job = check_job(rows, frozen, ds, tag, 1, a, dry=kind)
            done[tag] = job
            lens, dist = lit_dist(job["ta"], job["n_states"])
            if dist["over_3"]:
                refuse(f"{tag} seed 1: {dist['over_3']} clauses over {BUDGET} literals: the exact budget did not hold")
            s = score(job, a, tag, 1)
            print(f"ok {tag} seed 1 ({kind}): settings, bits, run record ({s['chunked_budget_steps']} chunked budget updates), model file,"
                  f" recomputed accuracy equal to the row's; literals per clause {dist}; train {s['train_s']} s, peak GPU {s['peak_gpu_mb']} MB",
                  flush=True)
            if kind == "size":
                ref = {"train_s": s["train_s"], "peak": s["peak_gpu_mb"]}
        if ds == PLAIN["ds"]:
            twin = check_job(rows, frozen, ds, PLAIN["tag"], 1, a, dry="tiny", select="plain")
            if not same_run(done[PLAIN["twin"]], twin):
                refuse(f"{PLAIN['tag']} seed 1: the plain-select twin differs from {PLAIN['twin']} (select_k_skip0 must change nothing)")
            print(f"ok {PLAIN['tag']} seed 1: equal to {PLAIN['twin']} array by array (ta, w), and in acc and history:"
                  " select_k_skip0 changed nothing on the GPU", flush=True)
        p = ROOT / "results" / "explore" / "e87" / f"readability_{ds}.json"
        if p.exists():
            v = R.l1_at_budget(grid_from_part_a(ds, a, frozen), 0.0)  # dry: the grid's path only, no extension, no rule
            print(f"ok {ds}: Part A's grid read; l1_at_budget path ran (dry, no rule: comparator C {v['C']})")
        else:
            print(f"ok {ds}: Part A's file not there yet; the kill rule's path is covered by tests/test_e87b_check.py")
    for job in retried(ROOT / "logs" / "e87b_dry_queue.log"):
        print(f"note (no rule): {job} was started twice by the queue (try 2 in logs/e87b_dry_queue.log)")
    if ref:
        regs = {ds: json.loads((ROOT / REG[ds] / "seed1.json").read_text())["train_s"] for ds in DS}
        factor = ref["train_s"] / SIZE["epochs"] * BASE_CFG["epochs"] / regs["star"]
        mins = sum(3 * (regs[ds] * factor + OVERHEAD[ds]) / 60 for ds in DS)
        print(f"projection (no rule): the exact budget's training time on STAR 16k is {factor:.2f} times R22's seed-1 run;"
              f" the 9 jobs then need about {mins:.0f} GPU-minutes (90 s per 16k job and 40 s per ABCD job for the view and the"
              f" files); peak GPU {ref['peak']} MB")
    print("all dry checks passed")


def first_evaluation(rows: dict, b_row: dict) -> str:
    """B's STAR seed 1 against the dry run's full-size STAR job (its first 2 epochs): the first dev evaluation must repeat."""
    dry = rows.get((SIZE["tag"], 1), [])
    if len(dry) != 1:
        return "B's STAR seed 1 cannot be compared with the dry run's full-size job (no single dry row)"
    a, b = dry[0]["history"][0], b_row["history"][0]
    same = (a["epoch"], a["val_acc"]) == (b["epoch"], b["val_acc"])
    return ("B's STAR seed 1 repeats the dry run's full-size job at its first evaluation (epoch "
            f"{b['epoch']}, the same dev accuracy)" if same else
            "WARNING: B's STAR seed 1 differs from the dry run's full-size job at its first evaluation: the full-size chunked exact budget"
            " is not reproducible bit for bit")


def main() -> None:
    ap = argparse.ArgumentParser(description="e87 Part B check")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--bar-only", action="store_true")
    a_ = ap.parse_args()
    frozen = note_frozen()
    rows = load_rows()
    if a_.dry:
        dry_mode(rows, frozen)
        return
    out = ROOT / "results" / "explore" / "e87" / ("partb_bar.json" if a_.bar_only else "partb.json")
    if out.exists():
        refuse(f"{out.relative_to(ROOT)} exists; not overwriting")
    res = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note_frozen_utc": frozen, "mode": "bar only" if a_.bar_only else "full",
           "datasets": {}}
    res["cells"], res["caveats"] = CELLS, CAVEATS
    prep = {}
    for ds, g in DS.items():  # pass 1: every check of every job and of Part A's files, before any number is printed
        a = load_a0(ds)
        jobs = {sd: check_job(rows, frozen, ds, g["tag"], sd, a) for sd in SEEDS}
        prep[ds] = (a, jobs, None if a_.bar_only else grid_from_part_a(ds, a, frozen))
    per_ds = {ds: {sd: score(jobs[sd], a, DS[ds]["tag"], sd) for sd in SEEDS} for ds, (a, jobs, _) in prep.items()}  # refuses on a mismatch
    print(f"e87 Part B; the cells: {CELLS}; {CAVEATS}", flush=True)
    for ds, g in DS.items():  # pass 2: the rules and the report
        a, _, grid = prep[ds]
        per = per_ds[ds]
        b = bar(per, a["lr_acc"])
        pm = float(np.mean([per[sd]["pruned"]["dev_acc"] for sd in SEEDS]))
        print(f"{ds} ({g['clauses'] // 1000}k, T {g['clauses']}, s {g['s']:g}, {g['neg']} negatives; exact budget {BUDGET}, start trim;"
              f" L2 LR dev {a['lr_acc']:.4f} from A0, bound {b['bound']:.4f}):", flush=True)
        for sd in SEEDS:
            r, o = per[sd], b["per_seed"][sd]
            if r["literals"]["over_3"]:
                print(f"  WARNING seed {sd}: {r['literals']['over_3']} clauses over {BUDGET} literals (the exact budget should give none);"
                      " pruned as the plan's registered reading says")
            print(f"  seed {sd}: pruned dev {r['pruned']['dev_acc']:.4f} ({'>=' if o['accuracy'] else '<'} bound), S1 point"
                  f" {r['pruned']['s1_point']:.3f} ({'>=' if o['coverage'] else '<'} 0.90): {'meets both' if o['both'] else 'does not meet both'};"
                  f" no rule: unpruned dev {r['unpruned']['dev_acc']:.4f}, unpruned S1 point {r['unpruned']['s1_point']:.3f},"
                  f" literals per clause {r['literals']}")
        print(f"  the bar: {'MET' if b['met'] else 'not met'} ({b['seeds_meeting']} of 3 seeds meet both); pruned mean over seeds {pm:.4f}")
        k = dict(NOT_EVALUATED) if a_.bar_only else None
        if a_.bar_only:
            print('  the kill rule (L1-LR at the same budget): not evaluated in this pass (Part A not run); never "not killed"')
        if grid is not None:
            k = kill(grid, pm, a)
            if k["C"] is None:
                print(f"  the kill rule: not testable ({k['why']}; added C {k['added']})")
            else:
                print(f"  the kill rule: {k['verdict']} (L1-LR at the budget: C {k['C']:g}, dev {k['dev_acc']:.4f}, against the pruned mean"
                      f" minus 0.5: {pm - 0.005:.4f}; added C {k['added']})")
        for line in follows(ds, b, k):
            print(f"  what follows: {line}")
        res["datasets"][ds] = {"tag": g["tag"], "lr_acc": a["lr_acc"], "lr_row": a["lr_row"], "seeds": {str(sd): per[sd] for sd in SEEDS},
                               "bar": {**b, "per_seed": {str(sd): v for sd, v in b["per_seed"].items()}}, "pruned_mean": pm,
                               "kill": k, "follows": follows(ds, b, k)}
    if a_.bar_only:
        print("  open in this pass: the plan's second question (can the agent also stay clear of L1-LR at the same budget?) and Part A")
    res["first_evaluation"] = first_evaluation(rows, prep["star"][1][1]["row"])
    print(f"determinism (no rule): {res['first_evaluation']}")
    res["retried"] = retried(ROOT / "logs" / "e87b_queue.log")
    for job in res["retried"]:
        print(f"note (no rule): {job} was started twice by the queue (try 2 in logs/e87b_queue.log)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
