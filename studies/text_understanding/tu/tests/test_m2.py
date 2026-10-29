#!/usr/bin/env python
"""Tests for M2's code (tu/PLAN_M2.md). A plain script, no pytest; CPU only, no GPU. It reads no MTOP label column and
never opens MTOP's en/test.txt: every run on real MTOP texts uses a stand-in of m2_standin.py (labels = sha256(text)
mod 20, every 40th en/eval label set to -1, test := val) under tu/m2_audit_run.py, which raises if en/test.txt is opened.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_m2.py [--feats]

- m2_verdict.py on synthetic rows, predictions, labels (with -1), test IDs, matched lists and overlap files:
  - H1 and H2 met / strong / not met; both probes needed; a bar met exactly at its boundary; H2 one-sided (a3B far
    above potion is met and strong); the headline of each case;
  - the claim sentences (review of draft 1, design MAJOR 1): H1 on point estimates with the interval clause only when
    STRONG; H2 "no more than 0.5 points below potion-base-32M ... (point estimates; one-sided; ...)"; not met per probe
    with its interval; never "within 0.5", "beats" or "failed replication"; the provenance with the teacher revision,
    the MTEB caveat and "never pooled with M1";
  - the per-probe table; "no seed level" for LR; the STRONG qualifier when the TM's seed-inclusive lower end misses;
  - the per-seed differences, the seed SE and SD; TM minus LR; the en/eval differences beside the test differences;
  - -1 labels as errors with their counts; the secondary analysis with the ID check and its per-pool wording; the
    c4_fit line; the paraphrase-exact line; identical numbers on a rerun;
  - refusals: a missing seed (with "NOT EVALUABLE"), a smoke row, s 16, a duplicate LR row, rows of another dataset or
    tag, a configuration not in M2, a table that does not fit its arm, an LR row whose arm is not its name, TM and LR
    rows of one arm from two caches, a row that disagrees with its predictions, a prediction file of the wrong length,
    an overwrite;
  - after the re-check of draft 2: the weak-evidence clause of a MET-not-STRONG sentence, the provenance tail of every
    claim sentence, the per-seed lines of the rule parts, and a matched list or overlap file that does not fit (another
    size, a differing ID, labels without IDs, a c4_fit position outside the list, another c4_fit count, an unreadable
    overlap file): exit 0, H1 and H2 written and equal to the unbroken case, the lines that use the file NOT EVALUABLE;
  - a real-size run (n 4,386, the frozen matched list and the real overlap file, about 110 classes), timed.
- m2_check_cache.py on a synthetic cache; m2_feats.load_ds's default loader.
- m2_codecheck.py: M1's own code-check rows (tu/m1/check) pass against 3c's rows; a changed val accuracy, test
  accuracy, best epoch, curve point, LR C or LR test accuracy, a missing or doubled row each fail.
- m2_freeze_check.py on small files: the plan's block equal to the pins (absolute and relative paths) passes, also with
  --plan and --pins given relative to the working directory; a block hash that differs, a path with two different
  hashes (a stale line placed before the right one), a changed file, pins without the plan, a 16-character pin, a
  relative pins line, a pinned file missing from the block, a block line under another heading, a plan without a block,
  a dry summary without pass, with another code hash or without a pinned file each refuse.
- m2_code_diff.sh regenerates tu/m2_code_diff_vs_m1.txt byte for byte, and its parent hashes equal M1's pins.
- With --feats (minutes; the CPU smoke of the data path):
  - tu/m2_dry_gpu.sh with M2_DRY_DEVICE=cpu (m2_standin.py:mtop_small_unseen, 56 labels -1): its own check passed
    (the -1 accounting row by row, the audit, network-free, tu/m2 untouched) and its summary names the code hashes;
  - the C4 and a3B steps imported no sentence_transformers, transformers or model2vec; the potion build did;
  - the hard-set rule; M1's m1_tm.py raising KeyError 'mtop_en' after its test write (before) and m2_tm.py writing
    its row on the same cache (after); the one-read refusals;
  - the verdict on the dry run's own predictions (seeds 7-9 made from the one smoke run per arm), under the audit.
Each run has its own folder in tu/tests/runs/<UTC stamp>_m2/.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
EMB = TU.parent
sys.path.insert(0, str(TU))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
PY = sys.executable
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_m2")
TABLES = {"C4": ("ppw_S1000_d500_e0.5", "e1771d9eaf4a07f7"), "a3B": ("a3B_d768", "4a1ed2aed7aa4ce7"),
          "potion": ("potion-base-32M", "minishlab/potion-base-32M@1e5a03f8eeb2c98b928fbbd846f22f816360919f")}
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def run(args: list[str], timeout: int = 900, env: dict | None = None, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-I", "-B", *args], capture_output=True, text=True, timeout=timeout, check=False, env=env,
                          cwd=cwd)


TAIL_H1 = ("Provenance: a3B is network-free at run time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time; "
           "C4 uses no neural network at any stage.")
TAIL_H2 = ("Provenance: a3B is network-free at run time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time; "
           "potion-base-32M@1e5a03f8 is distilled from bge-base-en-v1.5 by its authors, so the two share the teacher family.")
WEAK_H1 = ("; not STRONG: a point-estimate result, weak evidence (tu/PLAN_M2.md Power: at no true gain H1 is MET in "
           "0.4-10.5% of runs). ")
WEAK_H2 = ("; not STRONG: a point-estimate result, weak evidence (tu/PLAN_M2.md Power: at a true -1.0 H2 is MET in "
           "0.1-7.8% of runs). ")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


RUNS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- m2_verdict.py on synthetic rows
SEEDS = (7, 8, 9)


def make_synth(d0: Path, n: int, tmk: dict, lrk: dict, unseen: int = 0, matched_pos=(), para_pos=(),
               ids_in_labels: bool = True, c4_pos=None) -> None:
    """tmk: arm -> correct counts for seeds 7, 8, 9; lrk: arm -> correct count. Item i is predicted right (0) when
    i < k, else wrong (1); the labels are 0 except the last `unseen` items (-1: an intent not in train). Val accuracy
    = test accuracy + 0.01 for every row, so each en/eval difference equals its test difference."""
    (d0 / "preds").mkdir(parents=True)
    y = np.zeros(n, np.int64)
    if unseen:
        y[n - unseen:] = -1
    tm, lr = [], []
    for arm, ks in tmk.items():
        for sd, k in zip(SEEDS, ks):
            p = d0 / "preds" / f"m2_{arm}_mtop_en_r1_s{sd}.npy"
            pr = np.r_[np.zeros(k), np.ones(n - k)].astype(np.int16)
            np.save(p, pr)
            acc = round(float((pr == y).mean()), 4)
            tm.append({"tag": "m2", "dataset": "mtop_en", "arm": arm, "repeat": 1, "seed": sd, "s": 8.0, "smoke": False,
                       "cfg": {"n_clauses": 4000, "T": 4000, "seed": sd}, "test_acc": acc, "val_acc": round(acc + 0.01, 4),
                       "best_epoch": 30, "preds": str(p), "table": TABLES[arm][0], "table_sha256": TABLES[arm][1],
                       "kmeans_run_id": f"km_{arm}"})
    for arm, k in lrk.items():
        p = d0 / "preds" / f"lr_{arm}_mtop_en.npy"
        pr = np.r_[np.zeros(k), np.ones(n - k)].astype(np.int16)
        np.save(p, pr)
        acc = round(float((pr == y).mean()), 4)
        lr.append({"dataset": "mtop_en", "name": arm, "arm": arm, "lr_test": acc, "lr_val": round(acc + 0.01, 4), "C": 1.0,
                   "acc_by_C": {"1.0": acc}, "preds": str(p), "table": TABLES[arm][0], "table_sha256": TABLES[arm][1],
                   "kmeans_run_id": f"km_{arm}"})
    (d0 / "results_m2_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm))
    (d0 / "results_m2_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))
    ids_line = f"    d.ids = {{'test': ['id%d' % i for i in range({n})]}}\n" if ids_in_labels else ""
    (d0 / "labels.py").write_text(
        "import numpy as np\n\nclass D:\n    pass\n\n\ndef load(name):\n    assert name == 'mtop_en', name\n    d = D()\n"
        f"    y = np.zeros({n}, np.int64)\n    y[{n} - {unseen}:] = -1 if {unseen} else 0\n"
        f"    d.labels = {{'test': y, 'val': np.r_[np.zeros(90, np.int64), -np.ones(2, np.int64)]}}\n"
        "    d.intents = ['a', 'b', 'c']\n" + ids_line + "    return d\n")
    mp, pp = list(matched_pos), list(para_pos)
    m = {"n_test": n, "n_left_out": len(mp), "n_kept": n - len(mp), "positions": mp, "ids": [f"id{i}" for i in mp],
         "paraphrase_exact": {"n": len(pp), "positions": pp, "ids": [f"id{i}" for i in pp]},
         "per_pool_union": {"ours_train_val": 4, "paraphrases": 6, "c4_fit": len(c4_pos or [])}}
    (d0 / "matched.json").write_text(json.dumps(m))
    if c4_pos is not None:
        c4 = list(c4_pos)
        (d0 / "overlap.json").write_text(json.dumps({"mtop_sizes": {"test": n}, "hits": {"test": {"c4_fit": {
            "exact": c4[:1], "order_free": [], "one_edit": c4[1:]}}}}))


def verdict(d0: Path, overlap: bool = False) -> subprocess.CompletedProcess:
    return run([str(TU / "m2_verdict.py"), "--m2", str(d0), "--matched", str(d0 / "matched.json"), "--labels",
                f"{d0 / 'labels.py'}:load"] + (["--overlap", str(d0 / "overlap.json")] if overlap else []))


def jv(d0: Path) -> dict:
    p = d0 / "verdict_m2.json"
    return json.loads(p.read_text()) if p.exists() else {}


N = 1000
A_TM = {"C4": (880, 890, 900), "a3B": (920, 920, 920), "potion": (924, 924, 924)}
A_LR = {"C4": 890, "a3B": 925, "potion": 929}
vA = RUNS / "verdict_A"
make_synth(vA, N, A_TM, A_LR, matched_pos=range(10), para_pos=(0, 1, 2), c4_pos=(3, 4))
p = verdict(vA, overlap=True)
check("verdict A: exit 0", p.returncode == 0, p.stderr[-800:])
J = jv(vA)
if J:
    c, h = J["comparisons"], J["hypotheses"]
    check("A: points from the predictions (H1 TM +3.00, LR +3.50; H2 TM -0.40, LR -0.40)",
          abs(c["H1_tm"]["point"] - 3.0) < 1e-6 and abs(c["H1_lr"]["point"] - 3.5) < 1e-6
          and abs(c["H2_tm"]["point"] + 0.4) < 1e-6 and abs(c["H2_lr"]["point"] + 0.4) < 1e-6, json.dumps(c)[:400])
    check("A: H1 MET and STRONG; H2 MET, not strong (one-sided bar -0.5, lower end below it)",
          h["H1"]["met"] and h["H1"]["strong"] and not h["H1"]["strong_item_level_only"] and h["H2"]["met"]
          and not h["H2"]["strong"], str(h)[:300])
    check("A: the headline", J["headline"] == "VERDICT M2 (MTOP English test, first use): H1 MET (STRONG); H2 MET (not strong)",
          J["headline"])
    c1, c2 = h["H1"]["claim"], h["H2"]["claim"]
    check("A: the H1 sentence is on point estimates, with both probes' intervals, the interval clause (STRONG) and the provenance tail",
          c1.startswith("H1 MET: a3B - C4 is at least +0.5 points on MTOP English with the TM (+3.00 [")
          and "and with LR (+3.50 [" in c1 and "(point estimates); the lower ends of the item-level 95% intervals clear +0.5. " + TAIL_H1 in c1
          and c1.endswith(TAIL_H1) and "weak evidence" not in c1, c1)
    check("A: the H2 sentence says 'no more than 0.5 points below', one-sided, no interval clause, the weak-evidence clause "
          "(MET, not strong) and the provenance tail naming potion-base-32M@1e5a03f8 and the shared teacher family",
          c2.startswith("H2 MET: a3B is no more than 0.5 points below potion-base-32M on MTOP English with the TM and with LR "
                        "(point estimates; one-sided; TM -0.40 [") and "LR -0.40 [" in c2 and "lower ends" not in c2
          and WEAK_H2 + TAIL_H2 in c2 and c2.endswith(TAIL_H2), c2)
    txt = (vA / "verdict_m2.txt").read_text()
    check("A: the text never says 'within 0.5', 'beats' or 'failed replication'",
          "within 0.5" not in txt and "beats" not in txt and "failed replication" not in txt)
    check("A: the provenance names the teacher revision, potion's unknown one, the MTEB caveat and no pooling with M1",
          "bge-base-en-v1.5@a5beb1e3" in txt and "the teacher revision they used is not known" in txt and "MTEB" in txt
          and "never pooled with M1's three reused splits" in txt and "no neural network at any stage" in txt
          and "use count 0 -> 1" in txt and "No combined line" in txt, txt[:400])
    rp = next((ln for ln in txt.splitlines() if ln.startswith("  rule parts (per-seed differences and seed SE for TM): ")), "")
    check("A: under the table, each rule part with its per-seed differences and seed SE (TM) or 'no seed level' (LR)",
          "H1_tm +3.00 [" in rp and "per seed +4.00, +3.00, +2.00; seed SE 0.58" in rp and "H1_lr +3.50 [" in rp
          and "(no seed level; en/eval +3.50)" in rp and "H2_tm -0.40 [" in rp and "H2_lr -0.40 [" in rp, rp[:400])
    rows_t = [ln for ln in txt.splitlines() if ln.startswith("| H")]
    check("A: one per-probe table (H1 TM, H1 LR, H2 TM, H2 LR); LR rows say 'no seed level'",
          "| Rule | Probe | Point | Bar | Holds |" in txt and [r.split("|")[1:3] for r in rows_t] ==
          [[" H1 ", " TM "], [" H1 ", " LR "], [" H2 ", " TM "], [" H2 ", " LR "]]
          and all("no seed level" in r for r in rows_t if "| LR |" in r)
          and not any("no seed level" in r for r in rows_t if "| TM |" in r), str(rows_t))
    check("A: LR comparisons carry no seed-inclusive interval (None); TM and TM-LR ones do",
          c["H1_lr"]["ci95_seed_inclusive"] is None and c["H2_lr"]["ci95_seed_inclusive"] is None
          and c["H1_tm"]["ci95_seed_inclusive"] is not None and c["tm_minus_lr_C4"]["ci95_seed_inclusive"] is not None)
    check("A: per-seed differences (+4, +3, +2) and the seed SE 1/sqrt(3)",
          c["H1_tm"]["per_seed"] == [4.0, 3.0, 2.0] and abs(c["H1_tm"]["seed_se"] - 0.5774) < 1e-4, str(c["H1_tm"]))
    check("A: the C4 arm's seed SD is 1.0 and its mean 89.0", abs(J["tm"]["C4@1"]["test_seed_sd"] - 1.0) < 1e-6
          and abs(J["tm"]["C4@1"]["test"] - 89.0) < 1e-6, str(J["tm"]["C4@1"]))
    w1 = c["H1_tm"]["ci95"][1] - c["H1_tm"]["ci95"][0]
    w2 = c["H1_tm"]["ci95_seed_inclusive"][1] - c["H1_tm"]["ci95_seed_inclusive"][0]
    check(f"A: the seed-inclusive interval of H1 TM is wider than the registered one when the seeds differ ({w2:.2f} > {w1:.2f})",
          w2 > w1 + 0.1, f"{w1} {w2}")
    check("A: every interval holds its point", all(v["ci95"][0] <= v["point"] <= v["ci95"][1]
                                                   and (v["ci95_seed_inclusive"] is None
                                                        or v["ci95_seed_inclusive"][0] <= v["point"] <= v["ci95_seed_inclusive"][1])
                                                   for v in c.values()), json.dumps(c)[:300])
    check("A: TM minus LR per arm (C4 0.00, a3B -0.50, potion -0.50), no bar",
          abs(c["tm_minus_lr_C4"]["point"]) < 1e-6 and abs(c["tm_minus_lr_a3B"]["point"] + 0.5) < 1e-6
          and abs(c["tm_minus_lr_potion"]["point"] + 0.5) < 1e-6 and "bar" not in c["tm_minus_lr_a3B"], str(c["tm_minus_lr_a3B"]))
    check("A: the en/eval difference beside each test difference (val = test + 0.01, so H1 TM +3.00, H1 LR +3.50)",
          abs(c["H1_tm"]["val_point_best_epoch"] - 3.0) < 1e-6 and abs(c["H1_lr"]["val_point_best_epoch"] - 3.5) < 1e-6
          and "en/eval (best epoch)" in txt, str(c["H1_tm"]))
    s = J["secondary_without_matched"]
    check("A: secondary without the 10 matched items: n 990, C4 TM (870+880+890)/3/990, LR a3B 915/990",
          s["n_kept"] == 990 and s["n_left_out"] == 10 and abs(s["tm"]["C4@1"] - round(100 * 880 / 990, 4)) < 1e-6
          and abs(s["lr"]["a3B"] - round(100 * 915 / 990, 4)) < 1e-6 and s["rule"].startswith("none"), str(s)[:300])
    check("A: the secondary line names per pool which M2 arm saw it (c4_fit: a3B's teacher-fit input; the others: none)",
          "a3B's teacher-fit input" in s["pools"] and "no M2 arm trained on them" in s["pools"], s["pools"])
    check("A: secondary H1 TM point on the kept items (+30 of 990)", abs(s["comparisons"]["H1_tm"]["point"] - round(100 * 30 / 990, 4)) < 1e-6,
          str(s["comparisons"]["H1_tm"]))
    s4 = J["secondary_without_c4_fit"]
    check("A: the c4_fit line leaves out only the 2 c4_fit items (n 998)", s4 is not None and s4["n_kept"] == 998
          and s4["n_left_out"] == 2 and s4["pools"].startswith("c4_fit only"), str(s4)[:200])
    pe = J["paraphrase_exact_items"]
    check("A: the paraphrase-exact line (3 items, all right for every arm)", pe["n"] == 3 and pe["tm"]["a3B@1"] == 100.0
          and pe["lr"]["C4"] == 100.0, str(pe))
    check("A: the matched list and the overlap file are recorded with their sha256 (a synthetic list is not the frozen v1)",
          len(J["matched"]["sha256"]) == 64 and J["matched"]["is_frozen_v1"] is False and len(J["overlap"]["sha256"]) == 64)
    p2 = verdict(vA, overlap=True)
    check("verdict: refuses to overwrite (exit 2)", p2.returncode == 2, p2.stderr[-300:])
    vA2 = RUNS / "verdict_A2"
    make_synth(vA2, N, A_TM, A_LR, matched_pos=range(10), para_pos=(0, 1, 2), c4_pos=(3, 4))
    verdict(vA2, overlap=True)
    JA2 = jv(vA2)
    check("verdict: identical numbers on a rerun (fixed generators)",
          JA2.get("comparisons") == J["comparisons"] and JA2.get("secondary_without_matched") == J["secondary_without_matched"]
          and JA2.get("secondary_without_c4_fit") == J["secondary_without_c4_fit"])

# B: H1 needs both probes; H2 is one-sided (no upper bound); the not-met sentence per probe
vB = RUNS / "verdict_B"
make_synth(vB, N, {"C4": (900,) * 3, "a3B": (920,) * 3, "potion": (890,) * 3}, {"C4": 918, "a3B": 922, "potion": 892})
p = verdict(vB)
JB = jv(vB)
check("B: H1 NOT MET when only TM clears +0.5 (LR +0.40); H2 MET and STRONG at +3.0 (one-sided, no upper bound)",
      p.returncode == 0 and JB["hypotheses"]["H1"]["met"] is False and JB["hypotheses"]["H1"]["per_probe"]["H1_tm"]["holds"]
      and not JB["hypotheses"]["H1"]["per_probe"]["H1_lr"]["holds"] and JB["hypotheses"]["H2"]["met"] and JB["hypotheses"]["H2"]["strong"]
      and JB["headline"].endswith("H1 NOT MET; H2 MET (STRONG)"), (p.stderr[-300:] + JB.get("headline", "")))
if JB:
    cb = JB["hypotheses"]["H1"]["claim"]
    check("B: the not-met sentence names the rule and each probe with its interval (TM holds, LR below the bar)",
          cb.startswith("H1 NOT MET (the registered rule a3B - C4 >= +0.5 points with TM and with LR was not met): "
                        "with the TM, a3B - C4 is +2.00 [") and "]: holds; with LR, a3B - C4 is +0.40 [" in cb
          and "]: below the bar. A true difference inside the interval is not ruled out. " + TAIL_H1 in cb
          and cb.endswith(TAIL_H1) and "weak evidence" not in cb, cb)
    check("B: H2 STRONG adds the interval clause at -0.5 and the provenance tail, no weak-evidence clause",
          "the lower ends of the item-level 95% intervals clear -0.5. " + TAIL_H2 in JB["hypotheses"]["H2"]["claim"]
          and "weak evidence" not in JB["hypotheses"]["H2"]["claim"], JB["hypotheses"]["H2"]["claim"])
# C: exactly at the bars
vC = RUNS / "verdict_C"
make_synth(vC, N, {"C4": (900,) * 3, "a3B": (905,) * 3, "potion": (910,) * 3}, {"C4": 900, "a3B": 905, "potion": 910})
p = verdict(vC)
JC = jv(vC)
check("C: bars met exactly at the boundary count (H1 +0.50, H2 -0.50), not strong",
      p.returncode == 0 and abs(JC["comparisons"]["H1_tm"]["point"] - 0.5) < 1e-6 and abs(JC["comparisons"]["H2_lr"]["point"] + 0.5) < 1e-6
      and JC["hypotheses"]["H1"]["met"] and not JC["hypotheses"]["H1"]["strong"]
      and JC["hypotheses"]["H2"]["met"] and not JC["hypotheses"]["H2"]["strong"], JC.get("headline", p.stderr[-300:]))
if JC:
    check("C: H1 MET, not strong, carries the H1 weak-evidence clause; H2 the H2 one",
          WEAK_H1 + TAIL_H1 in JC["hypotheses"]["H1"]["claim"] and WEAK_H2 + TAIL_H2 in JC["hypotheses"]["H2"]["claim"],
          JC["hypotheses"]["H1"]["claim"])
# D: just short of the bars
vD = RUNS / "verdict_D"
make_synth(vD, N, {"C4": (900,) * 3, "a3B": (904,) * 3, "potion": (910,) * 3}, {"C4": 890, "a3B": 920, "potion": 910})
p = verdict(vD)
JD = jv(vD)
check("D: H1 TM +0.40 and H2 TM -0.60 miss (LR alone cannot carry them): H1 NOT MET; H2 NOT MET",
      p.returncode == 0 and JD["headline"].endswith("H1 NOT MET; H2 NOT MET")
      and "with the TM, a3B - potion is -0.60 [" in JD["hypotheses"]["H2"]["claim"], JD.get("headline", p.stderr[-300:]))
# E: labels -1 (intents not in train)
vE = RUNS / "verdict_E"
make_synth(vE, N, {"C4": (N,) * 3, "a3B": (N,) * 3, "potion": (N,) * 3}, {"C4": N, "a3B": N, "potion": N}, unseen=7)
p = verdict(vE)
JE = jv(vE)
check("E: 7 test items with label -1 count as errors for every arm (99.30) and are reported, with the val count and train intents",
      p.returncode == 0 and JE["tm"]["a3B@1"]["test"] == 99.3 and JE["lr"]["potion"]["test"] == 99.3
      and JE["counts"]["test_items_intent_not_in_train"] == 7 and JE["counts"]["val_items_intent_not_in_train"] == 2
      and JE["counts"]["train_intents"] == 3, str(JE.get("counts", p.stderr[-300:])))
# F: STRONG on the item-level interval while the TM's seed-inclusive lower end misses the bar
vF = RUNS / "verdict_F"
make_synth(vF, N, {"C4": (900,) * 3, "a3B": (900, 925, 950), "potion": (900,) * 3}, {"C4": 900, "a3B": 940, "potion": 900})
p = verdict(vF)
JF = jv(vF)
if JF:
    hf = JF["hypotheses"]["H1"]
    check("F: H1 STRONG on the item-level interval only (TM seeds +0.0 / +2.5 / +5.0): flag, qualifier and sentence",
          hf["met"] and hf["strong"] and hf["strong_item_level_only"]
          and JF["headline"].startswith("VERDICT M2 (MTOP English test, first use): H1 MET (STRONG on the item-level interval only)")
          and "(on the item-level interval only: the TM's seed-inclusive lower end is +" in hf["claim"]
          and hf["per_probe"]["H1_tm"]["seed_inclusive_lower_clears"] is False and hf["per_probe"]["H1_lr"]["seed_inclusive_lower_clears"] is None,
          JF.get("headline", "") + " " + str(JF["comparisons"]["H1_tm"]))
else:
    check("F: verdict exit 0", False, p.stderr[-500:])


def broken(name: str, edit, want: int, overlap: bool = False, need: str = "", **kw) -> None:
    d0 = RUNS / f"verdict_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
    make_synth(d0, N, A_TM, A_LR, matched_pos=kw.get("matched_pos", range(10)), para_pos=(0, 1, 2),
               ids_in_labels=kw.get("ids_in_labels", True), c4_pos=kw.get("c4_pos", (3, 4)))
    tm = [json.loads(x) for x in (d0 / "results_m2_tm.jsonl").read_text().splitlines()]
    lr = [json.loads(x) for x in (d0 / "results_m2_lr.jsonl").read_text().splitlines()]
    tm, lr = edit(tm, lr, d0)
    (d0 / "results_m2_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm))
    (d0 / "results_m2_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))
    p = verdict(d0, overlap=overlap)
    check(f"verdict refuses: {name} (exit {want})", p.returncode == want and not (d0 / "verdict_m2.json").exists()
          and need in p.stderr, f"exit {p.returncode} {p.stderr[-300:]}")


def short_preds(tm, lr, d0):
    np.save(lr[0]["preds"], np.zeros(N - 1, np.int16))
    return tm, lr


def edit_json(d0, name, f):
    m = json.loads((d0 / name).read_text())
    f(m)
    (d0 / name).write_text(json.dumps(m))


def other_size(tm, lr, d0):
    edit_json(d0, "matched.json", lambda m: m.update(n_test=4386))
    return tm, lr


def other_id(tm, lr, d0):
    edit_json(d0, "matched.json", lambda m: m["ids"].__setitem__(3, "id999999"))
    return tm, lr


def c4_outside(tm, lr, d0):
    edit_json(d0, "overlap.json", lambda m: m["hits"]["test"]["c4_fit"]["one_edit"].append(500))
    edit_json(d0, "matched.json", lambda m: m["per_pool_union"].update(c4_fit=3))
    return tm, lr


def c4_count(tm, lr, d0):
    edit_json(d0, "matched.json", lambda m: m["per_pool_union"].update(c4_fit=5))
    return tm, lr


broken("a missing seed", lambda tm, lr, d0: (tm[1:], lr), 3, need="NOT EVALUABLE")
broken("a smoke row", lambda tm, lr, d0: (tm + [{**tm[0], "smoke": True}], lr), 3)
broken("s 16", lambda tm, lr, d0: ([{**tm[0], "s": 16.0}] + tm[1:], lr), 3)
broken("a duplicate LR row", lambda tm, lr, d0: (tm, lr + [lr[0]]), 3)
broken("a TM row of another dataset", lambda tm, lr, d0: (tm + [{**tm[0], "dataset": "clinc150"}], lr), 3)
broken("a TM row with another tag", lambda tm, lr, d0: (tm + [{**tm[0], "tag": "dry"}], lr), 3)
broken("a configuration not in M2 (repeat 5)", lambda tm, lr, d0: (tm + [{**r, "repeat": 5} for r in tm[:3]], lr), 3)
broken("an LR row of another dataset", lambda tm, lr, d0: (tm, [{**lr[0], "dataset": "hwu64"}] + lr[1:]), 3)
broken("a TM row whose table does not fit its arm", lambda tm, lr, d0: ([{**tm[0], "table": "a3B_d768", "table_sha256": "4a1ed2aed7aa4ce7"}] + tm[1:], lr), 3,
       need="expected ppw_S1000_d500_e0.5")
broken("an LR row whose arm is not its name", lambda tm, lr, d0: (tm, [{**lr[0], "arm": "a3B"}] + lr[1:]), 3)
broken("TM and LR rows of one arm from two caches", lambda tm, lr, d0: (tm, [{**lr[0], "kmeans_run_id": "other"}] + lr[1:]), 3,
       need="2 feature caches")
broken("a row that disagrees with its predictions", lambda tm, lr, d0: ([{**tm[0], "test_acc": 0.5}] + tm[1:], lr), 4)
broken("a prediction file of the wrong length", short_preds, 4)


def bad_overlap(tm, lr, d0):
    (d0 / "overlap.json").write_text("not json")
    return tm, lr


def ne_case(name: str, edit, which: tuple, need: str, overlap: bool = True, **kw) -> None:
    """A matched list or overlap file that does not fit: the verdict is written (exit 0), H1 and H2 equal the unbroken
    case A, and the lines that use the file are NOT EVALUABLE with the reason (re-check of draft 2, MINOR E)."""
    d0 = RUNS / f"verdict_ne_{name.replace(' ', '_')}"
    make_synth(d0, N, A_TM, A_LR, matched_pos=range(10), para_pos=(0, 1, 2), ids_in_labels=kw.get("ids_in_labels", True),
               c4_pos=(3, 4))
    edit([], [], d0)
    p = verdict(d0, overlap=overlap)
    Jn = jv(d0)
    txt = (d0 / "verdict_m2.txt").read_text() if (d0 / "verdict_m2.txt").exists() else ""
    ok = (p.returncode == 0 and bool(Jn) and Jn["comparisons"] == J.get("comparisons") and Jn["hypotheses"]["H1"]["claim"]
          and all(Jn[w]["status"] == "not evaluable" and any(need in r for r in Jn[w]["reasons"]) for w in which)
          and "NOT EVALUABLE" in txt and "WARNING: secondary lines NOT EVALUABLE" in p.stderr)
    check(f"verdict writes H1 and H2 and marks {', '.join(which)} NOT EVALUABLE: {name}", ok,
          f"exit {p.returncode} {p.stderr[-300:]} {str({w: Jn.get(w) for w in which})[:300]}")


ne_case("a matched list of another size", other_size, ("secondary_without_matched", "paraphrase_exact_items", "secondary_without_c4_fit"),
        "n_test 4386")
ne_case("an ID that differs at a matched position", other_id, ("secondary_without_matched", "paraphrase_exact_items"),
        "IDs at the matched positions differ")
ne_case("labels without test IDs", lambda tm, lr, d0: (tm, lr), ("secondary_without_matched", "paraphrase_exact_items"), "no IDs",
        ids_in_labels=False)
ne_case("a c4_fit position outside the matched list", c4_outside, ("secondary_without_c4_fit",), "not in the matched list")
ne_case("a c4_fit count that differs from the list's", c4_count, ("secondary_without_c4_fit",), "per_pool_union c4_fit")
ne_case("an unreadable overlap file", bad_overlap, ("secondary_without_c4_fit",), "unreadable")
dn = RUNS / "verdict_ne_a_c4_fit_position_outside_the_matched_list"
Jn = jv(dn)
check("NOT EVALUABLE is per line: with only the overlap file at fault, the matched-list secondary and the paraphrase "
      "line are still evaluated", bool(Jn) and Jn["secondary_without_matched"]["status"] == "evaluated"
      and Jn["paraphrase_exact_items"]["status"] == "evaluated", str(Jn.get("secondary_without_matched", {}))[:200])

# ---------------------------------------------------------------- the verdict at real size: n 4,386, the frozen list
fm = json.loads((EMB / "fresh" / "mtop_test_matched_v1.json").read_text())
nn = fm["n_test"]
vR = RUNS / "verdict_realsize"
(vR / "preds").mkdir(parents=True)
rng = np.random.default_rng(20261025)
yR = rng.integers(0, 110, nn).astype(np.int64)
yR[rng.choice(nn, 9, replace=False)] = -1
idsR = [f"f{i}" for i in range(nn)]
for pos, i in zip(fm["positions"], fm["ids"]):
    idsR[pos] = i
np.save(vR / "y.npy", yR)
(vR / "ids.json").write_text(json.dumps(idsR))
(vR / "labels.py").write_text(
    "import json\nfrom pathlib import Path\n\nimport numpy as np\n\nHERE = Path(__file__).resolve().parent\n\n\nclass D:\n    pass\n\n\n"
    "def load(name):\n    assert name == 'mtop_en', name\n    d = D()\n    y = np.load(HERE / 'y.npy')\n"
    "    d.labels = {'test': y, 'val': y[:2235].copy()}\n    d.intents = [f'IN:{i}' for i in range(110)]\n"
    "    d.ids = {'test': json.loads((HERE / 'ids.json').read_text())}\n    return d\n")
tmR, lrR = [], []
for arm, acc in (("C4", 0.90), ("a3B", 0.92), ("potion", 0.92)):
    for sd in SEEDS:
        pr = np.where(rng.random(nn) < acc, yR, (yR + 1) % 110)
        pr = np.where(yR < 0, 0, pr).astype(np.int16)
        pth = vR / "preds" / f"m2_{arm}_mtop_en_r1_s{sd}.npy"
        np.save(pth, pr)
        a_ = round(float((pr == yR).mean()), 4)
        tmR.append({"tag": "m2", "dataset": "mtop_en", "arm": arm, "repeat": 1, "seed": sd, "s": 8.0, "smoke": False,
                    "cfg": {"n_clauses": 4000, "T": 4000, "seed": sd}, "test_acc": a_, "val_acc": a_, "best_epoch": 30,
                    "preds": str(pth), "table": TABLES[arm][0], "table_sha256": TABLES[arm][1], "kmeans_run_id": f"km_{arm}"})
    pr = np.where(rng.random(nn) < acc, yR, (yR + 1) % 110)
    pr = np.where(yR < 0, 0, pr).astype(np.int16)
    pth = vR / "preds" / f"lr_{arm}_mtop_en.npy"
    np.save(pth, pr)
    a_ = round(float((pr == yR).mean()), 4)
    lrR.append({"dataset": "mtop_en", "name": arm, "arm": arm, "lr_test": a_, "lr_val": a_, "C": 10.0, "preds": str(pth),
                "table": TABLES[arm][0], "table_sha256": TABLES[arm][1], "kmeans_run_id": f"km_{arm}"})
(vR / "results_m2_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tmR))
(vR / "results_m2_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrR))
t_0 = time.time()
p = run([str(TU / "m2_verdict.py"), "--m2", str(vR), "--matched", str(EMB / "fresh" / "mtop_test_matched_v1.json"),
         "--overlap", str(EMB / "fresh" / "mtop_overlap.json"), "--labels", f"{vR / 'labels.py'}:load"])
secs = time.time() - t_0
JR = jv(vR)
check(f"real size: the verdict runs at n 4,386 with the frozen list and the real overlap file ({secs:.0f} s)",
      p.returncode == 0 and JR.get("matched", {}).get("is_frozen_v1") is True and JR["secondary_without_matched"]["n_kept"] == 3710
      and JR["secondary_without_c4_fit"]["n_kept"] == 4278 and JR["paraphrase_exact_items"]["n"] == 55
      and JR["counts"]["test_items_intent_not_in_train"] == 9 and JR["counts"]["train_intents"] == 110,
      p.stderr[-600:] if p.returncode else str(JR.get("counts")))

# ---------------------------------------------------------------- m2_check_cache.py
src = RUNS / "cc_src"
src.mkdir()
np.savez(src / "clinc150__p32tok_K512.npz", train__exists=np.ones((4, 3), np.float32), val__exists=np.arange(6, dtype=np.float32).reshape(2, 3),
         y_train=np.arange(4), y_val=np.array([5, 6]))
(src / "clinc150__p32tok_K512.meta.json").write_text(json.dumps({"table": "a3B_d768", "arm": "orig"}))
cco = RUNS / "cc_out"
p = run([str(TU / "m2_check_cache.py"), "--src", str(src), "--out", str(cco)])
check("check cache: exit 0", p.returncode == 0, p.stderr[-500:])
if p.returncode == 0:
    zc = np.load(cco / "clinc150__p32tok_K512.npz")
    mc = json.loads((cco / "clinc150__p32tok_K512.meta.json").read_text())
    check("check cache: test := val, y_test := y_val, the meta says so (m2_code_check)",
          np.array_equal(zc["test__exists"], zc["val__exists"]) and list(zc["y_test"]) == [5, 6]
          and mc["m2_code_check"]["test_is_a_copy_of_val"] and mc["table"] == "a3B_d768")
    p2 = run([str(TU / "m2_check_cache.py"), "--src", str(src), "--out", str(cco)])
    check("check cache: refuses to overwrite", p2.returncode == 2, p2.stderr[-300:])
    p3 = run([str(TU / "m2_check_cache.py"), "--src", str(cco), "--out", str(RUNS / "cc_out2")])
    check("check cache: refuses a source that has test rows", p3.returncode != 0 and "has test rows" in p3.stderr, p3.stderr[-300:])

# ---------------------------------------------------------------- m2_feats.load_ds: the default loader
import m2_feats  # noqa: E402

try:
    m2_feats.load_ds("clinc150")
    check("load_ds: the default loader (m2_data.load) refuses any set but mtop_en", False)
except AssertionError:
    check("load_ds: the default loader (m2_data.load) refuses any set but mtop_en", True)

# ---------------------------------------------------------------- m2_codecheck.py against M1's own code-check rows
m1tm = [json.loads(x) for x in (TU / "m1" / "check" / "results_m1_tm.jsonl").read_text().splitlines() if x.strip()]
m1lr = [json.loads(x) for x in (TU / "m1" / "check" / "results_m1_lr.jsonl").read_text().splitlines() if x.strip()]
tm0 = {**next(r for r in m1tm if r.get("tag") == "m1-check"), "tag": "m2-check"}
lr0 = next(r for r in m1lr if r.get("name") == "check")


def codecheck(name: str, tm_rows: list, lr_rows: list | None, want: int, need: str = "") -> None:
    ck = RUNS / f"codecheck_{name.replace(' ', '_')}"
    ck.mkdir(parents=True)
    (ck / "results_m2_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm_rows))
    if lr_rows is not None:
        (ck / "results_m2_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr_rows))
    p = run([str(TU / "m2_codecheck.py"), "--c3", str(TU / "c3"), "--ck", str(ck)])
    check(f"codecheck: {name} (exit {want})", p.returncode == want and need in p.stdout, (p.stdout + p.stderr)[-400:])


codecheck("M1's own check rows pass against 3c's rows (val, test, best epoch 40, the 20-point curve, LR, C)", [tm0], [lr0], 0,
          need="curve 20 points equal to 3c-ref")
codecheck("a val accuracy off by 0.0001", [{**tm0, "val_acc": tm0["val_acc"] + 0.0001}], [lr0], 1, need="TM val_acc")
codecheck("a test accuracy not equal to val", [{**tm0, "test_acc": 0.9}], [lr0], 1, need="TM test_acc")
codecheck("another best epoch", [{**tm0, "best_epoch": 38}], [lr0], 1, need="TM best_epoch")
cv = [list(x) for x in tm0["curve"]]
cv[5][1] = round(cv[5][1] + 0.0001, 4)
codecheck("one curve point differs", [{**tm0, "curve": cv}], [lr0], 1, need="TM curve differs")
codecheck("another LR C", [tm0], [{**lr0, "C": 10.0}], 1, need="LR C")
codecheck("an LR test accuracy not equal to its val", [tm0], [{**lr0, "lr_test": 0.9}], 1, need="LR lr_test")
codecheck("no LR row", [tm0], None, 1, need="m2 LR (check): 0 rows")
codecheck("two TM rows", [tm0, tm0], [lr0], 1, need="m2 TM (m2-check): 2 rows")

# ---------------------------------------------------------------- m2_freeze_check.py on small files
fz = RUNS / "freeze"
fz.mkdir()
f1, f2 = fz / "f1.txt", fz / "f2.txt"
f1.write_text("one\n")
f2.write_text("two\n")
dry = fz / "dry_summary.json"


def write_dry(path: Path, passed: bool = True, code: dict | None = None) -> None:
    path.write_text(json.dumps({"pass": passed, "code_sha256": code if code is not None else {str(f1): sha(f1), str(f2): sha(f2)}}))


def write_plan(path: Path, lines: list[str], heading: str = "## Pins") -> None:
    path.write_text("# test plan\n\n## Tests\n\nno pins here\n\n" + heading + "\n\n```\n" + "\n".join(lines) + "\n```\n\n## Draft history\n")


def write_pins(path: Path, files: list[Path], short: bool = False) -> None:
    path.write_text("".join(f"check {f} {sha(f)[:16] if short else sha(f)}\n" for f in files))


def freeze(name: str, plan: Path, pins: Path, want: int, need: str = "", dry_path: Path | None = dry) -> None:
    p = run([str(TU / "m2_freeze_check.py"), "--plan", str(plan), "--pins", str(pins)] + (["--dry", str(dry_path)] if dry_path else []))
    check(f"freeze check: {name} (exit {want})", p.returncode == want and need in p.stdout, (p.stdout + p.stderr)[-400:])


rel2 = str(f2.relative_to(EMB))
write_dry(dry)
planF = fz / "plan.md"
write_plan(planF, [f"{sha(f1)}  {f1}", f"{sha(f2)}  {rel2}", f"{sha(dry)}  {dry}"])
pinsF = fz / "pins.sh"
write_pins(pinsF, [f1, f2, planF, dry])
freeze("the block equals the pins (an absolute and a relative path); the plan pins itself; the dry summary passed",
       planF, pinsF, 0, need="freeze check ok: 4 pins, 3 block lines, dry summary ok")
p_np = fz / "pins_noplan.sh"
write_pins(p_np, [f1, f2, dry])
freeze("pins that do not pin the plan", planF, p_np, 1, need="does not pin the plan itself")
p_16 = fz / "pins_16.sh"
write_pins(p_16, [f1, f2, planF, dry], short=True)
freeze("a 16-character pin", planF, p_16, 1, need="not a full sha256")
f3 = fz / "f3.txt"
f3.write_text("three\n")
p_x = fz / "pins_extra.sh"
write_pins(p_x, [f1, f2, f3, planF, dry])
freeze("a pinned file that the block does not name", planF, p_x, 1, need=f"not in the plan's block with the same sha256: {f3}")
plan_b = fz / "plan_badhash.md"
write_plan(plan_b, [f"{'0' * 64}  {f1}", f"{sha(f2)}  {rel2}", f"{sha(dry)}  {dry}"])
pins_b = fz / "pins_badhash.sh"
write_pins(pins_b, [f1, f2, plan_b, dry])
freeze("a block hash that differs from the pins", plan_b, pins_b, 1, need=f"not in the plan's block with the same sha256: {f1}")
plan_h = fz / "plan_otherheading.md"
plan_h.write_text(f"# p\n\n## Tests\n\n```\n{sha(f1)}  {f1}\n```\n\n## Pins\n\n```\n{sha(f2)}  {f2}\n{sha(dry)}  {dry}\n```\n\n## End\n")
pins_h = fz / "pins_otherheading.sh"
write_pins(pins_h, [f1, f2, plan_h, dry])
freeze("a hash line under another heading is not a pin", plan_h, pins_h, 1, need=f"not in the plan's block with the same sha256: {f1}")
plan_n = fz / "plan_noblock.md"
plan_n.write_text("# p\n\n## Pins\n\nnone yet\n")
freeze("a plan without a block", plan_n, pinsF, 1, need="no sha256sum block")
dry_f = fz / "dry_fail.json"
write_dry(dry_f, passed=False)
plan_df = fz / "plan_dryfail.md"
write_plan(plan_df, [f"{sha(f1)}  {f1}", f"{sha(f2)}  {f2}", f"{sha(dry_f)}  {dry_f}"])
pins_df = fz / "pins_dryfail.sh"
write_pins(pins_df, [f1, f2, plan_df, dry_f])
freeze("a dry summary without pass", plan_df, pins_df, 1, need="pass: true", dry_path=dry_f)
dry_c = fz / "dry_otherhash.json"
write_dry(dry_c, code={str(f1): "0" * 64, str(f2): sha(f2)})
plan_dc = fz / "plan_dryhash.md"
write_plan(plan_dc, [f"{sha(f1)}  {f1}", f"{sha(f2)}  {f2}", f"{sha(dry_c)}  {dry_c}"])
pins_dc = fz / "pins_dryhash.sh"
write_pins(pins_dc, [f1, f2, plan_dc, dry_c])
freeze("a file changed after the dry run", plan_dc, pins_dc, 1, need=f"the file changed after the dry run: {f1}", dry_path=dry_c)
dry_m = fz / "dry_missing.json"
write_dry(dry_m, code={str(f2): sha(f2)})
plan_dm = fz / "plan_drymissing.md"
write_plan(plan_dm, [f"{sha(f1)}  {f1}", f"{sha(f2)}  {f2}", f"{sha(dry_m)}  {dry_m}"])
pins_dm = fz / "pins_drymissing.sh"
write_pins(pins_dm, [f1, f2, plan_dm, dry_m])
freeze("a pinned file the dry run did not record", plan_dm, pins_dm, 1, need=f"did not record this pinned file: {f1}", dry_path=dry_m)
plan_dup = fz / "plan_dup.md"
write_plan(plan_dup, [f"{'0' * 64}  {f1}", f"{sha(f1)}  {f1}", f"{sha(f2)}  {f2}", f"{sha(dry)}  {dry}"])
pins_dup = fz / "pins_dup.sh"
write_pins(pins_dup, [f1, f2, plan_dup, dry])
freeze("a path with two different hashes (a stale line placed before the right one)", plan_dup, pins_dup, 1,
       need=f"the plan's block names this file with 2 different sha256: {f1}")
pins_rel = fz / "pins_relative.sh"
pins_rel.write_text(f"check {f1.relative_to(EMB)} {sha(f1)}\ncheck {f2} {sha(f2)}\ncheck {planF} {sha(planF)}\ncheck {dry} {sha(dry)}\n")
freeze("a relative pins line", planF, pins_rel, 1, need="a pins line with a relative path")
p = run([str(TU / "m2_freeze_check.py"), "--plan", "plan.md", "--pins", "pins.sh", "--dry", "dry_summary.json"], cwd=str(fz))
check("freeze check: --plan, --pins and --dry relative to the working directory are read from it (exit 0)",
      p.returncode == 0 and "freeze check ok: 4 pins, 3 block lines, dry summary ok" in p.stdout, (p.stdout + p.stderr)[-300:])
f2.write_text("two, changed\n")
freeze("a file changed after the pins", planF, pinsF, 1, need=f"block line differs from the file now: {f2}")

# ---------------------------------------------------------------- m2_code_diff.sh: the saved diff is current
pd = subprocess.run(["bash", str(TU / "m2_code_diff.sh")], capture_output=True, timeout=120, check=False)
saved = (TU / "m2_code_diff_vs_m1.txt").read_bytes() if (TU / "m2_code_diff_vs_m1.txt").exists() else b""
check("code diff: m2_code_diff.sh regenerates tu/m2_code_diff_vs_m1.txt byte for byte",
      pd.returncode == 0 and pd.stdout == saved and len(saved) > 1000, f"exit {pd.returncode}, {len(pd.stdout)} vs {len(saved)} bytes")
m1pins = {ln.split()[1].rsplit("/", 1)[-1]: ln.split()[2] for ln in (TU / "m1" / "pins.sh").read_text().splitlines() if ln.startswith("check ")}
parents = [ln.split() for ln in saved.decode().splitlines() if ln.startswith("parent ")]
check("code diff: 9 parents named; the 8 M1 parents' full sha256 begin with M1's frozen pins (tu/m1/pins.sh)",
      len(parents) == 9 and all(m1pins.get(nm) == hx[:16] for _p, hx, nm in parents if nm.startswith("m1_"))
      and sum(1 for _p, _h, nm in parents if nm.startswith("m1_")) == 8, str(parents)[:400])

# ---------------------------------------------------------------- --feats: the CPU smoke of the data path
if "--feats" in sys.argv:
    env = dict(os.environ, M2_DRY_DEVICE="cpu", M2_DRY_ROOT=str(RUNS / "dry"))
    t_0 = time.time()
    p = subprocess.run(["bash", str(TU / "m2_dry_gpu.sh")], capture_output=True, text=True, timeout=3600, env=env, check=False)
    dry_dirs = sorted((RUNS / "dry").glob("*_cpu"))
    dd = dry_dirs[-1] if dry_dirs else RUNS / "dry" / "missing"
    check(f"dry run on the CPU (m2_dry_gpu.sh, mtop_small_unseen stand-in): exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0,
          (p.stdout + p.stderr)[-600:] + ((dd / "dry.log").read_text()[-1500:] if (dd / "dry.log").exists() else ""))
    S = json.loads((dd / "dry_summary.json").read_text()) if (dd / "dry_summary.json").exists() else {}
    check("dry: its own check passed (test acc = val acc, the -1 accounting row by row, audit, network-free, tu/m2 untouched)",
          S.get("pass") is True and S.get("tu_m2_unchanged") is True and S.get("tm_rows") == 3 and S.get("lr_rows") == 3,
          str(S.get("errors")))
    check("dry: the -1 path ran: 56 en/eval (= test) labels are -1 (every 40th of 2,235)",
          S.get("unseen_items") == 56 and S.get("stand_in") == "mtop_small_unseen" and S.get("unseen_intents") == str({"val": 56, "test": 56}),
          str({k: S.get(k) for k in ("unseen_items", "stand_in", "unseen_intents")}))
    listed = subprocess.run(["bash", str(TU / "m2_write_pins.sh"), "--list"], capture_output=True, text=True, check=False).stdout.split()
    check("dry: the summary names the full sha256 of every listed file, equal to the files now (the plan aside: it may change before the freeze)",
          len(S.get("code_sha256", {})) == len(listed) == 58
          and all(S["code_sha256"].get(f) == sha(Path(f)) for f in listed if not f.endswith("/PLAN_M2.md")),
          f"{len(S.get('code_sha256', {}))} vs {len(listed)}")
    au = [json.loads(x) for x in (dd / "audit.jsonl").read_text().splitlines()] if (dd / "audit.jsonl").exists() else []
    check("audit: 9 steps; MTOP members read are en/train and en/eval only; en/test.txt never opened",
          len(au) == 9 and S.get("mtop_members_read") == ["mtop/en/eval.txt", "mtop/en/train.txt"]
          and all("mtop/en/test.txt" not in a["mtop_members"] for a in au), str(S.get("mtop_members_read")))
    word = [a for a in au if any("wordtab:" in x for x in a["args"]) or any(x.endswith("/C4") or x.endswith("/a3B") for x in a["args"])]
    pot = [a for a in au if a["script"] == "m2_feats.py" and "potion" in a["args"]]
    check("network-free: the 6 C4 and a3B steps imported no sentence_transformers, transformers or model2vec",
          len(word) == 6 and all(not any(a["nn_imported"].values()) for a in word), str([a["nn_imported"] for a in word]))
    check("audit sanity: the potion feature build did import sentence_transformers (the audit sees an import)",
          len(pot) == 1 and pot[0]["nn_imported"]["sentence_transformers"], str(pot))
    rows = [json.loads(x) for x in (dd / "run" / "results_m2_tm.jsonl").read_text().splitlines()] if (dd / "run" / "results_m2_tm.jsonl").exists() else []
    r0 = rows[0] if rows else {}
    check("m2_tm on MTOP: the hard-set rule (lost_fixed and won_kept null, the reason named), study M2, 20 stand-in classes",
          r0.get("lost_fixed") is None and r0.get("won_kept") is None and str(r0.get("hardset", "")).startswith("none")
          and r0.get("study") == "M2" and r0.get("n_classes") == 20 and r0.get("n_train") == 1500 and r0.get("smoke") is True
          and r0.get("unseen_intents") == str({"val": 56, "test": 56}),
          str({k: r0.get(k) for k in ("lost_fixed", "hardset", "study", "n_classes", "n_train", "unseen_intents")}))
    meta = json.loads((dd / "feats" / "a3B" / "mtop_en__p32tok_K512.meta.json").read_text()) if (dd / "feats" / "a3B").exists() else {}
    zf = np.load(dd / "feats" / "a3B" / "mtop_en__p32tok_K512.npz") if meta else None
    check("feats a3B on MTOP: test rows = val rows (the stand-in), 2,235 val texts, 56 labels -1 in y_test, the splits named",
          zf is not None and np.array_equal(zf["test__exists"], zf["val__exists"]) and zf["val__exists"].shape[0] == 2235
          and int((zf["y_test"] < 0).sum()) == 56 and meta["splits"] == ["train", "val", "test"]
          and meta["unseen_intents"] == str({"val": 56, "test": 56}) and "m2_feats_sha256" in meta,
          str({k: meta.get(k) for k in ("splits", "unseen_intents")}))
    # the bug M2 fixes: M1's m1_tm.py on an MTOP-named cache (before) and m2_tm.py on the same cache (after)
    nosync = RUNS / "nosync.py"
    nosync.write_text("import runpy\nimport sys\n\nimport torch\n\ntorch.cuda.synchronize = lambda *a, **k: None  # no GPU in this test\n"
                      "sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")
    stand = f"{TU / 'm2_standin.py'}:mtop_small_unseen"
    m1o = RUNS / "m1_before"
    pb = run([str(nosync), str(TU / "m1_tm.py"), "--dataset", "mtop_en", "--view", "existsT", "--seed", "7", "--s", "8",
              "--emb-dir", str(dd / "feats" / "C4"), "--tag", "before", "--out-dir", str(m1o), "--device", "cpu", "--smoke",
              "--load", stand], timeout=1800)
    left = sorted((m1o / "preds").glob("*.npy")) if (m1o / "preds").exists() else []
    check("BEFORE (M1's m1_tm.py): KeyError 'mtop_en' after the test predictions were written, and no row",
          pb.returncode != 0 and "KeyError: 'mtop_en'" in pb.stderr and len(left) == 1 and not (m1o / "results_m1_tm.jsonl").exists(),
          pb.stderr[-400:])
    m2o = RUNS / "m2_after"
    tm_args = [str(TU / "m2_audit_run.py"), "--cpu", str(TU / "m2_tm.py"), "--dataset", "mtop_en", "--view", "existsT", "--seed",
               "7", "--s", "8", "--emb-dir", str(dd / "feats" / "C4"), "--tag", "after", "--out-dir", str(m2o), "--device", "cpu",
               "--smoke", "--load", stand]
    pa = run(tm_args, timeout=1800)
    ra = [json.loads(x) for x in (m2o / "results_m2_tm.jsonl").read_text().splitlines()] if (m2o / "results_m2_tm.jsonl").exists() else []
    check("AFTER (m2_tm.py): the same run writes its row (hardset none) and its predictions",
          pa.returncode == 0 and len(ra) == 1 and ra[0]["lost_fixed"] is None and Path(ra[0]["preds"]).exists(), pa.stderr[-400:])
    p2 = run(tm_args, timeout=1800)
    check("m2_tm: a second run with the same key is refused (one test read per run)",
          p2.returncode != 0 and "one test read per run" in p2.stderr, p2.stderr[-300:])
    lr_args = [str(TU / "m2_audit_run.py"), "--cpu", str(TU / "m2_lr.py"), "--name", "C4", "--dataset", "mtop_en", "--emb-dir",
               str(dd / "feats" / "C4"), "--out-dir", str(dd / "run"), "--device", "cpu", "--load", stand]
    p3 = run(lr_args, timeout=1800)
    check("m2_lr: a second run of an arm is refused (one test read per arm)",
          p3.returncode != 0 and "one test read per arm" in p3.stderr, p3.stderr[-300:])
    # the verdict on the dry run's own predictions (with the -1 labels), under the audit
    import m2_standin  # noqa: E402

    sd_ds = m2_standin.mtop_small_unseen("mtop_en")
    vd = RUNS / "verdict_dry"
    vd.mkdir()
    drows = [json.loads(x) for x in (dd / "run" / "results_m2_tm.jsonl").read_text().splitlines() if x.strip()]
    lrows = [json.loads(x) for x in (dd / "run" / "results_m2_lr.jsonl").read_text().splitlines() if x.strip()]
    trows = [{**r, "tag": "m2", "seed": s_, "smoke": False, "cfg": {**r["cfg"], "seed": s_}} for r in drows if r["tag"] == "dry"
             for s_ in SEEDS]
    (vd / "results_m2_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trows))
    (vd / "results_m2_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lrows))
    (vd / "labels.py").write_text(f"import sys\nsys.path.insert(0, {str(TU)!r})\nimport m2_standin\n\n\n"
                                  "def load(name):\n    return m2_standin.mtop_small_unseen(name)\n")
    posd = list(range(20))
    (vd / "matched.json").write_text(json.dumps({"n_test": 2235, "n_left_out": 20, "n_kept": 2215, "positions": posd,
                                                 "ids": [sd_ds.ids["test"][q] for q in posd],
                                                 "paraphrase_exact": {"n": 2, "positions": [0, 1], "ids": [sd_ds.ids["test"][q] for q in (0, 1)]},
                                                 "per_pool_union": {"ours_train_val": 10, "paraphrases": 10, "c4_fit": 5}}))
    (vd / "overlap.json").write_text(json.dumps({"mtop_sizes": {"test": 2235}, "hits": {"test": {"c4_fit": {
        "exact": [0, 1, 2], "order_free": [], "one_edit": [3, 4]}}}}))
    env_v = dict(os.environ, M2_AUDIT_LOG=str(vd / "audit.jsonl"))
    pv = run([str(TU / "m2_audit_run.py"), "--cpu", str(TU / "m2_verdict.py"), "--m2", str(vd), "--matched", str(vd / "matched.json"),
              "--overlap", str(vd / "overlap.json"), "--labels", f"{vd / 'labels.py'}:load"], env=env_v, timeout=900)
    JV = jv(vd)
    av = [json.loads(x) for x in (vd / "audit.jsonl").read_text().splitlines()] if (vd / "audit.jsonl").exists() else []
    check("verdict on the dry predictions: exit 0; the 56 labels -1 are counted as errors and reported; the arms' "
          "accuracies equal the rows; en/test.txt never opened",
          pv.returncode == 0 and JV["counts"]["test_items_intent_not_in_train"] == 56
          and all(abs(JV["tm"][f"{r['arm']}@1"]["test"] / 100 - r["test_acc"]) <= 0.00005 + 1e-12 for r in drows if r["tag"] == "dry")
          and all(abs(JV["lr"][r["name"]]["test"] / 100 - r["lr_test"]) <= 0.00005 + 1e-12 for r in lrows)
          and len(av) == 1 and set(av[0]["mtop_members"]) <= {"mtop/en/train.txt", "mtop/en/eval.txt"},
          pv.stderr[-600:] if pv.returncode else str(JV.get("counts")))
else:
    print("feats part not run (no --feats)")

for x in ("verdict_A2",):
    shutil.rmtree(RUNS / x, ignore_errors=True)
print(f"m2 tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
