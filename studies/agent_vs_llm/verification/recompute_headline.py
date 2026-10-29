#!/usr/bin/env python
"""An independent recompute of REG_agent_vs_llm (glassbox-lm/studies/agent_vs_llm/REG_agent_vs_llm.md, sections 2-3).

Written from the registration text, not from reg_report.py or verify_reg.py (only their output key names were read):
- rows: sorted(random.Random(20261027).sample(range(N_test), n)), n = 4,000 STAR, 3,608 ABCD (whole split), 4,000
  MultiWOZ; LLM*'s preds.jsonl must answer exactly these rows;
- per row: AGENT seed s right (R22/R23/R24 seed files, pred == y; their y, convo_id and k must equal the test file's),
  LLM* right (pred == gold, an invalid answer is wrong), FT and TFIDF right (top-1 of test_preds.npz == gold);
- differences: means over the rows (AGENT = seed 1); 95% interval: bootstrap over dialogues (convo_id), numpy
  default_rng(0), DRAWS draws, percentiles 2.5 / 97.5 of (sum of the drawn dialogues' row differences / their rows);
- rules: NI lower end > -0.03; NI2 > -0.02; SUP difference >= +0.03 and lower end > 0; FT (FT - AGENT) lower end > 0;
- two-level (no rule): each draw resamples the 5 agent seeds with replacement (mean correctness per row), then the
  dialogues; the 5-seed mean difference is the mean over seeds of (AGENT_s - LLM*).
Compared with runs/reg/report_<ds>.json: every point estimate equal (1e-12); every rule outcome equal, or NEAR-BOUND
when either lower end lies within 0.003 of the bound or a SUP difference is +3.0 points up to 1e-9 (the registered
interval decides); interval ends listed beside (another RNG). Verdict MATCH or MISMATCH per dataset.

usage: studies/decision_layer/.venv-dl/bin/python -I recompute_headline.py \
           [--headline DIR] [--data DIR] [--confirm DIR] [--draws 10000] [--datasets star abcd multiwoz]
Prints JSON; exit 0 if every dataset read is MATCH (or NEAR-BOUND only), else 1.
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

HEADLINE = Path(f"{_REPO_ROOT}/studies/agent_vs_llm")
DATA = Path(f"{_REPO_ROOT}/studies/decision_layer/data")
CONFIRM = Path(f"{_REPO_ROOT}/studies/decision_layer/results/confirm")
N_ROWS = {"star": 4000, "abcd": 3608, "multiwoz": 4000}
SEED_DIRS = {"star": "R22star/tm_mem_c4_s16", "abcd": "R23abcd/tm_mem_c4_s16", "multiwoz": "R24mwoz/tm_mem_c4_nrand"}
ROW_SEED = 20261027
NEAR = 0.003


def rows_of_test(data: Path, ds: str):
    with gzip.open(data / ds / "examples_test.json.gz", "rt") as h:
        d = json.load(h)
    return d["rows"], d["buttons"]


def cluster_boot(diff: np.ndarray, convo: np.ndarray, draws: int, rng: np.random.Generator, chunk: int = 500):
    """Percentile bootstrap over dialogues of the row mean of diff (ratio of sums)."""
    _, inv = np.unique(convo, return_inverse=True)
    S = np.bincount(inv, weights=diff)
    C = np.bincount(inv).astype(float)
    D = len(S)
    out = []
    for start in range(0, draws, chunk):
        m = min(chunk, draws - start)
        cnt = rng.multinomial(D, np.full(D, 1.0 / D), size=m).astype(float)
        out.append(cnt @ S / (cnt @ C))
    return np.concatenate(out)


def two_level(agent_seeds: np.ndarray, other: np.ndarray, convo: np.ndarray, draws: int, rng: np.random.Generator,
              agent_first: bool = True):
    """Each draw: the seeds resampled with replacement (mean correctness per row), then the dialogues."""
    _, inv = np.unique(convo, return_inverse=True)
    C = np.bincount(inv).astype(float)
    D, k = len(C), agent_seeds.shape[0]
    out = np.empty(draws)
    for b in range(draws):
        w = rng.multinomial(k, np.full(k, 1.0 / k)) / k
        a = w @ agent_seeds
        d = a - other if agent_first else other - a
        S = np.bincount(inv, weights=d, minlength=D)
        cnt = rng.multinomial(D, np.full(D, 1.0 / D)).astype(float)
        out[b] = cnt @ S / (cnt @ C)
    return out


def pct(x, lo, hi):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def recompute(ds: str, headline: Path, data: Path, confirm: Path, draws: int, n_rows: int | None = None):
    test, buttons = rows_of_test(data, ds)
    y = np.array([int(r["y"]) for r in test])
    convo_all = np.array([r["convo_id"] for r in test])
    k_all = np.array([int(r["k"]) for r in test])
    n = n_rows or N_ROWS[ds]
    rows = np.array(sorted(random.Random(ROW_SEED).sample(list(range(len(test))), n)))
    problems = []
    # LLM*
    recs = {}
    for line in open(headline / "runs/reg" / f"llm_{ds}" / "preds.jsonl"):
        if line.strip():
            r = json.loads(line)
            recs[r["i"]] = r
    if sorted(recs) != rows.tolist():
        problems.append(f"LLM* rows are not the registered draw ({len(recs)} answered)")
    llm_ok = np.array([float(recs.get(int(i), {}).get("pred") is not None and recs[int(i)]["pred"] == buttons[y[i]])
                       for i in rows])
    gold_names = [recs[int(i)]["gold"] for i in rows if int(i) in recs]
    if gold_names != [buttons[y[i]] for i in rows if int(i) in recs]:
        problems.append("LLM* gold names differ from the test file")
    n_err = sum(r.get("error") is not None for r in recs.values())
    n_invalid = sum(r.get("pred") is None for r in recs.values())
    # AGENT seeds
    sd = confirm / SEED_DIRS[ds]
    seeds = sorted(int(p.stem[4:]) for p in sd.glob("seed*.npz"))
    if seeds != [1, 2, 3, 4, 5]:
        problems.append(f"agent seeds {seeds}, not 1-5")
    A = []
    for s in seeds:
        z = np.load(sd / f"seed{s}.npz")
        for key, ref in (("y", y), ("convo_id", convo_all), ("k", k_all)):
            if not np.array_equal(z[key], ref):
                problems.append(f"agent seed {s}: {key} differs from the test file")
        A.append((z["pred"][rows] == y[rows]).astype(float))
    A = np.stack(A)
    # FT, TFIDF
    arms = {}
    for arm in ("ft", "tfidf"):
        z = np.load(headline / "runs/reg" / f"{arm}_{ds}" / "test_preds.npz")
        if not np.array_equal(z["gold"].astype(int), y):
            problems.append(f"{arm}: gold differs from the test file")
        arms[arm] = (z["top_ids"][:, 0].astype(int)[rows] == y[rows]).astype(float)
    convo = convo_all[rows]
    rng = np.random.default_rng(0)
    a1 = A[0]
    res = {"dataset": ds, "n_rows": int(n), "n_dialogues": int(len(np.unique(convo))), "agent_seeds": seeds,
           "llm_errors": int(n_err), "llm_invalid": int(n_invalid),
           "acc_agent": float(a1.mean()), "acc_llm": float(llm_ok.mean()), "acc_ft": float(arms["ft"].mean()),
           "acc_tfidf": float(arms["tfidf"].mean()), "agent_acc_per_seed": [float(x) for x in A.mean(1)]}
    d1 = a1 - llm_ok
    b1 = cluster_boot(d1, convo, draws, rng)
    res["agent_minus_llm"] = float(d1.mean())
    res["ci_agent_minus_llm"] = pct(b1, 2.5, 97.5)
    res["ci90_agent_minus_llm"] = pct(b1, 5, 95)
    d2 = arms["ft"] - a1
    res["ft_minus_agent"] = float(d2.mean())
    res["ci_ft_minus_agent"] = pct(cluster_boot(d2, convo, draws, rng), 2.5, 97.5)
    d3 = arms["tfidf"] - llm_ok
    res["tfidf_minus_llm"] = float(d3.mean())
    res["ci_tfidf_minus_llm"] = pct(cluster_boot(d3, convo, draws, rng), 2.5, 97.5)
    d4 = a1 - arms["tfidf"]
    res["agent_minus_tfidf"] = float(d4.mean())
    res["ci_agent_minus_tfidf"] = pct(cluster_boot(d4, convo, draws, rng), 2.5, 97.5)
    lo1, lo2 = res["ci_agent_minus_llm"][0], res["ci_ft_minus_agent"][0]
    res["NI_met"], res["NI2_met"] = lo1 > -0.03, lo1 > -0.02
    res["SUP_met"] = res["agent_minus_llm"] >= 0.03 - 1e-12 and lo1 > 0
    res["FT_met"] = lo2 > 0
    res["two_level"] = {"agent_minus_llm_mean": float((A - llm_ok).mean()),
                        "ci": pct(two_level(A, llm_ok, convo, draws, rng), 2.5, 97.5),
                        "ft_minus_agent_ci": pct(two_level(A, arms["ft"], convo, draws, rng, agent_first=False), 2.5, 97.5),
                        "agent_minus_tfidf_ci": pct(two_level(A, arms["tfidf"], convo, draws, rng), 2.5, 97.5)}
    res["problems"] = problems
    return res


def compare(mine: dict, rep: dict):
    """Point estimates equal, rule outcomes equal or NEAR-BOUND; interval ends side by side."""
    out = {"points": {}, "rules": {}, "intervals": {}}
    tf = rep.get("tfidf_no_rule") or {}
    tl = rep.get("two_level") or {}
    pairs = {"acc_agent": rep.get("acc_agent"), "acc_llm": rep.get("acc_llm"), "acc_ft": rep.get("acc_ft"),
             "agent_minus_llm": rep.get("agent_minus_llm"), "ft_minus_agent": rep.get("ft_minus_agent"),
             "acc_tfidf": tf.get("acc_tfidf", rep.get("acc_tfidf")),
             "tfidf_minus_llm": tf.get("tfidf_minus_llm", rep.get("tfidf_minus_llm")),
             "agent_minus_tfidf": tf.get("agent_minus_tfidf", rep.get("agent_minus_tfidf")),
             "two_level.agent_minus_llm_mean": tl.get("agent_minus_llm_mean")}
    for k, v in pairs.items():
        m = mine["two_level"]["agent_minus_llm_mean"] if k.startswith("two_level") else mine[k]
        out["points"][k] = {"mine": m, "report": v, "equal": v is not None and abs(m - v) < 1e-12}
    bounds = {"NI_met": ("ci_agent_minus_llm", -0.03), "NI2_met": ("ci_agent_minus_llm", -0.02),
              "SUP_met": ("ci_agent_minus_llm", 0.0), "FT_met": ("ci_ft_minus_agent", 0.0)}
    for rule, (ci_key, bound) in bounds.items():
        lo_m, lo_r = mine[ci_key][0], (rep.get(ci_key) or [None])[0]
        near = lo_r is not None and (abs(lo_m - bound) < NEAR or abs(lo_r - bound) < NEAR)
        if rule == "SUP_met":
            near = near or abs(mine["agent_minus_llm"] - 0.03) < 1e-9
        same = rep.get(rule) == mine[rule]
        out["rules"][rule] = {"mine": mine[rule], "report": rep.get(rule),
                              "status": "equal" if same else ("NEAR-BOUND" if near else "DIFFERENT")}
    for k in ("ci_agent_minus_llm", "ci90_agent_minus_llm", "ci_ft_minus_agent"):
        out["intervals"][k] = {"mine": mine[k], "report": rep.get(k)}
    out["intervals"]["two_level.ci"] = {"mine": mine["two_level"]["ci"], "report": tl.get("ci")}
    ok = (all(p["equal"] for p in out["points"].values()) and not mine["problems"]
          and all(r["status"] != "DIFFERENT" for r in out["rules"].values()))
    out["verdict"] = "MATCH" if ok else "MISMATCH"
    if ok and any(r["status"] == "NEAR-BOUND" for r in out["rules"].values()):
        out["verdict"] = "MATCH (NEAR-BOUND: the registered interval decides)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", type=Path, default=HEADLINE)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--confirm", type=Path, default=CONFIRM)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--datasets", nargs="+", default=["star", "abcd", "multiwoz"])
    ap.add_argument("--n-rows", type=json.loads, default=None, help="tests only: {dataset: n}")
    a = ap.parse_args(argv)
    out, all_ok = {}, True
    for ds in a.datasets:
        rp = a.headline / "runs/reg" / f"report_{ds}.json"
        if not rp.exists():
            out[ds] = {"verdict": "no report"}
            continue
        mine = recompute(ds, a.headline, a.data, a.confirm, a.draws, (a.n_rows or {}).get(ds))
        cmp_ = compare(mine, json.loads(rp.read_text()))
        out[ds] = {"recompute": mine, "compare": cmp_, "verdict": cmp_["verdict"]}
        all_ok = all_ok and cmp_["verdict"].startswith("MATCH")
    print(json.dumps(out, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
