"""The registered statistic of REG_agent_vs_llm: per dataset, on the rows the LLM* run answered.

usage: python -I reg_report.py --dataset star --split test --n ROWS --reg REG_agent_vs_llm.md \
           --agent <R22 seed1.npz> --agent-seeds <seed1.npz ... seed5.npz> --llm runs/reg/llm_star/preds.jsonl \
           --ft runs/reg/ft_star/test_preds.npz --tfidf runs/reg/tfidf_star/test_preds.npz --out runs/reg/report_star.json
       (--split dev for tests: the same code on dev rows; the row draw and the freeze are checked on test only)
Prediction files: either top_ids + gold (ft_baseline.py, tfidf_reg.py, agent_preds.py) or pred + y (the registered
seed files of R22, R23 and R24, written by decision_layer's scripts/abcd_confirm.py); column 0 / pred is the decision.
Rules (each per dataset; percentile intervals from a bootstrap over dialogues, seed 20261027, 10,000 draws):
  NI:  AGENT - LLM* has the lower end of its 95% interval above -0.03 (the registered margin).
  NI2: the same above -0.02 (the margin of drafts v3-v5; registered beside NI).
  SUP: AGENT - LLM* >= +0.03 and the lower end of its 95% interval above 0.
  FT:  FT - AGENT has the lower end of its 95% interval above 0.
No rule: the 90% interval of AGENT - LLM* and whether it lies inside (-0.03, +0.03) (equivalence at 3 points); the
two-level bootstrap (agent seeds, then dialogues) for AGENT - LLM*, FT - AGENT and AGENT - TFIDF; TFIDF - LLM*; the
LLM's median seconds per row. On test the report refuses: no freeze record or a registration that differs from it,
LLM rows that are not the registered draw, or LLM rows with a request error.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

SEED = 20261027
MARGIN, MARGIN2 = 0.03, 0.02


def boot_draws(diff, convo, draws, seed=SEED):
    """Means of diff over dialogues drawn with replacement (the same draws for every interval of one pair)."""
    ids = np.array(sorted(set(convo.tolist())))
    groups = [np.where(convo == c)[0] for c in ids]
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for t in range(draws):
        pick = rng.integers(0, len(ids), len(ids))
        rows = np.concatenate([groups[j] for j in pick])
        out[t] = diff[rows].mean()
    return out


def interval(draws, level):
    lo = (100 - level) / 2
    return [float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo))]


def rules(d1_mean, ci95_d1, ci90_d1, ci95_d2, margin=MARGIN, margin2=MARGIN2):
    """The registered rules (NI, NI2, SUP, FT) and the no-rule equivalence reading."""
    return {"NI_met": bool(ci95_d1[0] > -margin),
            "NI2_met": bool(ci95_d1[0] > -margin2),
            "SUP_met": bool(d1_mean >= margin and ci95_d1[0] > 0),
            "FT_met": bool(ci95_d2[0] > 0),
            "EQ_inside_no_rule": bool(ci90_d1[0] > -margin and ci90_d1[1] < margin)}


def registered_rows(n_rows, n, seed=SEED):
    return sorted(random.Random(seed).sample(range(n_rows), min(n, n_rows)))


def decisions(path):
    """(decision per row, gold per row) from either file format."""
    z = np.load(path)
    if "pred" in z.files:
        return np.asarray(z["pred"]).astype(np.int64), np.asarray(z["y"]).astype(np.int64)
    return np.asarray(z["top_ids"])[:, 0].astype(np.int64), np.asarray(z["gold"]).astype(np.int64)


def freeze_guard(reg):
    """The test split is read only under a frozen registration whose text is unchanged."""
    fr = os.path.splitext(os.path.abspath(reg))[0] + ".frozen.json"
    if not os.path.exists(fr):
        raise SystemExit(f"refused: no freeze record {fr}")
    rec = json.load(open(fr))
    have = hashlib.sha256(open(reg, "rb").read()).hexdigest()
    if have != rec["sha256"]:
        raise SystemExit(f"refused: {reg} differs from its freeze record ({have[:16]} != {rec['sha256'][:16]})")
    return {"frozen_utc": rec["frozen_utc"], "sha256": rec["sha256"]}


def two_level(seed_oks, other, convo, draws, agent_first=True):
    """Seeds within the agent arm, then dialogues; other is a 0/1 vector (one run)."""
    ids = np.array(sorted(set(convo.tolist())))
    groups = [np.where(convo == c)[0] for c in ids]
    rng = np.random.default_rng(SEED)
    dd = np.empty(draws)
    for t in range(draws):
        sp = rng.integers(0, len(seed_oks), len(seed_oks))
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))])
        a_, o_ = seed_oks[sp][:, rows].mean(), other[rows].mean()
        dd[t] = a_ - o_ if agent_first else o_ - a_
    return [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--tfidf", help="TFIDF arm (no rule)")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--n", type=int, default=None, help="registered rows of this dataset (runs/reg_rows.json); test only")
    ap.add_argument("--reg", default=None, help="the registration (test only: its freeze record is checked)")
    ap.add_argument("--agent-seeds", nargs="*", default=[], help="the agent's seed files (two-level bootstrap)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    frozen = None
    if a.split == "test":
        assert a.n and a.reg, "--n and --reg are required on the test split"
        frozen = freeze_guard(a.reg)
    with gzip.open(os.path.join(F.DATA, a.dataset, f"examples_{a.split}.json.gz"), "rt") as h:
        d = json.load(h)
    nid = {b: i for i, b in enumerate(d["buttons"])}
    gold_all = np.array([r["y"] for r in d["rows"]])
    convo_all = np.array([r["convo_id"] for r in d["rows"]])
    recs = sorted((json.loads(x) for x in open(a.llm) if x.strip()), key=lambda r: r["i"])
    idx = np.array([r["i"] for r in recs])
    assert len(set(idx.tolist())) == len(idx), "a row answered twice"
    n_err = sum(r.get("error") is not None for r in recs)
    draw_ok = None
    if a.split == "test":
        draw_ok = idx.tolist() == registered_rows(len(gold_all), a.n)
        assert draw_ok, "the LLM rows are not the registered draw (missing, extra or other rows)"
        assert n_err == 0, f"{n_err} LLM rows hold a request error: ask them again before the report"
    llm = np.array([nid.get(r["pred"], -1) for r in recs])
    arms = {}
    for name, p in (("agent", a.agent), ("ft", a.ft)) + ((("tfidf", a.tfidf),) if a.tfidf else ()):
        dec, gold = decisions(p)
        assert np.array_equal(gold, gold_all), f"{name}: rows not in split order"
        arms[name] = dec
    g, convo = gold_all[idx], convo_all[idx]
    ag = (arms["agent"][idx] == g).astype(float)
    ll = (llm == g).astype(float)
    ft = (arms["ft"][idx] == g).astype(float)
    d1, d2 = ag - ll, ft - ag
    b1 = boot_draws(d1, convo, a.draws)
    b2 = boot_draws(d2, convo, a.draws)
    ci1, ci1_90, ci2 = interval(b1, 95), interval(b1, 90), interval(b2, 95)
    res = {"registration": "REG_agent_vs_llm", "frozen": frozen, "dataset": a.dataset, "split": a.split,
           "n_rows": int(len(idx)), "rows_are_registered_draw": draw_ok, "llm_errors": int(n_err),
           "n_dialogues": int(len(set(convo.tolist()))), "acc_agent": float(ag.mean()), "acc_llm": float(ll.mean()),
           "acc_ft": float(ft.mean()), "llm_invalid": int((llm < 0).sum()),
           "llm_median_s_per_row": float(np.median([r["s"] for r in recs])),
           "agent_minus_llm": float(d1.mean()), "ci_agent_minus_llm": ci1, "ci90_agent_minus_llm": ci1_90,
           "ft_minus_agent": float(d2.mean()), "ci_ft_minus_agent": ci2, "margin": MARGIN, "margin2": MARGIN2,
           "files": {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in
                     (("agent", a.agent), ("llm", a.llm), ("ft", a.ft)) + ((("tfidf", a.tfidf),) if a.tfidf else ())}}
    res.update(rules(d1.mean(), ci1, ci1_90, ci2))
    tf = None
    if a.tfidf:  # no rule
        tf = (arms["tfidf"][idx] == g).astype(float)
        d3, d4 = tf - ll, ag - tf
        res["tfidf_no_rule"] = {"acc_tfidf": float(tf.mean()),
                                "tfidf_minus_llm": float(d3.mean()), "ci_tfidf_minus_llm": interval(boot_draws(d3, convo, a.draws), 95),
                                "agent_minus_tfidf": float(d4.mean()), "ci_agent_minus_tfidf": interval(boot_draws(d4, convo, a.draws), 95)}
    if a.agent_seeds:  # two-level bootstrap, no rule
        oks = []
        for p in a.agent_seeds:
            dec, gold = decisions(p)
            assert np.array_equal(gold, gold_all)
            oks.append((dec[idx] == g).astype(float))
        oks = np.array(oks)
        res["two_level"] = {"agent_seeds": len(oks), "agent_acc_per_seed": [float(x) for x in oks.mean(1)],
                            "agent_acc_mean": float(oks.mean()),
                            "agent_minus_llm_mean": float(oks.mean() - ll.mean()),
                            "ci": two_level(oks, ll, convo, a.draws),
                            "ft_minus_agent_ci": two_level(oks, ft, convo, a.draws, agent_first=False),
                            "note": "the FT, LLM and TFIDF arms are one run each: their seed variance is not included"}
        if tf is not None:
            res["two_level"]["agent_minus_tfidf_ci"] = two_level(oks, tf, convo, a.draws)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
