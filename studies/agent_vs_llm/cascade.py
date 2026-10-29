"""Cascade of the readable agent and a neural partner on dev (note NOTE_cascade_dev.md). CPU only, standard numpy.

usage: python -I cascade.py --dataset star --agent runs/agent_star_c4_tm_s1/tm.npz \
           --partner runs/ft_star_roberta_s1/dev_preds.npz --out runs/cascade_star_roberta.json
       (--partner-llm runs/llm_star/preds.jsonl in place of --partner: only the rows the LLM answered)
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import json
import os
import sys

import numpy as np

DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
SEED = 20261026


def halves(convo, seed=SEED):
    ids = np.array(sorted(set(convo)))
    rng = np.random.default_rng(seed)
    ids = ids[rng.permutation(len(ids))]
    a = set(ids[: len(ids) // 2].tolist())
    return np.array([c in a for c in convo])


def pick(margin, agent_ok, partner_ok, tol=0.005):
    """Largest coverage (1% steps) whose cascade accuracy is at least partner accuracy - tol; returns (c, thr)."""
    order = np.argsort(-margin, kind="stable")
    n = len(margin)
    best = (0, np.inf)
    p_acc = partner_ok.mean()
    for c in range(1, 101):
        k = int(round(c / 100 * n))
        if k == 0:
            continue
        use = np.zeros(n, bool)
        use[order[:k]] = True
        acc = np.where(use, agent_ok, partner_ok).mean()
        if acc >= p_acc - tol:
            best = (c, margin[order[k - 1]])
    return best


def apply(margin, thr, agent_ok, partner_ok):
    use = margin >= thr
    return use, np.where(use, agent_ok, partner_ok)


def boot_diff(casc_ok, partner_ok, convo, draws=10000, seed=SEED):
    ids = np.array(sorted(set(convo)))
    idx = {c: np.where(convo == c)[0] for c in ids}
    rng = np.random.default_rng(seed)
    d = np.empty(draws)
    for t in range(draws):
        pick_ids = ids[rng.integers(0, len(ids), len(ids))]
        rows = np.concatenate([idx[c] for c in pick_ids])
        d[t] = casc_ok[rows].mean() - partner_ok[rows].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd"], required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--partner")
    ap.add_argument("--partner-llm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args(argv)
    with gzip.open(os.path.join(DATA, a.dataset, "examples_dev.json.gz"), "rt") as h:
        d = json.load(h)
    rows, buttons = d["rows"], d["buttons"]
    convo = np.array([r["convo_id"] for r in rows])
    gold = np.array([r["y"] for r in rows])
    A = np.load(a.agent)
    assert np.array_equal(A["gold"], gold), "agent rows are not in dev order"
    agent_pred = A["top_ids"][:, 0].astype(int)
    margin = (A["top_scores"][:, 0] - A["top_scores"][:, 1]).astype(float)
    keep = np.ones(len(rows), bool)
    if a.partner:
        P = np.load(a.partner)
        assert np.array_equal(P["gold"], gold), "partner rows are not in dev order"
        partner_pred = P["top_ids"][:, 0].astype(int)
    else:
        with open(a.partner_llm) as h:
            recs = [json.loads(x) for x in h if x.strip()]
        name_to_id = {b: i for i, b in enumerate(buttons)}
        partner_pred = np.full(len(rows), -1)
        keep[:] = False
        for r in recs:
            keep[r["i"]] = True
            partner_pred[r["i"]] = name_to_id.get(r["pred"], -1)
    sel = np.where(keep)[0]
    gold, convo, agent_pred, margin, partner_pred = gold[sel], convo[sel], agent_pred[sel], margin[sel], partner_pred[sel]
    agent_ok, partner_ok = agent_pred == gold, partner_pred == gold
    inA = halves(convo)
    out = {"note": "NOTE_cascade_dev.md", "dataset": a.dataset, "agent": a.agent, "partner": a.partner or a.partner_llm,
           "n": int(len(sel)), "agent_acc": float(agent_ok.mean()), "partner_acc": float(partner_ok.mean()),
           "agent_right_partner_wrong": float((agent_ok & ~partner_ok).mean()),
           "partner_right_agent_wrong": float((partner_ok & ~agent_ok).mean()), "readings": []}
    for fit_half, score_half in ((inA, ~inA), (~inA, inA)):
        c, thr = pick(margin[fit_half], agent_ok[fit_half], partner_ok[fit_half])
        use, casc = apply(margin[score_half], thr, agent_ok[score_half], partner_ok[score_half])
        lo, hi = boot_diff(casc, partner_ok[score_half], convo[score_half], a.draws)
        out["readings"].append({"coverage_fit": c, "threshold": float(thr), "n_score": int(score_half.sum()),
                                "coverage_score": float(use.mean()), "cascade_acc": float(casc.mean()),
                                "partner_acc": float(partner_ok[score_half].mean()),
                                "diff": float(casc.mean() - partner_ok[score_half].mean()), "ci95": [lo, hi]})
    out["claim_holds"] = all(r["diff"] >= -0.005 and r["ci95"][0] > -0.015 for r in out["readings"])
    order = np.argsort(-margin, kind="stable")
    curve = []
    for c in range(0, 101, 5):
        k = int(round(c / 100 * len(sel)))
        use = np.zeros(len(sel), bool)
        use[order[:k]] = True
        curve.append({"coverage": c, "acc": float(np.where(use, agent_ok, partner_ok).mean()),
                      "agent_acc_on_covered": float(agent_ok[use].mean()) if k else None})
    out["curve_all_dev"] = curve
    with open(a.out, "w") as h:
        json.dump(out, h, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "curve_all_dev"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
