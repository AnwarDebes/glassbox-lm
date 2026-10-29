"""Independent recompute of REG_agent_vs_llm's statistic (own code, own random generator).

usage: python -I recompute_reg.py --dataset star --split test --n 4000 --llm PREDS.jsonl --agent SEED1.npz \
           [--agent-seeds S1.npz ... S5.npz] --ft FT.npz [--tfidf TFIDF.npz] [--rows-file runs/reg_rows.json] \
           [--seed 20261028] [--draws 10000] --out OUT.json

Same data and rule definitions as reg_report.py; a different implementation: labels come from the split's example
file (each LLM row's own gold string is checked against it), the registered row draw is rebuilt from its seed, and
the bootstrap over dialogues uses multinomial dialogue weights from numpy's PCG64 with its own seed (reg_report.py
draws dialogue indices with default_rng(20261027); verify_reg.py uses the standard library).
Rules: NI lower 95% end of AGENT-LLM above -0.03; NI2 above -0.02; SUP mean >= +0.03 and lower end above 0;
FT lower 95% end of FT-AGENT above 0; no rule: 90% interval inside (-0.03, +0.03).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import argparse
import gzip
import hashlib
import json
import os
import random

import numpy as np

DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
REG_SEED = 20261027


def sha(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def load_decisions(path):
    z = np.load(path)
    if "pred" in z.files:
        return np.asarray(z["pred"]).astype(np.int64), np.asarray(z["y"]).astype(np.int64)
    return np.asarray(z["top_ids"])[:, 0].astype(np.int64), np.asarray(z["gold"]).astype(np.int64)


class DialogueBoot:
    """Bootstrap over dialogues with multinomial weights: mean = sum_c w_c S_c / sum_c w_c n_c."""

    def __init__(self, convo, seed, draws, chunk=500):
        ids, inv = np.unique(convo, return_inverse=True)
        self.inv, self.d = inv, len(ids)
        self.n = np.bincount(inv, minlength=self.d).astype(float)
        rng = np.random.Generator(np.random.PCG64(seed))
        p = np.full(self.d, 1.0 / self.d)
        self.w = [rng.multinomial(self.d, p, size=min(chunk, draws - k)).astype(float) for k in range(0, draws, chunk)]
        self.seed_rng = np.random.Generator(np.random.PCG64(seed + 1))

    def sums(self, x):
        return np.bincount(self.inv, weights=x, minlength=self.d)

    def means(self, x):
        s = self.sums(x)
        return np.concatenate([(w @ s) / (w @ self.n) for w in self.w])

    def two_level(self, seed_ok, other, agent_first=True):
        """Draw seeds with replacement, then the same dialogue weights; the agent side is the drawn seeds' mean."""
        k = seed_ok.shape[0]
        s_seed = np.stack([self.sums(r) for r in seed_ok])  # k x d
        s_other = self.sums(other)
        out = []
        for w in self.w:
            m = self.seed_rng.multinomial(k, np.full(k, 1.0 / k), size=w.shape[0]).astype(float) / k  # draws x k
            den = w @ self.n
            a = ((w @ s_seed.T) * m).sum(1) / den  # the drawn seeds' mean, under the same dialogue weights
            o = (w @ s_other) / den
            out.append(a - o if agent_first else o - a)
        return np.concatenate(out)


def ci(draws, level):
    lo = (100 - level) / 2
    return [float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["star", "abcd", "multiwoz"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-seeds", nargs="*", default=[])
    ap.add_argument("--ft", required=True)
    ap.add_argument("--tfidf")
    ap.add_argument("--rows-file")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with gzip.open(os.path.join(DATA, a.dataset, f"examples_{a.split}.json.gz"), "rt") as h:
        d = json.load(h)
    buttons = d["buttons"]
    y = np.array([r["y"] for r in d["rows"]], dtype=np.int64)
    convo = np.array([str(r["convo_id"]) for r in d["rows"]])
    recs = [json.loads(x) for x in open(a.llm) if x.strip()]
    recs.sort(key=lambda r: r["i"])
    rows = np.array([r["i"] for r in recs], dtype=np.int64)
    checks = {"rows_unique": len(set(rows.tolist())) == len(rows),
              "llm_errors": sum(r.get("error") is not None for r in recs),
              "llm_gold_matches_split": all(r["gold"] == buttons[y[r["i"]]] for r in recs)}
    if a.split == "test":
        draw = sorted(random.Random(REG_SEED).sample(range(len(y)), min(a.n, len(y))))
        checks["rows_are_rebuilt_draw"] = rows.tolist() == draw
        if a.rows_file:  # runs/reg_rows.json: the registered row count and test split size per dataset
            rf = json.load(open(a.rows_file))["datasets"][a.dataset]
            checks["rows_file_n"] = rf["rows"] == a.n == len(rows)
            checks["rows_file_split_size"] = rf["n_test_split"] == len(y)
    name_to_id = {b: i for i, b in enumerate(buttons)}
    llm_dec = np.array([name_to_id.get(r["pred"], -1) if r["pred"] is not None else -1 for r in recs])
    yy, cc = y[rows], convo[rows]
    ok = {"llm": (llm_dec == yy).astype(float)}
    files = {"llm": a.llm, "agent": a.agent, "ft": a.ft}
    if a.tfidf:
        files["tfidf"] = a.tfidf
    for k in ("agent", "ft", "tfidf"):
        if k not in files:
            continue
        dec, gold = load_decisions(files[k])
        checks[f"{k}_gold_matches_split"] = bool(len(gold) == len(y) and np.array_equal(gold, y))
        ok[k] = (dec[rows] == yy).astype(float)
    boot = DialogueBoot(cc, a.seed, a.draws)
    res = {"dataset": a.dataset, "split": a.split, "n_rows": int(len(rows)), "n_dialogues": boot.d,
           "seed": a.seed, "draws": a.draws, "checks": checks,
           "acc": {k: float(v.mean()) for k, v in ok.items()},
           "llm_invalid": int((llm_dec < 0).sum()),
           "files": {k: sha(v) for k, v in files.items()}}
    pairs = {"agent_minus_llm": ok["agent"] - ok["llm"], "ft_minus_agent": ok["ft"] - ok["agent"]}
    if a.tfidf:
        pairs["tfidf_minus_llm"] = ok["tfidf"] - ok["llm"]
        pairs["agent_minus_tfidf"] = ok["agent"] - ok["tfidf"]
    for k, v in pairs.items():
        b = boot.means(v)
        res[k] = {"mean": float(v.mean()), "ci95": ci(b, 95), "ci90": ci(b, 90)}
    d1, d2 = res["agent_minus_llm"], res["ft_minus_agent"]
    res["rules"] = {"NI_met": d1["ci95"][0] > -0.03, "NI2_met": d1["ci95"][0] > -0.02,
                    "SUP_met": d1["mean"] >= 0.03 and d1["ci95"][0] > 0, "FT_met": d2["ci95"][0] > 0,
                    "EQ_inside_no_rule": d1["ci90"][0] > -0.03 and d1["ci90"][1] < 0.03}
    if a.agent_seeds:
        seed_ok = []
        for p in a.agent_seeds:
            dec, gold = load_decisions(p)
            assert np.array_equal(gold, y), f"{p}: rows not in split order"
            seed_ok.append((dec[rows] == yy).astype(float))
        seed_ok = np.stack(seed_ok)
        res["two_level"] = {"agent_acc_per_seed": [float(x) for x in seed_ok.mean(1)],
                            "agent_minus_llm_mean": float(seed_ok.mean() - ok["llm"].mean()),
                            "ci_agent_minus_llm": ci(boot.two_level(seed_ok, ok["llm"]), 95),
                            "ci_ft_minus_agent": ci(boot.two_level(seed_ok, ok["ft"], agent_first=False), 95)}
        if a.tfidf:
            res["two_level"]["ci_agent_minus_tfidf"] = ci(boot.two_level(seed_ok, ok["tfidf"]), 95)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps({k: res[k] for k in ("dataset", "split", "n_rows", "acc", "checks")}))


if __name__ == "__main__":
    main()
