"""How readable the agent's dev decisions are (descriptive, no rule; dev rows only; CPU).

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python -I readability_dev.py --dataset abcd \
           --model runs/dry/agent_abcd_s1/tm_model.npz --result runs/dry/agent_abcd_s1/result.json \
           --out runs/readability/abcd_s1.json [--examples 3]
Definitions (those of decision_layer's e87 plan, TM part only):
- a clause fires on x when all its included literals hold; class sums S = the summed weights of the firing clauses;
  decision c = argmax S, runner-up r = argmax over the other classes (lowest index on ties, as np.argmax);
- receipt: the firing clauses with d_k = w[k, c] - w[k, r] > 0, ranked by d_k (descending; ties: lower clause index);
  its size is the shortest prefix (at most 20 clauses) whose own sums give argmax c; no receipt within 20 is recorded;
- reported: firing clauses per decision, literals per clause, receipt size in clauses and literals, the share of
  decisions with a receipt of 1, at most 3, 5, 10 and 20 clauses, and example receipts printed with the bit names.
Checks first: the CPU class sums give the run's dev accuracy exactly (result.json arms.tm.dev_acc).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = f"{_REPO_ROOT}/studies/decision_layer"
sys.path.insert(0, ROOT + "/src")
PARTS = ("mem", "words", "xcust", "xrecent")


def receipts(model, X, max_k=20, budget=8_000_000):
    """Per row: decision, runner-up, firing count, receipt clause indices (kept-clause numbering) or None."""
    from dl import cpu_tm_fast
    W = model["w"]
    chunk = max(1, budget // max(1, model["n_literals"]))
    out = []
    for s in range(0, X.shape[0], chunk):
        F = cpu_tm_fast.fired(model, X[s:s + chunk]).astype(bool)
        for i in range(F.shape[0]):
            idx = np.flatnonzero(F[i])
            S = W[idx].sum(0)
            c = int(S.argmax())
            S2 = S.copy()
            S2[c] = np.iinfo(np.int64).min
            r = int(S2.argmax())
            d = W[idx, c] - W[idx, r]
            pos = idx[d > 0]
            dpos = d[d > 0]
            order = np.lexsort((pos, -dpos))
            ranked = pos[order][:max_k]
            rec = None
            P = np.zeros(W.shape[1], np.int64)
            for k, cl in enumerate(ranked, 1):
                P = P + W[cl]
                if int(P.argmax()) == c:
                    rec = ranked[:k]
                    break
            out.append({"c": c, "r": r, "n_fired": int(len(idx)), "receipt": rec,
                        "d": dict(zip(ranked.tolist(), dpos[order][:max_k].tolist()))})
    return out


def clause_words(model, k, names):
    lits = model["lits"][model["starts"][k]:model["starts"][k] + model["counts"][k]]
    f = model["n_features"]
    return [names[j] if j < f else "NOT " + names[j - f] for j in lits]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--result", help="the run's result.json (arms.tm.dev_acc) or a registered seedK.json (dev_acc)")
    ap.add_argument("--dev-acc", type=float, help="the expected dev accuracy, instead of --result")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    from dl import count_text, cpu_tm_fast
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples
    tr, blist = load_examples("train", a.dataset)
    dv, _ = load_examples("dev", a.dataset)
    ydv = np.array([r["y"] for r in dv])
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, {}, blist)
    Xdv = view.transform(dv, {})
    z = np.load(a.model)
    m = cpu_tm_fast.pack(z["ta"], z["w"])
    m["counts"] = np.diff(np.concatenate([m["starts"], [len(m["lits"])]]))
    assert len(view.names) == m["n_features"], (len(view.names), m["n_features"])
    R = receipts(m, Xdv)
    pred = np.array([x["c"] for x in R])
    acc = float((pred == ydv).mean())
    want = (a.dev_acc if a.dev_acc is not None else (lambda r: r["arms"]["tm"]["dev_acc"] if "arms" in r else r["dev_acc"])(json.load(open(a.result))))
    assert abs(acc - want) < 1e-12, (acc, want)
    sizes = np.array([len(x["receipt"]) if x["receipt"] is not None else 0 for x in R])
    lit_sizes = np.array([int(m["counts"][x["receipt"]].sum()) if x["receipt"] is not None else 0 for x in R])
    has = sizes > 0
    fired = np.array([x["n_fired"] for x in R])
    res = {"dataset": a.dataset, "model": a.model, "table": meta, "n_dev": len(dv), "dev_acc": acc,
           "dev_acc_equals_run": True, "n_clauses_kept": m["n_clauses_kept"], "n_features": m["n_features"],
           "literals_per_clause": {"median": float(np.median(m["counts"])), "p90": float(np.percentile(m["counts"], 90)),
                                   "max": int(m["counts"].max())},
           "fired_per_decision": {"median": float(np.median(fired)), "p10": float(np.percentile(fired, 10)),
                                  "p90": float(np.percentile(fired, 90))},
           "receipt_found_within_20": float(has.mean()),
           "receipt_share": {f"<= {k} clauses": float(((sizes >= 1) & (sizes <= k)).mean()) for k in (1, 3, 5, 10, 20)},
           "receipt_literals_median": float(np.median(lit_sizes[has])) if has.any() else None,
           "receipt_literals_p90": float(np.percentile(lit_sizes[has], 90)) if has.any() else None,
           "examples": []}
    rng = np.random.default_rng(0)
    ok = np.flatnonzero((pred == ydv) & has & (sizes <= 3))
    for i in sorted(rng.choice(ok, size=min(a.examples, len(ok)), replace=False).tolist()):
        x = R[i]
        res["examples"].append({"row": int(i), "decision": blist[x["c"]], "runner_up": blist[x["r"]],
                                "n_fired": x["n_fired"],
                                "receipt": [{"clause": int(k), "weight_margin": int(x["d"][int(k)]),
                                             "literals": clause_words(m, int(k), view.names)} for k in x["receipt"]]})
    res["seconds"] = round(time.time() - t0, 1)
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
    for e in res["examples"]:
        print(f"row {e['row']}: {e['decision']} (runner-up {e['runner_up']}; {e['n_fired']} clauses fired)")
        for c in e["receipt"]:
            lits = c["literals"]
            shown = "; ".join(lits[:8]) + (f"; ... ({len(lits)} literals)" if len(lits) > 8 else "")
            print(f"   +{c['weight_margin']} IF {shown}")


if __name__ == "__main__":
    main()
