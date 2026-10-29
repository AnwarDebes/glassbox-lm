"""CPU time of one decision of a saved readable agent (one thread, batch 1), as scripts/decision_speed_c4.py measures it.

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -I agent_speed.py --dataset abcd --model DIR/tm_model.npz \
           --result DIR/result.json --out DIR/speed.json [--n 1000]
Checks before timing: the fast CPU decision gives the same argmax as the reference CPU class sums on every dev row,
and the resulting dev accuracy equals the run's (result.json arms.tm.dev_acc). Dev rows only; rough timing on a
shared machine (read the order of magnitude).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
import sys
import time

import numpy as np

ROOT = f"{_REPO_ROOT}/studies/decision_layer"
sys.path.insert(0, ROOT + "/src")
PARTS = ("mem", "words", "xcust", "xrecent")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--result", help="the run's result.json (arms.tm.dev_acc) or a registered seedK.json (dev_acc)")
    ap.add_argument("--dev-acc", type=float, help="the expected dev accuracy, instead of --result")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from dl import count_text, cpu_tm, cpu_tm_fast
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", a.dataset)
    dv, _ = load_examples("dev", a.dataset)
    ydv = np.array([r["y"] for r in dv])
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, {}, blist)
    Xdv = view.transform(dv, {})
    z = np.load(a.model)
    m0, m1 = cpu_tm.pack(z["ta"], z["w"]), cpu_tm_fast.pack(z["ta"], z["w"])
    S0 = cpu_tm.class_sums(m0, Xdv)
    fast = np.array([cpu_tm_fast.decide(m1, Xdv[i]) for i in range(len(dv))])
    assert np.array_equal(S0.argmax(1), fast), "fast and reference CPU decisions differ"
    acc = float((fast == ydv).mean())
    want = (a.dev_acc if a.dev_acc is not None else (lambda r: r["arms"]["tm"]["dev_acc"] if "arms" in r else r["dev_acc"])(json.load(open(a.result))))
    idx = np.random.default_rng(0).choice(len(dv), size=min(a.n, len(dv)), replace=False)
    tv, tt = [], []
    for warm in (True, False):
        for i in (idx[:20] if warm else idx):
            t0 = time.perf_counter()
            x = view.transform([dv[i]], {})
            t1 = time.perf_counter()
            cpu_tm_fast.decide(m1, x[0])
            t2 = time.perf_counter()
            if not warm:
                tv.append((t1 - t0) * 1000)
                tt.append((t2 - t1) * 1000)
    tot = np.array(tv) + np.array(tt)
    res = {"dataset": a.dataset, "model": a.model, "table": meta, "n": len(idx), "cpu_dev_acc": acc,
           "run_dev_acc": want, "acc_equal": abs(acc - want) < 1e-12, "n_clauses_kept": m1["n_clauses_kept"],
           "n_literals": m1["n_literals"], "view_median_ms": float(np.median(tv)), "tm_median_ms": float(np.median(tt)),
           "total_median_ms": float(np.median(tot)), "total_p90_ms": float(np.percentile(tot, 90)),
           "label": "rough: shared machine, one CPU thread, batch 1",
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
