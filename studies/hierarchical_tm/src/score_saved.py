"""Score a saved model without retraining (models pickled by htm_driver.py --save-model / --save-at).

Usage: python score_saved.py RUN_DIR MODEL_FILE dev
       python score_saved.py RUN_DIR MODEL_FILE test --final-test   (the one final test, gates.md; use it once)
The model is rebuilt with the run's CONFIG (htm_driver.build), the package's load() restores its state, and the
split is scored like the driver does (same dev subset, same partner-shuffle).

On dev, this script writes the full output set of a normal run AT THE MODEL'S PASS P, into $TMLM_RUN_DIR
(or --out), so a model saved at pass P counts as a complete run stopped at P (gates.md, pass-count addendum):
  - SCORE line: acc, AUC, tuned accuracy, pred_pos, and partner-shuffle acc / AUC / pred_pos, all at P;
  - the source run's per-pass history truncated at P, the best pass among passes <= P, and the stability window
    P-4..P (Check 3: accuracy >= 0.58 and pred_pos in [0.2, 0.8] at each pass);
  - rules.txt and include_bits.npz from the P model;
  - a check that the re-scored dev AUC equals the source run's own dev AUC at pass P.
Note: when P is chosen on this seed's own dev curve, these dev numbers are slightly optimistic (selection effect).

Package pitfall: MultiClassTsetlinMachine.load() rebuilds each class machine WITHOUT binary_inference,
and_group_normalization, constant_update_p and no_clipping, so a model trained with any of them on would be scored
wrongly. This script refuses such models.
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htm_driver as hd  # noqa: E402


DERIVED = {"best_pass_by_dev_auc", "window_passes", "window_min_acc", "window_pred_pos", "window_stable",
           "rescore_equals_in_run", "selection_note", "in_run_dev_auc_at_P", "bootstrap"}


def article_bootstrap(score, pred, ps_score, ps_pred, y, art, n=1000, seed=0):
    """gates.md 'Article bootstrap': resample whole articles with all their pairs, with replacement, 1,000 times;
    95% percentile intervals. A pair belongs to the article of its first paragraph (art_a). Tuned accuracy is
    re-tuned on each resample, as its definition says (best cut-off on the pairs scored)."""
    uniq, inv = np.unique(art, return_inverse=True)
    rows = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng, stats = np.random.default_rng(seed), []
    for _ in range(n):
        i = np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])
        stats.append((hd.tuned_acc(score[i], y[i]), hd.auc(score[i], y[i]), (pred[i] == y[i]).mean(),
                      (ps_pred[i] == y[i]).mean(), hd.auc(ps_score[i], y[i])))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    names = ("tuned_acc", "auc", "acc", "partner_shuffle_acc", "partner_shuffle_auc")
    return {"articles": len(uniq), "resamples": n, "seed": seed, "pair_owner": "art_a",
            **{f"{k}_ci95": [round(float(l), 6), round(float(h), 6)] for k, l, h in zip(names, lo, hi)}}


def write_metrics(out_dir, score, hist):
    """metrics.json in the run_logged.py format: {name: {"value": ..., "label": measured|derived}}."""
    m = {k: {"value": v, "label": "derived" if k in DERIVED else "measured"} for k, v in score.items()}
    if hist is not None:
        m["history_to_P"] = {"value": hist, "label": "measured"}
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("model", help="file name inside RUN_DIR, e.g. model_final.pkl or model_p150.pkl")
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--final-test", action="store_true", help="required to score the untouched final-test split")
    ap.add_argument("--out", default=".", help="output dir when not run under run_logged.py")
    ap.add_argument("--max-minutes", type=float, default=0, help="accepted from the queue runner; scoring takes seconds")
    x = ap.parse_args()
    if x.split == "test" and not x.final_test:
        raise SystemExit("the final-test split is used once (gates.md); pass --final-test only for that")
    log = open(os.path.join(x.run_dir, "log.txt")).read().splitlines()
    cfg = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG "))
    a = argparse.Namespace(**cfg)
    if a.label_mode != "pair":
        raise SystemExit("only real pair labels are supported")
    for sw in ("binary_inference", "and_group_normalization", "constant_update_p", "no_clipping"):
        if getattr(a, sw):
            raise SystemExit(f"refusing: {sw} is on, and the package's load() drops it")
    hist_all = [json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("EPOCH ")]
    m = re.match(r"model_p(\d+)\.pkl$", x.model)
    P = int(m.group(1)) if m else hist_all[-1]["pass"]
    out_dir = os.environ.get("TMLM_RUN_DIR", x.out)
    os.makedirs(out_dir, exist_ok=True)

    tm, _ = hd.build(a)
    tm.load(pickle.load(open(os.path.join(x.run_dir, x.model), "rb")))
    X, y, vocab, f = hd.load(a, x.split)
    if x.split == "dev":  # the driver's dev subset
        keep = np.random.default_rng(1).permutation(len(y))[:a.n_dev or None]
        X, y = X[keep], y[keep]
    r = hd.evaluate(tm, X, y)
    perm = np.random.default_rng(a.seed + 7).permutation(len(y))
    ps = hd.evaluate(tm, np.hstack([X[:, :a.F], X[perm, a.F:]]), y)
    score = {"run": os.path.basename(x.run_dir), "model": x.model, "pass": P, "split": x.split, "n": len(y),
             "data_file": f, **r, "partner_shuffle_acc": ps["acc"], "partner_shuffle_auc": ps["auc"],
             "partner_shuffle_pred_pos": ps["pred_pos"]}
    if not a.data.endswith(".npz"):  # Wikipedia pairs: per-pair scores and the article bootstrap (Check 3, Check 4b)
        cs, cps = hd.class_sums(tm, X), hd.class_sums(tm, np.hstack([X[:, :a.F], X[perm, a.F:]]))
        sc, pr, pss, psp = cs[1] - cs[0], np.argmax(cs, 0), cps[1] - cps[0], np.argmax(cps, 0)
        d = np.load(f)
        art = d["art_a"][keep] if x.split == "dev" else d["art_a"]
        np.savez_compressed(os.path.join(out_dir, f"scores_{x.split}.npz"), score=sc, pred=pr, ps_score=pss,
                            ps_pred=psp, y=y, art_a=art, par_a=d["par_a"][keep] if x.split == "dev" else d["par_a"])
        score["bootstrap"] = article_bootstrap(sc, pr, pss, psp, y, art)
    if x.split == "dev":
        hist = [h for h in hist_all if h["pass"] <= P]
        win = [h for h in hist if h["pass"] > P - 5]
        at_p = next((h for h in hist if h["pass"] == P), None)
        score.update({
            "best_pass_by_dev_auc": hist[int(np.nanargmax([h["dev_auc"] for h in hist]))]["pass"] if hist else None,
            "window_passes": [h["pass"] for h in win],
            "window_min_acc": min(h["dev_acc"] for h in win) if win else None,
            "window_pred_pos": [h["pred_pos"] for h in win],
            "window_stable": bool(len(win) == 5 and all(h["dev_acc"] >= 0.58 and 0.2 <= h["pred_pos"] <= 0.8 for h in win)),
            "in_run_dev_auc_at_P": at_p["dev_auc"] if at_p else None,
            "rescore_equals_in_run": bool(at_p and at_p["dev_auc"] == r["auc"] and at_p["dev_acc"] == r["acc"]),
            "selection_note": "P chosen on this seed's dev curve: dev numbers at P are slightly optimistic" if m else "",
        })
        inc1 = hd.include_bits(tm.tms[1])
        Xdg, ydg = X[:a.n_diag], y[:a.n_diag]
        fire = (hd.or_sums(inc1, Xdg, a.F, a.A, a.K) > 0).all(axis=(2, 3))
        text = "\n".join(hd.rules(inc1, fire, ydg, vocab, a.A, a.K, a.top))
        open(os.path.join(out_dir, "rules.txt"), "w").write(text + "\n")
        np.savez_compressed(os.path.join(out_dir, "include_bits.npz"), class1=np.packbits(inc1, axis=-1),
                            class0=np.packbits(hd.include_bits(tm.tms[0]), axis=-1), F=a.F, A=a.A, K=a.K,
                            model=a.model, clauses=a.clauses, T=a.T, s=a.s, vocab=np.array(vocab), pass_P=P)
        write_metrics(out_dir, score, hist)
    else:
        write_metrics(out_dir, score, None)
    print("SCORE", json.dumps(score))


if __name__ == "__main__":
    main()
