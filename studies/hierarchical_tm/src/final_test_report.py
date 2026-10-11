"""Main table of the one final test, and its dry run on dev (the frozen settings in docs/registrations.md: "Baselines on
the same test pairs", "Pre-registered comparison", "Pre-registered 5-seed reporting").

  final_test_report.py --split dev --group sa sa RUN... --group np np RUN... --group ns_all ns_all RUN...
                       --shuffled ns_all RUN... [--reference THURSDAY/checks.json ...] [--hashes FILE]
  final_test_report.py --split test --final-test --best-from DEVRUN/dev_report.json --group ... --shuffled ...

--group NAME DATA RUN [RUN ...]: DATA is sa, np or ns_all. Each RUN is a score_saved.py output dir (scores_{split}.npz
  and a SCORE line in log.txt); the settings and n_dev come from the CONFIG of the training run it re-scored.
--shuffled NAME RUN [RUN ...]: shuffled-label runs of group NAME, the no-signal check. On dev only, a RUN may also be a
  training-run dir without scores: then its in-run dev AUC at the group's pass P is read from its log (no CI).
Without --final-test the test split is refused, and np.load refuses any file named test.npz in this process.

Pairs: pairdata.load_pairs(DATA, split, 1024), put in the scores files' row order (dev: the driver's subset
np.random.default_rng(1).permutation(n)[:n_dev]; test: file order). It refuses unless y, art_a and par_a equal each
scores file's exactly, and unless all runs share one setting and one pass P.
Baselines: data_checks.py's functions, unchanged, fitted on train only, F = 1,024: jaccard, tfidf_cosine, shared_count
and ppmi_svd_cosine (the similarity ranks the pairs), lr_A_AND_B and lr_A_B_AandB (lr_baseline's decision_function).
idf2 and the PPMI vectors W come from train paragraphs, built as in data_checks.main().
Best baseline per data set = the highest dev AUC. It is chosen on dev and written to dev_report.json; on test it is read
from --best-from, never chosen on test.
AUC = htm_driver.auc (ties count 1/2); tuned accuracy = htm_driver.tuned_acc (re-tuned in each resample).
Article bootstrap: 1,000 resamples of whole articles (a pair belongs to art_a), numpy seed 0, drawn exactly as in
score_saved.article_bootstrap, so each seed's CIs equal its SCORE line's. The SAME resamples serve every seed,
baseline and shuffled run of a group. 95% percentile intervals.
Paired comparison with baseline X: in each resample, the mean over seeds of [AUC(seed) - AUC(X)]. "beats X" if the
interval's lower end is above 0, "below X" if its upper end is below 0, otherwise "not distinguishable". It is run
against jaccard and against the best-on-dev baseline, with each seed's own paired interval. On the synonym data
(ns, ns_all) shared words were removed, so word overlap is 0.50 by construction: it gets no verdict ("not a comparison
(by construction)"; the script checks that no evaluated pair shares a word), and the comparisons are the best-on-dev
baseline and lr_A_B_AandB, the word-matching classifier, helpless without shared words.
Writes report.json, metrics.json (run_logged format) and, on dev, dev_report.json to --out, else $TMLM_RUN_DIR.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import data_checks as dc  # noqa: E402  (the baseline definitions of Thursday's dev table)
import htm_driver as hd  # noqa: E402  (auc and tuned_acc, as in every TM score)
from pairdata import DATA, load_pairs  # noqa: E402

BASELINES = ("jaccard", "tfidf_cosine", "shared_count", "ppmi_svd_cosine", "lr_A_AND_B", "lr_A_B_AandB")
N_RESAMPLES, RESAMPLE_SEED = 1000, 0  # the same as score_saved.article_bootstrap
SETTINGS = ("model", "F", "clauses", "T", "s", "A", "K", "batch", "n_train", "label_mode", "binary_inference",
            "and_group_normalization", "constant_update_p", "no_clipping")
CODE = ("final_test_report.py", "data_checks.py", "htm_driver.py", "pairdata.py")
NO_SHARED_WORDS = ("ns", "ns_all")  # synonym data: shared words removed, so word overlap is 0.50 by construction
NOT_A_COMPARISON = "not a comparison (by construction)"


def get_args():
    ap = argparse.ArgumentParser(description="the final test's main table (see the module docstring)")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--final-test", action="store_true", help="required for --split test: the one final test")
    ap.add_argument("--group", nargs="+", action="append", required=True, help="NAME DATA RUN [RUN ...]")
    ap.add_argument("--shuffled", nargs="+", action="append", default=[], help="NAME RUN [RUN ...]")
    ap.add_argument("--best-from", help="the dev run's dev_report.json (required on test)")
    ap.add_argument("--reference", nargs="+", default=[], help="dev only: Thursday's checks.json files to reproduce")
    ap.add_argument("--hashes", help="frozen_settings_hashes.json: refuse if a file read here differs from it")
    ap.add_argument("--out", help="output dir (default: $TMLM_RUN_DIR, else .)")
    a = ap.parse_args()
    if a.split == "test" and not a.final_test:
        raise SystemExit("refusing: the test split is the one final test (gates.md); pass --final-test only for that")
    if a.split == "dev" and a.final_test:
        raise SystemExit("refusing: --final-test goes with --split test only")
    if a.split == "test" and (not a.best_from or a.reference):
        raise SystemExit("refusing: on test the best baseline comes from the dev run "
                         "(--best-from DEVRUN/dev_report.json), and --reference is not used")
    if any(len(g) < 3 for g in a.group) or any(len(s) < 2 for s in a.shuffled):
        raise SystemExit("usage: --group NAME DATA RUN [RUN ...] and --shuffled NAME RUN [RUN ...]")
    return a


def refuse_test_files():
    """Belt and braces: without --final-test, no file named test.npz can be opened through np.load here."""
    load = np.load

    def guarded(file, *args, **kw):
        if os.path.basename(str(file)) == "test.npz":
            raise SystemExit(f"refusing to open {file} without --final-test")
        return load(file, *args, **kw)
    np.load = guarded


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def log_json(run, tag):
    """The JSON of the first 'TAG {...}' line in RUN/log.txt."""
    with open(os.path.join(run, "log.txt")) as f:
        for ln in f:
            if ln.startswith(tag + " "):
                return json.loads(ln.split(" ", 1)[1])
    raise SystemExit(f"{run}: no {tag} line in log.txt")


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def load_run(run, split, data, shuffled):
    """A score_saved.py output dir: its SCORE line, per-pair scores and the source run's CONFIG. On dev, a
    shuffled-label training-run dir without scores is accepted too: then only its in-run dev history is read."""
    run = os.path.abspath(run)
    f = os.path.join(run, f"scores_{split}.npz")
    if os.path.exists(f):
        sc = log_json(run, "SCORE")
        cfg = log_json(os.path.join(os.path.dirname(run), sc["run"]), "CONFIG")
        if sc["split"] != split or os.path.normpath(sc["data_file"]) != os.path.join(DATA, data, f"{split}.npz"):
            raise SystemExit(f"refusing: {run} scored {sc['data_file']} ({sc['split']}), not {data}/{split}")
        z = np.load(f)
        r = {"run": os.path.basename(run), "source_run": sc["run"], "sc": sc, "P": sc["pass"], "score": z["score"],
             "y": z["y"], "art_a": z["art_a"], "par_a": z["par_a"], "scores_sha256": sha256(f)}
    elif split == "dev" and shuffled:
        cfg = log_json(run, "CONFIG")
        with open(os.path.join(run, "log.txt")) as fh:
            hist = [json.loads(ln.split(" ", 1)[1]) for ln in fh if ln.startswith("EPOCH ")]
        r = {"run": os.path.basename(run), "source_run": os.path.basename(run), "score": None, "hist": hist}
    else:
        raise SystemExit(f"refusing: {run} has no scores_{split}.npz (a score_saved.py output dir is needed)")
    if cfg["data"] != data or bool(cfg["shuffle_labels"]) != shuffled or cfg["label_mode"] != "pair":
        raise SystemExit(f"refusing: {run}: data {cfg['data']}, shuffle_labels {cfg['shuffle_labels']}, "
                         f"label_mode {cfg['label_mode']}; expected {data}, {shuffled}, pair")
    r.update(seed=cfg["seed"], n_dev=cfg["n_dev"], settings={k: cfg[k] for k in SETTINGS})
    return r


def lr_baseline_scores(X_tr, y_tr, X_ev, y_ev):
    """data_checks.lr_baseline, unchanged; its decision_function values are captured where it hands them to report()."""
    got, report = [], dc.report

    def capture(pred, score, y):
        got.append(score)
        return report(pred, score, y)
    dc.report = capture
    try:
        out = dc.lr_baseline(X_tr, y_tr, X_ev, y_ev)
    finally:
        dc.report = report
    return got[0], out


def resamples(art):
    """N_RESAMPLES resamples of whole articles, drawn exactly as score_saved.article_bootstrap draws them."""
    uniq, inv = np.unique(art, return_inverse=True)
    rows = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(RESAMPLE_SEED)
    for _ in range(N_RESAMPLES):
        yield np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])


def run_group(name, data, main, shuf, a, idf2, W, best_file):
    ev, tr = load_pairs(data, a.split, dc.F), load_pairs(data, "train", dc.F)
    n, P = len(ev["y"]), main[0]["P"]

    def rows(r):  # the scores file's row order (score_saved.py)
        return np.random.default_rng(1).permutation(n)[:r["n_dev"] or None] if a.split == "dev" else np.arange(n)
    order = rows(main[0])
    for r in main + shuf:
        if not np.array_equal(rows(r), order):
            raise SystemExit(f"refusing: {r['run']} scored other {a.split} pairs than {main[0]['run']}")
        for k in ("y", "art_a", "par_a") if r["score"] is not None else ():
            if not np.array_equal(r[k], ev[k][order]):
                raise SystemExit(f"refusing: {r['run']}: {k} differs from the {data} {a.split} pairs in its row order")
    y = ev["y"][order]

    # baselines, fitted on train only (on the whole split in file order, as Thursday; then put in the TM's row order)
    s_tr, s_ev = dc.scores(tr, idf2, W), dc.scores(ev, idf2, W)
    cut = {k: dc.threshold_baseline(s_tr[k], tr["y"], s_ev[k], ev["y"]) for k in s_tr}
    for k, kind in (("lr_A_AND_B", "AND"), ("lr_A_B_AandB", "ALL")):
        s_ev[k], cut[k] = lr_baseline_scores(dc.feats(tr, kind), tr["y"], dc.feats(ev, kind), ev["y"])

    # one set of article resamples for every seed, baseline and shuffled run of this group
    items = {f"seed{r['seed']}": r["score"] for r in main}
    items.update({k: s_ev[k][order] for k in BASELINES})
    items.update({f"shuffled_seed{r['seed']}": r["score"] for r in shuf if r["score"] is not None})
    items["self_check"] = main[0]["score"].copy()  # the first seed again, through the baseline path
    keys = list(items)
    B, T = np.empty((N_RESAMPLES, len(keys))), np.empty((N_RESAMPLES, len(main)))
    for b, i in enumerate(resamples(ev["art_a"][order])):
        B[b] = [hd.auc(items[k][i], y[i]) for k in keys]
        T[b] = [hd.tuned_acc(r["score"][i], y[i]) for r in main]
    col = {k: B[:, j] for j, k in enumerate(keys)}
    auc = {k: hd.auc(items[k], y) for k in keys}

    seeds = []
    for j, r in enumerate(main):
        k, sc, t = f"seed{r['seed']}", r["sc"], hd.tuned_acc(r["score"], y)
        c_auc, c_t = ci(col[k]), ci(T[:, j])
        seeds.append({"seed": r["seed"], "run": r["run"], "source_run": r["source_run"], "auc": auc[k],
                      "auc_ci95": c_auc, "tuned_acc": t, "tuned_acc_ci95": c_t,
                      "equals_score_line": auc[k] == sc["auc"] and t == sc["tuned_acc"],
                      "ci_equals_score_line": [round(x, 6) for x in c_auc + c_t] ==
                      sc["bootstrap"]["auc_ci95"] + sc["bootstrap"]["tuned_acc_ci95"]})
    v, u = np.array([s["auc"] for s in seeds]), np.array([s["tuned_acc"] for s in seeds])
    sd = (lambda x: float(x.std(ddof=1))) if len(v) > 1 else (lambda x: float("nan"))
    summary = {"n_seeds": len(v), "auc_mean": float(v.mean()), "auc_sd": sd(v), "tuned_acc_mean": float(u.mean()),
               "tuned_acc_sd": sd(u), "sd": "sample SD (ddof 1) over seeds"}
    base = {k: {"auc": auc[k], "auc_ci95": ci(col[k]), "tuned_acc": hd.tuned_acc(items[k], y),
                "train_cutoff_report": cut[k]} for k in BASELINES}

    if a.split == "dev":  # chosen here, on dev; with --best-from, the file must agree
        best, source = max(BASELINES, key=lambda k: auc[k]), "highest AUC on these dev pairs"
        if best_file:
            if best_file["best_baseline"][data] != best:
                raise SystemExit(f"refusing: {a.best_from} names {best_file['best_baseline'][data]}, dev gives {best}")
            source += f"; equals {a.best_from}"
    else:  # never chosen on test
        best, source = best_file["best_baseline"][data], f"read from {a.best_from}, chosen on dev"
    S = np.stack([col[f"seed{r['seed']}"] for r in main], 1)

    def paired(x, what):
        lo, hi = ci((S - col[x][:, None]).mean(1))
        return {"vs": x, "what": what, "ci95": [lo, hi],
                "mean_diff": float(np.mean([auc[f"seed{r['seed']}"] - auc[x] for r in main])),
                "verdict": f"beats {x}" if lo > 0 else f"below {x}" if hi < 0 else "not distinguishable",
                "per_seed": {f"seed{r['seed']}": {"diff": auc[f"seed{r['seed']}"] - auc[x],
                                                  "ci95": ci(S[:, j] - col[x])} for j, r in enumerate(main)}}
    if data in NO_SHARED_WORDS:  # word overlap is 0.50 by construction: no verdict against it (decision of 2 Oct 2026)
        if s_ev["shared_count"][order].any():
            raise SystemExit(f"refusing: {data} pairs share words, so word overlap is not 0.50 by construction")
        comparisons = {"word_overlap": {"vs": "jaccard", "what": "word overlap", "value": NOT_A_COMPARISON,
                                        "note": "by construction (shared words removed), not a comparison"},
                       "best_on_dev": paired(best, "best on dev"),
                       "word_matching_classifier": paired("lr_A_B_AandB",
                                                          "word-matching classifier, helpless without shared words")}
    else:
        comparisons = {"word_overlap": paired("jaccard", "word overlap"), "best_on_dev": paired(best, "best on dev")}
    shuffled = []
    for r in shuf:
        if r["score"] is not None:
            k = f"shuffled_seed{r['seed']}"
            shuffled.append({"seed": r["seed"], "run": r["run"], "auc": auc[k], "auc_ci95": ci(col[k]),
                             "equals_score_line": auc[k] == r["sc"]["auc"], "source": "per-pair scores"})
        else:
            h = next((h for h in r["hist"] if h["pass"] == P), None)
            if h is None:
                raise SystemExit(f"refusing: {r['run']} has no dev evaluation at pass {P}")
            shuffled.append({"seed": r["seed"], "run": r["run"], "auc": h["dev_auc"], "auc_ci95": None,
                             "source": f"in-run dev AUC at pass {P} from the training log; no per-pair scores, so no "
                                       "CI (the final test re-scores the saved model with score_saved.py)"})
    ref = None
    for f in a.reference:
        with open(f) as fh:
            res = json.load(fh)["results"]
        if data in res:  # Thursday's logged values: every field of every baseline must be identical
            ref = {"file": f, "differences": {k: {x: [cut[k].get(x), v] for x, v in res[data]["baselines"][k].items()
                                                  if cut[k].get(x) != v} for k in BASELINES}}
            ref["identical"] = not any(ref["differences"].values())
    return {"name": name, "data": data, "split": a.split, "n_pairs": len(y), "n_pos": int(y.sum()),
            "n_articles": int(len(np.unique(ev["art_a"][order]))), "pass": P, "settings": main[0]["settings"],
            "seeds": seeds, "summary": summary, "baselines": base, "best_on_dev": best, "best_source": source,
            "comparisons": comparisons,
            "self_check_ci95": ci(col[f"seed{main[0]['seed']}"] - col["self_check"]), "shuffled": shuffled,
            "thursday": ref,
            "scores_sha256": {r["run"]: r["scores_sha256"] for r in main + shuf if "scores_sha256" in r}}


def f4(x, c=None, sign=False):
    s = "{:+.4f}" if sign else "{:.4f}"
    return s.format(x) + (" [" + ", ".join(s.format(e) for e in c) + "]" if c else "")


def print_group(g):
    p, sm = print, g["summary"]
    p(f"\n=== {g['name']} (data {g['data']}, {g['split']} split): {g['n_pairs']} pairs ({g['n_pos']} positive) "
      f"in {g['n_articles']} articles; TM at pass {g['pass']}")
    p(f"{'TM seed':<9}{'AUC [95% CI]':<27}{'tuned acc [95% CI]':<27}run")
    for s in g["seeds"]:
        p(f"{s['seed']:<9}{f4(s['auc'], s['auc_ci95']):<27}{f4(s['tuned_acc'], s['tuned_acc_ci95']):<27}{s['run']}")
    p(f"mean (SD) over {sm['n_seeds']} seeds: AUC {sm['auc_mean']:.4f} ({sm['auc_sd']:.4f}), "
      f"tuned acc {sm['tuned_acc_mean']:.4f} ({sm['tuned_acc_sd']:.4f})")
    p(f"{'baseline (train only)':<22}{'AUC [95% CI]':<27}{'acc, train cut-off':<20}tuned acc (TM's rule)")
    for k, b in g["baselines"].items():
        p(f"{k:<22}{f4(b['auc'], b['auc_ci95']):<27}{b['train_cutoff_report']['acc']:<20.4f}{b['tuned_acc']:.4f}")
    p(f"best baseline on dev: {g['best_on_dev']} ({g['best_source']})")
    p("paired article bootstrap, mean over seeds of AUC(seed) - AUC(X):")
    for c in g["comparisons"].values():
        if "verdict" not in c:
            p(f"  vs {c['vs']}: {c['note']}")
            continue
        p(f"  vs {c['vs']} ({c['what']}): {f4(c['mean_diff'], c['ci95'], True)} -> {c['verdict']}")
        p("     each seed: " + "; ".join(f"{s} {f4(d['diff'], d['ci95'], True)}" for s, d in c["per_seed"].items()))
    for s in g["shuffled"]:
        p(f"shuffled-label seed {s['seed']}: AUC {f4(s['auc'], s['auc_ci95'])} ({s['source']}) {s['run']}")


def main():
    a = get_args()
    if not a.final_test:
        refuse_test_files()
    t0, out_dir = time.time(), a.out or os.environ.get("TMLM_RUN_DIR") or "."
    os.makedirs(out_dir, exist_ok=True)
    for f in CODE:
        shutil.copy(os.path.join(SRC, f), out_dir)
    root = os.path.dirname(SRC)
    files = [os.path.join(DATA, d, f"{s}.npz") for d in sorted({g[1] for g in a.group}) for s in ("train", a.split)]
    files += [os.path.join(DATA, f) for f in ("paragraphs_F3072.npz", "splits.json")]
    files += [os.path.join(SRC, f) for f in CODE]
    inputs = {os.path.relpath(f, root): sha256(f) for f in files}
    if a.hashes:
        with open(a.hashes) as fh:
            h = json.load(fh)
        listed = {**h.get("data_sha256", {}), **h.get("scripts_sha256", {})}
        bad = [f for f in inputs if f in listed and listed[f] != inputs[f]]
        if bad:
            raise SystemExit(f"refusing: these files differ from {a.hashes}: {bad}")
        print(f"hashes equal to {a.hashes}: {[f for f in inputs if f in listed]}; "
              f"not listed there: {[f for f in inputs if f not in listed]}")
    names = [g[0] for g in a.group]
    best_file = None
    if a.best_from:
        with open(a.best_from) as fh:
            best_file = json.load(fh)
        if best_file.get("split") != "dev" or any(best_file["best_baseline"].get(g[1]) not in BASELINES
                                                  for g in a.group):
            raise SystemExit(f"refusing: {a.best_from} is not a dev report with a best baseline for every group")
    shuf = {}
    for s in a.shuffled:
        if s[0] not in names:
            raise SystemExit(f"refusing: --shuffled {s[0]}: no such group")
        shuf.setdefault(s[0], []).extend(s[1:])
    if len(set(names)) < len(names):
        raise SystemExit("refusing: two groups have the same name")
    # every run is read and checked before any pairs are loaded, so a wrong run dir costs no read of the split
    runs = {}
    for name, data, *dirs in a.group:
        runs[name] = (data, sorted((load_run(r, a.split, data, False) for r in dirs), key=lambda r: r["seed"]),
                      sorted((load_run(r, a.split, data, True) for r in shuf.get(name, [])), key=lambda r: r["seed"]))
    first = runs[names[0]][1][0]
    for name, (data, m_runs, s_runs) in runs.items():
        for rs in (m_runs, s_runs):
            if len({r["seed"] for r in rs}) < len(rs):
                raise SystemExit(f"refusing: {name}: a seed appears twice")
        for r in m_runs + s_runs:
            if r["settings"] != first["settings"] or r.get("P", first["P"]) != first["P"]:
                raise SystemExit(f"refusing: {r['run']}: settings or pass differ from {first['run']}")

    P = dc.train_paragraphs()  # paragraphs of train articles only; idf2 and W exactly as in data_checks.main()
    df = np.asarray(P.sum(0)).ravel()
    idf2 = np.log(P.shape[0] / np.maximum(df, 1)) ** 2
    W = dc.ppmi_vectors(P)
    res = {name: run_group(name, data, m_runs, s_runs, a, idf2, W, best_file)
           for name, (data, m_runs, s_runs) in runs.items()}

    g0 = next(iter(res.values()))
    print(f"split {a.split}; settings of every run: {g0['settings']}, pass P {g0['pass']}; article bootstrap: "
          f"{N_RESAMPLES} resamples of whole articles, numpy seed {RESAMPLE_SEED}, the same for every seed and "
          "baseline")
    for g in res.values():
        print_group(g)
    checks = {
        "tm_equals_score_lines": all(s["equals_score_line"] for g in res.values() for s in g["seeds"]),
        "tm_ci_equals_score_lines": all(s["ci_equals_score_line"] for g in res.values() for s in g["seeds"]),
        "shuffled_equals_score_lines": all(s.get("equals_score_line", True)
                                           for g in res.values() for s in g["shuffled"]),
        "self_pair_interval_zero": all(g["self_check_ci95"] == [0.0, 0.0] for g in res.values()),
    }
    if a.reference:
        checks["baselines_equal_thursday"] = all(g["thursday"] and g["thursday"]["identical"] for g in res.values())
    print("\nCHECKS")
    for g in res.values():
        print(f"{g['name']}: TM AUC and tuned acc equal the SCORE lines: "
              f"{[s['equals_score_line'] for s in g['seeds']]}; "
              f"CIs equal them (6 decimals): {[s['ci_equals_score_line'] for s in g['seeds']]}; "
              f"seed {g['seeds'][0]['seed']} vs itself: {g['self_check_ci95']}")
        if g["thursday"]:
            print(f"{g['name']}: baselines vs Thursday ({g['thursday']['file']}): "
                  f"{'identical in every field' if g['thursday']['identical'] else g['thursday']['differences']}")
    print(json.dumps(checks))

    report = {"split": a.split, "argv": sys.argv, "n_resamples": N_RESAMPLES, "resample_seed": RESAMPLE_SEED,
              "inputs_sha256": inputs,
              "best_from": a.best_from and {"file": a.best_from, "sha256": sha256(a.best_from)},
              "checks": checks, "groups": res, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    m = {}
    for name, g in res.items():
        for s in g["seeds"]:
            m[f"{name}_seed{s['seed']}_auc"] = {"value": s["auc"], "label": "measured", "ci95": s["auc_ci95"],
                                               "run": s["run"]}
            m[f"{name}_seed{s['seed']}_tuned_acc"] = {"value": s["tuned_acc"], "label": "measured",
                                                     "ci95": s["tuned_acc_ci95"]}
        for k in ("auc", "tuned_acc"):
            m[f"{name}_mean_{k}"] = {"value": g["summary"][f"{k}_mean"], "label": "derived",
                                     "sd": g["summary"][f"{k}_sd"], "n_seeds": g["summary"]["n_seeds"]}
        for k, b in g["baselines"].items():
            m[f"{name}_baseline_{k}_auc"] = {"value": b["auc"], "label": "measured", "ci95": b["auc_ci95"]}
        m[f"{name}_best_on_dev_baseline"] = {"value": g["best_on_dev"], "label": "derived", "source": g["best_source"]}
        for key, c in g["comparisons"].items():
            if "verdict" in c:
                m[f"{name}_vs_{key}_mean_auc_diff"] = {**c, "value": c["mean_diff"], "label": "derived"}
            else:  # no beats/below verdict for a comparison that holds by construction
                m[f"{name}_vs_{key}"] = {"value": c["value"], "label": "derived", "vs": c["vs"], "note": c["note"]}
        for s in g["shuffled"]:
            m[f"{name}_shuffled_seed{s['seed']}_auc"] = {"value": s["auc"], "label": "measured",
                                                        "ci95": s["auc_ci95"], "source": s["source"]}
    for k, v in checks.items():
        m[f"check_{k}"] = {"value": v, "label": "derived"}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    if a.split == "dev":
        dev = {"split": "dev", "made_by": "src/final_test_report.py", "out_dir": os.path.abspath(out_dir),
               "rule": "best baseline per data set = the highest dev AUC of the 6 baselines (frozen settings)",
               "best_baseline": {g["data"]: g["best_on_dev"] for g in res.values()},
               "dev_auc": {g["data"]: {k: b["auc"] for k, b in g["baselines"].items()} for g in res.values()},
               "tm_dev_auc": {g["data"]: {f"seed{s['seed']}": s["auc"] for s in g["seeds"]} for g in res.values()}}
        with open(os.path.join(out_dir, "dev_report.json"), "w") as fh:
            json.dump(dev, fh, indent=1)
    print(f"wall {time.time() - t0:.0f}s; wrote report.json, metrics.json"
          f"{', dev_report.json' if a.split == 'dev' else ''} to {out_dir}")


if __name__ == "__main__":
    main()
