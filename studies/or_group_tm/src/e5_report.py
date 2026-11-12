"""Checks and report of E5 (registrations/E5_wikipedia_test.md): the registered test of the OR-group rule (arm TB of
E4) on the held-back test pairs of the hierarchical_tm study.

  .venv/bin/python -I src/e5_report.py --frozen registrations/E5_wikipedia_test.frozen.json --out logs/e5_report.json
  .venv/bin/python -I src/e5_report.py --dry --frozen registrations/dry/E5_dry.frozen.json --out logs/e5_dry_report.json

The CHECKS come first:
  - the record is the record of this plan, and the plan, every file under src/ and tests/ and every pinned file are
    unchanged since the freeze (e5_common.preflight);
  - no run in this folder was scored on the test pairs except the registered ones (every config.json and every
    scores_test.npz under runs/, runs_dry/ and logs/tests/ is looked at);
  - the report's own process: the interpreter .venv/bin/python and every package of env/requirements.txt;
  - the run log: its text up to the freeze is unchanged (length and sha256 stored by the job at the freeze);
  - for each data version and registered seed: exactly one complete run folder (the five files that the job also
    requires), the registered setting, data, seed, name, split and output folder, its row in the run log, the record
    named, the engine, data and word list of the record, the run's interpreter, Python version and packages, a start
    after the freeze, 40,000 training pairs and the expected number of scored pairs, the pairs of the split scored
    after the last pass only, a partner-shuffle AUC within 0.47 to 0.53, a score file with the labels of the split
    that gives the AUCs that the run logged.
Then the numbers, step by step, each written to the output file as soon as it is done:
  1. the rules (R1-sa, R1-np, R1-ns_all: the vote sum; met only when both intervals are above 0), from
     src/wiki_report.py in the unit reading, the per-seed rows, and the fixed sentences;
  2. the ratio reading (no rule);
  3. the unit reading without the pairs that touch a duplicate (registrations/E5_duplicate_articles.json; no rule);
  4. the word groups (src/score_groups.py; no rule);
  5. the recomputed baselines and stock AUCs next to the numbers that the study printed (no rule).
A failure in steps 2 to 5 is written under "readings_failed" and leaves the rules in place.
If a check fails, the numbers are still computed (with the runs that are there) when the record passed, because only
then does src/wiki_report.py open the test pairs; they are written under "registered run void: <checks>" and the exit
status is 1. With --out the file of an attempt is <out>.part_<UTC>.json while the report runs, and becomes <out> (all
checks passed) or <out>.FAILED_<UTC>.json at the end; the report is deterministic, and a stopped attempt keeps its part
file and is simply run again.

With --dry the same code checks the dry run of the pipeline: DEV pairs, the throwaway plan registrations/dry/E5_dry.md,
runs under runs_dry/, one seed per data version. Nothing is skipped in dry mode.
--checks-only prints the checks and stops (no number). --steps rules computes step 1 only and says so (the tests use
it to try the void path quickly); the registered job uses all steps.
"""
import argparse
import contextlib
import datetime
import glob
import io
import json
import os
import platform
import re
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True
import e5_common as e5
from metrics import auc as auc_of

LABEL = {"sa": "Same-article pairs (sa)", "np": "Next-paragraph pairs (np)",
         "ns_all": "Next-paragraph pairs with every shared word removed (ns_all)"}
NAMES = {"jaccard": "word overlap (Jaccard)", "tfidf_cosine": "tf-idf cosine", "ppmi_svd_cosine": "PPMI cosine",
         "lr_A_B_AandB": "the logistic regression on both paragraphs and their shared words",
         "the stock model": "the stock model"}
SUBSET_TEXT = {"whole_copy_articles": "articles that are whole copies of a training article",
               "articles_with_a_duplicated_paragraph": "articles with a paragraph equal to a training paragraph"}
STUDY_TOLERANCE = 1e-8  # the study printed its AUCs with 8 or 9 decimals
STEPS = ("rules", "ratio", "duplicates", "groups", "study_numbers")


def same(a, b):
    return a is not None and a == b


def utc(text, fmt):
    return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.UTC)


def test_traces(root):
    """Every folder under runs/, runs_dry/ and logs/tests/ that shows a scoring on the test pairs."""
    out = []
    for base in ("runs", "runs_dry", os.path.join("logs", "tests")):
        for dirpath, _, filenames in os.walk(os.path.join(root, base)):
            hit = "scores_test.npz" in filenames
            if "config.json" in filenames:
                try:
                    with open(os.path.join(dirpath, "config.json")) as f:
                        hit = hit or json.load(f).get("split") == "test"
                except (OSError, ValueError, AttributeError):
                    hit = True  # an unreadable config is not waved through
            if hit:
                out.append(os.path.relpath(dirpath, root))
    return sorted(out)


def allowed_test(rel, dry):
    """True iff the folder is the place of a registered run (or of a crashed one that was kept)."""
    if dry:
        return False
    m = re.fullmatch(rf"{e5.RUNS_ROOT[False]}/\d{{8}}T\d{{6}}Z_{e5.PREFIX[False]}({'|'.join(e5.DATA)})_s(\d+)(_CRASHED)?",
                     rel.replace(os.sep, "/"))
    return bool(m) and int(m.group(2)) in e5.SEEDS[False]


def environment_ok(env):
    """The interpreter of the study and every package of env/requirements.txt at its version."""
    req = e5.requirements()
    return (isinstance(env, dict) and bool(req) and env.get("packages") == req and env.get("executable") == e5.EXECUTABLE
            and env.get("python") == platform.python_version())


def seed_row(root, rec, record, dry, data, s, runlog, labels):
    """(checks, row, folder) of one data version and seed. The row holds the run's own numbers whenever its folder is
    complete, whatever the checks say; labels: the labels of the split, or None when the record did not pass."""
    split, name = e5.SPLIT[dry], e5.run_name(data, s, dry)
    c = {}
    folders = sorted(glob.glob(os.path.join(root, e5.RUNS_ROOT[dry], f"*_{name}")))
    need = e5.RUN_FILES + (f"scores_{split}.npz",)
    complete = [d for d in folders if all(os.path.isfile(os.path.join(d, f)) for f in need)]
    c["one_complete_run"] = len(folders) == 1 and len(complete) == 1
    if len(complete) != 1:
        return c, None, None
    d = complete[0]
    try:
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(d, "metrics.json")) as f:
            met = json.load(f)
        m, hist = met["final"], met["history"]
    except (OSError, ValueError, KeyError, TypeError):
        c["files_readable"] = False
        return c, None, None
    ext, src = rec.get("extras_sha256", {}), rec.get("sources_sha256", {})
    c["setting"] = all(cfg.get(k) == v and isinstance(cfg.get(k), bool) == isinstance(v, bool) for k, v in e5.SETTING.items())
    c["data_seed_name_split"] = (cfg.get("data") == data and cfg.get("seed") == s and cfg.get("name") == name
                                 and cfg.get("split") == split)
    c["logged_in_default_folder"] = (cfg.get("no_log") is False and cfg.get("out_root") == e5.RUNS_ROOT[dry]
                                     and f"| {os.path.basename(d)} |" in runlog)
    c["names_this_record"] = bool(cfg.get("registered")) and os.path.abspath(os.path.join(root, cfg["registered"])) == os.path.abspath(record)
    c["engine_of_record"] = same(cfg.get("orgroup_tm_sha256"), src.get("src/orgroup_tm.py"))
    hashes = cfg.get("data_sha256", {})
    c["data_of_record"] = all(same(hashes.get(k), ext.get(f"{e5.H}/data/{p}")) for k, p in (
        (f"{data}/train.npz", f"{data}/train.npz"), (f"{data}/{split}.npz", f"{data}/{split}.npz"),
        ("vocab_F1024.json", "vocab_F1024.json")))
    c["environment_of_the_run"] = environment_ok(cfg.get("versions"))
    try:
        c["started_after_freeze"] = utc(cfg["run_start_utc"], "%Y%m%dT%H%M%SZ") >= utc(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError, TypeError):
        c["started_after_freeze"] = False
    c["pair_counts"] = m.get("n_train") == e5.N_TRAIN and m.get("n_eval") == e5.N_EVAL[dry]
    c["scored_once_after_the_last_pass"] = (len(hist) == e5.SETTING["passes"]
                                            and [h.get("pass") for h in hist if f"{split}_auc" in h] == [e5.SETTING["passes"]])
    ps = m.get("partner_shuffle_auc")
    c["partner_shuffle_in_bounds"] = isinstance(ps, float) and e5.SHUFFLE_BOUNDS[0] <= ps <= e5.SHUFFLE_BOUNDS[1]
    try:
        z = np.load(os.path.join(d, f"scores_{split}.npz"))
        y = z["y"]
        c["score_file_of_this_run"] = bool(
            len(y) == e5.N_EVAL[dry] and 2 * int(y.sum()) == len(y) and (labels is None or np.array_equal(y, labels))
            and auc_of(z["votes"], y) == m.get(f"{split}_auc") and auc_of(z["ratio"], y) == m.get(f"{split}_auc_ratio")
            and auc_of(z["weight"], y) == m.get(f"{split}_auc_weight") and auc_of(z["logodds"], y) == m.get(f"{split}_auc_logodds"))
    except (OSError, KeyError, ValueError):
        c["score_file_of_this_run"] = False
    row = {"data": data, "seed": s, "run": os.path.basename(d), "run_start_utc": cfg.get("run_start_utc"),
           "checks_failed": sorted(k for k, v in c.items() if not v)}
    for k, key in (("auc_unit", f"{split}_auc"), ("auc_ratio", f"{split}_auc_ratio"), ("auc_weight", f"{split}_auc_weight"),
                   ("auc_logodds", f"{split}_auc_logodds"), ("tuned_acc_unit", f"{split}_tuned_acc"),
                   ("tuned_acc_ratio", f"{split}_tuned_acc_ratio"), ("partner_shuffle_auc", "partner_shuffle_auc"),
                   ("train_auc", "train_auc"), ("members", "members"), ("fit_s_per_pass", "fit_s_per_pass"), ("wall_s", "wall_s")):
        row[k] = m.get(key)
    return c, row, d


def quiet(fn, argv):
    """fn(argv) with its printed output kept out of the report."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(argv)


def wiki(root, runs, d, reading, record, dry, subsets=False):
    import wiki_report
    argv = ["--data", d, "--split", e5.SPLIT[dry], "--runs", *runs, "--stock",
            *[os.path.join(root, p) for p in e5.stock_dirs(dry)[d]], "--reading", reading]
    if subsets:
        argv += ["--subsets", os.path.join(root, e5.DUPLICATES), "--subsets-only"]
    return quiet(wiki_report.main, argv + ([] if dry else ["--registered", record]))


def rules(res):
    out = {}
    for name, (d, comp) in e5.RULES.items():
        c = res[d][e5.PRIMARY_READING]["comparisons"][comp]
        each = {k: bool(c[k][0] > 0) for k in e5.RULE_INTERVALS}
        out[name] = {"data": d, "vs": c["vs"], "reading": e5.PRIMARY_READING, "mean_diff": c["mean_diff"],
                     **{k: c[k] for k in e5.RULE_INTERVALS}, "above_0_in_each_interval": each, "met": all(each.values())}
    return out


def study_numbers(root, res, dry):
    """The recomputed baseline AUCs and stock AUCs next to those the study printed (DEV: its dev check of the final
    report; test: its final-test report)."""
    out = {}
    with open(os.path.join(root, e5.DEV_CHECK_REPORT if dry else e5.FINAL_TEST_REPORT)) as f:
        study = json.load(f)
    for d in e5.DATA:
        r = res[d][e5.PRIMARY_READING]
        if dry:
            base = study["dev_auc"][d]
            stock = [study["tm_dev_auc"][d][f"seed{i + 1}"] for i in range(len(e5.STOCK_DEV[d]))]
        else:
            base = {k: v["auc"] for k, v in study["groups"][d]["baselines"].items()}
            stock = [s["auc"] for s in study["groups"][d]["seeds"]]
        diffs = {k: abs(r["baselines"][k]["auc"] - base[k]) for k in base}
        diffs.update({f"stock_{i + 1}": abs(r["stock"][f"stock_{i + 1}"]["auc"] - v) for i, v in enumerate(stock)})
        out[d] = {"largest_difference": max(diffs.values()), "equal_to_the_printed_digits": max(diffs.values()) <= STUDY_TOLERANCE,
                  "compared": sorted(diffs)}
    return out


def word(c):
    """The study's interval in words, with no verdict: above 0, below 0 or includes 0."""
    return "interval above 0" if c["ci95"][0] > 0 else "interval below 0" if c["ci95"][1] < 0 else "interval includes 0"


def ivs(c):
    return (f"{c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
            f"[{c['ci95_seeds_and_articles'][0]:+.3f}, {c['ci95_seeds_and_articles'][1]:+.3f}]")


def sentences(res, rl, dry, void=(), subsets=None):
    """The fixed wording of the registration with the numbers filled in."""
    split = e5.SPLIT[dry]
    what = "DEV pairs (dry run)" if dry else "held-back test pairs"
    k = len(e5.SEEDS[dry])
    out = [f"registered run void: {', '.join(void)}. The numbers below are reported in full and decide nothing."] if void else []
    for d in e5.DATA:
        if e5.PRIMARY_READING not in res.get(d, {}):
            continue
        r = res[d][e5.PRIMARY_READING]
        st, r1 = r["comparisons"]["stock"], rl[f"R1-{d}"]
        text = (f"{LABEL[d]}, {r['n_pairs']:,} {what}: the mean AUC of {k} {'dry' if dry else 'new'} seed{'s' if k > 1 else ''} "
                f"of the OR-group rule in the vote sum (the score E4 named) is {r['ours_mean_auc']:.3f}, and the mean AUC of the 5 "
                f"seeds of the study's stock model (its registered setting) on the same pairs is {r['stock_mean_auc']:.3f}; the "
                f"difference is {st['mean_diff']:+.3f} [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] (paired article bootstrap, "
                f"95%) and [{st['ci95_seeds_and_articles'][0]:+.3f}, {st['ci95_seeds_and_articles'][1]:+.3f}] with the seeds "
                f"resampled, so R1-{d} (above the stock model) is {'met' if r1['met'] else 'not met'}; predicted from DEV: met.")
        q = res[d].get(e5.OTHER_READING)
        if q is not None:
            text += (" In the ratio reading (a Jaccard index over the learned word groups, defined on the DEV pairs after "
                     f"the go-bar G2 of E4 was missed in the vote sum; no rule) the mean AUC is {q['ours_mean_auc']:.3f}; ")
            if "ci95" in q["comparisons"]["word_overlap"]:
                text += (f"the ratio reading minus word overlap (AUC {q['baselines']['jaccard']['auc']:.3f}) is "
                         f"{ivs(q['comparisons']['word_overlap'])}, and the vote sum minus word overlap is "
                         f"{ivs(r['comparisons']['word_overlap'])}.")
            else:
                text += f"the ratio reading minus the stock model is {ivs(q['comparisons']['stock'])}."
        out.append(text)
    out.append(e5.CAVEAT)
    for d in e5.DATA:
        for reading in (e5.PRIMARY_READING, e5.OTHER_READING):
            q = res.get(d, {}).get(reading)
            if q is None:
                continue
            best, parts, two = q["comparisons"]["best_on_dev"]["vs"], [], []
            for key in ("word_overlap", "tfidf_cosine", "ppmi_svd_cosine", "best_on_dev", "stock"):
                c = q["comparisons"][key]
                if "ci95" not in c or (key == "best_on_dev" and c["vs"] == "ppmi_svd_cosine"):
                    continue
                tag = " (the baseline that was best on DEV in the study)" if c["vs"] == best else ""
                a = q["stock_mean_auc"] if key == "stock" else q["baselines"][c["vs"]]["auc"]
                parts.append(f"minus {NAMES.get(c['vs'], c['vs'])}{tag}, AUC {a:.3f}: {ivs(c)}, {word(c)}")
                if "ci95_two_way" in c:
                    two.append(f"{NAMES.get(c['vs'], c['vs'])} [{c['ci95_two_way'][0]:+.3f}, {c['ci95_two_way'][1]:+.3f}]")
            name = "the vote sum" if reading == "unit" else "the ratio reading"
            out.append(f"Readings, {d} ({split}), {name}, mean AUC {q['ours_mean_auc']:.3f}: " + "; ".join(parts)
                       + (". With the articles of paragraphs A and B resampled (seeds fixed): " + ", ".join(two) if two else "")
                       + " (no rule).")
    for d in e5.DATA:
        for sname, s in ((subsets or {}).get(d) or {}).items():
            c, w = s["comparisons"]["stock"], s["comparisons"]["word_overlap"]
            text = (f"Without the {s['n_pairs_dropped']:,} {split} pairs of {d} that touch one of the {s['articles_listed']} "
                    f"{SUBSET_TEXT.get(sname, sname)} ({s['n_pairs']:,} pairs left), the vote sum minus the stock model is {ivs(c)}")
            if "ci95" in w:
                text += f" and minus word overlap {ivs(w)}"
            out.append(text + " (no rule).")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--dry", action="store_true", help="the dry run of the pipeline: DEV pairs, runs under runs_dry/")
    ap.add_argument("--checks-only", action="store_true", help="print the checks and stop")
    ap.add_argument("--steps", choices=["all", "rules"], default="all")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    root, dry = e5.ROOT, a.dry
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = now.replace("-", "").replace(":", "")
    rec, problems = e5.preflight(a.frozen, dry)
    checks = {"record_plan_sources_and_pinned_files": not problems}
    stray = [t for t in test_traces(root) if not allowed_test(t, dry)]
    checks["no_test_scoring_outside_the_registered_runs"] = not stray
    checks["report_environment"] = environment_ok(e5.environment())
    checks["run_log_prefix_unchanged"] = e5.runlog_prefix_unchanged(dry)
    log, runlog = os.path.join(root, e5.RUNLOG), ""
    if os.path.isfile(log):
        with open(log) as f:
            runlog = f.read()
    rows, runs = [], {d: [] for d in e5.DATA}
    if rec is not None:
        for data in e5.DATA:
            labels = None
            if not problems:  # the labels of the split are read only under a record that passed
                labels = np.load(os.path.join(root, e5.H, "data", data, f"{e5.SPLIT[dry]}.npz"))["y"]
            for s in e5.SEEDS[dry]:
                c, row, folder = seed_row(root, rec, a.frozen, dry, data, s, runlog, labels)
                checks.update({f"{data}_seed{s}_{k}": v for k, v in c.items()})
                if row is not None:
                    rows.append(row)
                    runs[data].append(folder)
    all_ok = all(checks.values()) and len(rows) == len(e5.DATA) * len(e5.SEEDS[dry])
    void = sorted(k for k, v in checks.items() if not v) if not all_ok else []
    out = {"registration": e5.PLAN[dry], "split": e5.SPLIT[dry], "dry": dry, "report_utc": now,
           "frozen_utc": (rec or {}).get("frozen_utc"), "plan_sha256": (rec or {}).get("plan_sha256"),
           "record_sha256": e5.sha(a.frozen) if os.path.isfile(a.frozen) else None,
           "e5_report_py_sha256": e5.sha(os.path.abspath(__file__)), "steps": a.steps,
           "checks": checks, "problems": problems, "test_scoring_outside_the_registered_runs": stray,
           "checks_all_passed": all_ok, "rows": rows}
    if void:
        out["registered_run_void"] = [f"registered run void: {k}" for k in void]
    final = (a.out if all_ok else f"{a.out}.FAILED_{stamp}.json") if a.out else ""
    work = f"{a.out}.part_{stamp}.json" if a.out else ""
    if final and os.path.exists(final):
        print(f"refusing to overwrite {final}", file=sys.stderr)
        return 2

    def save():
        if work:
            with open(work, "w") as f:
                f.write(json.dumps(out, indent=1) + "\n")
    save()
    numbers = (not a.checks_only and rec is not None and not problems and all(runs[d] for d in e5.DATA))
    if not a.checks_only and not numbers:
        out["numbers"] = "not computed: " + ("the record did not pass, so the test pairs stay closed" if problems or rec is None
                                             else "a data version has no complete run folder")
    if numbers:
        res, failed = {d: {} for d in e5.DATA}, {}

        def step(name, fn):
            try:
                fn()
            except (Exception, SystemExit) as ex:  # noqa: BLE001 - a failed step is reported, never hidden
                msg = str(ex).replace(root + os.sep, "").replace(os.path.dirname(root) + os.sep, "../")
                failed[name] = f"{type(ex).__name__}: {msg}"
                out["readings_failed"] = failed
            save()

        def rule_step():
            for d in e5.DATA:
                res[d][e5.PRIMARY_READING] = wiki(root, runs[d], d, e5.PRIMARY_READING, a.frozen, dry)
            out["primary_reading"] = e5.PRIMARY_READING
            out["rules"] = rules(res)
            out["readings"] = res
            out["sentences"] = sentences(res, out["rules"], dry, void)
        step("rules", rule_step)
        if "rules" not in failed and a.steps == "all":
            def ratio_step():
                for d in e5.DATA:
                    res[d][e5.OTHER_READING] = wiki(root, runs[d], d, e5.OTHER_READING, a.frozen, dry)
                out["sentences"] = sentences(res, out["rules"], dry, void)
            subsets = {}

            def duplicates_step():
                for d in e5.DATA:
                    subsets[d] = wiki(root, runs[d], d, e5.PRIMARY_READING, a.frozen, dry, subsets=True)["subsets"]
                out["readings_without_duplicates"] = subsets
                out["sentences"] = sentences(res, out["rules"], dry, void, subsets)

            def groups_step():
                import score_groups
                out["groups"] = {d: quiet(score_groups.main, ["--data", d, "--runs", *runs[d], "--stock",
                                                              *[os.path.join(root, p) for p in e5.group_dirs()[d]]])
                                 for d in e5.DATA}

            def study_step():
                out["study_numbers"] = study_numbers(root, res, dry)
            step("ratio", ratio_step)
            step("duplicates", duplicates_step)
            step("groups", groups_step)
            step("study_numbers", study_step)
            out["cost"] = {d: {k: [r[k] for r in rows if r["data"] == d] for k in ("fit_s_per_pass", "wall_s", "members")}
                           for d in e5.DATA}
        if "rules" in failed:
            all_ok = False
            if final == a.out and a.out:
                final = f"{a.out}.FAILED_{stamp}.json"
    text = json.dumps(out, indent=1)
    print(text)
    if failed_steps := out.get("readings_failed"):
        print("readings failed: " + ", ".join(sorted(failed_steps)), file=sys.stderr)
    if work:
        save()
        os.replace(work, final)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
