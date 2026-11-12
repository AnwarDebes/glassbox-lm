"""Checks and report of E3 (registrations/E3_slide10_text.md): the registered confirmation of the slide-10 clause
on the EVAL sentences, direct-label version.

  .venv/bin/python -I src/e3_report.py --frozen registrations/E3_slide10_text.frozen.json --out logs/e3_report.json
  .venv/bin/python -I src/e3_report.py --dry --frozen registrations/dry/E3_dry.frozen.json --out logs/e3_dry_report.json

The CHECKS come first; the rule and the readings are printed only when every one of them passes:
  - the record is the record of this plan, and the plan, every file under src/ and tests/ and every pinned file
    (data, scorer, job script, package list) are unchanged since the freeze (e3_common.preflight);
  - no run in this folder was scored on the EVAL sentences except the registered ones (every config.json under
    runs/, runs_dry/ and logs/tests/ is read);
  - each registered seed has exactly one complete run, started after the freeze, with the registered setting, name,
    split and output folder, written to the run log, made with the engine, the data and the scorer of the record;
    the scorer saw the expected numbers of sentences; the exported rule equals the model's own firing; the probe
    file belongs to the run and its intact share equals the scorer's coverage.
The same checks run for the 20 further seeds, which carry no rule; if one of those fails, the further seeds are
left out and the reason is printed.

With --dry the same code checks the dry run of the pipeline: DEV sentences, the throwaway plan
registrations/dry/E3_dry.md, runs under runs_dry/. Nothing is skipped in dry mode.

If a check fails the output goes to <out>.FAILED_<UTC>.json and the exit status is 1.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

from scipy.stats import beta

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import e3_common as e3

RUN_FILES = ("config.json", "metrics.json", "state.npz", "include_bits.npz")
COMPANION = "The pair version (the procedure of the proposal) met none of the bars on the DEV sentences."


def exact_interval(k, n, conf=0.95):
    """Exact (Clopper-Pearson) interval of a share from k of n."""
    lo = 0.0 if k == 0 else float(beta.ppf((1 - conf) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def same(a, b):
    return a is not None and a == b


def utc(text, fmt):
    return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.UTC)


def eval_traces(root):
    """Every folder under runs/, runs_dry/ and logs/tests/ that shows a scoring on the EVAL sentences."""
    out = []
    for base in ("runs", "runs_dry", os.path.join("logs", "tests")):
        for dirpath, _, filenames in os.walk(os.path.join(root, base)):
            hit = any(f in filenames for f in ("probe_eval.json", "check2b_eval.txt"))
            if "config.json" in filenames:
                try:
                    with open(os.path.join(dirpath, "config.json")) as f:
                        hit = hit or json.load(f).get("split") == "eval"
                except (OSError, ValueError):
                    hit = True  # an unreadable config is not waved through
            if hit:
                out.append(os.path.relpath(dirpath, root))
    return sorted(out)


def allowed_eval(rel, dry):
    """True iff the folder is the place of a registered run (or of a crashed one that was kept)."""
    if dry:
        return False
    m = re.fullmatch(rf"{e3.RUNS_ROOT[False]}/\d{{8}}T\d{{6}}Z_{e3.PREFIX[False]}(\d+)(_CRASHED)?", rel.replace(os.sep, "/"))
    return bool(m) and int(m.group(1)) in e3.SEEDS[False] + e3.EXTRA_SEEDS[False]


def seed_row(root, rec, record, dry, s):
    """(checks, row) of one seed; row is None when a check that the row needs has failed."""
    split, prefix = e3.SPLIT[dry], e3.PREFIX[dry]
    c = {}
    folders = sorted(glob.glob(os.path.join(root, e3.RUNS_ROOT[dry], f"*_{prefix}{s}")))
    complete = [d for d in folders if all(os.path.isfile(os.path.join(d, f)) for f in RUN_FILES)]
    c["one_complete_run"] = len(folders) == 1 and len(complete) == 1
    if not c["one_complete_run"]:
        return c, None
    d = folders[0]
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)["final"]
    chk = m["check2b"]
    ext, src = rec.get("extras_sha256", {}), rec.get("sources_sha256", {})
    c["setting"] = all(cfg.get(k) == v for k, v in e3.SETTING.items())
    c["seed_name_split"] = cfg.get("seed") == s and cfg.get("name") == f"{prefix}{s}" and cfg.get("split") == split
    c["logged_in_default_folder"] = cfg.get("no_log") is False and cfg.get("out_root") == e3.RUNS_ROOT[dry]
    c["names_this_record"] = bool(cfg.get("registered")) and os.path.abspath(os.path.join(root, cfg["registered"])) == os.path.abspath(record)
    c["engine_of_record"] = same(cfg.get("orgroup_tm_sha256"), src.get("src/orgroup_tm.py"))
    c["data_of_record"] = all(same(cfg.get("data", {}).get(k), ext.get(v)) for k, v in e3.RUN_HASHES.items())
    c["scorer_of_record"] = (same(cfg.get("slide10_py_sha256"), ext.get("../hierarchical_tm/src/slide10.py"))
                             and same(cfg.get("htm_driver_py_sha256"), ext.get("../hierarchical_tm/src/htm_driver.py")))
    try:
        c["started_after_freeze"] = utc(cfg["run_start_utc"], "%Y%m%dT%H%M%SZ") >= utc(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        c["started_after_freeze"] = False
    c["sentence_counts"] = all(chk.get(f"n_{split}_{k}") == n for k, n in e3.N_SPLIT[dry].items())
    c["export_equals_model"] = m.get("export_equals_model_on_train_and_dev") is True
    pf = os.path.join(d, f"probe_{split}.json")
    c["probe_file"] = os.path.isfile(pf)
    if not c["probe_file"]:
        return c, None
    with open(pf) as f:
        pr = json.load(f)
    n_fam = e3.N_SPLIT[dry]["slide10"]
    c["probe_of_this_run"] = (pr.get("run") == os.path.basename(d) and pr.get("split") == split and pr.get("n_intact") == n_fam
                              and pr.get("n_deletion_texts") == 6 * n_fam)
    c["probe_intact_equals_coverage"] = abs(pr.get("best_rule_accepts_intact", -1.0) - chk.get(f"coverage_{split}_288", -2.0)) < 1e-12
    if not all(c.values()):
        return c, None
    tied = m["tied_best_rules"]
    row = {"seed": s, "run": os.path.basename(d),
           "coverage_288": chk[f"coverage_{split}_288"], "coverage_216": chk[f"coverage_{split}_216"],
           "fires_negated": chk[f"fires_{split}_negated"], "fires_nearmiss": chk[f"fires_{split}_nearmiss"],
           "fires_broken": chk[f"fires_{split}_broken"], "fires_max": chk[f"fires_{split}_max"],
           "brackets_288": chk["brackets_recovered_288"], "brackets_216": chk["brackets_recovered_216"],
           "bracket_jaccard_288": chk["bracket_jaccard_288"], "bracket_jaccard_216": chk["bracket_jaccard_216"],
           "bracket_jaccard_strict_288": chk["bracket_jaccard_strict_288"],
           "brackets_strict_288": int(sum(j >= 0.8 for j in chk["bracket_jaccard_strict_288"])),
           "flag_288": chk[f"check2b_{split}_pass_288"], "flag_216": chk[f"check2b_{split}_pass_216"],
           "tied_best_rules": len(tied), "tied_meeting_both_flags": int(sum(t["pass_288"] and t["pass_216"] for t in tied)),
           "tied_brackets_288": [t["brackets_288"] for t in tied], "tied_brackets_216": [t["brackets_216"] for t in tied],
           "dev_plain_acc": m["dev_plain_acc"], "members": m["members"], "wall_s": m["wall_s"]}
    row["cover_and_fire"] = bool(row["coverage_288"] >= 0.9 and row["coverage_216"] >= 0.9 and row["fires_max"] <= 0.05)
    row["both_flags"] = bool(row["flag_288"] and row["flag_216"])
    row["reading_288"] = bool(row["cover_and_fire"] and row["brackets_288"] >= 4)
    for k in ("best_rule_accepts_intact", "best_rule_accepts_deletions", "best_rule_accepts_deletions_per_bracket",
              "best_rule_accepts_deletions_not_training_negatives", "best_rule_reading", "best_rule_accepts_one_word",
              "best_rule_accepts_one_word_per_bracket", "model_accepts_intact", "model_accepts_deletions",
              "model_accepts_deletions_per_bracket", "model_accepts_deletions_not_training_negatives", "model_reading",
              "model_accepts_one_word", "template_like_clauses", "n_deletion_texts_not_training_negatives", "n_one_word_texts"):
        row[k] = pr[k]
    return c, row


def counts(rows):
    n = len(rows)
    out: dict = {"n_seeds": n}
    for name, key in (("both_flags", lambda r: r["both_flags"]), ("reading_288", lambda r: r["reading_288"]),
                      ("flag_288", lambda r: r["flag_288"]), ("flag_216", lambda r: r["flag_216"]),
                      ("cover_and_fire", lambda r: r["cover_and_fire"]),
                      ("best_rule_template_like", lambda r: r["best_rule_reading"] == "template-like"),
                      ("model_template_like", lambda r: r["model_reading"] == "template-like")):
        k = int(sum(bool(key(r)) for r in rows))
        out[name] = {"seeds": k, "of": n, "exact_95_interval_of_the_share": exact_interval(k, n) if n else None}
    return out


def span(rows, key, pick=None):
    v = [(r[key] if pick is None else r[key][pick]) for r in rows]
    return [round(min(v), 3), round(max(v), 3)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--dry", action="store_true", help="the dry run of the pipeline: DEV sentences, runs under runs_dry/")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    root, dry = e3.ROOT, a.dry
    rec, problems = e3.preflight(a.frozen, dry)
    checks = {"record_plan_sources_and_pinned_files": not problems}
    stray = [t for t in eval_traces(root) if not allowed_eval(t, dry)]
    checks["no_eval_scoring_outside_the_registered_runs"] = not stray
    rows, extra_rows, extra_checks = [], [], {}
    if rec is not None:
        for s in e3.SEEDS[dry]:
            c, row = seed_row(root, rec, a.frozen, dry, s)
            checks.update({f"seed{s}_{k}": v for k, v in c.items()})
            if row is not None:
                rows.append(row)
        for s in e3.EXTRA_SEEDS[dry]:
            c, row = seed_row(root, rec, a.frozen, dry, s)
            extra_checks.update({f"seed{s}_{k}": v for k, v in c.items()})
            if row is not None:
                extra_rows.append(row)
    all_ok = all(checks.values()) and len(rows) == len(e3.SEEDS[dry])
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {"registration": e3.PLAN[dry], "split": e3.SPLIT[dry], "dry": dry, "report_utc": now,
           "frozen_utc": (rec or {}).get("frozen_utc"), "plan_sha256": (rec or {}).get("plan_sha256"),
           "record_sha256": e3.sha(a.frozen) if os.path.isfile(a.frozen) else None,
           "e3_report_py_sha256": e3.sha(os.path.abspath(__file__)),
           "checks": checks, "problems": problems, "eval_scoring_outside_the_registered_runs": stray,
           "checks_all_passed": all_ok}
    if all_ok:
        n = len(rows)
        cnt = counts(rows)
        j, k = cnt["both_flags"]["seeds"], cnt["reading_288"]["seeds"]
        out["rows"] = rows
        out["counts_registered_seeds"] = cnt
        out["R1_both_flags_met"] = j >= 3
        out["reading_288_at_least_3"] = k >= 3
        what = "DEV sentences, dry run" if dry else "held-back sentences"
        result = (
            f"With direct family labels and a setting chosen on the DEV sentences, on {k} of {n} "
            f"{'design' if dry else 'new'} seeds the best single clause of the model (picked by the study's scorer on the "
            f"training sentences) accepts at least 90% of the {e3.N_SPLIT[dry]['slide10']} {what} of the slide-10 template "
            f"family (and of the {e3.N_IN216[dry]} among them that are in the 216 view), fires on at most 5% of the negated, "
            f"near-miss and broken sentences, and recovers at least 4 of the 6 brackets in the 288 reading; {j} of {n} seeds "
            "when both flags of the scorer are required.")
        if j >= 3:
            rule = "R1 is met: both flags of the scorer on at least 3 of 5 seeds."
        elif k >= 3:
            rule = ("R1 is not met: both flags of the scorer on fewer than 3 of 5 seeds. In the 288 reading of the brackets, "
                    f"which was fixed after the DEV rows were seen, the bars are met on {k} of {n} seeds.")
        else:
            rule = ("R1 is not met: both flags of the scorer on fewer than 3 of 5 seeds. The DEV numbers of the chosen arm "
                    "did not hold on new seeds.")
        lo, hi = span(rows, "model_accepts_deletions")
        lo6, hi6 = span(rows, "model_accepts_deletions_per_bracket", 5)
        whole = (f"The model as a whole (at least one of its 20 clauses fires) accepts {lo:.3f} to {hi:.3f} of the deletion "
                 f"texts (bracket 6: {lo6:.3f} to {hi6:.3f}).")
        out["sentences"] = [result, rule, whole, COMPANION]
        out["ranges_registered_seeds"] = {key: span(rows, key) for key in (
            "coverage_288", "coverage_216", "fires_max", "brackets_288", "brackets_216", "best_rule_accepts_deletions",
            "model_accepts_deletions", "best_rule_accepts_one_word", "template_like_clauses", "tied_best_rules", "dev_plain_acc")}
        extra_ok = all(extra_checks.values()) and len(extra_rows) == len(e3.EXTRA_SEEDS[dry])
        out["extra_seeds_checks_passed"] = extra_ok
        if extra_ok:
            out["extra_rows"] = extra_rows
            out["counts_extra_seeds"] = counts(extra_rows)
            out["counts_all_seeds"] = counts(rows + extra_rows)
        else:
            out["extra_seeds_void"] = sorted(k for k, v in extra_checks.items() if not v) or ["a further seed has no row"]
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        name = a.out if all_ok else f"{a.out}.FAILED_{now.replace('-', '').replace(':', '')}.json"
        if os.path.exists(name):
            print(f"refusing to overwrite {name}", file=sys.stderr)
            return 2
        with open(name, "x") as f:
            f.write(text + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
