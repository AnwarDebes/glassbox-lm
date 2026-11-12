"""What the registration E3 fixes, in one place: the setting, the seeds, the files that must be pinned, and the check
that a record is E3's own and still matches the files on disk. run_slide10.py and probe_slide10.py call these
checks BEFORE a model is scored on an EVAL sentence; e3_report.py calls them again before it prints a rule.

The dry run of the pipeline uses the same code with the DEV sentences, its own throwaway plan, another run folder
and other run names.

Check a record by hand (from the study folder):
  .venv/bin/python -I src/e3_common.py --check registrations/E3_slide10_text.frozen.json
  .venv/bin/python -I src/e3_common.py --check registrations/dry/E3_dry.frozen.json --dry
"""
import hashlib
import json
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

PLAN = {False: "registrations/E3_slide10_text.md", True: "registrations/dry/E3_dry.md"}
RUNS_ROOT = {False: "runs", True: "runs_dry"}
PREFIX = {False: "e3_text_s", True: "e3dry_text_s"}
SPLIT = {False: "eval", True: "dev"}
SEEDS = {False: (11, 12, 13, 14, 15), True: (1, 2, 3, 4, 5)}
EXTRA_SEEDS = {False: tuple(range(16, 36)), True: (6, 7)}  # reported as counts, with no rule
SETTING = {"version": "text", "clauses": 20, "K": 12, "T": 8.0, "mode": "near", "init_state": -1, "seed_state": 0, "r": 2,
           "p_rec": 1.0, "p_rep": 1.0, "q_seed": 1.0, "max_seed": 1, "reject": "weakest", "reinforce": "critical",
           "neg_frac": 0.0, "p_decay": 0.0, "seed_all": True, "tied": False, "passes": 20, "n_train": 0}
# files outside src/ and tests/ that the registered runs read or that fix how they are run
REQUIRED_EXTRAS = ("../hierarchical_tm/data/slide10/pairs_slide10.npz", "../hierarchical_tm/data/slide10/slide10_meta.json",
                   "../hierarchical_tm/data/slide10/vocab_slide10.json", "../hierarchical_tm/data/slide10/text/train.npz",
                   "../hierarchical_tm/data/slide10/text/dev.npz", "../hierarchical_tm/src/slide10.py",
                   "../hierarchical_tm/src/htm_driver.py", "../hierarchical_tm/src/recovered/e3fb_deletion_probe.py",
                   "registrations/E3_job.sh", "env/requirements.txt")
# the config keys of a run (run_slide10.py) and the pinned file each one must equal
RUN_HASHES = {"pairs_slide10.npz": REQUIRED_EXTRAS[0], "slide10_meta.json": REQUIRED_EXTRAS[1],
              "vocab_slide10.json": REQUIRED_EXTRAS[2], "text/train.npz": REQUIRED_EXTRAS[3], "text/dev.npz": REQUIRED_EXTRAS[4]}
# the numbers of sentences of each kind that the scorer must report (read from the data on 2026-11-12, no model
# involved): DEV for the dry run, EVAL for the registered runs
N_SPLIT = {True: {"slide10": 101, "negated": 101, "nearmiss": 221, "broken": 977},
           False: {"slide10": 101, "negated": 101, "nearmiss": 216, "broken": 972}}
N_IN216 = {True: 70, False: 80}  # family sentences of the split that are in the 216 view


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record_path(dry=False):
    return os.path.join(ROOT, os.path.splitext(PLAN[dry])[0] + ".frozen.json")


def is_dry_record(path):
    return os.path.abspath(path) == os.path.abspath(record_path(True))


def source_hashes():
    """SHA-256 of every file freeze.py hashes by itself (src/ and tests/), computed now."""
    out = {}
    for folder in ("src", "tests"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, folder)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".py", ".sh", ".md", ".txt", ".json")):
                    p = os.path.join(dirpath, name)
                    out[os.path.relpath(p, ROOT)] = sha(p)
    return out


def preflight(path, dry=False):
    """(record, problems). No problem means: the file is the record of this plan at its fixed place, the plan text
    is unchanged, the files under src/ and tests/ are exactly those of the record with the same content, and every
    required extra file is in the record and unchanged."""
    problems = []
    if not path or not os.path.isfile(path):
        return None, [f"no record file: {path!r}"]
    with open(path) as f:
        rec = json.load(f)
    if os.path.abspath(path) != os.path.abspath(record_path(dry)):
        problems.append(f"the record is not at {os.path.splitext(PLAN[dry])[0]}.frozen.json")
    if rec.get("plan") != PLAN[dry]:
        problems.append(f"the record is of {rec.get('plan')!r}, not of {PLAN[dry]!r}")
    text_path = os.path.join(ROOT, rec.get("plan") or "missing")
    if not os.path.isfile(text_path) or sha(text_path) != rec.get("plan_sha256"):
        problems.append("the plan text differs from the record")
    now = source_hashes()
    was = rec.get("sources_sha256", {})
    for k in sorted(set(now) | set(was)):
        if now.get(k) != was.get(k):
            problems.append(f"source differs from the record: {k}")
    extras = rec.get("extras_sha256", {})
    for k in REQUIRED_EXTRAS:
        p = os.path.join(ROOT, k)
        if k not in extras:
            problems.append(f"required file not pinned in the record: {k}")
        elif not os.path.isfile(p) or sha(p) != extras[k]:
            problems.append(f"pinned file differs from the record: {k}")
    for k, v in extras.items():
        p = os.path.join(ROOT, k)
        if k not in REQUIRED_EXTRAS and (not os.path.isfile(p) or sha(p) != v):
            problems.append(f"pinned file differs from the record: {k}")
    return rec, problems


def check_run_args(a, dry=False):
    """Problems with the arguments of a registered run (run_slide10.py): setting, seed, name, split, output folder."""
    problems = []
    root = os.path.join(ROOT, RUNS_ROOT[dry])
    for k, v in SETTING.items():
        if getattr(a, k) != v:
            problems.append(f"setting {k} is {getattr(a, k)!r}, registered {v!r}")
    if a.seed not in SEEDS[dry] + EXTRA_SEEDS[dry]:
        problems.append(f"seed {a.seed} is not a registered seed")
    if a.name != f"{PREFIX[dry]}{a.seed}":
        problems.append(f"name must be {PREFIX[dry]}{a.seed}")
    if a.split != SPLIT[dry]:
        problems.append(f"split must be {SPLIT[dry]}")
    if os.path.abspath(a.out_root) != os.path.abspath(root) or a.no_log:
        problems.append(f"a registered run writes to {RUNS_ROOT[dry]}/ and to the run log")
    if os.path.isdir(root):
        done = sorted(d for d in os.listdir(root) if d.endswith(f"_{PREFIX[dry]}{a.seed}"))
        if done:
            problems.append(f"a run of this seed exists already: {done[0]}")
    return problems


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "--check":
        raise SystemExit(__doc__)
    dry = "--dry" in args[2:]
    rec, problems = preflight(args[1], dry)
    print(json.dumps({"record": args[1], "dry": dry, "frozen_utc": (rec or {}).get("frozen_utc"),
                      "plan_sha256": (rec or {}).get("plan_sha256"), "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
