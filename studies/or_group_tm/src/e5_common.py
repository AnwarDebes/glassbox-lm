"""What the registration E5 fixes, in one place: the setting, the data versions, the seeds, the stock score folders,
the files that must be pinned, and the check that a record is E5's own and still matches the files on disk.
run_wiki.py and wiki_report.py call these checks BEFORE a model or a baseline is scored on a test pair;
e5_report.py calls them again before it prints a rule.

The dry run of the pipeline uses the same code with the DEV pairs, its own throwaway plan, another run folder and
other run names.

Check a record by hand (from the study folder):
  .venv/bin/python -I src/e5_common.py --check registrations/E5_wikipedia_test.frozen.json
  .venv/bin/python -I src/e5_common.py --check registrations/dry/E5_dry.frozen.json --dry
List the files that the freeze must pin (registrations/E5_job.sh passes them to src/freeze.py):
  .venv/bin/python -I src/e5_common.py --extras [--dry]
Length and sha256 of the run log now (the job stores them at the freeze):
  .venv/bin/python -I src/e5_common.py --runlog-prefix
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
H = "../hierarchical_tm"

PLAN = {False: "registrations/E5_wikipedia_test.md", True: "registrations/dry/E5_dry.md"}
RUNS_ROOT = {False: "runs", True: "runs_dry"}
PREFIX = {False: "e5_", True: "e5dry_"}
SPLIT = {False: "test", True: "dev"}
DATA = ("sa", "np", "ns_all")
SEEDS = {False: (11, 12, 13, 14, 15), True: (1,)}
# pairs per data version and split (the study's data card and final-test report; dev also read from the DEV runs)
N_TRAIN = 40000
N_EVAL = {False: 10000, True: 20000}
SHUFFLE_BOUNDS = (0.47, 0.53)  # partner-shuffle AUC of every run (E4), else the run is void
# arm TB of E4 with every flag of src/run_wiki.py; eval_every = passes: the pairs of the split are scored once, at the end
SETTING = {"F": 1024, "clauses": 2000, "K": 1, "T": 40.0, "mode": "side", "init_state": -3, "seed_state": 0, "r": 2,
           "p_rec": 1.0, "p_rep": 1.0, "q_seed": 1.0, "max_seed": 1, "reject": "weakest", "reinforce": "share",
           "p_decay": 0.0, "seed_all": False, "tied": True, "wmax": 64, "neg_frac": 0.0, "smin": -32, "smax": 31,
           "passes": 10, "eval_every": 10, "n_train": 0, "n_dev": 0, "top": 40}
RUN_FILES = ("config.json", "metrics.json", "state.npz", "groups.txt")
# the reading of the rules: the vote sum ("unit"), the score that E4 named. The ratio reading was defined on the DEV
# pairs after the tied rows of seed 1 were below the go-bar G2 of E4 in the vote sum; it is printed beside it, no rule.
PRIMARY_READING, OTHER_READING = "unit", "ratio"
# the rules, each standing alone: name -> (data version, comparison of src/wiki_report.py). A rule is met only when
# the whole 95% interval is above 0 in both intervals: the study's article bootstrap (seeds fixed) and the bootstrap
# that also resamples the seeds (ci95_seeds_and_articles). There is no rule against word overlap (E4: G2 missed).
RULES = {"R1-sa": ("sa", "stock"), "R1-np": ("np", "stock"), "R1-ns_all": ("ns_all", "stock")}
RULE_INTERVALS = ("ci95", "ci95_seeds_and_articles")
CAVEAT = ("The ratio reading was defined on the DEV pairs at 16:51:28Z, after the tied rows of seed 1 were below the "
          "bar G2 of E4 in the vote sum that E4 named (sa 0.8255 against 0.830, np 0.8567 against 0.875). G2 is missed in "
          "the vote sum (means of seeds 1 to 3: 0.8272 and 0.8567) and passed only in the ratio reading (0.8470 and "
          "0.8787); by E4, the comparison with word overlap carries no rule.")
# the stock model of the study, five seeds per data version (its docs/results.md, Check 3 and "The final test"):
# the folders that scored each saved model on the DEV pairs (they also hold its include bits) and on the test pairs
STOCK_DEV = {
    "sa": ("20261002T050235Z_e4_head_sa_s1_scoreP100", "20261002T052930Z_e4_head_sa_s2_score",
           "20261002T060727Z_e4_head_sa_s3_score", "20261002T145933Z_e4_head_sa_s4_score",
           "20261002T152719Z_e4_head_sa_s5_score"),
    "np": ("20261002T062612Z_e4_head_np_s1_score", "20261002T070833Z_e4_head_np_s2_score",
           "20261002T071028Z_e4_head_np_s3_score", "20261002T152755Z_e4_head_np_s4_score",
           "20261002T155611Z_e4_head_np_s5_score"),
    "ns_all": ("20261002T082456Z_e4s_nsall_s100_s1_score", "20261002T084106Z_e4s_nsall_s100_s2_score",
               "20261002T090952Z_e4s_nsall_s100_s3_score", "20261002T155646Z_e4s_nsall_s100_s4_score",
               "20261002T164352Z_e4s_nsall_s100_s5_score")}
STOCK_TEST = {
    "sa": ("20261002T192818Z_e4_head_sa_s1_testscore", "20261002T192838Z_e4_head_sa_s2_testscore",
           "20261002T192858Z_e4_head_sa_s3_testscore", "20261002T192918Z_e4_head_sa_s4_testscore",
           "20261002T192938Z_e4_head_sa_s5_testscore"),
    "np": ("20261002T192959Z_e4_head_np_s1_testscore", "20261002T193019Z_e4_head_np_s2_testscore",
           "20261002T193039Z_e4_head_np_s3_testscore", "20261002T193059Z_e4_head_np_s4_testscore",
           "20261002T193119Z_e4_head_np_s5_testscore"),
    "ns_all": ("20261002T193140Z_e4s_nsall_s100_s1_testscore", "20261002T193200Z_e4s_nsall_s100_s2_testscore",
               "20261002T193220Z_e4s_nsall_s100_s3_testscore", "20261002T193240Z_e4s_nsall_s100_s4_testscore",
               "20261002T193301Z_e4s_nsall_s100_s5_testscore")}
# the training run of each stock seed, named by the "run" entry of the metrics.json of both score folders
STOCK_SOURCE = {
    "sa": ("20261002T035511Z_e4_curve_sa_granmo_s100_p200", "20261002T050310Z_e4_head_sa_s2", "20261002T053211Z_e4_head_sa_s3",
           "20261002T142452Z_e4_head_sa_s4", "20261002T145158Z_e4_head_sa_s5"),
    "np": ("20261002T055851Z_e4_head_np_s1", "20261002T062537Z_e4_head_np_s2", "20261002T064353Z_e4_head_np_s3",
           "20261002T150009Z_e4_head_np_s4", "20261002T151944Z_e4_head_np_s5"),
    "ns_all": ("20261002T074540Z_e4s_nsall_s100_s1", "20261002T081320Z_e4s_nsall_s100_s2", "20261002T084141Z_e4s_nsall_s100_s3",
               "20261002T152830Z_e4s_nsall_s100_s4", "20261002T155721Z_e4s_nsall_s100_s5")}
FINAL_TEST_REPORT = f"{H}/results/20261002T193513Z_final_test_main_table/report.json"
# the article lists of the reading without the pairs that touch a duplicate (src/e5_duplicates.py)
DUPLICATES = "registrations/E5_duplicate_articles.json"
# the CPU rule of the machine, measured by the job before each run and before the report
CPU_LOAD = "../decision_layer/scripts/cpu_load.py"
# the versions of src/run_wiki.py and src/wiki_report.py that the record of E3 pins (E3 ended before E5 changed them)
E3_SOURCES = ("verification/e3_sources/run_wiki.py", "verification/e3_sources/wiki_report.py")
# the run log: the job stores the length and sha256 of its text at the freeze; the report checks that prefix
RUNLOG = "logs/RUNS_wiki.md"
RUNLOG_PREFIX = {False: "logs/e5/runs_wiki_prefix.json", True: "logs/e5_dry/runs_wiki_prefix.json"}
EXECUTABLE = ".venv/bin/python"  # the interpreter of every run and of the report, relative to the study folder
DEV_CHECK_REPORT = f"{H}/results/20261002T164442Z_final_test_report_devcheck_5seeds_exploratory/dev_report.json"
WORDNET = "../tm_language_model/data/concept_eval/nltk_data/corpora/wordnet.zip"


def stock_dirs(dry):
    """{data: [score folders of the five stock seeds on the split of this mode]}, paths relative to the study folder."""
    names = STOCK_DEV if dry else STOCK_TEST
    return {d: [f"{H}/results/{n}" for n in names[d]] for d in DATA}


def group_dirs():
    """{data: [the five stock DEV score folders]}: their include bits are the groups of the five stock models."""
    return {d: [f"{H}/results/{n}" for n in STOCK_DEV[d]] for d in DATA}


def _group_files():
    """The files of the stock folders that src/score_groups.py reads (include bits, and the log and metrics that the
    study's config_of reads), with the log of the training run that a score folder names."""
    out = []
    for d in DATA:
        for n, src in zip(STOCK_DEV[d], STOCK_SOURCE[d]):
            out += [f"{H}/results/{n}/include_bits.npz", f"{H}/results/{n}/log.txt", f"{H}/results/{n}/metrics.json",
                    f"{H}/results/{src}/log.txt"]
    return out


def required_extras(dry):
    """Files outside src/ and tests/ that the runs and the report of this mode read or that fix how they are run."""
    split = SPLIT[dry]
    out = [f"{H}/data/{d}/{s}.npz" for d in DATA for s in ("train", split)]
    out += [f"{H}/data/vocab_F1024.json", f"{H}/data/vocab_F3072.json", f"{H}/data/paragraphs_F3072.npz",
            f"{H}/data/splits.json", f"{H}/src/data_checks.py", f"{H}/src/htm_driver.py", f"{H}/src/pairdata.py",
            f"{H}/src/npmi_groups.py", f"{H}/src/wiki_groups.py", WORDNET]
    out += [f"{p}/scores_{split}.npz" for d in DATA for p in stock_dirs(dry)[d]]
    out += _group_files()
    out += [DEV_CHECK_REPORT if dry else FINAL_TEST_REPORT, "registrations/E5_job.sh", "env/requirements.txt", DUPLICATES,
            CPU_LOAD, *E3_SOURCES, RUNLOG_PREFIX[dry]]
    if not dry:
        out.append("logs/e5_dry_report.json")  # the report of the dry run is pinned with the registration
    return tuple(dict.fromkeys(out))


REQUIRED_EXTRAS = {False: required_extras(False), True: required_extras(True)}


def run_name(data, seed, dry=False):
    return f"{PREFIX[dry]}{data}_s{seed}"


def requirements():
    """{package: version} of env/requirements.txt ({} without the file)."""
    out, path = {}, os.path.join(ROOT, "env", "requirements.txt")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if "==" in line:
                    k, v = line.strip().split("==", 1)
                    out[k.lower()] = v
    return out


def environment():
    """Python version, interpreter (relative to the study folder) and the installed version of every package of
    env/requirements.txt, read in this process."""
    pk = {}
    for k in requirements():
        try:
            pk[k] = importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:
            pk[k] = None
    return {"python": platform.python_version(), "executable": os.path.relpath(sys.executable, ROOT), "packages": pk}


def runlog_prefix():
    """Length and sha256 of the run log as it is now (the job writes this at the freeze)."""
    with open(os.path.join(ROOT, RUNLOG), "rb") as f:
        text = f.read()
    return {"file": RUNLOG, "length": len(text), "sha256": hashlib.sha256(text).hexdigest()}


def runlog_prefix_unchanged(dry):
    """True iff the first `length` bytes of the run log still give the sha256 stored at the freeze."""
    p = os.path.join(ROOT, RUNLOG_PREFIX[dry])
    try:
        with open(p) as f:
            pin = json.load(f)
        with open(os.path.join(ROOT, RUNLOG), "rb") as f:
            text = f.read()
    except (OSError, ValueError):
        return False
    return (isinstance(pin, dict) and pin.get("file") == RUNLOG and isinstance(pin.get("length"), int)
            and len(text) >= pin["length"] and hashlib.sha256(text[:pin["length"]]).hexdigest() == pin.get("sha256"))


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
    try:
        with open(path) as f:
            rec = json.load(f)
    except ValueError:
        return None, [f"the record is not readable: {path!r}"]
    if not isinstance(rec, dict):
        return None, [f"the record is not readable: {path!r}"]
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
    for k in REQUIRED_EXTRAS[dry]:
        p = os.path.join(ROOT, k)
        if k not in extras:
            problems.append(f"required file not pinned in the record: {k}")
        elif not os.path.isfile(p) or sha(p) != extras[k]:
            problems.append(f"pinned file differs from the record: {k}")
    for k, v in extras.items():
        p = os.path.join(ROOT, k)
        if k not in REQUIRED_EXTRAS[dry] and (not os.path.isfile(p) or sha(p) != v):
            problems.append(f"pinned file differs from the record: {k}")
    return rec, problems


def check_run_args(a, dry=False):
    """Problems with the arguments of a registered run (run_wiki.py): setting, data, seed, name, split, output folder."""
    problems = []
    root = os.path.join(ROOT, RUNS_ROOT[dry])
    for k, v in SETTING.items():
        if getattr(a, k) != v or isinstance(getattr(a, k), bool) != isinstance(v, bool):
            problems.append(f"setting {k} is {getattr(a, k)!r}, registered {v!r}")
    if a.data not in DATA:
        problems.append(f"data {a.data!r} is not a registered data version")
    if a.seed not in SEEDS[dry]:
        problems.append(f"seed {a.seed} is not a registered seed")
    name = run_name(a.data, a.seed, dry)
    if a.name != name:
        problems.append(f"name must be {name}")
    if a.split != SPLIT[dry]:
        problems.append(f"split must be {SPLIT[dry]}")
    if os.path.abspath(a.out_root) != os.path.abspath(root) or a.no_log:
        problems.append(f"a registered run writes to {RUNS_ROOT[dry]}/ and to the run log")
    if os.path.isdir(root):
        done = sorted(d for d in os.listdir(root) if d.endswith(f"_{name}"))
        if done:
            problems.append(f"a run of this data version and seed exists already: {done[0]}")
    return problems


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--extras" and set(args[1:]) <= {"--dry"}:
        print("\n".join(REQUIRED_EXTRAS["--dry" in args[1:]]))
        return 0
    if args == ["--runlog-prefix"]:
        print(json.dumps(runlog_prefix()))
        return 0
    if len(args) < 2 or args[0] != "--check":
        raise SystemExit(__doc__)
    dry = "--dry" in args[2:]
    rec, problems = preflight(args[1], dry)
    print(json.dumps({"record": args[1], "dry": dry, "frozen_utc": (rec or {}).get("frozen_utc"),
                      "plan_sha256": (rec or {}).get("plan_sha256"), "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
