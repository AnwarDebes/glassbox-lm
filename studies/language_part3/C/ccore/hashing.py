"""SHA-256 of the files a C job depends on, and the check against the frozen R1 record.

Tracked (enforced): this arm's code (ccore/, scripts/, tests/), the imported read-only code (tally
ptb.py and floor/; since the step-9 amendment also dl tm.py and cpu_tm_fast.py, which step 9 imports)
and the input data (PTB train and validation text, the C4 word table, T0's frozen per-position
outputs and metrics). A real stage refuses to start when the frozen record is missing or any tracked
file changed or disappeared since the freeze. Since the step-9 amendment (code review of 2026-10-21,
issue 7) a real stage also refuses any tracked file that is not in the record, and a record whose own
SHA-256 is not in LOG.md (the R1 [C-core] line, or the dated amendment line that names a record
written by scripts/freeze.py --amend). Dry runs list new files and do not need the LOG.md line.
Recorded (not enforced): the plan, proposal C, the CPU gate script and dl tm_init.py; their hashes
at the freeze are kept in the record, so a later edit of the plan for another arm does not block C's
stages.
"""
import glob
import hashlib
import json
import os

from . import paths

LOGMD = paths.TMLANG / "LOG.md"     # the binding R1 [C-core] line and its dated amendment lines
T0_PATTERNS = ("*_t0_final_coal_c2000_s20_seed{s}", "*_t0_final_flat_c100_s5_seed{s}",
               "*_t0_logreg_of_final_coal_c2000_s20_seed{s}")
SEEDS = tuple(range(1, 11))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def t0_runs():
    """{(kind, seed): run folder} for the 30 frozen T0 runs (exactly one folder per pattern)."""
    out = {}
    for kind, pat in zip(("coal", "flat", "logreg"), T0_PATTERNS):
        for s in SEEDS:
            hits = sorted(glob.glob(str(paths.TALLY_RESULTS / pat.format(s=s))))
            if len(hits) != 1:
                raise SystemExit(f"T0 {kind} seed {s}: expected one run folder, found {hits}")
            out[(kind, s)] = hits[0]
    return out


def code_files():
    files = []
    for d in ("ccore", "scripts", "tests"):
        for ext in ("*.py", "*.sh"):
            files += sorted(glob.glob(str(paths.C / d / ext)))
    tally = paths.TALLY_SRC / "tally"
    files += [str(tally / "__init__.py"), str(tally / "ptb.py")]
    files += sorted(glob.glob(str(tally / "floor" / "*.py")))
    files += [str(paths.DL / "src" / "dl" / "tm.py"), str(paths.DL / "src" / "dl" / "cpu_tm_fast.py")]  # step 9
    return files


def reference_files():
    return [str(paths.PLAN), str(paths.PROPOSAL), str(paths.CPU_LOAD), str(paths.DL / "src" / "dl" / "tm_init.py")]


def reference_snapshot():
    return {rel(p): sha256(p) for p in reference_files()}


def input_files():
    files = [str(paths.PTB_RAW / "train" / "text.utf8"), str(paths.PTB_RAW / "validation" / "text.utf8"),
             str(paths.C4DIR / "words.txt"), str(paths.C4DIR / "vectors.npy"),
             str(paths.C4DIR / "MANIFEST.sha256")]
    for (_, _), d in sorted(t0_runs().items()):
        files += [os.path.join(d, "bits.npz"), os.path.join(d, "metrics.json")]
    return files


def rel(p):
    p = os.path.realpath(p)
    root = os.path.realpath(paths.ROOT)
    return os.path.relpath(p, root)


def snapshot():
    return {rel(p): sha256(p) for p in code_files() + input_files()}


def check_frozen(require=True):
    """Compare the tracked files with the frozen R1 record. Returns (record, new_files). A real stage
    (require=True) also refuses new tracked files and a record whose SHA-256 LOG.md does not hold."""
    if not paths.FROZEN.exists():
        if require:
            raise SystemExit(f"no frozen registration ({paths.FROZEN}); real stages run only after R1")
        return None, []
    rec = json.loads(paths.FROZEN.read_text())
    if sha256(paths.REG) != rec["registration_sha256"]:
        raise SystemExit("the registration text changed after the freeze")
    now = snapshot()
    changed = [k for k, v in rec["files"].items() if k in now and now[k] != v]
    gone = [k for k in rec["files"] if k not in now]
    new = sorted(k for k in now if k not in rec["files"])
    if changed or gone:
        raise SystemExit(f"frozen files changed: {changed}; gone: {gone}")
    if require:
        if new:
            raise SystemExit(f"tracked files that are not in the frozen record: {new}; a dated amendment line and "
                             "scripts/freeze.py --amend are needed first")
        h = sha256(paths.FROZEN)
        if not LOGMD.exists() or h not in LOGMD.read_text():
            raise SystemExit(f"{LOGMD} has no line with the frozen record's SHA-256 {h}: write the R1 or amendment "
                             "line first")
    return rec, new


def manifest(files):
    """{relative path: sha256} for output files of a stage."""
    return {rel(p): sha256(p) for p in files}


def write_json(path, obj):
    tmp = str(path) + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
