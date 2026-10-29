"""Paths of the C-core arm. Inputs are read only; every output goes under C/."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
import sys
from pathlib import Path

ROOT = Path(f"{_WS_ROOT}")
TMLANG = ROOT / "glassbox-lm" / "studies" / "language_part3"
C = TMLANG / "C"
TMP = C / "tmp"
DATA = C / "data"
DATA_DRY = C / "data_dry"
RUNS = C / "runs"
LISTINGS = RUNS / "listings"
REG = C / "R1_C-core.md"
FROZEN = C / "R1_C-core.frozen.json"
PLAN = TMLANG / "PLAN_part3.md"
PROPOSAL = TMLANG / "proposal_C-core.md"

STUDIES = ROOT / "glassbox-lm" / "studies"
TALLY = STUDIES / "tally"
TALLY_SRC = TALLY / "src"
TALLY_RESULTS = TALLY / "results"
DL = STUDIES / "decision_layer"
C4DIR = DL / "data" / "tables" / "c4words_S1000_v1"
PTB_RAW = STUDIES / "tm_language_model" / "data" / "ptb" / "raw"
PTB_TEST_FILE = PTB_RAW / "test" / "text.utf8"  # never opened by this package (no function reads it)
CPU_LOAD = DL / "scripts" / "cpu_load.py"
VENV_TM = STUDIES / "tm_language_model" / ".venv-tm" / "bin" / "python"
VENV_DL = DL / ".venv-dl" / "bin" / "python"

# Folders listed before and after every job (PLAN_part3.md section 6, "Files and run rules"), plus
# tally src/tally/ and its __pycache__/, which this arm imports (ptb.py).
LISTED_DIRS = [
    TALLY_SRC / "tally",
    TALLY_SRC / "tally" / "__pycache__",
    TALLY_SRC / "tally" / "floor",
    TALLY_SRC / "tally" / "floor" / "__pycache__",
    DL / "src" / "dl",
    DL / "src" / "dl" / "__pycache__",
    STUDIES / "tm_language_model" / "src" / "tmlm" / "eval" / "baselines",
    C4DIR,
    C4DIR / "__pycache__",
    ROOT / "tools" / "kenlm" / "bin",
]


def setup():
    """Put tally's src and this arm's folder on sys.path (python -I ignores PYTHONPATH)."""
    for p in (str(TALLY_SRC), str(C)):
        if p not in sys.path:
            sys.path.insert(0, p)
