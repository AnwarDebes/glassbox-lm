"""Every script runs the way the stages run it (revision 3). run_e4_gpu.sh starts each script with python -I -B, and -I
leaves the script's own folder off sys.path. Revision 3's first end-to-end dry run stopped at STAGE=summary because
summarize_e4.py imported check_pins with only the E4 folder on sys.path (ModuleNotFoundError); the unit tests import
the scripts with scripts/ on the path, so they could not see it. These tests close that gap:
  - static: every script that imports a sibling script module (scripts/<name>.py) anywhere, even inside a function,
    puts scripts/ on sys.path at module level, before any module-level sibling import; every script that imports
    e4guard puts the E4 folder there;
  - dynamic: every script with a command line answers --help under python -I -B with the interpreter its stage uses
    (module-level imports resolve in isolated mode);
  - the two gobar.json texts run_e4_gpu.sh writes itself at STAGE=train_guard ("not testable" on train_guard.py exit 3,
    "void" on a MILP failure, exit 5) are valid JSON that summarize_e4.gate_outcome reads as such;
  - no API key reaches a client: offline_env.sh and e4_run.local_only_env remove every variable ending in _API_KEY,
    _API_BASE or _BASE_URL (no provider is named in the code) and send other HTTP to a dead proxy.
No server, no GPU, no network.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_scripts_isolated.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"
DLPY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
DL_SCRIPTS = ("train_guard.py", "milp_proof.py")            # run_e4_gpu.sh and stage2_offline_rev2.sh use .venv-dl
NO_COMMAND_LINE = ("code_hashes.py",)                        # runs at import (an old helper; no stage calls it)
SCRIPTS = sorted(p for p in (E4 / "scripts").glob("*.py"))
SIBLINGS = {p.stem for p in SCRIPTS}


def _env():
    return {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9", "TMPDIR": str(E4 / ".tmp")}


def _imported(tree) -> list[tuple[str, bool, int]]:
    """(top-level module name, imported at module level?, statement index at module level or -1)."""
    out, top = [], {id(s): i for i, s in enumerate(tree.body)}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for n in names:
            out.append((n, id(node) in top, top.get(id(node), -1)))
    return out


def _path_inserts(tree) -> list[tuple[int, str]]:
    """(statement index, source) of every module-level sys.path.insert(...)."""
    out = []
    for i, s in enumerate(tree.body):
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            src = ast.unparse(s.value)
            if src.startswith("sys.path.insert("):
                out.append((i, src))
    return out


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_sibling_and_e4guard_imports_have_their_path(path):
    tree = ast.parse(path.read_text())
    imps = _imported(tree)
    ins = _path_inserts(tree)
    sib = [(n, top, i) for n, top, i in imps if n in SIBLINGS and n != path.stem]
    if sib:
        at = [i for i, src in ins if "'scripts'" in src]
        assert at, f"{path.name} imports {sorted({n for n, _, _ in sib})} but never puts scripts/ on sys.path"
        for n, top, i in sib:
            assert not top or i > min(at), f"{path.name}: module-level import of {n} before scripts/ is on sys.path"
    if any(n == "e4guard" for n, _, _ in imps):
        # the published scripts put it there as f"{_REPO_ROOT}/studies/agent_guards/e4" (found from their own place)
        at = [i for i, src in ins if src.rstrip(")").endswith("str(E4") or str(E4) in src
              or "{_REPO_ROOT}/studies/agent_guards/e4'" in src]
        assert at, f"{path.name} imports e4guard but never puts the E4 folder on sys.path"


def test_the_failure_this_closes_is_caught_by_the_static_check():
    """The pre-fix summarize_e4.py: E4 on sys.path, check_pins imported inside main(). The check must flag it."""
    bad = ("import sys\nfrom pathlib import Path\nE4 = Path('/x')\nsys.path.insert(0, str(E4))\n"
           "def main():\n    from check_pins import FROZEN\n")
    tree = ast.parse(bad)
    sib = [n for n, _, _ in _imported(tree) if n in SIBLINGS]
    assert sib == ["check_pins"] and not [i for i, src in _path_inserts(tree) if "'scripts'" in src]


@pytest.mark.parametrize("path", [p for p in SCRIPTS if p.name not in NO_COMMAND_LINE], ids=lambda p: p.name)
def test_help_under_isolated_mode(path):
    py = DLPY if path.name in DL_SCRIPTS else PY
    p = subprocess.run([py, "-I", "-B", str(path), "--help"], capture_output=True, text=True, timeout=180, env=_env(),
                       cwd=str(E4 / ".tmp"))
    assert p.returncode == 0 and "usage:" in p.stdout, (p.returncode, p.stderr[-800:])


def _gobar_literals() -> dict:
    sh = (E4 / "scripts/run_e4_gpu.sh").read_text()
    out = {}
    for m in re.finditer(r'gobar_write "(\{.*?\})"\n', sh):
        txt = m.group(1).replace('\\"', '"').replace("$MIN_CLASS", "10").replace("$sd", "1")
        d = json.loads(txt)
        out[d["outcome"]] = d
    return out


def test_the_scripts_own_gobar_texts_are_json_the_summary_reads():
    import summarize_e4 as S

    lit = _gobar_literals()
    assert set(lit) == {"not testable", "void"}, sorted(lit)
    assert S.gate_outcome(lit["not testable"]) == "not testable" and lit["not testable"]["GO"] is False
    assert S.gate_outcome(lit["void"]) == "void" and lit["void"]["failed_check"] == "IC2b" and lit["void"]["seed"] == 1
    assert S.ic2b({}, "not testable")["ok"] is None                       # no integrity flag on the 'not testable' path


def test_offline_env_removes_every_api_key_variable():
    env = dict(_env(), FOO_API_KEY="x", BAR_BASE_URL="http://example.invalid", BAZ_API_BASE="y", KEEP_ME="1")
    p = subprocess.run(["bash", str(E4 / "scripts/offline_env.sh"), "env"], capture_output=True, text=True, timeout=60,
                       env=env)
    assert p.returncode == 0
    seen = dict(x.split("=", 1) for x in p.stdout.splitlines() if "=" in x)
    assert not [k for k in seen if k.endswith(("_API_KEY", "_API_BASE", "_BASE_URL"))], sorted(seen)
    assert seen.get("KEEP_ME") == "1" and seen.get("HTTP_PROXY") == "http://127.0.0.1:9" and seen.get(
        "CUDA_VISIBLE_DEVICES") == ""


def test_local_only_env_removes_every_api_key_variable(monkeypatch):
    import e4_run

    for k, v in (("FOO_API_KEY", "x"), ("BAR_BASE_URL", "http://example.invalid"), ("BAZ_API_BASE", "y"),
                 ("KEEP_ME", "1")):
        monkeypatch.setenv(k, v)
    saved = dict(os.environ)
    try:
        base = "http://127.0.0.1:18082/v1"
        e4_run.local_only_env(base)
        keys = [k for k in os.environ if k.endswith(("_API_KEY", "_API_BASE", "_BASE_URL"))]
        assert all(os.environ[k] in ("sk-local-no-key", base) for k in keys), {k: os.environ[k] for k in keys}
        assert "FOO_API_KEY" not in os.environ and "BAR_BASE_URL" not in os.environ and "BAZ_API_BASE" not in os.environ
        assert os.environ["KEEP_ME"] == "1" and os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"
        assert os.environ["NO_PROXY"] == "127.0.0.1,localhost"
    finally:                                   # the test process keeps its own environment for the other tests
        os.environ.clear()
        os.environ.update(saved)
