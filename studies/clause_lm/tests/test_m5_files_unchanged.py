"""The three files that ran M5 step 1 on 2026-10-14 keep the code of their run-folder copies.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_m5_files_unchanged.py

Run from studies/clause_lm. CPU only, under a second. Needs the local run folders of 2026-10-14; skipped without them.
src/m5_run.py got new docstring text on 2026-10-25 (its first paragraph was stale); src/m5_report.py and src/runlog.py
were not edited. The run records hold copies of all three.

Asserts:
- src/m5_report.py and src/runlog.py are byte identical to the copies in the report run folder;
- src/m5_run.py has the same code as the copies in the 13 training run folders of 2026-10-14 (all 13 copies are one
  file): the parsed syntax trees are equal once the docstrings are removed;
- the copy's SHA-256 is the one recorded before the edit (23d53448...), and the published file now differs from it.
"""

import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
RES = toyutil.STUDY / "results"
OLD_M5_RUN = "23d53448afcb76c675c4dcf5600b3508edf7e6f0fe6a9a9fe4109de78d73f6b3"


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    copies = sorted(RES.glob("20261014T*_m5_*_torchgpu/m5_run.py"))
    report = RES / "20261014T222730Z_m5_summary_sorted_torchgpu"
    if not copies or not report.exists():
        print("skip: the run folders of 2026-10-14 are not on this machine")
        check.done()
    src = toyutil.SRC
    check(sha(src / "m5_report.py") == sha(report / "m5_report.py"), "src/m5_report.py is byte identical to its copy")
    check(sha(src / "runlog.py") == sha(report / "runlog.py") and all(sha(c.parent / "runlog.py") == sha(src / "runlog.py")
                                                                      for c in copies),
          "src/runlog.py is byte identical to its copies")
    hashes = {sha(c) for c in copies}
    check(len(copies) == 13 and hashes == {OLD_M5_RUN}, f"the 13 run-folder copies of m5_run.py are one file ({len(copies)})")
    now = (src / "m5_run.py").read_text()
    then = copies[0].read_text()
    check(strip_docstrings(ast.parse(now)) == strip_docstrings(ast.parse(then)),
          "src/m5_run.py has the same code as the run-folder copies (docstrings removed)")
    check(sha(src / "m5_run.py") != OLD_M5_RUN, f"src/m5_run.py now differs in its docstring ({sha(src / 'm5_run.py')})")
    check.done()


if __name__ == "__main__":
    main()
