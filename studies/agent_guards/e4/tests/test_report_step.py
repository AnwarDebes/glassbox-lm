"""The report step (revision 3; re-review item 7; the decision 6): section 11.5 of the registration and the
last comment of scripts/run_e4_gpu.sh are the same text, line for line; there is one campaign name (e4reg_<yyyymmdd>);
the stage loop stops after a failed stage.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_report_step.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
REG = E4 / "REGISTRATION_E4.md"
SH = E4 / "scripts/run_e4_gpu.sh"


def block_11_5(text: str) -> list[str]:
    i = text.index("\n### 11.5 ")
    j = text.index("```", i)
    k = text.index("\n", j)
    e = text.index("\n```", k)
    return text[k + 1:e].split("\n")


def script_last_comment(text: str) -> list[str]:
    lines = text.rstrip("\n").split("\n")
    i = max(n for n, x in enumerate(lines) if x.startswith("# THE REPORT STEP"))
    out = []
    for x in lines[i:]:
        assert x == "#" or x.startswith("# "), f"not a comment line at the script's end: {x!r}"
        out.append(x[2:] if x.startswith("# ") else "")
    return out


def test_section_11_5_is_the_scripts_last_comment():
    a, b = block_11_5(REG.read_text()), script_last_comment(SH.read_text())
    assert a == b, "\n".join(f"{x!r} | {y!r}" for x, y in zip(a, b) if x != y) or (len(a), len(b))


def test_one_campaign_name_and_the_loop_stops_after_a_failure():
    block = "\n".join(script_last_comment(SH.read_text()))
    assert "C=e4reg_<yyyymmdd>" in block and "|| break" in block
    for loop_line in [x for x in block.split("\n") if "bash scripts/run_e4_gpu.sh" in x and "$s" in x]:
        assert loop_line.rstrip().endswith("|| break")
    for text in (REG.read_text(), SH.read_text()):
        assert "e4_<yyyymmdd>" not in text and "e4_$(date" not in text      # the old name is gone
    assert "SLOT_MINUTES=<slot> bash" not in block                       # the slot enters only at the freeze
