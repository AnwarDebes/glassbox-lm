"""scripts/check_numbers_have_logs.py on temp fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(scripts_dir: Path, name: str, root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(scripts_dir / name), "--root", str(root)],
                          capture_output=True, text=True, timeout=60)


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ check_numbers_have_logs


def make_run(root: Path, name: str = "20260101T000000Z_e1") -> str:
    write(root, f"results/{name}/metrics.json", "{}\n")
    return f"results/{name}"


def test_numbers_pass_when_every_claim_cites_an_existing_log(scripts_dir, tmp_path):
    run = make_run(tmp_path)
    write(tmp_path, "docs/05-experiments.md",
          f"# E1\n\nTop-1 is 12.5% [measured, {run}/metrics.json].\n\n"
          f"Throughput 3100 ex/s [measured] (log: `{run}`).\n\n"
          "An estimate: 50k ex/s [derived], and 1 GB [unverified].\n")
    write(tmp_path, "CHANGELOG.md", f"- E1 done, top-1 12.5% [measured: {run}/metrics.json]\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert "OK" in proc.stdout


def test_numbers_paragraph_scope(scripts_dir, tmp_path):
    run = make_run(tmp_path)
    write(tmp_path, "docs/table.md",
          "| model | top-1 |\n"
          "|---|---|\n"
          "| TM | 12.5% [measured] |\n"
          "| bigram | 9.1% [measured] |\n"
          f"Source of both rows: {run}/metrics.json\n"
          "\n"
          "Next paragraph: 7.7% [measured] has no path of its own.\n"
          "\n"
          f"The log is in {run}, but that is another paragraph.\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 1
    lines = proc.stdout.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("docs/table.md:7: [measured without a log: no results/ path")
    assert "FAIL, 1 claim" in lines[1]


def test_numbers_fail_for_missing_and_nonexistent_paths(scripts_dir, tmp_path):
    make_run(tmp_path)
    write(tmp_path, "docs/a.md", "Accuracy 99% [measured].\n")
    write(tmp_path, "docs/notes/b.md", "Speed 5x [measured, results/20260101T000000Z_nope/log.txt]\n")
    write(tmp_path, "CHANGELOG.md", "fine\n\n- 3 points [measured, see results/]\n")
    write(tmp_path, "docs/c.md", "Escape 1% [measured, results/../docs/a.md]\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 1
    out = proc.stdout
    assert "docs/a.md:1: [measured without a log: no results/ path" in out
    assert "docs/notes/b.md:1: [measured without a log: cited path does not exist" in out
    assert "results/20260101T000000Z_nope/log.txt" in out
    assert "CHANGELOG.md:3:" in out
    assert "docs/c.md:1:" in out
    assert "FAIL, 4 claim" in out


def test_numbers_inline_code_mention_is_not_a_claim(scripts_dir, tmp_path):
    write(tmp_path, "docs/00-README.md",
          "Labels: `[measured]` needs a log, `[derived]` is arithmetic.\n\n"
          "But this is a claim: 5% [measured].\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 1
    lines = proc.stdout.splitlines()
    assert len(lines) == 2 and lines[0].startswith("docs/00-README.md:3:")


def test_numbers_label_definition_item_is_not_a_claim(scripts_dir, tmp_path):
    write(tmp_path, "docs/00-README.md",
          "## Labelling convention\n"
          "- [measured]: produced by a run on this machine; carries a log path.\n"
          "- [derived]: arithmetic.\n"
          "1. [measured]: same, in a numbered list.\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 0, proc.stdout

    write(tmp_path, "docs/sneaky.md",
          "- [measured]: a definition that smuggles a claim, 9% [measured].\n"
          "- [measured: nothing] is not a definition.\n"
          "Prose [measured]: not a list item.\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 1
    flagged = [line.split(":")[1] for line in proc.stdout.splitlines()[:-1]]
    assert flagged == ["1", "2", "3"]


def test_numbers_only_docs_and_changelog_are_scanned(scripts_dir, tmp_path):
    write(tmp_path, "RULES.md", "Protocol rule 2: numbers are labelled [measured], derived, or unverified.\n")
    write(tmp_path, "src/x.py", "# 5% [measured]\n")
    write(tmp_path, "docs/readme.txt", "5% [measured]\n")
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", tmp_path)
    assert proc.returncode == 0, proc.stdout


def test_numbers_real_project_tree_is_clean(scripts_dir, repo_root):
    proc = run_script(scripts_dir, "check_numbers_have_logs.py", repo_root)
    assert proc.returncode == 0, proc.stdout
