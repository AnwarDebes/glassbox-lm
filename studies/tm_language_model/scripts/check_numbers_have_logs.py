#!/usr/bin/env python3
"""Lint for protocol rules 2 and 3: a number labelled "measured" must point at a log.

    check_numbers_have_logs.py [--root DIR]

Scans docs/**/*.md and CHANGELOG.md for the token "[measured". Each line that carries
the token must cite a path that starts with results/ and EXISTS under the project root,
either on the same line or in the same paragraph (a paragraph is a run of non-blank
lines, so a whole markdown table counts as one paragraph).

Two exemptions, so that documents can describe the labelling convention itself:
  - a token written as inline code, with a backtick directly before the bracket, is a
    mention and not a claim. Example: `[measured]`.
  - a list item that DEFINES the label, that is a line starting with a list marker
    followed by the closed label and a colon. Example: "- [measured]: produced by a run".
    Only that leading label is exempt, the rest of the line is still checked.

Exit 0 when every claim has a log. Exit 1 with one `path:line: reason` line per
violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOKEN = "[measured"
PATH_RE = re.compile(r"results/[A-Za-z0-9_./+=@-]+")
CLAIM_RE = re.compile(r"(?<!`)" + re.escape(TOKEN))
DEFINITION_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[measured\]\s*:")
TRAILING = ".,;:)]}'\"`"


def results_paths(text: str) -> list[str]:
    """Every results/ path cited in `text`, with sentence punctuation stripped."""
    return [match.rstrip(TRAILING) for match in PATH_RE.findall(text)]


def path_exists(root: Path, cited: str) -> bool:
    """True when `cited` names an existing file or directory INSIDE <root>/results."""
    results_root = (root / "results").resolve()
    target = (root / cited).resolve()
    if target != results_root and results_root not in target.parents:
        return False
    return target.exists() and target != results_root


def has_claim(line: str) -> bool:
    """True when `line` carries the token as a claim (see the exemptions above)."""
    return bool(CLAIM_RE.search(DEFINITION_RE.sub("", line, count=1)))


def paragraphs(lines: list[str]) -> list[tuple[int, int]]:
    """Half-open (start, end) index ranges of the runs of non-blank lines."""
    spans = []
    start = None
    for index, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def check_file(root: Path, path: Path) -> list[tuple[int, str]]:
    """(line number, reason) for every measured claim without an existing log path."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    problems = []
    for start, end in paragraphs(lines):
        block = lines[start:end]
        block_ok = any(path_exists(root, cited) for cited in results_paths("\n".join(block)))
        for offset, line in enumerate(block):
            if not has_claim(line):
                continue
            if any(path_exists(root, cited) for cited in results_paths(line)) or block_ok:
                continue
            cited = results_paths("\n".join(block))
            if cited:
                reason = "cited path does not exist: " + ", ".join(sorted(set(cited)))
            else:
                reason = "no results/ path on this line or in this paragraph"
            problems.append((start + offset + 1, reason))
    return problems


def target_files(root: Path) -> list[Path]:
    files = sorted((root / "docs").rglob("*.md")) if (root / "docs").is_dir() else []
    changelog = root / "CHANGELOG.md"
    if changelog.is_file():
        files.append(changelog)
    return files


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Every [measured number needs a results/ path.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()

    files = target_files(root)
    total = 0
    for path in files:
        for line_no, reason in check_file(root, path):
            total += 1
            print(f"{path.relative_to(root).as_posix()}:{line_no}: {TOKEN} without a log: {reason}")
    if total:
        print(f"check_numbers_have_logs: FAIL, {total} claim(s) without an existing results/ path")
        return 1
    print(f"check_numbers_have_logs: OK, {len(files)} files scanned")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
