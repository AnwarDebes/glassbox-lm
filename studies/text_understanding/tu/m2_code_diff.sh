#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M2: the exact code changes from M1 (added after the review of draft 1, 2026-10-25: data MAJOR 1, design MINOR 4).
# For each of the 8 pairs tu/m1_X -> tu/m2_X it prints the parent's and the copy's full sha256 and `diff -u` with fixed
# labels (no file times), so the output depends on the file contents only; then the same for m2_data.py against its
# first version (2026-10-23, kept as tu/archive/m2_data_draft0_f4d2a44b.py), and the list of files with no parent.
# Saved once as tu/m2_code_diff_vs_m1.txt (pinned); tests/test_m2.py and the verify step regenerate it and compare
# byte for byte.
# Run: bash tu/m2_code_diff.sh > tu/m2_code_diff_vs_m1.txt
set -u
tu=${REPO_ROOT}/studies/text_understanding/tu
pair() { local a=$1 b=$2
  echo "=== $a -> $b"
  echo "parent $(sha256sum "$tu/$a" | cut -c1-64)  $a"
  echo "copy   $(sha256sum "$tu/$b" | cut -c1-64)  $b"
  diff -u --label "$a" --label "$b" "$tu/$a" "$tu/$b"
  echo; }
for p in feats.py tm.py lr.py verdict.py chain.sh write_pins.sh check_cache.py standin.py; do pair "m1_$p" "m2_$p"; done
pair archive/m2_data_draft0_f4d2a44b.py m2_data.py
echo "=== files with no M1 parent: m2_audit_run.py m2_dry_gpu.sh m2_codecheck.py m2_freeze_check.py m2_code_diff.sh"
exit 0
