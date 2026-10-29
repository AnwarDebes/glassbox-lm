#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# usage: log.sh <tool> <query-or-url> <hits/result>
# appends one row to search_log.md with the UTC time read now
L=${REPO_ROOT}/studies/text_understanding/novelty/n3d_2026-10-21/search_log.md
N=$(( $(grep -c '^| [0-9]' "$L") + 1 ))
T=$(date -u '+%Y-%m-%d %H:%M:%S')
q=$(printf '%s' "$2" | tr '|' '/')
r=$(printf '%s' "$3" | tr '|' '/')
printf '| %s | %s | %s | %s | %s |\n' "$N" "$T" "$1" "$q" "$r" >> "$L"
echo "logged #$N at $T"
