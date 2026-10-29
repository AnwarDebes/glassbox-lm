#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# make_root.sh DEST: a private run root for the GPU benchmark jobs (bench/jobs.tsv), so that a benchmark run of the decision_layer
# UNEDITED entry scripts can never write into the decision_layer results.
#
# Why: scripts/abcd_explore.py appends its row to <ROOT>/results/explore/abcd.jsonl and, with --save, (over)writes
# <ROOT>/results/explore/abcd_models/<tag>_seed<k>.npz, where ROOT = the script's own resolved parents[1]. Run from the
# study folder, a "before" rerun would append to the decision_layer record and overwrite the reference model it is compared with.
#
# Layout (everything under DEST; nothing outside DEST is created or changed):
#   scripts/{abcd_explore,table_run,newtable_run}.py  REAL byte copies (resolve() must not lead back to the study);
#                                                    MANIFEST.copies lists both SHA-256 and the run refuses on a mismatch
#   src -> the study's src                           imports only (run python with -B: no .pyc is written there);
#                                                    dl.abcd / dl.count_text find the study's data from their own files
#   data/emb -> the study's data/emb                 encoder caches (read-only)
#   data/tables/c4words_a3B_v1/                      a REAL folder (newtable_run.py checks that its parent is
#       MANIFEST.sha256, table.npz, tokenize_ref.py   <ROOT>/data/tables) holding links to the three files it reads
#   results/explore/                                 the private record: abcd.jsonl and abcd_models/ of benchmark runs
set -euo pipefail
dl=${REPO_ROOT}/studies/decision_layer
dest="${1:?usage: make_root.sh DEST}"
case "$(realpath -m "$dest")" in
  "$dl"|"$dl"/*) echo "refused: DEST is inside the study folder" >&2; exit 1 ;;
esac
[ -e "$dest" ] && { echo "refused: $dest exists (make a fresh root)" >&2; exit 1; }
mkdir -p "$dest/scripts" "$dest/data/tables/c4words_a3B_v1" "$dest/results/explore"
for f in abcd_explore.py table_run.py newtable_run.py; do
  cp -p "$dl/scripts/$f" "$dest/scripts/$f"
done
ln -s "$dl/src" "$dest/src"
ln -s "$dl/data/emb" "$dest/data/emb"
for f in MANIFEST.sha256 table.npz tokenize_ref.py; do
  ln -s "$dl/data/tables/c4words_a3B_v1/$f" "$dest/data/tables/c4words_a3B_v1/$f"
done
{
  for f in abcd_explore.py table_run.py newtable_run.py; do
    a=$(sha256sum "$dest/scripts/$f" | cut -c1-64); b=$(sha256sum "$dl/scripts/$f" | cut -c1-64)
    [ "$a" = "$b" ] || { echo "copy differs: $f" >&2; exit 1; }
    echo "$a  scripts/$f  (= $dl/scripts/$f)"
  done
} > "$dest/MANIFEST.copies"
echo "made $dest"; cat "$dest/MANIFEST.copies"
