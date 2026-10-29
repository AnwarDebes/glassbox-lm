#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1: download C4 en train shards 00000 and 00001 (allenai/c4 on Hugging Face, pinned revision) into their own
# folder data_untrusted/c4_en/, check each file's sha256 against the LFS object id that the Hugging Face API lists
# for that revision, and write SOURCE.txt (first line "repo@revision utc", then "sha256  file" lines, the format of
# the other data_untrusted folders, so `tail -n +2 SOURCE.txt | sha256sum -c` re-checks them). Data only: nothing in
# the folder is ever run. Licence (dataset card, cardData.license): odc-by; C4 is also subject to Common Crawl's terms of use.
set -euo pipefail
rev=1588ec454efa1a09f29cd18ddd04fe05fc8653a2
dir=${REPO_ROOT}/studies/text_understanding/data_untrusted/c4_en
declare -A want=(
  [c4-train.00000-of-01024.json.gz]=8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9
  [c4-train.00001-of-01024.json.gz]=b945059cd1a343cabe311881b7840a6f0363f570e745a0eff0e687e266f6b55d
)
mkdir -p "$dir"
[ -e "$dir/SOURCE.txt" ] && { echo "SOURCE.txt exists: refusing to overwrite"; exit 1; }
for f in c4-train.00000-of-01024.json.gz c4-train.00001-of-01024.json.gz; do
  curl -sSfL --retry 3 -o "$dir/$f.part" "https://huggingface.co/datasets/allenai/c4/resolve/$rev/en/$f"
  got=$(sha256sum "$dir/$f.part" | cut -d' ' -f1)
  [ "$got" = "${want[$f]}" ] || { echo "sha256 mismatch for $f: $got"; exit 1; }
  mv "$dir/$f.part" "$dir/$f"
done
{
  echo "allenai/c4@$rev $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  (cd "$dir" && sha256sum c4-train.00000-of-01024.json.gz c4-train.00001-of-01024.json.gz)
} > "$dir/SOURCE.txt"
cat "$dir/SOURCE.txt"
