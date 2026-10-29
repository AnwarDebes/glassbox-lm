#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1-S: download C4 en train shards 00002-00007 (allenai/c4 @1588ec45) into data_untrusted/c4_en/, check each sha256
# against the LFS id the Hugging Face API lists for that revision, write SOURCE2.txt (repo@revision utc, then
# "sha256  file" lines). Data only, never run. Licence ODC-BY, also bound by the Common Crawl terms of use.
set -euo pipefail
rev=1588ec454efa1a09f29cd18ddd04fe05fc8653a2
dir=${REPO_ROOT}/studies/text_understanding/data_untrusted/c4_en
declare -A want=(
  [c4-train.00002-of-01024.json.gz]=2967dc7e587ced6ecb9ba617ad2d4c44901467969de5bf5b0f5a9e5b70555d75
  [c4-train.00003-of-01024.json.gz]=b79d9abef5741578929be0d59db9ca652a8276207ef18a944b7a5f11fef5beb6
  [c4-train.00004-of-01024.json.gz]=cd9f98eac2bc6062f55d9a36bd744cc924a78ea2fd998830e0034e4456f5d014
  [c4-train.00005-of-01024.json.gz]=8ac5907a54dbc7ab9c14624448c7c3f6afed33af9d0a855f1eae955e62e255b9
  [c4-train.00006-of-01024.json.gz]=8fd9b9a4b74c9414466b245ebda7db041e7bd8603971de51b5db782bd758aac7
  [c4-train.00007-of-01024.json.gz]=41dd377a1ba6b72eab0260c39c626fe45ab6b649d42d57b311d3ba21a0337cd0
)
[ -e "$dir/SOURCE2.txt" ] && { echo "SOURCE2.txt exists: refusing to overwrite"; exit 1; }
files=""
for n in 00002 00003 00004 00005 00006 00007; do
  f=c4-train.$n-of-01024.json.gz
  curl -sSfL --retry 3 -o "$dir/$f.part" "https://huggingface.co/datasets/allenai/c4/resolve/$rev/en/$f"
  got=$(sha256sum "$dir/$f.part" | cut -d' ' -f1)
  [ "$got" = "${want[$f]}" ] || { echo "sha256 mismatch for $f: $got"; exit 1; }
  mv "$dir/$f.part" "$dir/$f"
  files="$files $f"
done
{ echo "allenai/c4@$rev $(date -u +%Y-%m-%dT%H:%M:%SZ)"; (cd "$dir" && sha256sum $files); } > "$dir/SOURCE2.txt"
cat "$dir/SOURCE2.txt"
