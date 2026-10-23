#!/usr/bin/env bash
# r4b_chains.sh : after the running R4b jobs end, run the missing R4b seeds in three sequential
# chains (two for the bge-base cells, one for the MiniLM cells) to bound GPU memory; each job is
# the registered command; expandable segments reduce allocator fragmentation (no effect on results).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
gpu="${here}/../tm_language_model/scripts/gpu_run.sh"
while ps -eo args | grep -q "[r]egistration R4b"; do sleep 30; done
.venv-dl/bin/python scripts/missing_seeds.py R4b > logs/r4b_missing_before_chains.txt
python_jobs() {  # cell seeds...
  local cell="$1"; shift
  local ds="${cell%%_*}" enc="${cell#*_}"
  echo ".venv-dl/bin/python scripts/confirm_router.py --registration R4b --clauses 16000 --dataset $ds --encoder $enc --name $cell --seeds $*"
}
chain() {  # file
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$gpu" shared -- bash -c "$line" >> logs/R4b.out 2>&1
  done < "$1"
}
.venv-dl/bin/python - <<'PY' > /dev/null
import re
from pathlib import Path
lines = [l for l in Path("logs/jobs_R4b_missing.txt").read_text().splitlines() if l.strip()]
bge = [l for l in lines if "bge-base" in l]
mini = [l for l in lines if "minilm" in l]
def split_pairs(l):
    seeds = re.search(r"--seeds (.*)$", l).group(1).split()
    base = l[: l.index("--seeds")]
    return [base + "--seeds " + " ".join(seeds[i:i + 2]) for i in range(0, len(seeds), 2)]
bge = [x for l in bge for x in split_pairs(l)]
mini = [x for l in mini for x in split_pairs(l)]
Path("logs/r4b_chain1.txt").write_text("\n".join(bge[0::2]) + "\n")
Path("logs/r4b_chain2.txt").write_text("\n".join(bge[1::2]) + "\n")
Path("logs/r4b_chain3.txt").write_text("\n".join(mini) + "\n")
PY
chain logs/r4b_chain1.txt & chain logs/r4b_chain2.txt & chain logs/r4b_chain3.txt &
wait
