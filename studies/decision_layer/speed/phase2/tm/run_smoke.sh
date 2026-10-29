#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# run_smoke.sh NAME CMD...   one QUICK smoke test (under 2 minutes) behind the CPU gate of this phase:
#   cpu_load.py 10 must show at most 4.5 cores (else exit 3, nothing started); CUDA hidden; 2 threads; nice 19;
#   HF_HUB_OFFLINE=1; no bytecode; a 170 s timeout. Output: smokes/NAME.out (gate line, UTC start and end, the command,
#   its output, its exit code); prints NAME, the exit code and the output's sha256. Refuses an existing NAME.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dl=${REPO_ROOT}/studies/decision_layer
py="$dl/.venv-dl/bin/python"
name="${1:?usage: run_smoke.sh NAME CMD...}"; shift
[ "$#" -ge 1 ] || { echo "usage: run_smoke.sh NAME CMD..." >&2; exit 2; }
out="$here/smokes/$name.out"
[ -e "$out" ] && { echo "refused: $out exists" >&2; exit 2; }
mkdir -p "$here/smokes"
m=$(CUDA_VISIBLE_DEVICES='' nice -n 19 "$py" -I -B "$dl/scripts/cpu_load.py" 10); c=$(awk '{print $3}' <<< "$m")
awk -v c="$c" 'BEGIN { exit !(c <= 4.5) }' || { echo "gate closed (> 4.5 cores): $m"; exit 3; }
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
{ echo "gate: $m"; echo "start $(date -u +%FT%TZ): $*"; } > "$out"
timeout 170 nice -n 19 "$@" >> "$out" 2>&1
rc=$?
echo "end $(date -u +%FT%TZ) exit $rc" >> "$out"
echo "$name exit $rc"
sha256sum "$out"
