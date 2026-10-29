#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# dry_bg2_probe.sh ROOT OUT_PREFIX   CPU dry test of BG2's sampling path (no GPU anywhere).
#   Starts review/bench/run_gpu_job.sh ROOT B1 patch_R1.py --dry-cpu in the background (CUDA hidden by the script's dry
#   branch: the job builds its view on the CPU, then stops at the first CUDA call, writing no row), retries
#   spin_check.py --descendant-of RUNNER every 0.5 s (at most 40 tries) until the runner's subtree holds exactly one
#   python, samples that python for 3 s while it builds the view, waits for the runner and reports both exit codes.
#   Files: OUT_PREFIX.spin.json (spin_check's JSON), OUT_PREFIX.runner.txt (run_gpu_job.sh's stdout / stderr),
#   OUT_PREFIX.tries.txt (one line per try). Only processes started here are waited for; none is signalled.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
bench=${REPO_ROOT}/studies/decision_layer/speed/review/bench
root="${1:?usage: dry_bg2_probe.sh ROOT OUT_PREFIX}"; pre="${2:?usage: dry_bg2_probe.sh ROOT OUT_PREFIX}"
for f in "$pre.spin.json" "$pre.runner.txt" "$pre.tries.txt"; do
  [ -e "$f" ] && { echo "refused: $f exists" >&2; exit 2; }
done
bash "$bench/run_gpu_job.sh" "$root" B1 "$here/patch_R1.py" --dry-cpu > "$pre.runner.txt" 2>&1 &
runner=$!
echo "$(date -u +%FT%TZ) runner pid $runner (run_gpu_job.sh B1 patch_R1.py --dry-cpu)" > "$pre.tries.txt"
spin_rc=9
for i in $(seq 1 40); do
  "$py" -I -B "$here/spin_check.py" --descendant-of "$runner" --seconds 3 --out "$pre.spin.json.tmp" > /dev/null 2>&1
  spin_rc=$?
  echo "$(date -u +%FT%TZ) try $i spin_check exit $spin_rc" >> "$pre.tries.txt"
  [ "$spin_rc" -eq 0 ] && break
  kill -0 "$runner" 2> /dev/null || break
  sleep 0.5
done
[ -e "$pre.spin.json.tmp" ] && mv "$pre.spin.json.tmp" "$pre.spin.json"
wait "$runner"
runner_rc=$?
echo "$(date -u +%FT%TZ) runner exit $runner_rc; spin_check exit $spin_rc" >> "$pre.tries.txt"
echo "runner exit $runner_rc; spin_check exit $spin_rc"
cat "$pre.runner.txt"
[ -e "$pre.spin.json" ] && sha256sum "$pre.spin.json"
sha256sum "$pre.tries.txt" "$pre.runner.txt"
