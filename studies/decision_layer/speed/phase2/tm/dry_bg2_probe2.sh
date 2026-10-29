#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# dry_bg2_probe2.sh ROOT OUT_PREFIX   CPU dry test of the slot-2 BG2 path on the REAL process tree (no GPU anywhere).
#   ROOT: a fresh private root from review/bench/make_root.sh.
#   1. starts review/bench/run_gpu_job.sh ROOT B1 patch_R1.py --dry-cpu in the background (stdout -> OUT_PREFIX.runner.out,
#      stderr -> OUT_PREFIX.runner.err). Its dry branch hides CUDA: the job builds its view on the CPU (about 15 s, 2
#      threads, nice 19), then stops at its first CUDA call with rc 1 and writes no row;
#   2. runs bg2_sample2.sh beside it, keyed on the patch's IMPORT line (printed before the view build; on the GPU the
#      sampler keys on the construction line instead), with SETTLE 1, N 2, WINDOW 3, GAP 1, so that both tools sample the
#      job's python (the runner's pgrep -P child) while it builds the view;
#   3. waits for the runner, then runs check_job_log.py on the runner's last stdout line with --expect-rc 1 and
#      --sample-log (-> OUT_PREFIX.logcheck.json).
#   Expected: sampler exit 0 with both tools on one pid, runner exit 1 (the CUDA stop), check_job_log PASS.
#   Only processes started here are waited for; none is signalled. Refuses existing outputs.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
bench=${REPO_ROOT}/studies/decision_layer/speed/review/bench
root="${1:?usage: dry_bg2_probe2.sh ROOT OUT_PREFIX}"; pre="${2:?usage: dry_bg2_probe2.sh ROOT OUT_PREFIX}"
for f in "$pre.runner.out" "$pre.runner.err" "$pre.sample.log" "$pre.logcheck.json"; do
  [ -e "$f" ] && { echo "refused: $f exists" >&2; exit 2; }
done
bash "$bench/run_gpu_job.sh" "$root" B1 "$here/patch_R1.py" --dry-cpu > "$pre.runner.out" 2> "$pre.runner.err" &
runner=$!
echo "$(date -u +%FT%TZ) runner pid $runner (run_gpu_job.sh B1 patch_R1.py --dry-cpu)"
SETTLE=1 N=2 WINDOW=3 GAP=1 POLL=1 WAIT_MAX=60 bash "$here/bg2_sample2.sh" "$root" "$runner" "$pre" 'B1_patch_R1_2*.log' \
  'speed_patch R1: dl.tm_init.CoTM'
sample_rc=$?
wait "$runner"
runner_rc=$?
echo "$(date -u +%FT%TZ) sampler exit $sample_rc; runner exit $runner_rc"
"$py" -I -B "$here/check_job_log.py" --runner-out "$pre.runner.out" --root "$root" --job B1 --impl patch_R1 --expect-rc 1 \
  --sample-log "$pre.sample.log" --out "$pre.logcheck.json"
check_rc=$?
echo "$(date -u +%FT%TZ) check_job_log exit $check_rc"
cat "$pre.sample.log"
[ -e "$pre.sample.json" ] && cat "$pre.sample.json"
sha256sum "$pre".* 2> /dev/null
[ "$sample_rc" -eq 0 ] && [ "$runner_rc" -eq 1 ] && [ "$check_rc" -eq 0 ]
