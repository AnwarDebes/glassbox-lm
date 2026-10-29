#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# MiniWoB++ smoke test runner (scripted DOM policy, no LLM, no GPU).
# Usage: [NONET=1] ./run_smoke.sh [extra args for scripts/smoke_test.py]   (NONET=1: run with no network at all)
# The browser and its driver start through browser/chrome-headless-shell and browser/chromedriver:
# wrappers that give ONLY those processes the conda-forge userspace libraries (LD_LIBRARY_PATH, fonts).
# This Python process keeps its normal environment.
set -euo pipefail
F=${REPO_ROOT}/studies/agent_guards/miniwob
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$F/tmp" SE_CACHE_PATH="$F/tmp/selenium-cache" SE_AVOID_STATS=true SE_OFFLINE=true
export MINIWOB_CHROME_BINARY="$F/browser/chrome-headless-shell"
export MINIWOB_CHROMEDRIVER="$F/browser/chromedriver"
mkdir -p "$TMPDIR" "$F/runs"
# wait (up to ~10 min) until the cgroup load is <= 3.5 cores
"$F/scripts/cpu_gate.sh" 60 || { echo "cgroup load stayed above 3.5 cores: not starting"; exit 3; }
OUT="$F/runs/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
cd "$F/scripts"
# taskset pins the browser + driver + python to 2 CPUs (Chromium is multi-process)
if [ "${NONET:-0}" = "1" ]; then
  # no network at all: a fresh user + network namespace with only a private loopback (scripts/netns_exec.py)
  taskset -c 0,1 nice -n 19 unshare --user --map-current-user --net --keep-caps \
    "$F/.venv/bin/python" -I "$F/scripts/netns_exec.py" "$F/.venv/bin/python" -I "$F/scripts/smoke_test.py" --outdir "$OUT" "$@"
else
  taskset -c 0,1 nice -n 19 "$F/.venv/bin/python" -I "$F/scripts/smoke_test.py" --outdir "$OUT" "$@"
fi
echo "results in $OUT"
