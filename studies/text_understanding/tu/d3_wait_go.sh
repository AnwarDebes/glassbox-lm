#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# 3d: writes d3_chain.sh's go file (tu/.d3/gpu_go) once the GPU slot is ours: the process the queue names is gone
# (ps -p on its pid, never a pattern match), then 3 empty-GPU checks in a row, 60 s apart (no compute process in
# nvidia-smi and at most 500 MiB used); a busy check resets the count. It starts nothing on the GPU itself.
# Usage: bash tu/d3_wait_go.sh <pid> "<who gave the slot, and when>"
# Test hooks (tests/test_d3_wait_go.sh only): D3W_GO, D3W_SLEEP, D3W_NVSMI (a command that prints "<processes> <MiB>").
set -u
pid=${1:?pid}; why=${2:?reason}
go=${D3W_GO:-${REPO_ROOT}/studies/text_understanding/tu/.d3/gpu_go}
sl=${D3W_SLEEP:-60}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gpu() { local n m
  if [ -n "${D3W_NVSMI:-}" ]; then $D3W_NVSMI; return; fi
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$n $m"; }
[ -e "$go" ] && { echo "$(ts) the go file exists already: $go"; exit 1; }
trap 'exit 143' TERM
echo "$(ts) waiting for pid $pid to end"
while ps -p "$pid" > /dev/null 2>&1; do sleep "$sl" & wait $!; done
echo "$(ts) pid $pid gone"
ok=0
while [ "$ok" -lt 3 ]; do
  read -r n m <<< "$(gpu)"
  if [ "$n" = 0 ] && [ -n "$m" ] && [ "$m" -le 500 ]; then ok=$((ok + 1)); echo "$(ts) empty-GPU check $ok of 3 ($n processes, $m MiB)"
  else ok=0; echo "$(ts) GPU busy ($n processes, $m MiB): the count starts again"; fi
  if [ "$ok" -lt 3 ]; then sleep "$sl" & wait $!; fi
done
mkdir -p "$(dirname "$go")"
echo "go $(ts): $why; pid $pid gone, then 3 empty-GPU checks" > "$go"
echo "$(ts) wrote $go"
