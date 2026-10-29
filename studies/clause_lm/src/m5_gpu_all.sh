#!/usr/bin/env bash
# RETIRED on 2026-10-25 (finish pass): this script exits 3 at once and starts nothing. The text below the exit is
# kept as the record of the launcher written on 2026-10-02.
# - Its wait can never end in the published tree: it reads ../hierarchical_tm/queue/{jobs.txt,started.txt,queue.log},
#   local queue state that was not moved with the reorganisation of 2026-10-11, so thesis_idle() fails and the
#   script gave up after MAX_WAIT_MIN (exit 3).
# - The work it was written for is complete: the amended Gate 0, the 11-run grid and the report ran on 2026-10-14
#   (docs/results.md). On the machine that ran them, src/m5_run.py refuses to run any of those 12 settings again
#   (their metrics.json files are in work/m5/runs).
# - The 2026-10-14 runs were started by research/roadmap/run_gate0.sh of the project folder, outside this repository
#   (sha256 32b61c95b32f9a7177f50e11cd0ca72dfaf435a7baf1feb0d743b102af7fcaa8; last modified at 22:15:14Z, after
#   the first attempt, whose version of the file was not kept). It runs the steps below without the wait and
#   without the 10-minute grace period, puts a python3 link first on PATH for scripts/mem_guard.py, and starts
#   Gate 0 with --after-crash. Its output: research/roadmap/gate0.attempt1_crash.out (the first attempt, exit 127)
#   and research/roadmap/gate0.out.
echo "m5_gpu_all.sh: retired on 2026-10-25; M5 step 1 is complete (see the header of this file)" >&2
exit 3

# M5 step 1 on the GPU (docs/prereg_M5_step1.md, amendment 2026-10-02 21:11 UTC).
# Thesis first: waits until the hierarchical_tm queue is EMPTY (no unstarted job, no thesis process,
# last queue.log line "QUEUE EMPTY"/"MPS daemon stopped"), then 10 more idle minutes, then:
#   CUDA sanity check -> Gate 0 (s 8, seed 42, torch_gpu) -> only if PASS: the 11-run grid -> report.
# Every GPU step holds the tm_language_model GPU lock (gpu_run.sh exclusive), which the thesis queue also uses.
# Gives up waiting after MAX_WAIT_MIN (exit 3).
set -u
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LL=$STUDY_DIR; Q=$STUDY_DIR/../hierarchical_tm/queue
PY=$LL/.venv-cdllm/bin/python; PYG=$LL/.venv-cdllm-gpu/bin/python
GPU_RUN=$STUDY_DIR/../tm_language_model/scripts/gpu_run.sh
MAX_WAIT_MIN="${MAX_WAIT_MIN:-100}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

thesis_idle() {
  local pending procs last
  pending=$(grep -v '^#' "$Q/jobs.txt" | grep '|' | cut -d'|' -f1 | while read -r n; do
              grep -qx "$n" "$Q/started.txt" || echo "$n"; done | wc -l)
  procs=$(pgrep -f "htm_driver.py|score_saved.py|run_queue_mps.sh|run_queue.sh|text_driver.py" | wc -l)
  last=$(tail -1 "$Q/queue.log")
  [ "$pending" = 0 ] && [ "$procs" = 0 ] || return 1
  case "$last" in *"QUEUE EMPTY"*|*"MPS daemon stopped"*) return 0;; *) return 1;; esac
}

start=$(date +%s)
wait_idle() {
  until thesis_idle; do
    if [ $(( ($(date +%s) - start) / 60 )) -ge "$MAX_WAIT_MIN" ]; then
      echo "== $(ts) STILL WAITING after ${MAX_WAIT_MIN} min: thesis queue busy; exit 3 (re-launch later)"; exit 3
    fi
    sleep 60
  done
}

echo "== $(ts) waiting for the thesis queue to be empty"
wait_idle
echo "== $(ts) idle; 10-minute grace period"
sleep 600
wait_idle
echo "== $(ts) thesis queue idle; CUDA check under the GPU lock"
"$GPU_RUN" exclusive -- "$PYG" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(512,512,device='cuda'); print('matmul ok', float((x@x).abs().sum())>0)" || { echo "== $(ts) CUDA CHECK FAILED"; exit 4; }

echo "== $(ts) Gate 0 (torch_gpu)"
"$PY" "$LL/src/m5_run.py" --s 8 --seed 42 --gate0 --backend torch_gpu
g=$(ls -1td "$LL"/results/*_m5_gate0_s8_seed42_sorted_torchgpu/ 2>/dev/null | head -1)
pass=$("$PY" -c "import json,sys; m=json.load(open('${g}metrics.json')); print(m.get('gate0_pass'), m.get('test_accuracy'), m.get('oos_auroc'), m.get('process_wall_s'))" 2>/dev/null)
echo "== $(ts) Gate 0 result (pass, test_acc, oos_auroc, wall_s): $pass  [$g]"
case "$pass" in
  True*) ;;
  *) echo "== $(ts) GATE 0 NOT PASSED: no grid (prereg)"; exit 5;;
esac

echo "== $(ts) grid (11 runs, torch_gpu)"
bash "$LL/src/m5_grid.sh" torch_gpu
echo "== $(ts) report"
"$PY" "$LL/src/m5_report.py" --label-order sorted_torchgpu
echo "== $(ts) ALL DONE"
