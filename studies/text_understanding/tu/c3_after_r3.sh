#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Handoff R3 -> 3c (the order of 01:00Z: "3c directly after R3. Create tu/.c3/gpu_go yourself when R3 logs
# its done line, after c3_base.py has written base.json. Cite this message (01:00Z) in the marker."). Waits for
# "R3 done" in tu/r3/runs_r3.log (a STOP there ends this script without a go), then runs c3_base.py --fa off (it
# writes tu/c3/base.json by the registered rule: 3a PASS -> a3B at R3's pick), then writes tu/.c3/gpu_go. The 3c chain
# (waiting since 00:43:53Z) starts at its next 60 s check. Log tu/c3/after_r3.log.
T=${REPO_ROOT}/studies/text_understanding/tu
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$T/c3/after_r3.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p $T/c3
echo "$(ts) waiting for R3's done line" >> "$log"
until grep -q "R3 done\|STOP" $T/r3/runs_r3.log; do sleep 30; done
if grep -q "STOP" $T/r3/runs_r3.log; then echo "$(ts) R3 STOPPED: no 3c go" >> "$log"; exit 1; fi
echo "$(ts) R3 done: $(grep "R3 done" $T/r3/runs_r3.log | tail -n 1)" >> "$log"
[ -e $T/c3/base.json ] && { echo "$(ts) base.json exists already: not rewritten" >> "$log"; } || \
  (cd /tmp && CUDA_VISIBLE_DEVICES= "$py" -I -B $T/c3_base.py --fa off >> "$log" 2>&1) || { echo "$(ts) c3_base.py failed: no 3c go" >> "$log"; exit 1; }
echo "$(ts) base.json $(sha256sum $T/c3/base.json | cut -c1-16)" >> "$log"
[ -e $T/.c3/gpu_go ] && { echo "$(ts) gpu_go exists already" >> "$log"; exit 1; }
printf "go 01:00Z ('3c directly after R3, without waiting for e80; tu/.c3/gpu_go is created when R3 logs its done line, after c3_base.py has written base.json. Cite this message (01:00Z) in the marker.'); R3 done line seen and base.json written; marker written %s; GPU then: %s\n" \
  "$(ts)" "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" > $T/.c3/gpu_go
echo "$(ts) gpu_go written: $(cat $T/.c3/gpu_go | head -c 400)" >> "$log"
