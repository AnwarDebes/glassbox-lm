#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# slot1_next.sh: job slot 1 after T4 lane A (pid 1877379) ends: the registered control val runs (t5_queue.sh controls)
T5=${REPO_ROOT}/studies/text_understanding/t5
while [ -e /proc/1877379 ]; do sleep 60; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [slot1] T4 lane A gone; start t5_queue.sh controls" >> "$T5/runs_t5.log"
bash "$T5/t5_queue.sh" controls
