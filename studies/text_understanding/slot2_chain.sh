#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# slot2_chain.sh: job slot 2 for T4 (REGISTRATION_T4.md): the test 5 timing pilot (CPU, one thread), then T4 lane B
here=${REPO_ROOT}/studies/text_understanding
echo "$(date -u +%H:%M:%SZ) slot 2: start the test 5 pilot" >> "$here/runs_T4t.log"
bash "$here/t5/pilot_queue.sh"
echo "$(date -u +%H:%M:%SZ) slot 2: pilot queue ended, start T4 lane B" >> "$here/runs_T4t.log"
bash "$here/t4t_queue.sh" B
