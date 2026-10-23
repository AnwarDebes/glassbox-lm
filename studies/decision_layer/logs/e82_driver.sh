#!/usr/bin/env bash
# e82's GPU stages back to back (frozen plan logs/plan_e82_refined_table_agent.md; dated note logs/plan_e82_note_2026-10-21.md)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=.venv-dl/bin/python
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for the running Stage A job (PID 775720) to end"
while kill -0 775720 2>/dev/null; do sleep 20; done
log "Stage A job 3 ended; the rest of Stage A"
bash scripts/seq_queue.sh logs/jobs_e82_rest.txt logs/e82_queue.log
log "R4 check"
$P scripts/e82_check.py --name c4words-a3B-v1 --table-sha 4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855 > logs/e82_check.out 2>&1
tail -n 4 logs/e82_check.out
if ! grep -q "TAKEN into the agent" logs/e82_check.out; then log "R4: not taken (or a check failed); e82 stops here"; exit 0; fi
log "e82-R3 screens"
bash scripts/seq_queue.sh logs/jobs_e82r3a.txt logs/e82r3a_queue.log
$P scripts/e82_pick.py --stage screen > logs/e82_pick_screen.out 2>&1 || { log "pick screen failed"; tail -n 3 logs/e82_pick_screen.out; exit 1; }
tail -n 5 logs/e82_pick_screen.out
if [ "$(grep -vc '^#' logs/jobs_e82r3b.txt)" -gt 0 ]; then log "e82-R3 16k"; bash scripts/seq_queue.sh logs/jobs_e82r3b.txt logs/e82r3b_queue.log; fi
$P scripts/e82_pick.py --stage final > logs/e82_pick_final.out 2>&1 || { log "pick final failed"; tail -n 3 logs/e82_pick_final.out; exit 1; }
tail -n 3 logs/e82_pick_final.out
if [ "$(grep -vc '^#' logs/jobs_e82c.txt)" -gt 0 ]; then log "Stage C"; bash scripts/seq_queue.sh logs/jobs_e82c.txt logs/e82c_queue.log; fi
$P scripts/e82_pick.py --stage gobars > logs/e82_gobars.out 2>&1 || { log "gobars failed"; tail -n 3 logs/e82_gobars.out; exit 1; }
tail -n 4 logs/e82_gobars.out
log "e82 GPU stages done"
