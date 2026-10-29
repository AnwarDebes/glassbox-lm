#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stage 1 (offline, no LLM): scripted episodes in tau2's pipeline (A0, A1), A0 transparency, gpt-4.1 stand-in rows.
set -u
E4=${REPO_ROOT}/studies/agent_guards/e4
TB=${REPO_ROOT}/studies/agent_guards/taubench
PY="$TB/.venv/bin/python"
S=$TB/data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992
OFF="$E4/scripts/offline_env.sh"
LOG=$E4/results/stage1.log
cd $E4/.tmp
run() { echo "$(date -u +%FT%TZ) start $1" >> $LOG; shift; "$@" > "$E4/.tmp/stage1_last.out" 2> "$E4/.tmp/stage1_last.err"; rc=$?;
        echo "$(date -u +%FT%TZ) end rc=$rc" >> $LOG; tail -c 3000 "$E4/.tmp/stage1_last.out" >> $LOG; return $rc; }
run "offline A0" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A0 --out $E4/results/offline_A0.json --rows-out $E4/results/rows_scripted_A0.json --log $E4/results/guardlog_offline_A0.jsonl
run "offline A1" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A1 --out $E4/results/offline_A1.json --rows-out $E4/results/rows_scripted_A1.json --log $E4/results/guardlog_offline_A1.jsonl
run "A0 transparency" $OFF $PY -I -B $E4/scripts/test_a0_transparent.py --out $E4/results/a0_transparent.json
run "gpt41 stand-in rows" $OFF $PY -I -B $E4/scripts/build_rows_gpt41.py --results $S/data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json --split-file $S/data/tau2/domains/retail/split_tasks.json --out-rows $E4/results/rows_gpt41_standin.json --out-stats $E4/results/gpt41_standin_stats.json
echo "$(date -u +%FT%TZ) stage1 done" >> $LOG
