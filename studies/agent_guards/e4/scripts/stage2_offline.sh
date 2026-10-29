#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stage 2 (offline, no LLM), on the FINAL code: scripted episodes A0/A1 (all sets), A0 transparency, gpt-4.1 stand-in
# rows, all-call bits for the domain check, STAND-IN training (CoTM seeds 1-3, LR, tree), MILP proofs, scripted
# episodes A2/A3/A3t with the stand-in models, the fake-server dry run of e4_run.py, the unit tests.
set -u
E4=${REPO_ROOT}/studies/agent_guards/e4
TB=${REPO_ROOT}/studies/agent_guards/taubench
PY="$TB/.venv/bin/python"
DLPY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
S=${REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992
OFF="$E4/scripts/offline_env.sh"
LOG=$E4/results/stage2.log
R=$E4/results
cd $E4/.tmp
run() { echo "$(date -u +%FT%TZ) start $1" >> $LOG; shift; "$@" > "$E4/.tmp/stage2_last.out" 2> "$E4/.tmp/stage2_last.err"; rc=$?;
        echo "$(date -u +%FT%TZ) end rc=$rc" >> $LOG; tail -c 2500 "$E4/.tmp/stage2_last.out" >> $LOG; echo >> $LOG;
        [ $rc -ne 0 ] && tail -c 1500 "$E4/.tmp/stage2_last.err" >> $LOG; return 0; }
rm -f $R/guardlog_final_*.jsonl
run "offline A0 final" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A0 --out $R/offline_A0_final.json --rows-out $R/rows_scripted_A0_final.json --all-bits-out $R/bits_allcalls_scripted_A0.json --log $R/guardlog_final_A0.jsonl
run "offline A1 final" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A1 --out $R/offline_A1_final.json --rows-out $R/rows_scripted_A1_final.json --log $R/guardlog_final_A1.jsonl
run "A0 transparency final" $OFF $PY -I -B $E4/scripts/test_a0_transparent.py --out $R/a0_transparent_final.json
run "gpt41 stand-in rows final" $OFF $PY -I -B $E4/scripts/build_rows_gpt41.py --results $S/data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json --split-file $S/data/tau2/domains/retail/split_tasks.json --out-rows $R/rows_gpt41_standin.json --out-stats $R/gpt41_standin_stats.json
run "all-call bits gpt-4.1" $OFF $PY -I -B $E4/scripts/dump_call_bits.py --results $S/data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json --out $R/bits_allcalls_gpt41.json
run "assemble stand-in rows" $OFF $PY -I -B -c "
import json
from collections import Counter
a = json.load(open('$R/rows_gpt41_standin.json'))
g = json.load(open('$R/rows_scripted_A1_final.json'))
assert a['names'] == g['names']
rows = a['rows'] + [r for r in g['rows'] if r['source'] in ('scripted:gold', 'scripted:wrong')]
json.dump({'names': a['names'], 'stand_in': True, 'rows': rows}, open('$R/rows_standin.json', 'w'))
print(json.dumps({'rows': len(rows), 'by_source_label_split': {'|'.join(map(str, k)): v for k, v in sorted(Counter((r['source'], r['label'], r['split']) for r in rows).items(), key=str)}}, indent=1))
"
run "train stand-in" $OFF $DLPY -I -B $E4/scripts/train_guard.py --rows $R/rows_standin.json --out-dir $E4/models/standin --tag standin --stand-in --seeds 1 2 3
for sd in 1 2 3; do
  run "milp stand-in seed $sd" $OFF $DLPY -I -B $E4/scripts/milp_proof.py --model $E4/models/standin/tm_seed${sd}.json --rows $R/rows_standin.json $R/bits_allcalls_scripted_A0.json $R/bits_allcalls_gpt41.json --out $R/milp_standin_tm_seed${sd}.json
done
for arm in A2:tm_seed1.json A3:lr.json A3t:tree.json; do
  a=${arm%%:*}; m=${arm#*:}
  run "offline $a (stand-in $m)" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm $a --learned $E4/models/standin/$m --sets gold,violations,constructed,wrong --out $R/offline_${a}_standin.json --log $R/guardlog_final_${a}.jsonl
done
run "dry run e4_run.py against a fake local server" $OFF $PY -I -B $E4/scripts/dry_run_fake_server.py --out $R/dry_run_fake_server.json
run "unit tests" $OFF $PY -I -B -m pytest -q -p no:cacheprovider --rootdir=$E4/tests -o addopts= -W ignore::DeprecationWarning $E4/tests
echo "$(date -u +%FT%TZ) stage2 done" >> $LOG
