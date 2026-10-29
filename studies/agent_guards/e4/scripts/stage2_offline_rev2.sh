#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stage 2, revision 2 (offline, no LLM), on the revised code (text-literal E4-R3, E4- names, veto-clean deduplicated
# rows, out-of-fold go-bar): the exhaustive equivalence check, scripted episodes A0/A1 (all sets), A0 transparency,
# gpt-4.1 stand-in rows (both R3 readings), all-call bits, STAND-IN training with out-of-fold predictions, MILP proofs,
# scripted episodes A2/A3/A3t with the stand-in models, the stand-in go-bar, the unit tests. Outputs carry "_rev2";
# revision 1's outputs are kept unchanged. Run behind scripts/when_cpu_free.sh (cgroup at most 3.5 cores).
set -u
E4=${REPO_ROOT}/studies/agent_guards/e4
TB=${REPO_ROOT}/studies/agent_guards/taubench
PY="$TB/.venv/bin/python"
DLPY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
S=$TB/data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992
G41=$S/data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json
SPLIT=$S/data/tau2/domains/retail/split_tasks.json
OFF="$E4/scripts/offline_env.sh"
LOG=$E4/results/stage2_rev2.log
R=$E4/results
cd $E4/.tmp
run() { echo "$(date -u +%FT%TZ) start $1" >> $LOG; shift; "$@" > "$E4/.tmp/stage2r2_last.out" 2> "$E4/.tmp/stage2r2_last.err"; rc=$?;
        echo "$(date -u +%FT%TZ) end rc=$rc" >> $LOG; tail -c 2500 "$E4/.tmp/stage2r2_last.out" >> $LOG; echo >> $LOG;
        [ $rc -ne 0 ] && tail -c 1500 "$E4/.tmp/stage2r2_last.err" >> $LOG; return 0; }
rm -f $R/guardlog_rev2_*.jsonl
run "equivalence (exhaustive)" $OFF $PY -I -B $E4/scripts/check_equivalence.py --out $R/equivalence_code_vs_veto_rev2.json
run "offline A0 rev2" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A0 --out $R/offline_A0_rev2.json --rows-out $R/rows_scripted_A0_rev2.json --all-bits-out $R/bits_allcalls_scripted_A0_rev2.json --log $R/guardlog_rev2_A0.jsonl
run "offline A1 rev2" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm A1 --out $R/offline_A1_rev2.json --rows-out $R/rows_scripted_A1_rev2.json --log $R/guardlog_rev2_A1.jsonl
run "A0 transparency rev2" $OFF $PY -I -B $E4/scripts/test_a0_transparent.py --out $R/a0_transparent_rev2.json
run "gpt41 stand-in rows rev2" $OFF $PY -I -B $E4/scripts/build_rows_gpt41.py --results $G41 --split-file $SPLIT --out-rows $R/rows_gpt41_standin_rev2.json --out-stats $R/gpt41_standin_stats_rev2.json
run "all-call bits gpt-4.1 rev2" $OFF $PY -I -B $E4/scripts/dump_call_bits.py --results $G41 --out $R/bits_allcalls_gpt41_rev2.json
run "assemble stand-in rows rev2 (veto-clean, deduplicated) and the stand-in A1 folder" $OFF $PY -I -B -c "
import json, sys
sys.path.insert(0, '$E4'); sys.path.insert(1, '$E4/scripts')
from pathlib import Path
from collections import Counter
from build_rows_from_results import select_training_rows
from e4guard import bits as B
import numpy as np
a = json.load(open('$R/rows_gpt41_standin_rev2.json'))
g = json.load(open('$R/rows_scripted_A1_rev2.json'))
assert a['names'] == g['names'] == B.NAMES
allr = []
for r in a['rows'] + [r for r in g['rows'] if r['source'] in ('scripted:gold', 'scripted:wrong')]:
    x = np.array(r['x'], np.uint8)
    r = dict(r, veto_clauses=B.veto_fires(x), veto_clean=not B.veto_fires(x), args=r['args'] + '|' + str(r.get('trial', r.get('variant'))))
    allr.append(r)
kept, c = select_training_rows(allr)
json.dump({'names': B.NAMES, 'stand_in': True, 'counts': dict(c), 'rows': kept}, open('$R/rows_standin_rev2.json', 'w'))
oof = [r for r in allr if r['source'] == 'gpt41_standin' and r.get('trial') == 0]
for r in oof: r['source'] = 'standin_A1_trial0'
json.dump({'names': B.NAMES, 'stand_in': True, 'rows': oof}, open('$R/rows_standin_oof_trial0_rev2.json', 'w'))
d = json.load(open('$G41'))
Path('$R/standin_A1_trial0').mkdir(exist_ok=True)
json.dump({'tasks': d['tasks'], 'simulations': [s for s in d['simulations'] if s['trial'] == 0]}, open('$R/standin_A1_trial0/results.json', 'w'))
print(json.dumps({'all': len(allr), 'counts': dict(c), 'oof_rows_trial0': len(oof), 'by_source_label_split': {'|'.join(map(str, k)): v for k, v in sorted(Counter((r['source'], r['label'], r['split']) for r in kept).items(), key=str)}}, indent=1))
"
run "train stand-in rev2 (with out-of-fold predictions)" $OFF $DLPY -I -B $E4/scripts/train_guard.py --rows $R/rows_standin_rev2.json --out-dir $E4/models/standin_rev2 --tag standin_rev2 --stand-in --seeds 1 2 3 --oof-rows $R/rows_standin_oof_trial0_rev2.json --oof-sources standin_A1_trial0 --oof-out $E4/models/standin_rev2/oof_rows.json
for sd in 1 2 3; do
  run "milp stand-in rev2 seed $sd" $OFF $DLPY -I -B $E4/scripts/milp_proof.py --model $E4/models/standin_rev2/tm_seed${sd}.json --rows $R/rows_standin_rev2.json $R/bits_allcalls_scripted_A0_rev2.json $R/bits_allcalls_gpt41_rev2.json --out $R/milp_standin_rev2_tm_seed${sd}.json
done
run "stand-in go-bar rev2 (code test; another model's data)" $OFF $PY -I -B $E4/scripts/gobar.py --oof $E4/models/standin_rev2/oof_rows.json --a1 $R/standin_A1_trial0 --split-file $SPLIT --out $R/gobar_standin_rev2.json
for arm in A2:tm_seed1.json A3:lr.json A3t:tree.json; do
  a=${arm%%:*}; m=${arm#*:}
  run "offline $a (stand-in rev2 $m)" $OFF $PY -I -B $E4/scripts/offline_episodes.py --arm $a --learned $E4/models/standin_rev2/$m --sets gold,violations,constructed,wrong --out $R/offline_${a}_standin_rev2.json --log $R/guardlog_rev2_${a}.jsonl
done
run "unit tests rev2" $OFF $PY -I -B -m pytest -q -p no:cacheprovider --rootdir=$E4/tests -o addopts= -W ignore::DeprecationWarning $E4/tests
echo "$(date -u +%FT%TZ) stage2 rev2 done" >> $LOG
