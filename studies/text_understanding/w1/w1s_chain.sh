#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1-S (rules in LOG.md before any download or run): lane A checks ppmi_svd2.py against pp_S10, builds pp2_S300,
# pp2_S1000 (potion's strings) and ppw_S1000 (C4 words), touches .w1s/tables_ready, then takes jobs; lane B waits until
# my CPU slot is free (n7 follow-ups done) and the tables are ready, then takes jobs. Jobs are claimed with mkdir (each
# runs once). Code hashes are checked before every step. Log runs_w1s.log.
lane=$1
here=${REPO_ROOT}/studies/text_understanding/w1
emb=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1s.log
cl=$here/.w1s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP lane $lane: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $here/ppmi_svd2.py fe37efe876686391; check $here/mo_tm.py 4d85663b1556ccf2; check $here/w1_feats.py 94f84ffeae4d441d
  check $here/wordtab_feats.py 001a93b259a1b34f; check $here/w1_exists_tm.py bd4e2bbae7f5af69; check $here/w1_n7.py c8bcdd17ab999579
  check $here/w1_n7w.py b54688135df9f106; check $emb/exists_tm.py 7da0aee9f4eef796; check $emb/n7_tm.py 3d715305b119b57b
  check $emb/glove_feats.py 7b5e8760e1f5e107; }
run() { echo "$(ts) lane $lane start $*" >> "$log"; "$gpu" shared -- "$py" -I -B "$@" >> "$log" 2>&1; local e=$?; echo "$(ts) lane $lane end $(basename "$1") exit $e" >> "$log"; return $e; }
mkdir -p "$cl"
if [ "$lane" = A ]; then
  checks
  run $here/ppmi_svd2.py --stream potion --slice S10 --tag check2_S10 --check-against tables/pp_S10_d500_e0.5.npz --mem 0.25 || stop "check failed to run"
  tail -n 1 $here/results_w1_checks.jsonl | grep -q '"passed": true' || stop "ppmi_svd2.py does not reproduce pp_S10 (rule: every row cosine >= 0.999)"
  run $here/ppmi_svd2.py --stream potion --slice S300 --tag pp2_S300 || stop "pp2_S300 failed"
  run $here/ppmi_svd2.py --stream potion --slice S1000 --tag pp2_S1000 || stop "pp2_S1000 failed"
  run $here/ppmi_svd2.py --stream words --slice S1000 --tag ppw_S1000 || stop "ppw_S1000 failed"
  touch "$cl/tables_ready"
else
  until grep -q 'followups done' $emb/runs_n7_followups.log 2>/dev/null && [ -e "$cl/tables_ready" ]; do sleep 30; done
fi
# jobs: suite 1 (features once per table and dataset, then seeds), then suite 2
for t in pp2_S1000_d500_e0.5 pp2_S300_d500_e0.5 ppw_S1000_d500_e0.5; do
  for ds in clinc150 banking77 hwu64; do
    mkdir "$cl/s1_${t}_${ds}" 2>/dev/null || continue
    checks
    if [ "$t" = ppw_S1000_d500_e0.5 ]; then
      run $here/wordtab_feats.py --dataset $ds --table wordtab:$here/tables/$t.npz --out-dir $here/feats/$t || stop "wordtab_feats $t $ds"
    else
      run $here/w1_feats.py --dataset $ds --table tables/$t.npz --out-dir $here/feats/$t || stop "w1_feats $t $ds"
    fi
    for seed in 1 2 3; do checks; run $here/w1_exists_tm.py --dataset $ds --view existsT --seed $seed --emb-dir $here/feats/$t --tag "W1S_${t}_${ds}_s$seed"; done
  done
done
for t in pp2_S1000_d500_e0.5 pp2_S300_d500_e0.5 ppw_S1000_d500_e0.5; do
  s=w1_n7.py; [ "$t" = ppw_S1000_d500_e0.5 ] && s=w1_n7w.py
  for ds in trec mr sst2 r52 r8; do
    mkdir "$cl/s2_${t}_${ds}" 2>/dev/null || continue
    checks
    run $here/$s --build --dataset $ds --table tables/$t.npz || stop "$s build $t $ds"
    for seed in 1 2 3; do checks; run $here/$s --dataset $ds --table tables/$t.npz --seed $seed; done
  done
done
echo "$(ts) lane $lane done" >> "$log"
