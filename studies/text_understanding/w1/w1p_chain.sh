#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W1-P (rules in LOG.md before any run): PPMI-SVD from C4 at S10 / S100 through existsT on both suites. ONE job at a
# time (my other slot runs n7_tmu_calib.py on the CPU), shared GPU lock, code hashes checked before every step.
here=${REPO_ROOT}/studies/text_understanding/w1
emb=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
gpu=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
log=$here/runs_w1p.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP w1p: $1" >> "$log"; exit 1; }
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || stop "$1 changed"; }
checks() { check $here/ppmi_svd.py 9132591108f61707; check $here/mo_tm.py 4d85663b1556ccf2; check $here/w1_feats.py 94f84ffeae4d441d
  check $here/w1_exists_tm.py bd4e2bbae7f5af69; check $here/w1_n7.py c8bcdd17ab999579; check $emb/exists_tm.py 7da0aee9f4eef796
  check $emb/n7_tm.py 3d715305b119b57b; }
run() { echo "$(ts) start $*" >> "$log"; "$gpu" shared -- "$py" -I -B "$@" >> "$log" 2>&1; local e=$?; echo "$(ts) end $(basename "$1") exit $e" >> "$log"; return $e; }
checks
run $here/ppmi_svd.py --slice S10 --tag pp_S10 --dims 500 --eigs 0.5 --features 20000 --mem 0.25 || stop "pp_S10 failed"
run $here/ppmi_svd.py --slice S100 --tag pp_S100 --dims 500 --eigs 0.5 --features 20000 --mem 0.4 || stop "pp_S100 failed"
ok2=1
for ds in trec mr sst2 r52 r8; do
  checks; run $here/w1_n7.py --check --dataset $ds || ok2=0
done
tail -n 5 $here/results_w1_checks.jsonl | grep -c '"passed": true' | grep -qx 5 || { ok2=0; echo "$(ts) suite-2 check did not pass on all 5 sets: suite 2 skipped" >> "$log"; }
for t in pp_S10_d500_e0.5 pp_S100_d500_e0.5; do
  for ds in clinc150 banking77 hwu64; do
    checks; run $here/w1_feats.py --dataset $ds --table tables/$t.npz --out-dir $here/feats/$t || stop "w1_feats $t $ds failed"
    for seed in 1 2 3; do
      checks; run $here/w1_exists_tm.py --dataset $ds --view existsT --seed $seed --emb-dir $here/feats/$t --tag "W1P_${t}_${ds}_s$seed"
    done
  done
done
echo "$(ts) suite 1 done" >> "$log"
if [ $ok2 = 1 ]; then
  for t in pp_S10_d500_e0.5 pp_S100_d500_e0.5; do
    for ds in trec mr sst2 r52 r8; do
      checks; run $here/w1_n7.py --build --dataset $ds --table tables/$t.npz || stop "w1_n7 build $t $ds failed"
      for seed in 1 2 3; do
        checks; run $here/w1_n7.py --dataset $ds --table tables/$t.npz --seed $seed
      done
    done
  done
  echo "$(ts) suite 2 done" >> "$log"
fi
echo "$(ts) w1p done" >> "$log"
