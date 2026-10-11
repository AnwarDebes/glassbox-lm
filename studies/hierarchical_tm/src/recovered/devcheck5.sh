#!/bin/bash
# 5-seed dev check of final_test_report.py before the final test (dev only; never --final-test).
cp "$0" "$TMLM_RUN_DIR/"
PY=.venv-htm/bin/python
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S=$STUDY/src/final_test_report.py
R=$STUDY/results
REF="$R/20261001T094249Z_data_checks_final_exploratory/checks.json $R/20261001T094347Z_data_checks_ns_all_v2_exploratory/checks.json"
echo "== STEP 1: --split test without --final-test must be refused"
$PY -u $S --split test --group sa sa $STUDY/results/20261002T050235Z_e4_head_sa_s1_scoreP100 $STUDY/results/20261002T052930Z_e4_head_sa_s2_score $STUDY/results/20261002T060727Z_e4_head_sa_s3_score $STUDY/results/20261002T145933Z_e4_head_sa_s4_score $STUDY/results/20261002T152719Z_e4_head_sa_s5_score 
echo "exit code $? (1 = refused)"
set -e
echo "== STEP 2: the 5-seed dev report"
$PY -u $S --split dev --group sa sa $STUDY/results/20261002T050235Z_e4_head_sa_s1_scoreP100 $STUDY/results/20261002T052930Z_e4_head_sa_s2_score $STUDY/results/20261002T060727Z_e4_head_sa_s3_score $STUDY/results/20261002T145933Z_e4_head_sa_s4_score $STUDY/results/20261002T152719Z_e4_head_sa_s5_score  --group np np $STUDY/results/20261002T062612Z_e4_head_np_s1_score $STUDY/results/20261002T070833Z_e4_head_np_s2_score $STUDY/results/20261002T071028Z_e4_head_np_s3_score $STUDY/results/20261002T152755Z_e4_head_np_s4_score $STUDY/results/20261002T155611Z_e4_head_np_s5_score  --group ns_all ns_all $STUDY/results/20261002T082456Z_e4s_nsall_s100_s1_score $STUDY/results/20261002T084106Z_e4s_nsall_s100_s2_score $STUDY/results/20261002T090952Z_e4s_nsall_s100_s3_score $STUDY/results/20261002T155646Z_e4s_nsall_s100_s4_score $STUDY/results/20261002T164352Z_e4s_nsall_s100_s5_score  --shuffled ns_all $STUDY/results/20261002T075700Z_e4s_nsall_s100_shuf_s1 $STUDY/results/20261002T082531Z_e4s_nsall_s100_shuf_s2 $STUDY/results/20261002T085502Z_e4s_nsall_s100_shuf_s3 $STUDY/results/20261002T154736Z_e4s_nsall_s100_shuf_s4 $STUDY/results/20261002T161542Z_e4s_nsall_s100_shuf_s5  --reference $REF --hashes $STUDY/frozen_settings_hashes.json
