#!/bin/bash
# Steps of final_test_report_devcheck_exploratory (dev only; never --final-test).
cp "$0" "$TMLM_RUN_DIR/"  # keep these steps with the run
PY=.venv-htm/bin/python
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S=$STUDY/src/final_test_report.py
R=$STUDY/results
SA="$R/20261002T050235Z_e4_head_sa_s1_scoreP100 $R/20261002T052930Z_e4_head_sa_s2_score $R/20261002T060727Z_e4_head_sa_s3_score"
NP="$R/20261002T062612Z_e4_head_np_s1_score $R/20261002T070833Z_e4_head_np_s2_score $R/20261002T071028Z_e4_head_np_s3_score"
NS="$R/20261002T082456Z_e4s_nsall_s100_s1_score $R/20261002T084106Z_e4s_nsall_s100_s2_score $R/20261002T090952Z_e4s_nsall_s100_s3_score"
SH="$R/20261002T075700Z_e4s_nsall_s100_shuf_s1 $R/20261002T082531Z_e4s_nsall_s100_shuf_s2 $R/20261002T085502Z_e4s_nsall_s100_shuf_s3"
REF="$R/20261001T094249Z_data_checks_final_exploratory/checks.json $R/20261001T094347Z_data_checks_ns_all_v2_exploratory/checks.json"
echo "== STEP 1 (check 4): --split test without --final-test must be refused before anything is read"
$PY -u $S --split test --group sa sa $SA
echo "check 4 exit code $? (1 = refused)"
echo "== STEP 1b: wrong inputs are refused before any pairs are loaded"
$PY -u $S --split dev --group sa np $SA
echo "exit code $? (1 = refused: sa runs given as np)"
$PY -u $S --split dev --group sa sa $SA --shuffled sa $SA
echo "exit code $? (1 = refused: normal runs given as shuffled-label runs)"
set -e
echo "== STEP 2: the dev report (sa, np, ns_all; shuffled ns_all from training logs; reference of 1 Oct 2026; frozen hashes)"
$PY -u $S --split dev --group sa sa $SA --group np np $NP --group ns_all ns_all $NS --shuffled ns_all $SH \
  --reference $REF --hashes $STUDY/frozen_settings_hashes.json
echo "== STEP 3: the --best-from path on dev (sa only), reading the dev_report.json just written"
$PY -u $S --split dev --group sa sa $SA --best-from "$TMLM_RUN_DIR/dev_report.json" --out "$TMLM_RUN_DIR/best_from_check"
