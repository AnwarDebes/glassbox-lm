#!/bin/bash
set -e
cp "$0" "$TMLM_RUN_DIR/"
D=$TMLM_RUN_DIR
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMLM_RUN_DIR=$D/report .venv-htm/bin/python -u $STUDY/src/recovered/e3fb_report.py flat768 $STUDY/results/20261003T005424Z_e3fb_slide10_text_flat_C768_s21_seed1 $STUDY/results/20261003T010445Z_e3fb_slide10_text_flat_C768_s21_seed2 $STUDY/results/20261003T012450Z_e3fb_slide10_text_flat_C768_s21_seed3
TMLM_RUN_DIR=$D/probe .venv-htm/bin/python -u $STUDY/src/recovered/e3fb_deletion_probe.py $STUDY/results/20261003T005424Z_e3fb_slide10_text_flat_C768_s21_seed1 $STUDY/results/20261003T010445Z_e3fb_slide10_text_flat_C768_s21_seed2 $STUDY/results/20261003T012450Z_e3fb_slide10_text_flat_C768_s21_seed3
