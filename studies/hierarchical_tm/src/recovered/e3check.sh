#!/bin/bash
cp "$0" "$TMLM_RUN_DIR/"
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for r in $STUDY/results/20261002T201414Z_e3_slide10_hier_C50_A4_K6_s12_seed1 $STUDY/results/20261002T201749Z_e3_slide10_hier_C50_A4_K6_s40_seed1 $STUDY/results/20261002T202109Z_e3_slide10_hier_C50_A4_K6_s150_seed1 ; do echo "== check2b dev $(basename $r)"; .venv-htm/bin/python -u $STUDY/src/slide10.py check2b $r --split dev; done
