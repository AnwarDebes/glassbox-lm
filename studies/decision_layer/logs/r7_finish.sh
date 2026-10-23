#!/usr/bin/env bash
# after the last R7 seed: apply the frozen rules, then the readability report (all datasets)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
until [ $(ls results/confirm/R7/clinc150 results/confirm/R7/banking77 results/confirm/R7/hwu64 | grep -c '^seed') -ge 30 ]; do sleep 15; done
date -u +%FT%TZ > logs/R7_rules.txt
.venv-dl/bin/python scripts/rules_check.py R7 >> logs/R7_rules.txt 2>&1
../tm_language_model/scripts/gpu_run.sh shared -- .venv-dl/bin/python scripts/r7_readability.py --receipts 8 2>&1 | grep -v -i -E "warning|loading weights" > logs/R7_readability.txt
date -u +%FT%TZ >> logs/R7_readability.txt
