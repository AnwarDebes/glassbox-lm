#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# After e84b's end line (logs/e84b_end.log) and an empty GPU: R24's dry runs, then r24_check.py --dry.
set -uo pipefail
S=${REPO_ROOT}/studies/decision_layer; cd "$S" || exit 1
[ -e logs/R24_dry_queue.log ] && { echo "R24 dry queue log exists: not starting"; exit 1; }
t0=$(date +%s); until [ -e logs/e84b_end.log ]; do [ $(( $(date +%s) - t0 )) -lt 18000 ] || exit 2; sleep 30; done
t1=$(date +%s); until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ]; do [ $(( $(date +%s) - t1 )) -lt 1800 ] || exit 3; sleep 20; done
echo "$(date -u +%FT%TZ) e84b ended ($(tail -n 1 logs/e84b_end.log)); R24 dry runs start"
bash scripts/seq_queue.sh logs/jobs_R24_dry.txt logs/R24_dry_queue.log
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python scripts/r24_check.py --dry > logs/R24_dry_check.out 2>&1; rc=$?
echo "$(date -u +%FT%TZ) r24_check --dry exit $rc"; tail -n 3 logs/R24_dry_check.out; exit $rc
