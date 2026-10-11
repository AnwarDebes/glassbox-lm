#!/usr/bin/env bash
# 2026-09-30: the c85 demo run results/20260925T091022Z_ssl_r2_hao_diag_demos_c85_exploratory hit its per-demo wall cap
# under 4-way GPU sharing and completed 0 runs of every demo; chain_hao_diag3 reused it, so the 2026-09-30 report shows
# the c85 hierarchical demos as not passing (0 of 10 runs). Rerun of the same stage (same command, same config, same
# cap) on a free GPU, as the HEAD demos were; then the report with the same run list, the new c85 demo run replacing
# the truncated one. Waits (no CPU slot) for the common-confirm retry, which uses the GPU, to finish.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=$LAUNCH_DIR
until grep -q 'RETRY DONE' $N/retry_confirm_common.out 2>/dev/null || ! pgrep -f retry_confirm_common.sh >/dev/null; do sleep 120; done
echo "retry finished $(date -u +%FT%TZ): taking a CPU slot"
cd "$LAUNCH_DIR/.." && exec scripts/cpu_slot.sh 1 -- bash -c '
source "$TMLM_ROOT/launch/lib.sh"
CFG=experiments/cvtm/configs/ssl_r2_hao_diagnostics.json
DG="-u experiments/ssl_r2_hao_diagnostics.py --config $CFG"
OLD=results/20260925T091022Z_ssl_r2_hao_diag_demos_c85_exploratory
step ssl_r2_hao_diag_demos_c85_exploratory $CFG 3 1 -- .venv-tm/bin/python $DG --stage demos || { echo "c85 demos FAILED"; exit 1; }
NEW=$(latest ssl_r2_hao_diag_demos_c85_exploratory)
RUNS=$(.venv-tm/bin/python -c "
import json,sys
c=json.load(open(\"results/20260930T041535Z_ssl_r2_hao_diag_report_exploratory/run.json\"))[\"command\"]
r=c[c.index(\"--runs\")+1].split(\",\"); assert \"$OLD\" in r
print(\",\".join(\"$NEW\" if x==\"$OLD\" else x for x in r))")
step ssl_r2_hao_diag_report_exploratory $CFG 1 0 -- .venv-tm/bin/python $DG --stage report --runs "$RUNS"
echo "C85 RERUN DONE $(date -u +%FT%TZ)"'
