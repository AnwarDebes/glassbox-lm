#!/usr/bin/env bash
# 2026-09-30: the configuration-arms report stage (chain_config_arms.sh "" reportonly) had never run; every input is clean.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
CFG=experiments/cvtm/configs/ssl_r1_config_arms.json
RC="$PY -u experiments/ssl_r1_config_arms.py --config $CFG"
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
PROBES=$(for S in 100 101; do for A in A3s8 A3s32 A3pos; do d=$(latest ssl_r1ca_probe_${A}_seed${S}_exploratory); ok "$d" && echo "$d"; done; for A in A3 A3T; do d=$(latest ssl_r1c_probe_${A}_seed${S}_exploratory); ok "$d" && echo "$d"; done; done | paste -sd, -)
QPRED=$(for S in 100 101; do for A in A3s8 A3s32 A3pos; do d=$(latest ssl_r1ca_qpred_${A}_seed${S}_exploratory); ok "$d" && echo "$d"; done; for A in A3 A3T; do d=$(latest ssl_r1c_qpred_${A}_seed${S}_exploratory); ok "$d" && echo "$d"; done; done | paste -sd, -)
step ssl_r1ca_report_exploratory $CFG 8 0 -- $RC --stage report --probe-runs "$PROBES" --qpred-runs "$QPRED"
