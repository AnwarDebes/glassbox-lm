#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# After S0 (README job 1) and before S1: append the frozen bit list's SHA-256 to LOG.md, with the time read by date -u
# in the same command (PLAN_part3.md section 3: "its SHA-256 goes into LOG.md before the first tune-cal score").
#   log_bitlist.sh [LOG.md]          (A_S0_DIR overrides A/out/s0, for the self-test only)
# Exit 2 when S0 is not done or bit_list.json changed after S0. Writes nothing else.
set -eu
source ${REPO_ROOT}/studies/language_part3/A/env.sh
LOGMD=${1:-${REPO_ROOT}/studies/language_part3/LOG.md}
S0=${A_S0_DIR:-$A/out/s0}
[ -f "$S0/done.json" ] || { echo "S0 is not done ($S0/done.json missing)"; exit 2; }
h=$(sha256sum "$S0/bit_list.json" | cut -d' ' -f1)
info=$("$PY_TM" -I -B -c '
import json, sys
d = json.load(open(sys.argv[1]))
c = d["counts"]
print(d["bit_list_sha256"], d["bits"], d["count_derived_bits"], d["bits_A_nf"], c["replies"], c["fit_replies"],
      c["tune_cal_replies"], c["tune_hold_replies"], c["fit_tokens_with_end"], c["vocab"])' "$S0/done.json")
read -r d bits cd nf rep fitr calr holdr fitt vocab <<< "$info"
[ "$h" = "$d" ] || { echo "bit_list.json changed after S0 ($h, done.json says $d)"; exit 2; }
echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): [A-Struct] S0 done; FROZEN BIT LIST ${S0#${WS_ROOT}}/bit_list.json sha256 $h ($bits bits, $cd count-derived, A-nf $nf); replies $rep (fit $fitr, tune-cal $calr, tune-hold $holdr), fit tokens with the end token $fitt, vocabulary $vocab. Logged before the first tune-cal score (S1)." >> "$LOGMD"
tail -n 1 "$LOGMD"
