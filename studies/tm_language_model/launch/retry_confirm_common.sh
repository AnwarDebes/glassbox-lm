#!/usr/bin/env bash
# 2026-09-30: rerun of the common confirm step, killed by mem_guard at 24 GiB on 2026-09-29 14:59 UTC after 174 s and
# BEFORE any counted access (no config=common line in data/test_access_log.jsonl; load_confirm records the access before
# it reads a file). Same command and inputs as the chain step; only the mem_guard limit changes, 24 -> 80 GiB (the
# full-pool probe of the same features peaked at 34.4 GiB RSS; nothing else runs; project cap 85 GiB). Then the report
# step again, with the chain's own argument rule. Runs inside one CPU slot; the test-access lock is held exclusively.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_DIR/lib.sh"
NIGHT=$LAUNCH_DIR
CFG=experiments/cvtm/configs/ssl_r1_confirm.json
RC="$PY -u experiments/ssl_r1_confirm.py --config $CFG"
SKIPPED=$NIGHT/r1_confirm_skipped.txt
ok() { [ -n "$1" ] && grep -q '"exit_status": 0' "$1/run.json" 2>/dev/null; }
echo "SKIP/FAIL note $(date -u +%FT%TZ): retry of ssl_r1cf_confirm_common_confirmatory at mem_guard 80 GiB (the chain gave it 24 GiB; killed at 24.28 GiB before any access)" >> "$SKIPPED"
P=results/20260926T204857Z_ssl_r1cf_probe_common_confirmatory
PF=results/20260929T000757Z_ssl_r1cf_probefull_common_confirmatory
ok "$P" && ok "$PF" || { echo "inputs not clean"; exit 1; }
(
  exec {lfd}>>"$NIGHT/r1_confirm_testaccess.lock"; flock -x "$lfd"; echo "LOCK -x held $(date -u +%FT%TZ)"
  step ssl_r1cf_confirm_common_confirmatory $CFG 80 1 -- $RC --stage confirm --probe-run "$P" --probe-full-run "$PF" --engine cuda
) || { echo "SKIP/FAIL note $(date -u +%FT%TZ): the retry of the common confirm failed, see $(latest ssl_r1cf_confirm_common_confirmatory)/log.txt" >> "$SKIPPED"; }
col() {  # the chain's rule: the latest clean run of every distinct run name of stage $1
  local seen=" " d n
  for d in $(ls -td results/*_ssl_r1cf_$1_*confirmatory 2>/dev/null); do
    n=${d#results/*Z_}; case "$seen" in *" $n "*) continue ;; esac
    ok "$d" && { seen="$seen$n "; echo "$d"; }
  done | paste -sd, -
}
PROBES=$(echo "$(col probe),$(col probefull)" | sed 's/^,//; s/,$//')
step ssl_r1cf_report_confirmatory $CFG 8 0 -- $RC --stage report --probe-runs "$PROBES" --confirm-runs "$(col confirm)" \
  --qpred-runs "$(col qpred)" --qcon-runs "$(col qcon)" --qint-runs "$(col qint)" --census-runs "$(col census)" \
  --skipped-file "$SKIPPED"
echo "RETRY DONE $(date -u +%FT%TZ)"
