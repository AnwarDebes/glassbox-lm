#!/usr/bin/env bash
# seq_queue.sh JOBFILE LOGFILE [MAXCORES]   (added 2026-10-19 for e72; plan logs/plan_e72_count_table.md)
# Runs the non-comment lines of JOBFILE (commands run from the study folder) one at a time, in order, each under the
# shared GPU lock, as master_queue.sh does. Before each start it waits until scripts/cpu_load.py (10 s window) shows
# at most MAXCORES cores (default 4.5), checking about once a minute. A job that exits non-zero is run once more with
# the same command; then the queue goes on. Every load check, start and exit is appended to LOGFILE.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${here}/../tm_language_model/scripts/gpu_run.sh"
jobfile="$1"; logf="$2"; maxc="${3:-4.5}"
cd "$here"
mapfile -t lines < <(grep -v -E '^[[:space:]]*(#|$)' "$jobfile")
echo "$(date -u +%FT%TZ) queue start: ${#lines[@]} jobs from $jobfile, max $maxc cores" >> "$logf"
for line in "${lines[@]}"; do
  while true; do
    m=$(.venv-dl/bin/python scripts/cpu_load.py 10)
    echo "$m" >> "$logf"
    c=$(awk '{print $3}' <<< "$m")
    awk -v c="$c" -v m="$maxc" 'BEGIN { exit !(c <= m) }' && break
    sleep 50
  done
  for try in 1 2; do
    echo "$(date -u +%FT%TZ) start (try $try): $line" >> "$logf"
    "$gpu" shared -- bash -c "cd '$here' && $line" < /dev/null
    rc=$?
    echo "$(date -u +%FT%TZ) exit $rc (try $try)" >> "$logf"
    [ "$rc" -eq 0 ] && break
  done
done
echo "$(date -u +%FT%TZ) queue done" >> "$logf"
