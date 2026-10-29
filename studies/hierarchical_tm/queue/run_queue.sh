#!/usr/bin/env bash
# Serial GPU queue for the hierarchical_tm study. One job per line in jobs.txt:  NAME|MINUTES|htm_driver.py args...
# Each job runs through src/launch.sh (timeout -> mem_guard -> run_logged -> gpu_run.sh exclusive).
# Finished jobs go to done.txt; a failed job is logged and the queue continues.
# PAUSE: create the file queue/STOP. The running job finishes; no new job starts until STOP is removed.
# Path variables are quoted since 2026-10-25 (HTM-B5); $args stays unquoted, because it is split into arguments.
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="$STUDY/queue"; L="$STUDY/src/launch.sh"; D="$STUDY/src/htm_driver.py"
touch "$Q/done.txt"
while true; do
  while [ -f "$Q/STOP" ]; do sleep 30; done
  job=$(grep -v '^\s*#' "$Q/jobs.txt" | grep -v '^\s*$' | while IFS= read -r l; do n=${l%%|*}; grep -qxF "$n" "$Q/done.txt" || { echo "$l"; break; }; done)
  [ -z "$job" ] && { echo "$(date -u +%FT%TZ) QUEUE EMPTY" >> "$Q/queue.log"; break; }
  name=${job%%|*}; rest=${job#*|}; min=${rest%%|*}; args=${rest#*|}
  echo "$(date -u +%FT%TZ) START $name (cap ${min} min)" >> "$Q/queue.log"
  out=$("$L" "$name" "$min" "$D" $args 2>&1); rc=$?
  rd=$(echo "$out" | grep -o 'run dir: .*' | head -1 | sed 's/run dir: //')
  fin=$(echo "$out" | grep '^FINAL' | grep -o '"dev_acc": [0-9.]*\|"dev_auc": [0-9.]*\|"partner_shuffle_auc": [0-9.]*' | tr '\n' ' ')
  echo "$(date -u +%FT%TZ) END $name exit=$rc $fin dir=$(basename "$rd")" >> "$Q/queue.log"
  echo "$name" >> "$Q/done.txt"
done
