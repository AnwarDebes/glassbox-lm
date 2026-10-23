#!/usr/bin/env bash
# master_queue.sh JOBFILE MAXJOBS LOGFILE
# Runs each non-comment line of JOBFILE (a command run from the study folder) under the shared
# GPU lock, starting a new one only while fewer than MAXJOBS of this study's python jobs run
# (counted over ALL queues, so several queues together respect the cap). Output of each job is
# appended to LOGFILE (last line only, warnings dropped).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${here}/../tm_language_model/scripts/gpu_run.sh"
jobfile="$1"; maxj="$2"; logf="$3"
count_jobs() {
  # every python job of this study (started from the study folder as "python scripts/<name>.py"), not only the
  # queued kinds: one-off scripts count too (changed 2026-10-17 18:03 UTC; before, only the listed script names);
  # plus this study's jobs still waiting for the shared GPU lock (gpu_run.sh has not exec'd them yet), so that a
  # wait behind an exclusive lock does not let the queue start everything at once (added 20:13 UTC)
  local p w
  p=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && $0 ~ /\.venv-dl\/bin\/python[[:space:]]+scripts\/[A-Za-z0-9_]+\.py/' | wc -l)
  w=$(ps -eo args= | awk '/gpu_run\.sh shared -- bash -c cd .*studies\/decision_layer/' | wc -l)
  echo $((p + w))
}
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
  while [ "$(count_jobs)" -ge "$maxj" ]; do sleep 5; done
  ( "$gpu" shared -- bash -c "cd '$here' && $line" 2>&1 | grep -v -i -E "warning|loading weights" | tail -1 >> "$logf" ) &
  sleep 3
done < "$jobfile"
wait
