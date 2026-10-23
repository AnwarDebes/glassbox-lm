#!/usr/bin/env bash
# queue.sh JOBFILE PARALLEL : run each line of JOBFILE (arguments of tm_explore.py or a full
# command starting with "CMD ") under the shared GPU lock, PARALLEL at a time.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${here}/../tm_language_model/scripts/gpu_run.sh"
py="${here}/.venv-dl/bin/python"
run_one() {
  line="$1"
  if [[ "$line" == CMD* ]]; then
    "$gpu" shared -- bash -c "cd '$here' && ${line#CMD }"
  else
    "$gpu" shared -- bash -c "cd '$here' && '$py' scripts/tm_explore.py $line" 2>&1 | grep -v -i warning | tail -1
  fi
}
export -f run_one; export here gpu py
grep -v '^\s*#' "$1" | grep -v '^\s*$' | xargs -d '\n' -P "$2" -I{} bash -c 'run_one "$@"' _ {}
