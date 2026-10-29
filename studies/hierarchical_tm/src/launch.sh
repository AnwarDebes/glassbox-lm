#!/usr/bin/env bash
# Run one GPU job the way of this study (study rule 4):
#   timeout -> mem_guard (12 GiB, group) -> run_logged.py -> gpu_run.sh exclusive -> .venv-htm python
# usage: launch.sh NAME MINUTES SCRIPT.py [script args...]
#   NAME     run name; use ..._exploratory for tests (results go to the results/ folder of this study)
#   MINUTES  hard wall-time cap, including the wait for the GPU lock
# The script finds its run dir in $TMLM_RUN_DIR.
# Interpreters (changed on 2026-10-25, bug HTM-B1): the job runs under $HTM_PY, and mem_guard.py and run_logged.py
# are started through $HTM_TOOL_PY instead of by name; src/interpreters.sh sets both and gives their defaults.
set -euo pipefail
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -lt 3 ]; then
  echo "usage: launch.sh NAME MINUTES SCRIPT.py [script args...]" >&2
  exit 2
fi
source "$STUDY/src/interpreters.sh"
NAME="$1"; MIN="$2"; shift 2
cfg=$(mktemp "${TMPDIR:-/tmp}/htmpoc_cfg_XXXXXX.json")
trap 'rm -f "$cfg"' EXIT
"$HTM_PY" - "$cfg" "$NAME" "$MIN" "$HTM_PY" "$@" <<'PY'
import json, sys
cfg, name, minutes, job_python, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
json.dump({"name": name, "timeout_min": float(minutes), "argv": argv, "launcher": "htm-poc/src/launch.sh",
           "job_python": job_python}, open(cfg, "w"))
PY
cd "$STUDY/../tm_language_model"
source env/env.sh
timeout --kill-after=60 "${MIN}m" "$HTM_TOOL_PY" scripts/mem_guard.py --scope group --limit-gib 12 -- \
  "$HTM_TOOL_PY" scripts/run_logged.py --name "$NAME" --config "$cfg" --results-dir "$STUDY/results" -- \
  scripts/gpu_run.sh exclusive -- "$HTM_PY" -u "$@"
