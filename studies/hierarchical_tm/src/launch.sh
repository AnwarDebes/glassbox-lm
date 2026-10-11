#!/usr/bin/env bash
# Run one GPU job the way of this study (study rule 4):
#   timeout -> mem_guard (12 GiB, group) -> run_logged.py -> gpu_run.sh exclusive -> .venv-htm python
# usage: launch.sh NAME MINUTES SCRIPT.py [script args...]
#   NAME     run name; use ..._exploratory for tests (results go to the results/ folder of this study)
#   MINUTES  hard wall-time cap, including the wait for the GPU lock
# The script finds its run dir in $TMLM_RUN_DIR.
set -euo pipefail
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"; MIN="$2"; shift 2
cfg=$(mktemp "${TMPDIR:-/tmp}/htmpoc_cfg_XXXXXX.json")
trap 'rm -f "$cfg"' EXIT
"$STUDY/../tm_language_model/.venv-htm/bin/python" - "$cfg" "$NAME" "$MIN" "$@" <<'PY'
import json, sys
cfg, name, minutes, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
json.dump({"name": name, "timeout_min": float(minutes), "argv": argv, "launcher": "htm-poc/src/launch.sh"}, open(cfg, "w"))
PY
cd "$STUDY/../tm_language_model"
source env/env.sh
timeout --kill-after=60 "${MIN}m" scripts/mem_guard.py --scope group --limit-gib 12 -- \
  scripts/run_logged.py --name "$NAME" --config "$cfg" --results-dir "$STUDY/results" -- \
  scripts/gpu_run.sh exclusive -- .venv-htm/bin/python -u "$@"
