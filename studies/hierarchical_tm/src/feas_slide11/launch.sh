#!/usr/bin/env bash
# usage: launch.sh <variant> <train_tm.py args...>
# Interpreters (changed on 2026-10-25, bug HTM-B1): the job runs under $HTM_PY, and mem_guard.py and run_logged.py
# are started through $HTM_TOOL_PY instead of by name; ../interpreters.sh sets both.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "usage: launch.sh <variant> <train_tm.py args...>" >&2; exit 2; fi
source "$D/../interpreters.sh"
V="$1"; shift
cd "$D/../../../tm_language_model"
source env/env.sh
"$HTM_PY" - "$V" "$HTM_PY" "$@" > "$D/$V.json" <<'PY'
import json, sys
print(json.dumps({"variant": sys.argv[1], "script": "train_tm.py", "args": sys.argv[3:], "purpose": "slide-11 feasibility probe, exploratory", "job_python": sys.argv[2]}))
PY
set +e; "$HTM_TOOL_PY" scripts/mem_guard.py --scope group --limit-gib 12 -- "$HTM_TOOL_PY" scripts/run_logged.py --name "feas_slide11_${V}_exploratory" --config "$D/$V.json" -- scripts/gpu_run.sh shared -- "$HTM_PY" -u "$D/train_tm.py" --tag "$V" "$@" > "$D/$V.log" 2>&1
rc=$?
RD=$(grep -o 'run dir: .*' "$D/$V.log" | head -1 | sed 's/run dir: //')
[ -n "$RD" ] && cp "$D/train_tm.py" "$D/prep_data.py" "$D/launch.sh" "$D/../interpreters.sh" "$RD/" 2>/dev/null || true
exit $rc
