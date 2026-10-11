#!/usr/bin/env bash
# usage: launch.sh <variant> <train_tm.py args...>
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="$1"; shift
cd "$D/../../../tm_language_model"
source env/env.sh
.venv-htm/bin/python - "$V" "$@" > "$D/$V.json" <<'PY'
import json, sys
print(json.dumps({"variant": sys.argv[1], "script": "train_tm.py", "args": sys.argv[2:], "purpose": "slide-11 feasibility probe, exploratory"}))
PY
set +e; scripts/mem_guard.py --scope group --limit-gib 12 -- scripts/run_logged.py --name "feas_slide11_${V}_exploratory" --config "$D/$V.json" -- scripts/gpu_run.sh shared -- .venv-htm/bin/python -u "$D/train_tm.py" --tag "$V" "$@" > "$D/$V.log" 2>&1
rc=$?
RD=$(grep -o 'run dir: .*' "$D/$V.log" | head -1 | sed 's/run dir: //')
[ -n "$RD" ] && cp "$D/train_tm.py" "$D/prep_data.py" "$D/launch.sh" "$RD/" 2>/dev/null || true
exit $rc
