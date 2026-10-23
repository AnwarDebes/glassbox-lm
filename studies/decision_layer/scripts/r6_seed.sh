#!/usr/bin/env bash
# r6_seed.sh SEED "TMQ_ARGS" "LINQ_ARGS" [ONLY] : the registered R6 learners for one seed, in order
# (the first call computes and caches the cross-fitted router of that seed). Test split, out
# confirm/R6. A learner already recorded for this seed in results/confirm/R6/r6.jsonl is skipped.
# ONLY = "rule" runs only the threshold rule (to build the router cache early).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
seed="$1"
py=".venv-dl/bin/python scripts/r6_run.py --seed $seed --eval test --out confirm/R6 --tag R6"
has() {  # has LEARNER MEMORY(true|false)
  [ -f results/confirm/R6/r6.jsonl ] && .venv-dl/bin/python - "$seed" "$1" "$2" <<'PY'
import json, sys
seed, learner, mem = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "true"
for l in open("results/confirm/R6/r6.jsonl"):
    r = json.loads(l)
    if r["seed"] == seed and r["learner"] == learner and r.get("memory", True) == mem:
        sys.exit(0)
sys.exit(1)
PY
}
has rule true || $py --learner rule
[ "${4:-}" = "rule" ] && exit 0
has route true || $py --learner route
has ask_route true || $py --learner ask_route
has tm_q true || $py --learner tm_q $2
has tm_myopic true || $py --learner tm_myopic $2
has tm_q false || $py --learner tm_q --no-memory $2
has lin_q true || $py --learner lin_q $3
