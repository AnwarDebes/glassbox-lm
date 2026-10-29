#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# shellcheck disable=SC2016  # the mutation patterns are literal code
# Shows that the unit tests can fail: each mutant copy of serve_qwen_agent.sh breaks one guard, and the unit tests
# must report that guard's test as FAIL (and fail overall). Mutants and their outputs in tests/runs/mutants_<stamp>/.
# Run: bash tests/mutation_check.sh
set -u
L=${REPO_ROOT}/studies/agent_guards/llm
T=$L/tests
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
W=$T/runs/mutants_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$W" || exit 1
caught=0; missed=0
mutant() { local name=$1 old=$2 new=$3 want=$4 m=$W/$1.sh out
  if ! "$PY" -I -c 'import sys; s = open(sys.argv[1]).read(); assert s.count(sys.argv[3]) == 1, "pattern not found once"
open(sys.argv[2], "w").write(s.replace(sys.argv[3], sys.argv[4]))' "$L/serve_qwen_agent.sh" "$m" "$old" "$new"; then
    echo "MISSED $name: the mutation could not be made"; missed=$((missed + 1)); return
  fi
  out=$(SERVE_SCRIPT=$m bash "$T/test_serve_qwen_agent.sh" 2>&1); echo "$out" > "$W/$name.out"
  if grep -qF "FAIL $want" <<< "$out" && ! grep -q "unit tests: [0-9]* passed, 0 failed" <<< "$out"; then
    echo "CAUGHT $name: $(grep -F "FAIL $want" <<< "$out" | head -1 | cut -c1-160); $(grep 'unit tests:' <<< "$out")"; caught=$((caught + 1))
  else
    echo "MISSED $name: no 'FAIL $want'; $(grep 'unit tests:' <<< "$out")"; missed=$((missed + 1))
  fi; }
mutant gpu_check '[ -z "$apps" ] || die 3' 'true || die 3' "refused with a GPU compute process"
mutant foreign_pid 'is_ours "$pid" || die 8' 'true || die 8' "stop refuses a foreign pid"
mutant ctx_total '-c "$((NP * CTX))"' '-c "$CTX"' "np 2 doubles -c"
mutant cpu_gate 'c + 0 <= m + 0' 'c + 0 >= 0' "refused by the CPU gate (5.20 > 4.5)"
mutant one_instance '[ -z "$others" ] || die 3 "a llama-server' 'true || die 3 "a llama-server' "refused beside another llama-server"
mutant props_check '(n_ctx != ctx)' '(False)' "a wrong n_ctx in /props is refused"
mutant build_pin '(p.get("build_info") != build)' '(False)' "a wrong build_info in /props is refused"
mutant default_reasoning 'REA=off; NOTE=""' 'REA=auto; NOTE=""' "print has ' -rea off '"
mutant q8_pin '[ "${line%% *}" = "$Q8_SHA" ] || die 5' 'true || die 5' "a matching file that is not the pinned Q8_0 is refused"
mutant list_pin '[ "$(sha256sum "$HASH_LIST" | cut -d'"' '"' -f1)" = "$LIST_SHA" ] || die 5' 'true || die 5' "a valid but different build list is refused by the pin"
pkill -f "$W/" 2>/dev/null
pkill -f "$T/stub/llama-server" 2>/dev/null
echo "mutants: $caught caught, $missed missed"
[ "$missed" = 0 ]
