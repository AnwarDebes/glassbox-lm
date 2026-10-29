#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# shellcheck disable=SC2015
# Read-only checks before 3c's chain restarts (decided 01:45Z: 3c restarts ALONE on the GPU, limit unchanged at 5 h,
# gen.jsonl kept as it is and its hash checked first; decided 03:39Z: after the E2-0 / E4-0 smokes, whose server
# must be stopped first). It starts, stops and signals nothing. Prints every check and READY or NOT READY (exit 1).
# The expected values are the recorded ones: gen.jsonl 2354 rows, sha256 60688a2c05b6a37c (tu/c3/NOTE_2026-10-21_stop.md);
# c3_chain.sh 597e50a5c5c0fa39 and c3_rate_monitor.sh 3d913800e8c81798 (LOG.md 01:33:07Z); base.json ec1bad4d8e54a0e3
# (runs_c3.log). Usage: bash tu/c3_restart_check.sh
tu=${REPO_ROOT}/studies/text_understanding/tu
c3=$tu/c3
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
serve=${REPO_ROOT}/studies/agent_guards/llm/serve
nbad=0
res() { if [ "$1" = ok ]; then echo "ok   $2"; else echo "BAD  $2"; nbad=$((nbad + 1)); fi; }
h16() { sha256sum "$1" | cut -c1-16; }
[ "$(h16 $c3/gen.jsonl)" = 60688a2c05b6a37c ] && res ok "gen.jsonl sha256 60688a2c05b6a37c" || res bad "gen.jsonl sha256 $(h16 $c3/gen.jsonl), recorded 60688a2c05b6a37c"
n=$(grep -c "" $c3/gen.jsonl); [ "$n" = 2354 ] && res ok "gen.jsonl 2354 rows" || res bad "gen.jsonl $n rows, recorded 2354"
"$py" -I -c "import json, sys; [json.loads(l) for l in open(sys.argv[1])]" $c3/gen.jsonl 2>/dev/null && res ok "every gen.jsonl row parses" || res bad "a gen.jsonl row does not parse"
[ "$(h16 $tu/c3_chain.sh)" = 597e50a5c5c0fa39 ] && res ok "c3_chain.sh 597e50a5c5c0fa39" || res bad "c3_chain.sh $(h16 $tu/c3_chain.sh)"
[ "$(h16 $tu/c3_rate_monitor.sh)" = 3d913800e8c81798 ] && res ok "c3_rate_monitor.sh 3d913800e8c81798" || res bad "c3_rate_monitor.sh $(h16 $tu/c3_rate_monitor.sh)"
[ "$(h16 $c3/base.json)" = ec1bad4d8e54a0e3 ] && res ok "base.json ec1bad4d8e54a0e3" || res bad "base.json $(h16 $c3/base.json)"
[ -e $tu/.c3/gpu_go ] && res ok "tu/.c3/gpu_go exists" || res bad "tu/.c3/gpu_go is missing"
p=$(pgrep -x llama-server | tr '\n' ' '); [ -z "$p" ] && res ok "no llama-server running" || res bad "llama-server running: pids $p"
[ -e $serve/current.pid ] && res bad "serve_qwen_agent.sh has a pid file ($(tr '\n' ' ' < $serve/current.env 2>/dev/null)); stop it first" || res ok "no agent server pid file"
pgrep -f "$tu/c3_chain.sh" > /dev/null && res bad "a c3_chain.sh is already running" || res ok "no c3_chain.sh running"
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1)
[ -z "$apps" ] && res ok "GPU: no compute process" || res bad "GPU compute processes (host PIDs): $(echo "$apps" | tr '\n' ';')"
(exec 3<>/dev/tcp/127.0.0.1/18081) 2>/dev/null && res bad "port 18081 is in use" || res ok "port 18081 free"
line=$("$py" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
awk -v c="$c" 'BEGIN { exit !(c ~ /^[0-9]+(\.[0-9]+)?$/ && c + 0 <= 4.5) }' && res ok "CPU gate: $line" || res bad "CPU gate: $line"
if [ "$nbad" = 0 ]; then echo "READY"; else echo "NOT READY ($nbad checks)"; exit 1; fi
