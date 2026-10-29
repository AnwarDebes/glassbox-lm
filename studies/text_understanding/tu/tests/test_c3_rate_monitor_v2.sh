#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# shellcheck disable=SC2015,SC2016  # the stubs are written as literal code
# Tests for c3_rate_monitor_v2.sh on fake files (2 s windows, sets a:60 and b:40, stub nvidia-smi and cpu_load).
# Nothing real is read or written: every case has its own folder in tests/runs/<stamp>/.
# Run: bash tu/tests/test_c3_rate_monitor_v2.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
M=$TU/c3_rate_monitor_v2.sh
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$W" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
expect() { if [ "$3" = "$2" ]; then ok "$1 (exit $3)"; else bad "$1: exit $3, wanted $2"; fi; }
has() { if grep -qF -- "$3" "$2"; then ok "$1"; else bad "$1: '$3' not in $2: $(tail -c 400 "$2")"; fi; }
hasnt() { if grep -qF -- "$3" "$2"; then bad "$1: '$3' found in $2"; else ok "$1"; fi; }
iso() { date -u -d "$1" +%Y-%m-%dT%H:%M:%SZ; }
setup() { local d=$W/$1
  mkdir -p "$d/pilot"
  for i in $(seq 1 10); do echo '{"set": "a", "usage": {"completion_tokens": 20}}'; echo '{"set": "b", "usage": {"completion_tokens": 20}}'; done > "$d/pilot/gen_pilot.jsonl"
  for i in $(seq 1 10); do echo '{"set": "a", "usage": {"completion_tokens": 20}}'; done > "$d/gen.jsonl"
  printf '%s server health: {"status":"ok"}\n%s server health: {"status":"ok"}\n' "$(iso '5 hours ago')" "$(iso '4 hours ago')" > "$d/runs_c3.log"
  printf '#!/usr/bin/env bash\n[ -n "${STUB_APPS:-}" ] && printf "%%s\\n" "$STUB_APPS"\nexit 0\n' > "$d/nvidia-smi"
  printf '#!/usr/bin/env bash\necho "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ) cores 1.00 over 10.0 s; throttled ${STUB_THR:-0} of 100 periods"\n' > "$d/cpu_load"
  chmod +x "$d/nvidia-smi" "$d/cpu_load"
  export C3MON_DIR=$d C3MON_OUT=$d C3MON_WINDOW=2 C3MON_SETS=a:60,b:40 C3MON_NVSMI=$d/nvidia-smi C3MON_LOAD=$d/cpu_load
  unset STUB_APPS STUB_THR; }
writer() { local d=$1 i  # 90 more rows (a to 60, then b to 40), 20 rows per second
  for i in $(seq 11 100); do
    if [ "$i" -le 60 ]; then echo '{"set": "a", "usage": {"completion_tokens": 20}}'; else echo '{"set": "b", "usage": {"completion_tokens": 20}}'; fi >> "$d/gen.jsonl"
    sleep 0.05
  done; }

echo "== complete, with the restart's server start as the anchor"
setup complete; d=$W/complete
echo "$(iso 'now') server health: {\"status\":\"ok\"}" >> "$d/runs_c3.log"
writer "$d" & wp=$!
timeout 120 bash "$M" 5 > "$d/stdout.txt" 2>&1; e=$?; wait $wp
expect "a finished generation ends the monitor" 0 $e
has "it says complete" "$d/rate_monitor_v2.log" "generation complete; monitor end"
has "the anchor is the new server start" "$d/rate_monitor_v2.log" "anchor $(sed -n 3p "$d/runs_c3.log" | cut -c1-20)"
hasnt "no flag" "$d/rate_monitor_v2.log" "FLAG"
has "rows of the total" "$d/rate_monitor_v2.log" "of 100 (+"
[ -s "$d/rate_baseline_v2.txt" ] && ok "a baseline was written" || bad "no baseline"

echo "== old server starts only: no anchor yet, so no projection flag (v1 would flag here)"
setup stale; d=$W/stale
writer "$d" & wp=$!
timeout 120 bash "$M" 5 > "$d/stdout.txt" 2>&1; e=$?; wait $wp
expect "it runs to the end" 0 $e
has "no anchor yet" "$d/rate_monitor_v2.log" "anchor none yet"
hasnt "no flag" "$d/rate_monitor_v2.log" "FLAG"

echo "== another GPU process"
setup gpu; d=$W/gpu
echo "$(iso 'now') server health: {\"status\":\"ok\"}" >> "$d/runs_c3.log"
STUB_APPS=$'706560, 9984 MiB\n742362, 4184 MiB' timeout 60 bash "$M" 1e12 > "$d/stdout.txt" 2>&1; e=$?
expect "a second GPU process flags" 3 $e
has "it names the processes" "$d/rate_monitor_v2.log" "FLAG: 2 GPU compute processes (sharing is OFF)"
hasnt "no baseline from a shared window" "$d/rate_baseline_v2.txt" "."
STUB_APPS='706560, 9984 MiB' timeout 60 bash "$M" 1e12 > "$d/stdout2.txt" 2>&1 & mp=$!  # no rows come, so a huge limit
sleep 5; kill $mp 2>/dev/null; wait $mp 2>/dev/null
hasnt "one GPU process (the server itself) does not flag" "$d/stdout2.txt" "FLAG"

echo "== projected end over the limit (an explicit anchor 5 h ago)"
setup late; d=$W/late
writer "$d" & wp=$!
timeout 60 bash "$M" 5 "$(iso '5 hours ago')" > "$d/stdout.txt" 2>&1; e=$?; wait $wp
expect "a late projection flags" 3 $e
has "it names the projection" "$d/rate_monitor_v2.log" "FLAG: projected end"
has "the given anchor is used" "$d/rate_monitor_v2.log" "monitor v2 start (limit 5 h, anchor $(iso '5 hours ago' | cut -c1-14)"

echo "== throttling, and a half-written last line"
setup thr; d=$W/thr
printf '{"set": "a", "usage": {"compl' >> "$d/gen.jsonl"
STUB_THR=3 timeout 60 bash "$M" 5 > "$d/stdout.txt" 2>&1; e=$?
expect "a throttled window ends the monitor" 4 $e
has "it says throttled" "$d/stdout.txt" "THROTTLED"
has "the half line is skipped, not fatal" "$d/rate_monitor_v2.log" "rows 10 of 100"

echo "== throttling with C3MON_THROTTLE=log: logged, not an exit"
setup thrlog; d=$W/thrlog
echo "$(iso 'now') server health: {\"status\":\"ok\"}" >> "$d/runs_c3.log"
writer "$d" & wp=$!
STUB_THR=3 C3MON_THROTTLE=log timeout 120 bash "$M" 5 > "$d/stdout.txt" 2>&1; e=$?; wait $wp
expect "it runs to the end despite throttling" 0 $e
has "the throttled periods are logged" "$d/rate_monitor_v2.log" "throttled 3 of 100 periods"
has "the mode is logged" "$d/rate_monitor_v2.log" "throttling: log)"
C3MON_THROTTLE=maybe bash "$M" 5 > "$d/stdout_bad.txt" 2>&1; expect "an unknown throttle mode is refused" 2 $?

echo "monitor v2 tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
