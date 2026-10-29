#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stub tests for tu/d3_wait_go.sh (no GPU is touched: nvidia-smi is a stub that reads its answers from a file).
# Cases: idle GPU after the pid ends; a busy GPU that frees later (the count resets); TERM while waiting; an existing
# go file is refused. Run: bash tu/tests/test_d3_wait_go.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)_d3wait
mkdir -p "$W" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
cat > "$W/nvsmi.sh" <<'EOS'
#!/usr/bin/env bash
# prints the first line of $NV_FILE and drops it (the last line repeats)
l=$(head -1 "$NV_FILE"); [ "$(wc -l < "$NV_FILE")" -gt 1 ] && sed -i 1d "$NV_FILE"; echo "$l"
EOS
export D3W_SLEEP=0.2 D3W_NVSMI="bash $W/nvsmi.sh"

echo "== idle"
sleep 1 & p=$!
export D3W_GO=$W/idle/go NV_FILE=$W/nv_idle; mkdir -p "$W/idle"; printf '0 10\n' > "$NV_FILE"
timeout 30 bash "$TU/d3_wait_go.sh" "$p" "test idle" > "$W/idle.out" 2>&1; e=$?
[ $e = 0 ] && ok "idle: exit 0" || bad "idle: exit $e"
grep -q "^go 20.*test idle; pid $p gone, then 3 empty-GPU checks$" "$D3W_GO" && ok "idle: the go file" || bad "idle: go file: $(cat "$D3W_GO" 2>/dev/null)"
[ "$(grep -c 'empty-GPU check' "$W/idle.out")" = 3 ] && ok "idle: 3 checks" || bad "idle: checks"
grep -q "pid $p gone" "$W/idle.out" && ok "idle: waited for the pid" || bad "idle: pid line"

echo "== busy, then free"
sleep 1 & p=$!
export D3W_GO=$W/busy/go NV_FILE=$W/nv_busy; mkdir -p "$W/busy"; printf '0 10\n0 10\n1 9000\n0 10\n0 10\n0 10\n' > "$NV_FILE"
timeout 30 bash "$TU/d3_wait_go.sh" "$p" "test busy" > "$W/busy.out" 2>&1; e=$?
[ $e = 0 ] && ok "busy: exit 0" || bad "busy: exit $e"
grep -q "GPU busy (1 processes, 9000 MiB): the count starts again" "$W/busy.out" && ok "busy: the count resets" || bad "busy: no reset line"
[ "$(grep -c 'empty-GPU check' "$W/busy.out")" = 5 ] && ok "busy: 2 + 3 checks" || bad "busy: $(grep -c 'empty-GPU check' "$W/busy.out") checks"
[ -e "$D3W_GO" ] && ok "busy: go file after the reset" || bad "busy: no go file"

echo "== memory above 500 MiB counts as busy"
sleep 0.5 & p=$!
export D3W_GO=$W/mem/go NV_FILE=$W/nv_mem; mkdir -p "$W/mem"; printf '0 501\n0 500\n0 500\n0 500\n' > "$NV_FILE"
timeout 30 bash "$TU/d3_wait_go.sh" "$p" "test mem" > "$W/mem.out" 2>&1
grep -q "GPU busy (0 processes, 501 MiB)" "$W/mem.out" && ok "mem: 501 MiB is busy, 500 is free" || bad "mem: $(head -3 "$W/mem.out")"

echo "== TERM while waiting for the pid"
sleep 30 & p=$!
export D3W_GO=$W/term/go NV_FILE=$W/nv_idle2; mkdir -p "$W/term"; printf '0 10\n' > "$NV_FILE"
bash "$TU/d3_wait_go.sh" "$p" "test term" > "$W/term.out" 2>&1 & w=$!
sleep 1; kill -TERM "$w"; wait "$w"; e=$?
kill "$p" 2>/dev/null
[ $e = 143 ] && ok "TERM: exit 143" || bad "TERM: exit $e"
[ ! -e "$D3W_GO" ] && ok "TERM: no go file" || bad "TERM: go file written"

echo "== an existing go file"
export D3W_GO=$W/idle/go
timeout 10 bash "$TU/d3_wait_go.sh" 1 "test exists" > "$W/exists.out" 2>&1; e=$?
[ $e = 1 ] && grep -q "exists already" "$W/exists.out" && ok "exists: refused" || bad "exists: exit $e"

echo "d3 wait tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
