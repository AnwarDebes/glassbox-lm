#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stub tests for tu/c3x_chain.sh's end line (the 11:39Z change): every way out appends
# "c3x chain end: <branch> <UTC> exit <code>" to the chain log. No GPU, no python step runs: the steps are a stub that
# records its arguments; the CPU gate and the pins are stubs too. Cases: done, a failing step, TERM during a step,
# TERM while waiting for the go file, a refused pin check, and a dry run. Each case has its own folder in
# tu/tests/runs/<UTC stamp>_c3xchain/.
# Run: bash tu/tests/test_c3x_chain.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
CH=$TU/c3x_chain.sh
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)_c3xchain
mkdir -p "$W" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
has() { if grep -qE -- "$3" "$2"; then ok "$1"; else bad "$1: '$3' not in $2: $(tail -c 300 "$2" 2>/dev/null)"; fi; }
cat > "$W/stub.sh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
case "$*" in *"${STUB_FAIL_ON:-@@none@@}"*) exit 1 ;; esac
case "$*" in *"${STUB_SLEEP_ON:-@@none@@}"*) sleep 30 ;; esac
exit 0
EOF
echo '{"s": {"clinc150": 8.0, "banking77": 8.0, "hwu64": 16.0}}' > "$W/base.json"
echo "pinned" > "$W/pinned.txt"
echo "check $W/pinned.txt $(sha256sum "$W/pinned.txt" | cut -c1-16)" > "$W/pins_ok.sh"
echo "check $W/pinned.txt 0000000000000000" > "$W/pins_bad.sh"
setup() { local d=$W/$1; mkdir -p "$d"; : > "$d/go"
  export C3X_DIR=$d C3X_GO=$d/go C3X_PINS=$W/pins_ok.sh C3X_BASE=$W/base.json C3X_STEP_CMD="bash $W/stub.sh" \
    C3X_LOAD="echo 2026-10-21T00:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods" C3X_GATE_SLEEP=1 C3X_GO_SLEEP=1 \
    C3X_SKIP_CODECHECK=1 C3X_NO_NVSMI=1 STUB_LOG=$d/steps.txt
  unset STUB_FAIL_ON STUB_SLEEP_ON; }

echo "== done"
setup "done"; d=$W/done
timeout 120 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "done: exit 0" || bad "done: exit $e"
has "done: the end line" "$d/runs_c3x.log" "^c3x chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
has "done: 'c3x done' before it" "$d/runs_c3x.log" "c3x done"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 55 ] && ok "done: 55 steps ran (3 feats, 40 TM, 11 LR, 1 report)" || bad "done: $n steps"
has "done: the chain and pins hashes are logged at the start" "$d/runs_c3x.log" "chain [0-9a-f]{16}, pins [0-9a-f]{16}"

echo "== a failing step"
setup fail; d=$W/fail; export STUB_FAIL_ON="--repeat 5"
timeout 120 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "fail: exit 1" || bad "fail: exit $e"
has "fail: STOP line" "$d/runs_c3x.log" "STOP: tm exit 1"
has "fail: the end line names the branch" "$d/runs_c3x.log" "^c3x chain end: tm 20[0-9-]+T[0-9:]+Z exit 1$"

echo "== TERM during a step"
setup term; d=$W/term; export STUB_SLEEP_ON="c3x_report.py"
setsid bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
for _ in $(seq 1 100); do grep -q "start report" "$d/runs_c3x.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM -- -"$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "TERM: exit 143" || bad "TERM: exit $e"
has "TERM: the end line names the branch" "$d/runs_c3x.log" "^c3x chain end: report 20[0-9-]+T[0-9:]+Z exit 143$"

echo "== TERM while waiting for the go file"
setup wait; d=$W/wait; rm -f "$d/go"
bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
sleep 2; kill -TERM "$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "wait-go TERM: exit 143" || bad "wait-go TERM: exit $e"
has "wait-go: the end line" "$d/runs_c3x.log" "^c3x chain end: wait-go 20[0-9-]+T[0-9:]+Z exit 143$"
[ ! -e "$d/steps.txt" ] && ok "wait-go: no step ran" || bad "wait-go: steps ran"

echo "== a refused pin check"
setup pins; d=$W/pins; export C3X_PINS=$W/pins_bad.sh
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "pins: exit 1" || bad "pins: exit $e"
has "pins: STOP names the file" "$d/runs_c3x.log" "STOP: $W/pinned.txt changed"
has "pins: the end line" "$d/runs_c3x.log" "^c3x chain end: pins 20[0-9-]+T[0-9:]+Z exit 1$"

echo "== dry run"
setup dry; d=$W/dry
timeout 60 bash "$CH" --dry > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "dry: exit 0" || bad "dry: exit $e"
n=$(grep -c "dry: would run" "$d/out.txt"); [ "$n" = 55 ] && ok "dry: 55 steps listed" || bad "dry: $n steps"
has "dry: the end line goes to stdout" "$d/out.txt" "^c3x chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
[ ! -s "$d/runs_c3x.log" ] && ok "dry: nothing written to the log" || bad "dry: log written"

echo "c3x chain tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
