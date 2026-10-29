#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stub tests for tu/m1_chain.sh. Every way out appends "m1 chain end: <branch> <UTC> exit <code>" to the chain log.
# No GPU and no python step runs: the steps are a stub that records its arguments, and the CPU gate, the pins and the
# step-0 file are stubs too.
# Cases, each with its own folder in tu/tests/runs/<UTC stamp>_m1chain/:
#   done; a failing step; TERM during a step; TERM while waiting for the go file; a refused pin check; no step-0 file;
#   a dry run.
# Run: bash tu/tests/test_m1_chain.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
CH=$TU/m1_chain.sh
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)_m1chain
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
echo "pinned" > "$W/pinned.txt"
echo "check $W/pinned.txt $(sha256sum "$W/pinned.txt" | cut -c1-16)" > "$W/pins_ok.sh"
echo "check $W/pinned.txt 0000000000000000" > "$W/pins_bad.sh"
setup() { local d=$W/$1; mkdir -p "$d"; : > "$d/go"; echo '{}' > "$d/overlap_m1.json"
  export M1_DIR=$d M1_GO=$d/go M1_PINS=$W/pins_ok.sh M1_STEP_CMD="bash $W/stub.sh" \
    M1_LOAD="echo 2026-10-21T00:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods" M1_GATE_SLEEP=1 M1_GO_SLEEP=1 \
    M1_SKIP_CODECHECK=1 M1_NO_NVSMI=1 STUB_LOG=$d/steps.txt
  unset STUB_FAIL_ON STUB_SLEEP_ON; }

echo "== done"
setup "done"; d=$W/done
timeout 120 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "done: exit 0" || bad "done: exit $e"
has "done: the end line" "$d/runs_m1.log" "^m1 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
has "done: 'm1 done' before it" "$d/runs_m1.log" "m1 done"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 94 ] && ok "done: 94 steps ran (9 feats, 3 code check, 63 TM, 18 LR, 1 verdict)" || bad "done: $n steps"
n=$(grep -c "m1_tm.py .*--tag m1 " "$d/steps.txt"); [ "$n" = 63 ] && ok "done: 63 TM runs (7 configurations x 3 seeds x 3 sets)" || bad "done: $n TM runs"
n=$(grep -c -- "--repeat 5" "$d/steps.txt"); [ "$n" = 9 ] && ok "done: 9 repeat-5 runs, all a3B" || bad "done: $n repeat-5 runs"
n=$(grep -- "--repeat 5" "$d/steps.txt" | grep -vc "feats/a3B "); [ "$n" = 0 ] && ok "done: every repeat-5 run reads the a3B cache" || bad "done: $n repeat-5 runs elsewhere"
n=$(grep -c -- "--seed 7 \|--seed 8 \|--seed 9 " "$d/steps.txt"); [ "$n" = 63 ] && ok "done: every M1 TM run uses seed 7, 8 or 9" || bad "done: $n"
n=$(grep -c "m1_tm.py.*--s 8 " "$d/steps.txt"); [ "$n" = 64 ] && ok "done: every TM run (with the code check's) at s 8" || bad "done: $n at s 8"
has "done: the code check runs on the stand-in" "$d/steps.txt" "m1_tm.py --dataset clinc150 .*--seed 4 .*--tag m1-check .*m1_standin.py:load"
has "done: the chain and pins hashes are logged at the start" "$d/runs_m1.log" "chain [0-9a-f]{16}, pins [0-9a-f]{16}"
n=$(grep -c "m1_feats.py" "$d/steps.txt"); [ "$n" = 9 ] && ok "done: 9 feature builds (3 tables x 3 sets)" || bad "done: $n feature builds"
has "done: the a3B build holds its three arms" "$d/steps.txt" "a3B:none,a3B_3c:[^ ]*kept_F.jsonl,a3B_twin:[^ ]*twin_kept.jsonl"

echo "== a failing step"
setup fail; d=$W/fail; export STUB_FAIL_ON="--repeat 5"
timeout 120 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "fail: exit 1" || bad "fail: exit $e"
has "fail: STOP line" "$d/runs_m1.log" "STOP: tm exit 1"
has "fail: the end line names the branch" "$d/runs_m1.log" "^m1 chain end: tm 20[0-9-]+T[0-9:]+Z exit 1$"

echo "== TERM during a step"
setup term; d=$W/term; export STUB_SLEEP_ON="m1_verdict.py"
setsid bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
for _ in $(seq 1 150); do grep -q "start verdict" "$d/runs_m1.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM -- -"$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "TERM: exit 143" || bad "TERM: exit $e"
has "TERM: the end line names the branch" "$d/runs_m1.log" "^m1 chain end: verdict 20[0-9-]+T[0-9:]+Z exit 143$"

echo "== TERM while waiting for the go file"
setup wait; d=$W/wait; rm -f "$d/go"
bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
sleep 2; kill -TERM "$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "wait-go TERM: exit 143" || bad "wait-go TERM: exit $e"
has "wait-go: the end line" "$d/runs_m1.log" "^m1 chain end: wait-go 20[0-9-]+T[0-9:]+Z exit 143$"
[ ! -e "$d/steps.txt" ] && ok "wait-go: no step ran" || bad "wait-go: steps ran"

echo "== a refused pin check"
setup pins; d=$W/pins; export M1_PINS=$W/pins_bad.sh
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "pins: exit 1" || bad "pins: exit $e"
has "pins: STOP names the file" "$d/runs_m1.log" "STOP: $W/pinned.txt changed"
has "pins: the end line" "$d/runs_m1.log" "^m1 chain end: pins 20[0-9-]+T[0-9:]+Z exit 1$"

echo "== no step-0 file"
setup nostep0; d=$W/nostep0; rm -f "$d/overlap_m1.json"
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "no step 0: exit 1" || bad "no step 0: exit $e"
has "no step 0: STOP line" "$d/runs_m1.log" "STOP: no step-0 overlap file"
[ ! -e "$d/steps.txt" ] && ok "no step 0: no step ran" || bad "no step 0: steps ran"

echo "== dry run"
setup dry; d=$W/dry
timeout 60 bash "$CH" --dry > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "dry: exit 0" || bad "dry: exit $e"
n=$(grep -c "dry: would run" "$d/out.txt"); [ "$n" = 94 ] && ok "dry: 94 steps listed" || bad "dry: $n steps"
has "dry: the end line goes to stdout" "$d/out.txt" "^m1 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
[ ! -s "$d/runs_m1.log" ] && ok "dry: nothing written to the log" || bad "dry: log written"

echo "m1 chain tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
