#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Tests for tu/d3_chain.sh with a stub for every step (tests/d3_chain_stub.py: it records each call and writes the
# outputs the real step would write, never creating an --out-dir). No GPU, no real step runs; the CPU gate and the pins
# are stubs; the M1 caches and 3c rows are small fakes made here. Every way out appends
# "d3 chain end: <branch> <UTC> exit <code>" to the chain log.
# Cases, each in its own folder in tu/tests/runs/<UTC stamp>_d3chain/:
#   done with every gate passing (148 steps); a restart (a new go, 0 steps); the go file used up; a failing step;
#   each gate failing (code check, fit check, fit not converged, feature check, a pair arm without pairs, Jwc below
#   JnoP); TERM during a step; TERM while waiting for the go file; refused pins; refused big pins; test hooks refused
#   in the real run (the real log untouched); a dry run.
# Run: bash tu/tests/test_d3_chain.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
CH=$TU/d3_chain.sh
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)_d3chain
mkdir -p "$W" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
has() { if grep -qE -- "$3" "$2"; then ok "$1"; else bad "$1: '$3' not in $2: $(tail -c 300 "$2" 2>/dev/null)"; fi; }
nsteps() { if [ -s "$1/steps.txt" ]; then grep -c "" "$1/steps.txt"; else echo 0; fi; }
echo "pinned" > "$W/pinned.txt"
echo "check $W/pinned.txt $(sha256sum "$W/pinned.txt" | cut -c1-16)" > "$W/pins_ok.sh"
echo "check $W/pinned.txt 0000000000000000" > "$W/pins_bad.sh"
# fake M1 caches (train and val arrays only) and fake 3c reference rows
"$PY" -I -c "
import sys
import numpy as np
from pathlib import Path
r = np.random.default_rng(0)
for d in ('clinc150', 'banking77', 'hwu64'):
    p = Path(sys.argv[1]) / 'feats' / 'a3B'
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / f'{d}__p32tok_K512.npz', train__exists=r.random((20, 4)).astype(np.float32),
             val__exists=r.random((8, 4)).astype(np.float32), y_train=np.arange(20), y_val=np.arange(8))
" "$W/fake_m1" || exit 1
mkdir -p "$W/fake_c3"
echo '{"tag": "3c-ref", "dataset": "clinc150", "seed": 4, "val_acc": 0.945}' > "$W/fake_c3/results_c3_val.jsonl"
echo '{"name": "ref_rebuilt", "dataset": "clinc150", "lr_val": 0.9503}' > "$W/fake_c3/results_c3_lr.jsonl"
setup() { local d=$W/$1; mkdir -p "$d"; : > "$d/go"
  export D3_DIR=$d D3_GO=$d/go D3_PINS=$W/pins_ok.sh D3_PINS_BIG=$W/pins_ok.sh D3_STEP_CMD="$PY -I -B $TU/tests/d3_chain_stub.py" \
    D3_LOAD="echo 2026-10-23T00:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods" D3_GATE_SLEEP=1 D3_GO_SLEEP=1 \
    D3_NO_NVSMI=1 D3_M1=$W/fake_m1 D3_C3=$W/fake_c3 STUB_LOG=$d/steps.txt OMP_NUM_THREADS=1
  unset D3_SKIP_CHECKS STUB_CODE_TM STUB_FIT_PASS STUB_FIT_EXIT STUB_FEAT_NOISE STUB_PAIR_BAD STUB_JWC_LOW STUB_SLEEP_ON; }
run() { timeout "${2:-600}" bash "$CH" ${3:-} > "$W/$1/out.txt" 2>&1; }

echo "== done, every gate passing"
setup "done"; d=$W/done
run "done"; e=$?
[ $e = 0 ] && ok "done: exit 0" || bad "done: exit $e ($(tail -2 "$d/runs_d3.log"))"
has "done: the end line" "$d/runs_d3.log" "^d3 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
for g in "code check passed" "fit check passed" "feature check passed" "pair-arm check passed" "d3 done"; do
  has "done: '$g'" "$d/runs_d3.log" "$g"; done
has "done: the code check compared the rows" "$d/runs_d3.log" "^code check 0.945 0.945 0.9503 0.9503$"
has "done: the feature check's difference is 0" "$d/runs_d3.log" "feature check: max abs diff 0.0$"
n=$(nsteps "$d"); [ "$n" = 148 ] && ok "done: 148 steps (2 code check, 1 fit check, 3 tables, 33 feats, 75 TM, 33 LR, 1 verdict)" || bad "done: $n steps"
n=$(grep -c "d3_exists_tm.py .*--tag d3 " "$d/steps.txt"); [ "$n" = 75 ] && ok "done: 75 TM runs (7 arms x 3 seeds + 4 k-means arms x 1, on 3 sets)" || bad "done: $n TM runs"
n=$(grep "d3_exists_tm.py .*--tag d3 " "$d/steps.txt" | grep -E -c -- "--emb-dir [^ ]*_k[12] "); [ "$n" = 12 ] && ok "done: 12 k-means spread runs" || bad "done: $n k-arm runs"
n=$(grep "d3_exists_tm.py .*--tag d3 " "$d/steps.txt" | grep -E -- "--emb-dir [^ ]*_k[12] " | grep -vc -- "--seed 7 "); [ "$n" = 0 ] && ok "done: the k-means spread runs use seed 7 only" || bad "done: $n"
n=$(grep -c "d3_exists_tm.py.*--s 8 " "$d/steps.txt"); [ "$n" = 76 ] && ok "done: every TM run (with the code check's) at s 8" || bad "done: $n at s 8"
n=$(grep -c "d3_feats.py" "$d/steps.txt"); [ "$n" = 33 ] && ok "done: 33 feature builds (11 arms x 3 sets)" || bad "done: $n feature builds"
has "done: J = the pair table, pair tokens on, all types, k-means seed 0" "$d/steps.txt" "--arm J --table [^ ]*/tables/d3J_d768.npz --pair-tokens on --kmeans all --kmeans-seed 0"
has "done: MID = the midpoint table" "$d/steps.txt" "--arm MID --table [^ ]*/tables/d3MID_d768.npz --pair-tokens on --kmeans all"
has "done: BJ = a3B words + J's pair rows" "$d/steps.txt" "--arm BJ --table [^ ]*/tables/d3BJ_d768.npz --pair-tokens on --kmeans all"
has "done: a k-means spread arm" "$d/steps.txt" "--arm J_k2 --table [^ ]*/tables/d3J_d768.npz --pair-tokens on --kmeans all --kmeans-seed 2"
n=$(grep -c "d3_lr.py" "$d/steps.txt"); [ "$n" = 34 ] && ok "done: 34 LR fits (33 + the code check)" || bad "done: $n LR fits"
n=$(sed "s#$TU/tests/runs##g" "$d/steps.txt" | grep -cwE "test|oos_test"); [ "$n" = 0 ] && ok "done: no step names a test split" || bad "done: $n steps name test"
[ ! -e "$d/go" ] && ls "$d"/go.used_* > /dev/null 2>&1 && ok "done: the go file was used up (renamed)" || bad "done: go file not renamed"
has "done: the test hooks are logged in full" "$d/runs_d3.log" "test hooks: D3_C3 D3_DIR .*D3_M1 D3_NO_NVSMI"
has "done: torch and CUDA versions are logged" "$d/runs_d3.log" "torch/cuda: "

echo "== a restart of the finished run"
cp "$d/steps.txt" "$d/steps_run1.txt"; : > "$d/steps.txt"; : > "$d/go"
run "done"; e=$?
[ $e = 0 ] && [ "$(nsteps "$d")" = 0 ] && ok "restart: exit 0 and 0 steps" || bad "restart: exit $e, $(nsteps "$d") steps"
[ "$(grep -c "^d3 chain end: done" "$d/runs_d3.log")" = 2 ] && ok "restart: a second end line" || bad "restart: end lines"

echo "== the gates fail"
gate_case() { local name=$1 knob=$2 want=$3; setup "$name"; export "${knob?}"; run "$name"; local e=$?
  [ $e = 1 ] && ok "$name: exit 1" || bad "$name: exit $e"
  has "$name: STOP line" "$W/$name/runs_d3.log" "$want"
  n=$(grep -c "d3_exists_tm.py .*--tag d3 " "$W/$name/steps.txt"); [ "$n" = 0 ] && ok "$name: no TM run after the stop" || bad "$name: $n TM runs"; }
gate_case codecheck STUB_CODE_TM=0.9447 "STOP: code check"
gate_case fitcheck STUB_FIT_PASS=false "STOP: fit check"
gate_case notconverged STUB_FIT_EXIT=4 "STOP: fit exit 4"
gate_case featcheck STUB_FEAT_NOISE=0.0002 "STOP: feature check"
gate_case pairarm STUB_PAIR_BAD=MW "STOP: pair-arm check"
gate_case jwclow STUB_JWC_LOW=1 "STOP: pair-arm check"
has "featcheck: the difference is logged" "$W/featcheck/runs_d3.log" "feature check: max abs diff 0.000(19|2)"
has "jwclow: the problem is named" "$W/jwclow/runs_d3.log" "Jwc', 'train exists below JnoP"

echo "== a failing step"
setup fail; d=$W/fail; export D3_STEP_CMD="bash -c exit_1"
run fail; e=$?
[ $e = 1 ] && ok "fail: exit 1" || bad "fail: exit $e"
has "fail: STOP line names the step" "$d/runs_d3.log" "STOP: tm exit"
has "fail: the end line names the branch" "$d/runs_d3.log" "^d3 chain end: code-check 20[0-9-]+T[0-9:]+Z exit 1$"

echo "== TERM during a step"
setup term; d=$W/term
cat > "$W/slow.sh" <<'EOS'
#!/usr/bin/env bash
case "$*" in *d3_verdict.py*) sleep 30 ;; esac
exec "$PY_STUB" -I -B "$STUB" "$@"
EOS
export PY_STUB=$PY STUB=$TU/tests/d3_chain_stub.py D3_STEP_CMD="bash $W/slow.sh"
setsid bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
for _ in $(seq 1 600); do grep -q "start verdict" "$d/runs_d3.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM -- -"$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "TERM: exit 143" || bad "TERM: exit $e"
has "TERM: the end line names the branch" "$d/runs_d3.log" "^d3 chain end: verdict 20[0-9-]+T[0-9:]+Z exit 143$"

echo "== TERM while waiting for the go file"
setup wait; d=$W/wait; rm -f "$d/go"
bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
sleep 2; kill -TERM "$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "wait-go TERM: exit 143" || bad "wait-go TERM: exit $e"
has "wait-go: the end line" "$d/runs_d3.log" "^d3 chain end: wait-go 20[0-9-]+T[0-9:]+Z exit 143$"
[ ! -e "$d/steps.txt" ] && ok "wait-go: no step ran" || bad "wait-go: steps ran"

echo "== refused pins"
setup pins; d=$W/pins; export D3_PINS=$W/pins_bad.sh
run pins 60; e=$?
[ $e = 1 ] && ok "pins: exit 1" || bad "pins: exit $e"
has "pins: STOP names the file" "$d/runs_d3.log" "STOP: $W/pinned.txt changed"
[ ! -e "$d/steps.txt" ] && ok "pins: no step ran" || bad "pins: steps ran"
setup bigpins; d=$W/bigpins; export D3_PINS_BIG=$W/pins_bad.sh
run bigpins 60; e=$?
[ $e = 1 ] && [ ! -e "$d/steps.txt" ] && ok "big pins: exit 1, no step ran" || bad "big pins: exit $e"

echo "== test hooks in the real run"
REAL_LOG=$TU/d3/runs_d3.log
before=$( [ -e "$REAL_LOG" ] && sha256sum "$REAL_LOG" || echo none)
( unset D3_DIR; export D3_SKIP_CHECKS=1; timeout 30 bash "$CH" > "$W/hooks_out.txt" 2>&1 ); e=$?
after=$( [ -e "$REAL_LOG" ] && sha256sum "$REAL_LOG" || echo none)
[ $e = 1 ] && ok "hooks: exit 1" || bad "hooks: exit $e"
has "hooks: refused on stderr, naming the hooks" "$W/hooks_out.txt" "STOP: test hooks set in the real run: .*D3_SKIP_CHECKS"
[ "$before" = "$after" ] && ok "hooks: the real log is untouched" || bad "hooks: the real log changed"

echo "== dry run"
setup dry; d=$W/dry
run dry 120 --dry; e=$?
[ $e = 0 ] && ok "dry: exit 0" || bad "dry: exit $e"
n=$(grep -c "dry: would run" "$d/out.txt"); [ "$n" = 148 ] && ok "dry: 148 steps listed" || bad "dry: $n steps"
has "dry: the end line goes to stdout" "$d/out.txt" "^d3 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
[ ! -s "$d/runs_d3.log" ] && ok "dry: nothing written to the log" || bad "dry: log written"

echo "d3 chain tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
