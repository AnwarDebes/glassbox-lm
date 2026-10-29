#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Stub tests for tu/m2_chain.sh and tu/m2_write_pins.sh. Every way out of the chain appends "m2 chain end: <branch>
# <UTC> exit <code>" to its log. No GPU and no M2 python step runs: the steps are a stub that records its arguments
# and its niceness and (with STUB_WRITE) writes minimal outputs, so the chain's skip-if-done branches and have_tm /
# have_lr really run; the CPU gate and the go file are stubs too. The pins are real check lines (full sha256), and the
# freeze check (tu/m2_freeze_check.py) runs for real against a small plan with a "## Pins" block, written per case.
# Cases, each with its own folder in tu/tests/runs/<UTC stamp>_m2chain/:
#   done (19 steps in the order code check, features, TM, LR, verdict; every step under nice 19); a second run of a
#   finished chain (all 19 steps skipped); a failing step, then a restart that runs only the 12 steps not done; TERM
#   during a step and while waiting for the go file; a refused pin; a pinned file that changes between two steps;
#   no pins file; no matched list; a 16-character pin (M1's form); the freeze check refusing a plan whose block differs
#   from the pins; a dry run (the freeze check included);
#   m2_write_pins.sh: --list, --block, the required dry summary, pins written only when the freeze check passes (a
#   plan block that differs, or a dry summary without pass, is refused and leaves no pins file), the chain's dry run
#   accepting the written pins, an earlier pins file kept, and (re-check of draft 2, MINOR D) relative dry-summary,
#   plan and output paths written as absolute paths, accepted by the chain from another working directory.
# Run: bash tu/tests/test_m2_chain.sh
set -u
TU=${REPO_ROOT}/studies/text_understanding/tu
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
CH=$TU/m2_chain.sh
W=$TU/tests/runs/$(date -u +%Y%m%dT%H%M%SZ)_m2chain
mkdir -p "$W" || exit 1
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
has() { if grep -qE -- "$3" "$2"; then ok "$1"; else bad "$1: '$3' not in $2: $(tail -c 300 "$2" 2>/dev/null)"; fi; }
cat > "$W/stub.sh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
[ -n "${STUB_NICE_LOG:-}" ] && ps -o ni= -p $$ | tr -d ' ' >> "$STUB_NICE_LOG"
case "$*" in *"${STUB_FAIL_ON:-@@none@@}"*) exit 1 ;; esac
case "$*" in *"${STUB_SLEEP_ON:-@@none@@}"*) sleep 30 ;; esac
case "$*" in *"${STUB_TOUCH_ON:-@@none@@}"*) echo changed >> "$STUB_TOUCH_FILE" ;; esac
[ -n "${STUB_WRITE:-}" ] || exit 0
script=$(basename "$1"); shift
declare -A A=()
while [ $# -gt 0 ]; do
  case "$1" in
    --*) k=${1#--}; if [ $# -gt 1 ] && [[ "$2" != --* ]]; then A[$k]=$2; shift 2; else A[$k]=1; shift; fi ;;
    *) shift ;;
  esac
done
case "$script" in
  m2_check_cache.py) mkdir -p "${A[out]}"; : > "${A[out]}/clinc150__p32tok_K512.npz" ;;
  m2_feats.py) arm=${A[arms]%%:*}; mkdir -p "${A[out-root]}/$arm"; : > "${A[out-root]}/$arm/${A[dataset]}__p32tok_K512.npz" ;;
  m2_tm.py) arm=$(basename "${A[emb-dir]}"); [ "${A[tag]}" = m2-check ] && arm=orig; mkdir -p "${A[out-dir]}"
    echo "{\"tag\": \"${A[tag]}\", \"arm\": \"$arm\", \"dataset\": \"${A[dataset]}\", \"seed\": ${A[seed]}, \"repeat\": ${A[repeat]:-1}}" \
      >> "${A[out-dir]}/results_m2_tm.jsonl" ;;
  m2_lr.py) mkdir -p "${A[out-dir]}"; echo "{\"name\": \"${A[name]}\", \"dataset\": \"${A[dataset]}\"}" >> "${A[out-dir]}/results_m2_lr.jsonl" ;;
  m2_verdict.py) : > "${A[m2]}/verdict_m2.json" ;;
esac
exit 0
EOF
echo '{}' > "$W/matched.json"
# mkplan PLANPATH FILE...: a small plan whose "## Pins" section holds the sha256sum lines of FILE...
mkplan() { local p=$1; shift; { echo "# test plan"; echo; echo "## Pins"; echo; echo '```'; sha256sum "$@"; echo '```'; echo; echo "## Draft history"; } > "$p"; }
# mkpins PINS PLANPATH FILE...: check lines for FILE... and for the plan itself
mkpins() { local o=$1 p=$2; shift 2; { for f in "$@" "$p"; do echo "check $f $(sha256sum "$f" | cut -c1-64)"; done; } > "$o"; }
echo "pinned" > "$W/pinned.txt"
mkplan "$W/plan_ok.md" "$W/pinned.txt"
mkpins "$W/pins_ok.sh" "$W/plan_ok.md" "$W/pinned.txt"
{ echo "check $W/pinned.txt 0000000000000000000000000000000000000000000000000000000000000000"; } > "$W/pins_bad.sh"
{ echo "check $W/pinned.txt $(sha256sum "$W/pinned.txt" | cut -c1-16)"; } > "$W/pins_16.sh"
setup() { local d=$W/$1; mkdir -p "$d"; : > "$d/go"
  export M2_DIR=$d M2_GO=$d/go M2_PINS=$W/pins_ok.sh M2_PLAN=$W/plan_ok.md M2_MATCHED=$W/matched.json \
    M2_STEP_CMD="bash $W/stub.sh" M2_LOAD="echo 2026-10-25T00:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods" \
    M2_GATE_SLEEP=1 M2_GO_SLEEP=1 M2_SKIP_CODECHECK=1 M2_NO_NVSMI=1 STUB_LOG=$d/steps.txt STUB_NICE_LOG=$d/nice.txt STUB_WRITE=1
  unset STUB_FAIL_ON STUB_SLEEP_ON STUB_TOUCH_ON STUB_TOUCH_FILE; }
line_of() { grep -n -- "$2" "$1" | head -1 | cut -d: -f1; }
last_of() { grep -n -- "$2" "$1" | tail -1 | cut -d: -f1; }

echo "== done"
setup "done"; d=$W/done
timeout 180 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "done: exit 0" || bad "done: exit $e $(tail -c 400 "$d/runs_m2.log")"
has "done: the end line" "$d/runs_m2.log" "^m2 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
has "done: 'm2 done' before it" "$d/runs_m2.log" "m2 done"
has "done: the freeze check ran before the go and passed" "$d/runs_m2.log" "^freeze check ok: 2 pins, 1 block lines$"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 19 ] && ok "done: 19 steps ran (3 code check, 3 feats, 9 TM, 3 LR, 1 verdict)" || bad "done: $n steps"
n=$(grep -c "m2_tm.py .*--tag m2 " "$d/steps.txt"); [ "$n" = 9 ] && ok "done: 9 TM runs (3 arms x seeds 7, 8, 9)" || bad "done: $n TM runs"
n=$(grep -c -- "--dataset mtop_en" "$d/steps.txt"); [ "$n" = 15 ] && ok "done: every feats, TM and LR step reads mtop_en (15)" || bad "done: $n mtop_en steps"
n=$(grep -c -- "--repeat 5\|kept_F\|twin_kept" "$d/steps.txt"); [ "$n" = 0 ] && ok "done: no repeat, paraphrase or twin arm" || bad "done: $n such steps"
for arm in C4 a3B potion; do
  n=$(grep -c "m2_tm.py .*--emb-dir [^ ]*/feats/$arm --tag m2 " "$d/steps.txt"); [ "$n" = 3 ] && ok "done: arm $arm has 3 TM seeds" || bad "done: arm $arm $n TM runs"
done
n=$(grep -c -- "--seed 7 \|--seed 8 \|--seed 9 " "$d/steps.txt"); [ "$n" = 9 ] && ok "done: every M2 TM run uses seed 7, 8 or 9" || bad "done: $n"
n=$(grep -c "m2_tm.py.*--s 8 " "$d/steps.txt"); [ "$n" = 10 ] && ok "done: every TM run (with the code check's) at s 8" || bad "done: $n at s 8"
has "done: the code check runs on the CLINC150 stand-in" "$d/steps.txt" "m2_tm.py --dataset clinc150 .*--seed 4 .*--tag m2-check .*m2_standin.py:load"
has "done: the chain and pins hashes (full sha256) are logged at the start" "$d/runs_m2.log" "chain [0-9a-f]{64}, pins [0-9a-f]{64}"
n=$(grep -c "m2_feats.py" "$d/steps.txt"); [ "$n" = 3 ] && ok "done: 3 feature builds (C4, a3B, potion)" || bad "done: $n feature builds"
has "done: the verdict reads the matched list and the overlap file" "$d/steps.txt" "m2_verdict.py --m2 $d --matched $W/matched.json --overlap ${REPO_ROOT}/studies/text_understanding/fresh/mtop_overlap.json"
cc=$(last_of "$d/steps.txt" "m2-check\|--name check\|m2_check_cache"); fe=$(line_of "$d/steps.txt" "m2_feats.py")
fl=$(last_of "$d/steps.txt" "m2_feats.py"); t1=$(line_of "$d/steps.txt" "--tag m2 "); tl=$(last_of "$d/steps.txt" "--tag m2 ")
l1=$(line_of "$d/steps.txt" "m2_lr.py --name C4"); ll=$(last_of "$d/steps.txt" "m2_lr.py --name"); v=$(line_of "$d/steps.txt" "m2_verdict.py")
if [ "$cc" = 3 ] && [ "$cc" -lt "$fe" ] && [ "$fl" -lt "$t1" ] && [ "$tl" -lt "$l1" ] && [ "$ll" -lt "$v" ] && [ "$v" = 19 ]; then
  ok "done: order: code check (steps 1-3) before any MTOP step, then features, TM, LR, verdict (step 19)"
else bad "done: order cc=$cc fe=$fe fl=$fl t1=$t1 tl=$tl l1=$l1 ll=$ll v=$v"; fi
n=$(grep -c " start " "$d/runs_m2.log"); m=$(grep -c "load check:" "$d/runs_m2.log")
[ "$n" = 19 ] && [ "$m" -ge 1 ] && ok "done: 19 starts logged, the CPU gate logged" || bad "done: $n starts, $m gate lines"
n=$(sort -u "$d/nice.txt" | tr '\n' ' '); [ "$n" = "19 " ] && ok "done: every step ran at niceness 19" || bad "done: niceness '$n'"
grep -q 'nice -n 19 "${STEP_CMD\[@\]}"' "$CH" && grep -q 'nice -n 19 "$py" -I -B "$tu/m2_codecheck.py"' "$CH" \
  && ok "done: step() and the code check call nice -n 19 themselves (static check; the test runner may already be at 19)" \
  || bad "done: nice -n 19 missing in m2_chain.sh"

echo "== a second run of the finished chain skips every step"
timeout 180 bash "$CH" > "$d/out2.txt" 2>&1; e=$?
[ $e = 0 ] && ok "rerun: exit 0" || bad "rerun: exit $e"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 19 ] && ok "rerun: no new step (still 19; have_tm, have_lr and the file checks skipped all)" || bad "rerun: $n steps"
n=$(grep -c "m2 done" "$d/runs_m2.log"); [ "$n" = 2 ] && ok "rerun: 'm2 done' twice in the log" || bad "rerun: $n 'm2 done' lines"

echo "== a failing step, then a restart"
setup fail; d=$W/fail; export STUB_FAIL_ON="--seed 8 "
timeout 180 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "fail: exit 1" || bad "fail: exit $e"
has "fail: STOP line" "$d/runs_m2.log" "STOP: tm exit 1"
has "fail: the end line names the branch" "$d/runs_m2.log" "^m2 chain end: tm 20[0-9-]+T[0-9:]+Z exit 1$"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 8 ] && ok "fail: 8 step calls (3 code check, 3 feats, C4 seed 7, the failing C4 seed 8)" || bad "fail: $n step calls"
unset STUB_FAIL_ON
timeout 180 bash "$CH" > "$d/out2.txt" 2>&1; e=$?
[ $e = 0 ] && ok "restart: exit 0" || bad "restart: exit $e"
n=$(grep -c "" "$d/steps.txt"); [ "$n" = 20 ] && ok "restart: 12 more steps (C4 seeds 8-9, a3B and potion x 3, 3 LR, the verdict); the 7 done were skipped" || bad "restart: $n step calls"
n=$(grep -c "m2_tm.py .*feats/C4 --tag m2 .*" "$d/steps.txt"); n7=$(grep -c "m2_tm.py .*--seed 7 .*feats/C4 --tag m2 " "$d/steps.txt")
[ "$n" = 4 ] && [ "$n7" = 1 ] && ok "restart: C4 seed 7 ran once, seed 8 twice (failed, then rerun before any write)" || bad "restart: C4 calls $n, seed 7 $n7"
has "restart: the end line" "$d/runs_m2.log" "^m2 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"

echo "== TERM during a step"
setup term; d=$W/term; export STUB_SLEEP_ON="m2_verdict.py"
setsid bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
for _ in $(seq 1 300); do grep -q "start verdict" "$d/runs_m2.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM -- -"$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "TERM: exit 143" || bad "TERM: exit $e"
has "TERM: the end line names the branch" "$d/runs_m2.log" "^m2 chain end: verdict 20[0-9-]+T[0-9:]+Z exit 143$"

echo "== TERM while waiting for the go file"
setup wait; d=$W/wait; rm -f "$d/go"
bash "$CH" > "$d/out.txt" 2>&1 & pid=$!
for _ in $(seq 1 100); do grep -q "freeze check ok" "$d/runs_m2.log" 2>/dev/null && break; sleep 0.2; done
sleep 1; kill -TERM "$pid"; wait "$pid"; e=$?
[ $e = 143 ] && ok "wait-go TERM: exit 143" || bad "wait-go TERM: exit $e"
has "wait-go: the end line" "$d/runs_m2.log" "^m2 chain end: wait-go 20[0-9-]+T[0-9:]+Z exit 143$"
[ ! -e "$d/steps.txt" ] && ok "wait-go: no step ran" || bad "wait-go: steps ran"

echo "== a refused pin check"
setup pins; d=$W/pins; export M2_PINS=$W/pins_bad.sh
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "pins: exit 1" || bad "pins: exit $e"
has "pins: STOP names the file" "$d/runs_m2.log" "STOP: $W/pinned.txt changed"
has "pins: the end line" "$d/runs_m2.log" "^m2 chain end: pins 20[0-9-]+T[0-9:]+Z exit 1$"
[ ! -e "$d/steps.txt" ] && ok "pins: no step ran" || bad "pins: steps ran"

echo "== a pinned file changes between two steps (pins checked before every step)"
setup midpin; d=$W/midpin; cp "$W/pinned.txt" "$d/pinned_mid.txt"
mkplan "$d/plan.md" "$d/pinned_mid.txt"; mkpins "$d/pins_mid.sh" "$d/plan.md" "$d/pinned_mid.txt"
export M2_PINS=$d/pins_mid.sh M2_PLAN=$d/plan.md STUB_TOUCH_ON="--emb-dir $d/feats/C4 --tag m2 --out-dir $d" STUB_TOUCH_FILE=$d/pinned_mid.txt
timeout 180 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "midpin: exit 1" || bad "midpin: exit $e"
has "midpin: STOP names the changed file" "$d/runs_m2.log" "STOP: $d/pinned_mid.txt changed"
has "midpin: the end line names the TM branch" "$d/runs_m2.log" "^m2 chain end: tm 20[0-9-]+T[0-9:]+Z exit 1$"
n=$(grep -c "m2_tm.py .*--tag m2 " "$d/steps.txt"); [ "$n" = 1 ] && ok "midpin: the step after the change did not run (1 TM step)" || bad "midpin: $n TM steps ran"

echo "== no pins file"
setup nopins; d=$W/nopins; export M2_PINS=$W/does_not_exist.sh
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "no pins: exit 1" || bad "no pins: exit $e"
has "no pins: STOP line" "$d/runs_m2.log" "STOP: no pins file"

echo "== no frozen matched list"
setup nomatched; d=$W/nomatched; export M2_MATCHED=$W/missing_matched.json
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "no matched list: exit 1" || bad "no matched list: exit $e"
has "no matched list: STOP line" "$d/runs_m2.log" "STOP: no frozen matched list"
[ ! -e "$d/steps.txt" ] && ok "no matched list: no step ran" || bad "no matched list: steps ran"

echo "== a 16-character pin (M1's form) is refused"
setup pin16; d=$W/pin16; export M2_PINS=$W/pins_16.sh
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "pin16: exit 1 (full sha256 required)" || bad "pin16: exit $e"
has "pin16: STOP names the file" "$d/runs_m2.log" "STOP: $W/pinned.txt changed"

echo "== the freeze check refuses a plan whose pins block differs from the pins file"
setup freeze; d=$W/freeze; sed -E 's/^[0-9a-f]{64}  /0000000000000000000000000000000000000000000000000000000000000000  /' "$W/plan_ok.md" > "$d/plan_bad.md"
mkpins "$d/pins.sh" "$d/plan_bad.md" "$W/pinned.txt"; export M2_PINS=$d/pins.sh M2_PLAN=$d/plan_bad.md
timeout 60 bash "$CH" > "$d/out.txt" 2>&1; e=$?
[ $e = 1 ] && ok "freeze: exit 1" || bad "freeze: exit $e"
has "freeze: the checker names the file" "$d/runs_m2.log" "pinned file not in the plan's block with the same sha256: $W/pinned.txt"
has "freeze: STOP line" "$d/runs_m2.log" "STOP: freeze check: the pins differ from the registration"
has "freeze: the end line (before the go)" "$d/runs_m2.log" "^m2 chain end: pins 20[0-9-]+T[0-9:]+Z exit 1$"
[ ! -e "$d/steps.txt" ] && ok "freeze: no step ran" || bad "freeze: steps ran"
mkplan "$d/plan_noself.md" "$W/pinned.txt"; { echo "check $W/pinned.txt $(sha256sum "$W/pinned.txt" | cut -c1-64)"; } > "$d/pins_noself.sh"
export M2_PINS=$d/pins_noself.sh M2_PLAN=$d/plan_noself.md
timeout 60 bash "$CH" --dry > "$d/out_noself.txt" 2>&1; e=$?
[ $e = 1 ] && grep -q "does not pin the plan itself" "$d/out_noself.txt" && ok "freeze: pins that do not pin the plan are refused (dry run)" || bad "freeze: noself exit $e"

echo "== dry run"
setup dry; d=$W/dry
timeout 60 bash "$CH" --dry > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && ok "dry: exit 0" || bad "dry: exit $e"
n=$(grep -c "dry: would run" "$d/out.txt"); [ "$n" = 19 ] && ok "dry: 19 steps listed" || bad "dry: $n steps"
has "dry: the freeze check ran" "$d/out.txt" "^freeze check ok: 2 pins, 1 block lines$"
has "dry: pins ok" "$d/out.txt" "^dry: pins ok$"
has "dry: the end line goes to stdout" "$d/out.txt" "^m2 chain end: done 20[0-9-]+T[0-9:]+Z exit 0$"
[ ! -s "$d/runs_m2.log" ] && ok "dry: nothing written to the log" || bad "dry: log written"

echo "== m2_write_pins.sh"
pw=$W/write_pins; mkdir -p "$pw"; export M2_PLAN=$pw/plan.md M2_PINS_OUT=$pw/pins.sh
echo "# placeholder" > "$pw/plan.md"
bash "$TU/m2_write_pins.sh" --list > "$pw/list.txt"; e=$?
n=$(grep -c "" "$pw/list.txt"); nf=$(grep -c "^$pw/plan.md$" "$pw/list.txt")
[ $e = 0 ] && [ "$n" = 58 ] && [ "$nf" = 1 ] && ok "write_pins --list: 58 files, the plan among them" || bad "write_pins --list: exit $e, $n files, plan $nf"
"$PY" -I -c "
import hashlib, json, sys
files = [l.strip() for l in open(sys.argv[1]) if l.strip() and l.strip() != sys.argv[2]]
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump({'pass': True, 'code_sha256': {f: h(f) for f in files}}, open(sys.argv[3], 'w'))
json.dump({'pass': False, 'code_sha256': {f: h(f) for f in files}}, open(sys.argv[4], 'w'))
" "$pw/list.txt" "$pw/plan.md" "$pw/dry_summary.json" "$pw/dry_fail.json"
bash "$TU/m2_write_pins.sh" --block "$pw/dry_summary.json" > "$pw/block.txt"; e=$?
n=$(grep -cE "^[0-9a-f]{64}  /" "$pw/block.txt"); np=$(grep -c "  $pw/plan.md$" "$pw/block.txt")
[ $e = 0 ] && [ "$n" = 58 ] && [ "$np" = 0 ] && grep -q "  $pw/dry_summary.json$" "$pw/block.txt" \
  && ok "write_pins --block: 58 sha256sum lines (57 files without the plan, plus the dry summary)" || bad "write_pins --block: exit $e, $n lines, plan $np"
{ echo "# synthetic plan"; echo; echo "## Pins"; echo; echo '```'; cat "$pw/block.txt"; echo '```'; echo; echo "## Draft history"; } > "$pw/plan.md"
bash "$TU/m2_write_pins.sh" > "$pw/out0.txt" 2>&1; e=$?
[ $e = 2 ] && [ ! -e "$pw/pins.sh" ] && ok "write_pins: refuses without the dry summary (exit 2)" || bad "write_pins: no-summary exit $e"
bash "$TU/m2_write_pins.sh" "$pw/dry_fail.json" > "$pw/out_fail.txt" 2>&1; e=$?
[ $e = 1 ] && [ ! -e "$pw/pins.sh" ] && grep -q "pass: true" "$pw/out_fail.txt" && ok "write_pins: a dry summary without pass is refused; no pins file" || bad "write_pins: dry_fail exit $e"
bash "$TU/m2_write_pins.sh" "$pw/dry_summary.json" > "$pw/out1.txt" 2>&1; e=$?
[ $e = 0 ] && grep -q "^freeze check ok: 59 pins, 58 block lines, dry summary ok$" "$pw/out1.txt" && ok "write_pins: the freeze check passed; pins written" || bad "write_pins: exit $e $(tail -c 400 "$pw/out1.txt")"
n=$(grep -c '^check ' "$pw/pins.sh" 2>/dev/null); [ "$n" = 59 ] && ok "write_pins: 59 check lines (58 files with the plan, plus the dry summary)" || bad "write_pins: $n lines"
badh=0
while read -r _c f h; do [ "$(sha256sum "$f" | cut -c1-64)" = "$h" ] && [ ${#h} = 64 ] || badh=$((badh + 1)); done < <(grep '^check ' "$pw/pins.sh")
[ "$badh" = 0 ] && ok "write_pins: every line holds the file's full sha256 (64 hex, read again here)" || bad "write_pins: $badh wrong lines"
for f in mtop.zip mtop_test_matched_v1.json mtop_overlap.json a3B_d768.npz ppw_S1000_d500_e0.5.npz m2_verdict.py m2_codecheck.py m2_freeze_check.py m2_code_diff_vs_m1.txt m1_verdict.py cpu_load.py torch/version.py transformers/__init__.py safetensors/__init__.py plan.md dry_summary.json; do
  grep -q "$f " "$pw/pins.sh" && ok "write_pins: pins $f" || bad "write_pins: $f not pinned"
done
setup wpdry; d=$W/wpdry; export M2_PINS=$pw/pins.sh M2_PLAN=$pw/plan.md
timeout 60 bash "$CH" --dry > "$d/out.txt" 2>&1; e=$?
[ $e = 0 ] && grep -q "^dry: pins ok$" "$d/out.txt" && ok "write_pins: the chain's dry run (with the freeze check) accepts the written pins" || bad "write_pins: chain dry exit $e $(tail -c 300 "$d/out.txt")"
cp "$pw/plan.md" "$pw/plan_good.md"; sed -i '0,/^[0-9a-f]\{64\}  \//s//0000000000000000000000000000000000000000000000000000000000000000  \//' "$pw/plan.md"
cp "$pw/pins.sh" "$pw/pins_before.sh"
bash "$TU/m2_write_pins.sh" "$pw/dry_summary.json" > "$pw/out2.txt" 2>&1; e=$?
[ $e = 1 ] && cmp -s "$pw/pins.sh" "$pw/pins_before.sh" && grep -q "freeze check REFUSED" "$pw/out2.txt" \
  && ok "write_pins: a plan block that differs is refused; the pins file is left as it was" || bad "write_pins: bad block exit $e"
cp "$pw/plan_good.md" "$pw/plan.md"
sleep 1
bash "$TU/m2_write_pins.sh" "$pw/dry_summary.json" > "$pw/out3.txt" 2>&1; e=$?
n=$(ls "$pw"/pins_*.sh 2>/dev/null | grep -vc pins_before); [ $e = 0 ] && [ "$n" = 1 ] && ok "write_pins: a rewrite keeps the earlier pins file as pins_<UTC>.sh" || bad "write_pins: rewrite exit $e, $n kept"
n=$(ls -a "$pw" | grep -c "^.pins_tmp"); [ "$n" = 0 ] && ok "write_pins: no temporary file left behind" || bad "write_pins: $n temporary files"
(cd "$pw" && bash "$TU/m2_write_pins.sh" --block dry_summary.json) > "$pw/block_rel.txt" 2>&1; e=$?
[ $e = 0 ] && cmp -s "$pw/block_rel.txt" "$pw/block.txt" && ok "write_pins --block: a relative dry-summary path gives the same block (absolute paths)" \
  || bad "write_pins --block relative: exit $e $(diff "$pw/block_rel.txt" "$pw/block.txt" | head -3)"
(cd "$pw" && M2_PLAN=plan.md M2_PINS_OUT=pins_rel.sh bash "$TU/m2_write_pins.sh" dry_summary.json) > "$pw/out_rel.txt" 2>&1; e=$?
[ $e = 0 ] && ! grep -q '^check [^/]' "$pw/pins_rel.sh" && grep -q "^check $pw/dry_summary.json " "$pw/pins_rel.sh" \
  && grep -q "^check $pw/plan.md " "$pw/pins_rel.sh" && ok "write_pins: relative dry-summary, plan and output paths are written as absolute paths" \
  || bad "write_pins relative: exit $e $(tail -c 300 "$pw/out_rel.txt")"
(cd / && M2_PINS=$pw/pins_rel.sh M2_PLAN=$pw/plan.md bash "$CH" --dry) > "$pw/out_rel_chain.txt" 2>&1; e=$?
[ $e = 0 ] && grep -q "^dry: pins ok$" "$pw/out_rel_chain.txt" && ok "write_pins: the chain's dry run accepts those pins from another working directory (/)" \
  || bad "write_pins relative chain: exit $e $(tail -c 300 "$pw/out_rel_chain.txt")"

echo "m2 chain tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
