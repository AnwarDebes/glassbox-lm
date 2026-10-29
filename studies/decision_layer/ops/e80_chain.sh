#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e80 after its deferral of 2026-10-21 (dated note logs/plan_e80_note_2026-10-25.md), one stage per start, by hand:
#   e80_chain.sh dry    GPU: 3 empty-GPU checks in a row; logs/jobs_e80_dry.txt (the 5 frozen job lines with
#                       --clauses 200 --epochs 2, tags e80dry_*) through scripts/seq_queue.sh; then the CPU gate and
#                       scripts/e80_dry_check.py into logs/e80_dry_check.out, which must end "all dry checks passed".
#   e80_chain.sh run    GPU: waits for "e80n dry end: done" and needs the dry check's pass line; 3 empty-GPU checks;
#                       logs/jobs_e80.txt (the 5 frozen job lines) through scripts/seq_queue.sh; then the CPU gate,
#                       scripts/e80_check.py --mwoz-s 8 into logs/e80_check.out and scripts/e80_report.py into
#                       logs/e80_report.out.
# Every stage:
# - holds a lock (logs/e80n_<stage>.lock), so a second chain of the same stage cannot start while one runs, and refuses
#   while an e87 GPU stage holds its lock (logs/e87n_a0.lock, e87n_bdry.lock, e87n_b.lock): the shared GPU lock allows
#   overlap, so the GPU stages of the two chains must not run or wait at the same time;
# - refuses without the note's freeze record (the SHA-256 of the note's text equals the record's), and unless the files
#   it runs equal the note's pins: every repository row of the note's hash table (sha256sum of the path in the first
#   column, a trailing parenthesis removed; rows marked "project notes" or "finish-pass notes" are outside the
#   repository), this script against its own row, the stage's job file present in the table, and no file on e80's path
#   (src/dl/, scripts/abcd_explore, table_run, seq_queue, cpu_load, e80_check, e80_dry_check, e80_report, diff_frozen)
#   changed since the note's freeze record (scripts/diff_frozen.py);
# - refuses a second start: its end log or an output exists. The one exception: a stage that ended before writing any
#   output (refused, a handoff or wait timeout, a gate timeout, killed while waiting) may start again; its end log keeps
#   the old line and gets a new one;
# - waits for 3 empty-GPU checks in a row (an nvidia-smi error counts as busy) and for the CPU gate before each CPU
#   command (a cpu_load.py error counts as busy);
# - writes "e80n <stage> end: <branch> <UTC> exit <code>" to logs/e80n_<stage>_end.log; a TERM, HUP or INT writes
#   "killed by signal <name>"; any end stops the running command's process group (setsid).
# Start a stage detached, for example:
#   setsid nohup bash e80_chain.sh dry > e80_chain_dry_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
S="${E80_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; Q="${QUEUE_CMD:-bash scripts/seq_queue.sh}"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python"
C="${CHECK_CMD:-$PY scripts/e80_check.py --mwoz-s 8}"; DC="${DRYCHECK_CMD:-$PY scripts/e80_dry_check.py}"; RP="${REPORT_CMD:-$PY scripts/e80_report.py}"
LOAD="${LOAD_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python -I scripts/cpu_load.py 10}"
DIFF="${DIFF_CMD:-env CUDA_VISIBLE_DEVICES= nice -n 19 .venv-dl/bin/python -I scripts/diff_frozen.py}"
POLL="${POLL_S:-20}"; GW="${GPU_WAIT_S:-21600}"; PW="${PRE_WAIT_S:-43200}"; GS="${GATE_S:-7200}"
NOTE=logs/plan_e80_note_2026-10-25; STAGE="${1:-}"
PATHRE='(src/dl/|scripts/(abcd_explore|table_run|seq_queue|cpu_load|e80_check|e80_dry_check|e80_report|diff_frozen))'
log() { echo "$(date -u +%FT%TZ) $*"; }
case "$STAGE" in
  dry) OUTS="logs/e80_dry_queue.log logs/e80_dry_check.out"; JOBS=logs/jobs_e80_dry.txt;;
  run) OUTS="logs/e80_queue.log logs/e80_check.out logs/e80_report.out"; JOBS=logs/jobs_e80.txt;;
  *) echo "usage: e80_chain.sh dry|run"; exit 1;;
esac
END="logs/e80n_${STAGE}_end.log"; NAME="e80n $STAGE end"
exec 9> "logs/e80n_$STAGE.lock"
flock -n 9 || { log "another e80n $STAGE chain is running: not starting"; exit 1; }
for f in $OUTS; do [ -e "$f" ] && { log "$f exists: not starting"; exit 1; }; done
if [ -e "$END" ]; then
  last=$(tail -n 1 "$END")
  case "$last" in
    "$NAME: refused"*|"$NAME: handoff"*|"$NAME: no 'e80n"*|"$NAME: CPU gate never"*|"$NAME: killed by signal"*)
      log "restart: no output exists and the last end line was: $last";;
    *) log "$END exists: not starting"; exit 1;;
  esac
fi
BR="refused"; SIG=""; CHILD=""; NAP=""
trap 'rc=$?; [ -n "$SIG" ] && BR="killed by signal $SIG"; if [ -n "$CHILD" ]; then kill -TERM -- "-$CHILD" 2> /dev/null; BR="$BR; its running job stopped"; fi
      echo "$NAME: $BR $(date -u +%FT%TZ) exit $rc" >> "$END"' EXIT
on_sig() { SIG="$1"; [ -n "$NAP" ] && kill "$NAP" 2> /dev/null; exit "$2"; }
trap 'on_sig TERM 143' TERM; trap 'on_sig HUP 129' HUP; trap 'on_sig INT 130' INT
nap() { sleep "$1" & NAP=$!; wait "$NAP"; NAP=""; }
job() { setsid -w "$@" 9>&- & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return "$rc"; }
[ -e "$NOTE.frozen.json" ] && [ "$(sha256sum "$NOTE.md" | cut -d' ' -f1)" = "$(jq -r .sha256 "$NOTE.frozen.json")" ] \
  || { BR="refused (no freeze record of the note's text)"; exit 1; }
pin() { grep -F "| $1 |" "$NOTE.md" | head -n 1 | cut -d'|' -f3 | tr -d ' '; }
[ -n "$(pin "$JOBS")" ] && [ "$(sha256sum "$JOBS" 2> /dev/null | cut -d' ' -f1)" = "$(pin "$JOBS")" ] || { BR="refused ($JOBS does not hash to the note's table)"; exit 1; }
while IFS= read -r row; do  # every repository row of the note's table
  first=$(cut -d'|' -f2 <<< "$row" | sed 's/^ *//; s/ *$//'); want=$(cut -d'|' -f3 <<< "$row" | tr -d ' ')
  case "$first" in *"(project notes"*|*"(finish-pass notes"*) continue;; esac
  f=$(sed 's/ (.*$//' <<< "$first")
  [ "$(sha256sum "$f" 2> /dev/null | cut -d' ' -f1)" = "$want" ] || { BR="refused ($f does not hash to the note's table)"; exit 1; }
done < <(grep -E '^\| [^|]+ \| [0-9a-f]{64} \|$' "$NOTE.md")
[ "$(sha256sum "$SELF" | cut -d' ' -f1)" = "$(pin "e80_chain.sh (project notes, not published)")" ] || { BR="refused (this chain does not hash to the note's table)"; exit 1; }
d=$($DIFF "$NOTE.frozen.json" 2> /dev/null) || { BR="refused (scripts/diff_frozen.py failed)"; exit 1; }
if grep -E -q "$PATHRE" <<< "$d"; then BR="refused (changed since the note's freeze:$(grep -E "$PATHRE" <<< "$d" | head -n 3 | tr '\n' ' ' | sed 's/^/ /'))"; exit 1; fi
for f in logs/e87n_a0.lock logs/e87n_bdry.lock logs/e87n_b.lock; do
  [ -e "$f" ] && ! flock -n "$f" true 2> /dev/null && { BR="refused (an e87 GPU stage is running or waiting: $f)"; exit 1; }
done
log "e80n $STAGE start; note frozen $(jq -r .frozen_utc "$NOTE.frozen.json"); pins checked"
wait_done() {  # wait_done STAGE LIMIT_S: until STAGE's end log ends "done"; refuse if it ended otherwise
  local f="logs/e80n_$1_end.log" t0; t0=$(date +%s); BR="no 'e80n $1 end: done' within $(( $2 / 60 )) min"
  until tail -n 1 "$f" 2> /dev/null | grep -q "^e80n $1 end: done "; do
    tail -n 1 "$f" 2> /dev/null | grep -q "^e80n $1 end: " && { BR="refused ($1 did not end done)"; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt "$2" ] || exit 2; nap "$POLL"
  done
}
gpu_empty() { local out; out=$($NVSMI 2> /dev/null) || return 1; [ -z "$(grep . <<< "$out")" ]; }
gpu_free() {
  local t1 k=0; t1=$(date +%s); BR="handoff: no 3 empty-GPU checks in a row within $(( GW / 60 )) min"
  while [ "$k" -lt 3 ]; do
    [ $(( $(date +%s) - t1 )) -lt "$GW" ] || exit 3
    if gpu_empty; then k=$((k + 1)); else k=0; fi
    [ "$k" -lt 3 ] && nap "$POLL"
  done
  log "handoff done: 3 empty-GPU checks"
}
gate_ok() { local m; m=$($LOAD 2> /dev/null) || return 1; awk '$2 == "cores" && $3 ~ /^[0-9.]+$/ {ok = ($3 + 0 <= 4.5)} END {exit !ok}' <<< "$m"; }
gate() {  # gate WHAT: the CPU gate, at most 4.5 cores over 10 s, within GATE_S
  local t1; t1=$(date +%s); BR="CPU gate never at or below 4.5 cores for $(( GS / 60 )) min ($1)"
  until gate_ok; do [ $(( $(date +%s) - t1 )) -lt "$GS" ] || exit 3; nap "$POLL"; done
}
case "$STAGE" in
  dry)
    gpu_free
    BR="dry queue running"; log "e80 dry queue"; job $Q "$JOBS" logs/e80_dry_queue.log 4.5
    gate "dry check"
    BR="dry check failed"; job $DC > logs/e80_dry_check.out 2>&1 || { tail -n 2 logs/e80_dry_check.out; exit 1; }
    [ "$(tail -n 1 logs/e80_dry_check.out)" = "all dry checks passed" ] || exit 1
    tail -n 2 logs/e80_dry_check.out; BR="done"; log "e80 dry done";;
  run)
    wait_done dry "$PW"
    [ "$(tail -n 1 logs/e80_dry_check.out 2> /dev/null)" = "all dry checks passed" ] || { BR="refused (the dry checks have not passed)"; exit 1; }
    gpu_free
    BR="queue running"; log "e80 queue"; job $Q "$JOBS" logs/e80_queue.log 4.5
    gate "check"
    BR="check failed"; log "e80 check"; job $C > logs/e80_check.out 2>&1 || { tail -n 2 logs/e80_check.out; exit 1; }
    BR="report failed"; job $RP > logs/e80_report.out 2>&1 || { tail -n 2 logs/e80_report.out; exit 1; }
    cat logs/e80_check.out logs/e80_report.out; BR="done"
    log "e80 done; logs/e80_check.out $(sha256sum logs/e80_check.out | cut -c1-8), logs/e80_report.out $(sha256sum logs/e80_report.out | cut -c1-8)";;
esac
