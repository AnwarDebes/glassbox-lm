#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# e87 after the stop of 2026-10-23 (dated note logs/plan_e87_note_2026-10-25.md), one stage per start, by hand:
#   e87_chain.sh a0      GPU: 3 empty-GPU checks in a row, then scripts/e87_bits.py for abcd, star, multiwoz under the
#                        GPU lock (gpu_run.sh shared) into logs/e87_bits.out (stops at the first failure); exits.
#   e87_chain.sh a       CPU: waits for "e87n a0 end: done"; per dataset the CPU gate (at most 4.5 cores over 10 s),
#                        then scripts/e87_readability.py into logs/e87_readability.out. Long: see the note. With
#                        A_PARALLEL=1 it first waits for "e87n b end: done", then starts each dataset after its own
#                        gate without waiting for the others, into logs/e87_readability_<ds>.out (the same commands,
#                        so the same results; less wall time).
#   e87_chain.sh bdry    GPU: waits for "e87n a0 end: done"; 3 empty-GPU checks; logs/jobs_e87b_dry.txt through
#                        scripts/seq_queue.sh; then the CPU gate and scripts/e87b_check.py --dry into
#                        logs/e87b_dry_check.out, which must end "all dry checks passed".
#   e87_chain.sh b       GPU: waits for "e87n bdry end: done"; 3 empty-GPU checks; logs/jobs_e87b.txt (9 jobs) through
#                        scripts/seq_queue.sh; then the CPU gate and scripts/e87b_check.py --bar-only into
#                        logs/e87b_bar.out.
#   e87_chain.sh bkill   CPU: waits for "e87n b end: done" and "e87n a end: done"; the CPU gate; then
#                        scripts/e87b_check.py (the bar and the kill rule) into logs/e87b_check.out.
# Every stage:
# - holds a lock (logs/e87n_<stage>.lock), so a second chain of the same stage cannot start while one runs;
# - refuses without the note's freeze record (the SHA-256 of the note's text equals the record's), and unless the files
#   it runs equal the note's pins: every repository row of the note's hash table (sha256sum of the path in the first
#   column, a trailing parenthesis removed; rows marked "project notes" or "finish-pass notes" are outside the
#   repository), this script against its own row, the stage's job file present in the table, and no file on e87's path
#   (src/dl/, scripts/e87*, abcd_explore, table_run, seq_queue, cpu_load, diff_frozen) changed since the note's freeze
#   record (scripts/diff_frozen.py);
# - a GPU stage (a0, bdry, b) refuses while an e80 chain stage holds its lock (logs/e80n_dry.lock, logs/e80n_run.lock):
#   the shared GPU lock allows overlap, so the GPU stages of the two chains must not run or wait at the same time;
# - refuses a second start: its end log or an output exists. The one exception: a stage that ended before writing any
#   output (refused, handoff, a wait or gate timeout, killed while waiting) may start again; its end log keeps the old
#   line and gets a new one.
# End line "e87n <stage> end: <branch> <UTC> exit <code>" in logs/e87n_<stage>_end.log (new names: the stopped chain's
# logs/e87_a0_end.log and logs/e87_a_end.log stay as they are). A TERM, HUP or INT writes "killed by signal <name>"
# there; any end stops the stage's running commands (their own process groups, setsid). The GPU and CPU checks fail
# closed: an nvidia-smi or cpu_load.py error counts as busy. Start a stage detached, for example:
#   setsid nohup bash e87_chain.sh a0 > e87_chain_a0_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
S="${E87_ROOT:-${REPO_ROOT}/studies/decision_layer}"; cd "$S" || exit 1
NVSMI="${NVSMI_CMD:-nvidia-smi --query-compute-apps=pid --format=csv,noheader}"; Q="${QUEUE_CMD:-bash scripts/seq_queue.sh}"
A0="${A0_CMD:-../tm_language_model/scripts/gpu_run.sh shared -- env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/e87_bits.py --dataset}"
A="${A_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python scripts/e87_readability.py --dataset}"
C="${CHECK_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python scripts/e87b_check.py}"
LOAD="${LOAD_CMD:-env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 .venv-dl/bin/python -I scripts/cpu_load.py 10}"
DIFF="${DIFF_CMD:-env CUDA_VISIBLE_DEVICES= nice -n 19 .venv-dl/bin/python -I scripts/diff_frozen.py}"
POLL="${POLL_S:-20}"; GW="${GPU_WAIT_S:-21600}"; PW="${PRE_WAIT_S:-43200}"; AW="${A_WAIT_S:-1814400}"; GS="${GATE_S:-7200}"
DSETS="${DATASETS:-abcd star multiwoz}"; NOTE=logs/plan_e87_note_2026-10-25; STAGE="${1:-}"
PATHRE='(src/dl/|scripts/(e87|abcd_explore|table_run|seq_queue|cpu_load|diff_frozen))'
log() { echo "$(date -u +%FT%TZ) $*"; }
case "$STAGE" in
  a0) OUTS="logs/e87_bits.out $(for d in $DSETS; do printf 'results/explore/e87/%s_c4.npz ' "$d"; done)"; JOBS="";;
  a) OUTS="logs/e87_readability.out $(for d in $DSETS; do printf 'logs/e87_readability_%s.out ' "$d"; done)"; JOBS="";;
  bdry) OUTS="logs/e87b_dry_queue.log logs/e87b_dry_check.out"; JOBS="logs/jobs_e87b_dry.txt";;
  b) OUTS="logs/e87b_queue.log logs/e87b_bar.out"; JOBS="logs/jobs_e87b.txt";;
  bkill) OUTS="logs/e87b_check.out"; JOBS="";;
  *) echo "usage: e87_chain.sh a0|a|bdry|b|bkill"; exit 1;;
esac
END="logs/e87n_${STAGE}_end.log"; NAME="e87n $STAGE end"
exec 9> "logs/e87n_$STAGE.lock"
flock -n 9 || { log "another e87n $STAGE chain is running: not starting"; exit 1; }
for f in $OUTS; do [ -e "$f" ] && { log "$f exists: not starting"; exit 1; }; done
if [ -e "$END" ]; then
  last=$(tail -n 1 "$END")
  case "$last" in
    "$NAME: refused"*|"$NAME: handoff"*|"$NAME: no 'e87n"*|"$NAME: CPU gate never"*|"$NAME: killed by signal"*)
      log "restart: no output exists and the last end line was: $last";;
    *) log "$END exists: not starting"; exit 1;;
  esac
fi
BR="refused"; SIG=""; CHILD=""; NAP=""; CHILDREN=""
stop_jobs() { local c; for c in $CHILD $CHILDREN; do kill -TERM -- "-$c" 2> /dev/null; done; }
trap 'rc=$?; [ -n "$SIG" ] && BR="killed by signal $SIG"; if [ -n "$CHILD$CHILDREN" ]; then stop_jobs; BR="$BR; its running job(s) stopped"; fi
      echo "$NAME: $BR $(date -u +%FT%TZ) exit $rc" >> "$END"' EXIT
on_sig() { SIG="$1"; [ -n "$NAP" ] && kill "$NAP" 2> /dev/null; exit "$2"; }
trap 'on_sig TERM 143' TERM; trap 'on_sig HUP 129' HUP; trap 'on_sig INT 130' INT
nap() { sleep "$1" & NAP=$!; wait "$NAP"; NAP=""; }
job() { setsid -w "$@" 9>&- & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return "$rc"; }
[ -e "$NOTE.frozen.json" ] && [ "$(sha256sum "$NOTE.md" | cut -d' ' -f1)" = "$(jq -r .sha256 "$NOTE.frozen.json")" ] \
  || { BR="refused (no freeze record of the note's text)"; exit 1; }
pin() { grep -F "| $1 |" "$NOTE.md" | head -n 1 | cut -d'|' -f3 | tr -d ' '; }
for f in $JOBS; do
  [ -n "$(pin "$f")" ] && [ "$(sha256sum "$f" 2> /dev/null | cut -d' ' -f1)" = "$(pin "$f")" ] || { BR="refused ($f does not hash to the note's table)"; exit 1; }
done
while IFS= read -r row; do  # every repository row of the note's table
  first=$(cut -d'|' -f2 <<< "$row" | sed 's/^ *//; s/ *$//'); want=$(cut -d'|' -f3 <<< "$row" | tr -d ' ')
  case "$first" in *"(project notes"*|*"(finish-pass notes"*) continue;; esac
  f=$(sed 's/ (.*$//' <<< "$first")
  [ "$(sha256sum "$f" 2> /dev/null | cut -d' ' -f1)" = "$want" ] || { BR="refused ($f does not hash to the note's table)"; exit 1; }
done < <(grep -E '^\| [^|]+ \| [0-9a-f]{64} \|$' "$NOTE.md")
[ "$(sha256sum "$SELF" | cut -d' ' -f1)" = "$(pin "e87_chain.sh (project notes, not published)")" ] || { BR="refused (this chain does not hash to the note's table)"; exit 1; }
d=$($DIFF "$NOTE.frozen.json" 2> /dev/null) || { BR="refused (scripts/diff_frozen.py failed)"; exit 1; }
if grep -E -q "$PATHRE" <<< "$d"; then BR="refused (changed since the note's freeze:$(grep -E "$PATHRE" <<< "$d" | head -n 3 | tr '\n' ' ' | sed 's/^/ /'))"; exit 1; fi
case "$STAGE" in a0|bdry|b)
  for f in logs/e80n_dry.lock logs/e80n_run.lock; do
    [ -e "$f" ] && ! flock -n "$f" true 2> /dev/null && { BR="refused (an e80 chain stage is running or waiting: $f)"; exit 1; }
  done;;
esac
log "e87n $STAGE start; note frozen $(jq -r .frozen_utc "$NOTE.frozen.json"); pins checked"
wait_done() {  # wait_done STAGE LIMIT_S: until STAGE's end log ends "done"; refuse if it ended otherwise
  local f="logs/e87n_$1_end.log" t0; t0=$(date +%s); BR="no 'e87n $1 end: done' within $(( $2 / 60 )) min"
  until tail -n 1 "$f" 2> /dev/null | grep -q "^e87n $1 end: done "; do
    tail -n 1 "$f" 2> /dev/null | grep -q "^e87n $1 end: " && { BR="refused ($1 did not end done)"; exit 1; }
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
gate() {  # gate WHAT [LIMIT_S]: the CPU gate, at most 4.5 cores over 10 s
  local t1 lim="${2:-$GS}"; t1=$(date +%s); BR="CPU gate never at or below 4.5 cores for $(( lim / 60 )) min ($1)"
  until gate_ok; do [ $(( $(date +%s) - t1 )) -lt "$lim" ] || exit 3; nap "$POLL"; done
}
case "$STAGE" in
  a0)
    gpu_free
    for ds in $DSETS; do
      BR="A0 running ($ds)"; log "A0 $ds"
      job $A0 "$ds" >> logs/e87_bits.out 2>&1 || { BR="A0 failed ($ds)"; tail -n 2 logs/e87_bits.out; exit 1; }
    done
    BR="done"; log "A0 done; the GPU is free";;
  a)
    wait_done a0 "$PW"
    if [ "${A_PARALLEL:-0}" = 1 ]; then
      wait_done b "$AW"
      declare -A PID
      for ds in $DSETS; do
        gate "$ds" "$AW"; log "A $ds (parallel)"
        setsid -w $A "$ds" >> "logs/e87_readability_$ds.out" 2>&1 9>&- & PID[$ds]=$!; CHILDREN="$CHILDREN $!"
        BR="A running (parallel:$(for d in "${!PID[@]}"; do printf ' %s' "$d"; done))"
      done
      bad=""; for ds in $DSETS; do wait "${PID[$ds]}" || bad="$bad${bad:+, }$ds"; done; CHILDREN=""
      [ -z "$bad" ] || { BR="A failed ($bad)"; for ds in $DSETS; do tail -n 1 "logs/e87_readability_$ds.out"; done; exit 1; }
    else
      for ds in $DSETS; do
        gate "$ds"; BR="A running ($ds)"; log "A $ds"
        job $A "$ds" >> logs/e87_readability.out 2>&1 || { BR="A failed ($ds)"; tail -n 2 logs/e87_readability.out; exit 1; }
      done
    fi
    BR="done"; log "A done";;
  bdry)
    wait_done a0 "$PW"; gpu_free
    BR="dry run running"; log "e87b dry run"; job $Q logs/jobs_e87b_dry.txt logs/e87b_dry_queue.log 4.5
    gate "dry check"
    BR="dry check failed"; job $C --dry > logs/e87b_dry_check.out 2>&1 || { tail -n 2 logs/e87b_dry_check.out; exit 1; }
    [ "$(tail -n 1 logs/e87b_dry_check.out)" = "all dry checks passed" ] || exit 1
    tail -n 2 logs/e87b_dry_check.out; BR="done"; log "e87b dry done";;
  b)
    wait_done bdry "$PW"
    [ "$(tail -n 1 logs/e87b_dry_check.out 2> /dev/null)" = "all dry checks passed" ] || { BR="refused (the dry checks have not passed)"; exit 1; }
    gpu_free
    BR="queue running"; log "e87b queue"; job $Q logs/jobs_e87b.txt logs/e87b_queue.log 4.5
    gate "bar check"
    BR="bar check failed"; job $C --bar-only > logs/e87b_bar.out 2>&1 || { tail -n 2 logs/e87b_bar.out; exit 1; }
    cat logs/e87b_bar.out; BR="done"; log "e87b jobs and bar done; logs/e87b_bar.out $(sha256sum logs/e87b_bar.out | cut -c1-8)";;
  bkill)
    wait_done b "$AW"; wait_done a "$AW"; gate "check"
    BR="check failed"; log "e87b check"; job $C > logs/e87b_check.out 2>&1 || { tail -n 2 logs/e87b_check.out; exit 1; }
    cat logs/e87b_check.out; BR="done"; log "e87b check done; logs/e87b_check.out $(sha256sum logs/e87b_check.out | cut -c1-8)";;
esac
