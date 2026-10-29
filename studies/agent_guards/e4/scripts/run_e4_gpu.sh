#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# E4 GPU run: tau2-bench v1.0.1 retail, the local Qwen3.5-9B Q8_0 as agent AND user simulator (and post-hoc judge),
# arms A0, A1, A2 (TM seeds 1-3), A3, A3t of REGISTRATION_E4.md revision 6.
#
# NOT RUN BY THE PREPARER against a real server. It needs the shared llama-server on 127.0.0.1 (started with
# llm/serve_qwen_agent.sh in a GPU slot of the queue). This script NEVER starts, stops or signals a server and runs
# everything it starts with CUDA_VISIBLE_DEVICES= (the tau2 client and the guard need no GPU).
# Before every stage: cpu_ok (the cgroup gate) and then check_pins.py, which compares the registration's literal
# sha256 values with the disk and, at every registered stage, the registration with the write-once freeze record
# runs/E4_frozen.json (exit 7 on a difference or before the freeze). Every e4_run.py / judge / repeat call compares
# the live server with pins/server_expected.json when it starts (/props, the process's command line, its build
# files, the GGUF) and again when it ends (pid, /props, argv; exit 2 on any difference).
#
# Usage:  CAMPAIGN=e4reg_<yyyymmdd> STAGE=<stage> bash run_e4_gpu.sh     (the exact sequence: the last comment below)
# Stages, in order (each one resumes: tau2 resumes a results file, the replay store answers stored requests):
#   smoke        E4-0, its OWN campaign SMOKE_<CAMPAIGN> and store, the first 10 train tasks of split_tasks.json,
#                extended (revision 4, section 11.2 S0) one train task at a time in split order while A0's label
#                candidates are fewer than 10, to at most 20 tasks: A0, then A1 (replays A0), STOCK (tau2's
#                llm_agent, replay only), replay-only re-runs of A0 and A1 on the final task set, the repeat check
#                (50 agent + 50 user requests), the judge's health, smoke_report.py (gates S0-S9 and the budget;
#                BLIND: no reward is printed). No slot here: the slot is set at the freeze (section 11.3), so the
#                smoke's decision reads PENDING until the labels and the slot are in (section 11.4). Refused once the
#                freeze record exists (exit 7; revision 4).
#   a0           A0 (Qwen alone; guard in monitor mode) on all 114 tasks; the registered store starts empty
#   a1           A1 (code gates E4-R1..E4-R10) on all 114 tasks; replays A0 until its first denial
#   rerun        rule 9: an arm with MORE than 3 infra/timeout ends gets those tasks re-run once from the store into
#                <arm>_rerun/ (run after a1 and again after a2a3; a folder that exists is never re-run)
#   train_guard  CPU only: rows = veto-clean executed writes of A0 + A1 on the 74 train tasks, deduplicated;
#                CoTM seeds 1-3, LR, tree; MILP proofs on every TM seed; the go-bar (gobar.py: G >= 8 of 74 and
#                G - L >= 4, out of fold, TM seed 1). gobar.json's "outcome": go | no-go (exit 0) | not testable
#                (fewer than MIN_CLASS rows of a class: train_guard.py exit 3; exit 0) | void (the MILP check, IC2(b),
#                failed: STOP, exit 5)
#   a2a3         only after GO: A2 seed 1, A3, A2 seed 2, A2 seed 3, A3t on the 40 test tasks (replay from A1), minus
#                the arms the frozen cut names (A3t first, then A2 seeds 2-3; never A0, A1, A2 seed 1, A3)
#   judge        post-hoc NL judge (local Qwen, self-judged, unvalidated) for every arm folder present
#   summary      CPU only: summarize_e4.py -> runs/$CAMPAIGN/summary.json (every bar in one table; readings;
#                integrity checks IC1-IC6, each scoped to the bars it touches)
# Exit codes: 0 done (a registered outcome); 1 usage or a refused hook; 2 the server (not healthy, not the pinned one,
# or changed during a call); 3 a client failed; 4 the CPU gate; 5 the MILP check (IC2(b)); 6 train_guard failed
# without a registered outcome; 7 a pin or the freeze record differs, no freeze yet, a smoke after the freeze, a
# stage other than judge and summary after an integrity stop of this campaign, or (revision 6) a run log that the
# integrity or attempt guard cannot read ([refused], never counted); 8 (revision 5) the operator's stop (TERM or INT
# to this script, or a client ended by TERM or INT), at a slot's end. What follows each: section 11.6.
# Every STOP line of run_e4_gpu.log carries a tag (revision 4): [identity_changed] and [pins_changed] are the
# integrity stops of section 9.1 (check_pins.stop_effects reads them); [restart_same_identity], [server], [client]
# and [refused] are not. After an integrity stop that voids a bar, only judge and summary run in that campaign.
# Revision 5 (section 11.6; round-4 review M1): [slot_end] (exit 8) is not a failure and is never counted: the stage
# resumes. [client] (exit 3; also any step that fails without a tagged stop: the exit trap tags it) and [train_guard]
# (exit 6) are failed attempts: a registered stage runs at most 3 times in all (check_pins.py --attempts); from then on
# its start logs 'SKIP [stage_failed]' and exits 0, so the loop goes on. Every attempt is logged ('attempt k of 3'),
# and every end with exit 0 ('DONE'). Revision 6 (sections 4.1, 11.6; round-5 review Major 1): the operator's stop and
# a failed client give the stopped call its end check AT THE STOP (stop_end_check), while the server that served it is
# still up; "identity_changed" there is [identity_changed] (exit 2). At a run's next start e4_run.py runs the full start
# check FIRST (a server that is not the pinned one is [server], exit 2, the record untouched) and only then the
# deferred check of a record still without its end (a kill -9 or a crash of this script).
# Test hooks (refused unless CAMPAIGN starts with DRYRUN_): PINS, REG, FROZEN, SMOKE_TASKS, BASE_TASKS, TEST_TASKS,
# MIN_CLASS, DRY_FORCE_GO, SMOKE_MIN_TASKS, SMOKE_MIN_CANDIDATES, SMOKE_MAX_TASKS, CPU_GATE.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
E4=${REPO_ROOT}/studies/agent_guards/e4
TB=${REPO_ROOT}/studies/agent_guards/taubench
PY="$TB/.venv/bin/python"
DLPY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
SPLIT_FILE=$TB/data_untrusted/tau2-bench/extracted/tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/retail/split_tasks.json
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=e4reg_<yyyymmdd> (the UTC date of the smoke)}"
STAGE="${STAGE:?set STAGE: smoke | a0 | a1 | rerun | train_guard | a2a3 | judge | summary}"
case "$STAGE" in smoke|a0|a1|rerun|train_guard|a2a3|judge|summary) ;; *) echo "unknown STAGE $STAGE"; exit 1 ;; esac
DRY=0
case "$CAMPAIGN" in DRYRUN_*) DRY=1 ;; SMOKE_*) echo "REFUSING: CAMPAIGN must not start with SMOKE_"; exit 1 ;; esac
if [ "$DRY" = 0 ]; then
  for v in PINS REG FROZEN SMOKE_TASKS BASE_TASKS TEST_TASKS MIN_CLASS DRY_FORCE_GO SMOKE_MIN_TASKS SMOKE_MIN_CANDIDATES SMOKE_MAX_TASKS CPU_GATE; do
    if [ -n "${!v:-}" ]; then echo "REFUSING: $v is a dry-run hook (CAMPAIGN must start with DRYRUN_)"; exit 1; fi
  done
  case "$CAMPAIGN" in e4reg_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]|e4reg_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_icr) ;;
    *) echo "REFUSING: CAMPAIGN must be e4reg_<yyyymmdd> (or its integrity re-run e4reg_<yyyymmdd>_icr)"; exit 1 ;; esac
fi
if [ -n "${SLOT_MINUTES:-}" ]; then
  echo "REFUSING: SLOT_MINUTES: the slot is set at the freeze (smoke_report.py --slot-minutes, section 11.4)"; exit 1
fi
PINS="${PINS:-$E4/pins/server_expected.json}"
FROZEN="${FROZEN:-$E4/runs/E4_frozen.json}"
if [ "$STAGE" = smoke ]; then
  REG="${REG:-$E4/REGISTRATION_E4.md}"
  case "$CAMPAIGN" in *_icr) echo "REFUSING: an integrity re-run has no smoke (section 9.1)"; exit 1 ;; esac
  # revision 4 (round-3 review m3): the smoke's files are H6 files; once the freeze record exists, no smoke runs
  [ -f "$FROZEN" ] && { echo "REFUSING: $FROZEN exists: the smoke runs only before the freeze (section 11.4)"; exit 7; }
elif [ "$DRY" = 0 ]; then
  [ -f "$FROZEN" ] || { echo "REFUSING: $FROZEN does not exist: registered stages run only after the freeze (11.4)"; exit 7; }
  REG="$("$PY" -I -B -c "import json, sys; print(json.load(open(sys.argv[1]))['registration'])" "$FROZEN")"
else
  REG="${REG:-$E4/REGISTRATION_E4.md}"
fi
CPU_GATE="${CPU_GATE:-3.5}"
awk -v c="$CPU_GATE" 'BEGIN{exit !(c > 0 && c <= 4.5)}' || { echo "REFUSING: CPU_GATE $CPU_GATE (0 < gate <= 4.5)"; exit 1; }
PORT="${PORT:-18082}"
BASE="http://127.0.0.1:${PORT}/v1"
SMOKE_TASKS="${SMOKE_TASKS:-0 1 2 3 4 6 7 8 10 11}"
SMOKE_MIN_TASKS="${SMOKE_MIN_TASKS:-10}"
SMOKE_MIN_CANDIDATES="${SMOKE_MIN_CANDIDATES:-10}"
SMOKE_MAX_TASKS="${SMOKE_MAX_TASKS:-20}"
MIN_CLASS="${MIN_CLASS:-10}"
RUNS="$E4/runs/$CAMPAIGN"
SCAMP="SMOKE_${CAMPAIGN%_icr}"
SRUNS="$E4/runs/$SCAMP"
mkdir -p "$RUNS" "$E4/.tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$E4/.tmp" LITELLM_LOCAL_MODEL_COST_MAP=True
STOPPED=0 ATTEMPTS_MAX=3 ATTEMPT="" CUR_PID="" CUR_RUN="" CALL_REC="" CALL_SINCE=""
say() {  # one line of run_e4_gpu.log; a STOP, REFUSING or SKIP line marks this end as tagged (on_exit)
  case "$*" in "STOP "*|REFUSING*|"SKIP "*) STOPPED=1 ;; esac
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$STAGE] $*" | tee -a "$RUNS/run_e4_gpu.log"
}

failnote() {  # revision 5 (section 11.6): the note of a failed attempt (exit 3 or 6) of a registered stage
  [ -n "$ATTEMPT" ] || return 0
  if [ "$ATTEMPT" -ge "$ATTEMPTS_MAX" ]; then
    printf '%s' " [failed attempt $ATTEMPT of $ATTEMPTS_MAX: final; this stage's next start logs SKIP [stage_failed] (section 11.6)]"
  else
    printf '%s' " [failed attempt $ATTEMPT of $ATTEMPTS_MAX: resumed whatever the cause, once it is logged (section 11.6)]"
  fi
}

slot_end() {  # revision 5 (sections 11.3, 11.6; round-4 review M1 (a)): the operator's stop, TERM or INT to this script
  # revision 6 (round-5 review Major 1 (a)): the stopped call's end check AT THE STOP, right after its client has ended
  trap '' TERM INT
  local run="${CUR_RUN:-none}"
  if [ -n "$CUR_PID" ]; then kill -TERM "$CUR_PID" 2>/dev/null || true; wait "$CUR_PID" 2>/dev/null || true; fi
  say "STOP [slot_end] run=$run: the operator stopped stage $STAGE (SIG$1): not a failure and never counted; the stage resumes in the next GPU slot; the stopped call's end is checked now, at the stop (sections 4.1, 11.3, 11.6)"
  if [ -n "$CUR_RUN" ]; then stop_end_check "$run"; fi
  exit 8
}

stop_end_check() {  # revision 6 (sections 4.1, 11.6; round-5 review Major 1 (a)): the stopped call's end check AT THE
  # STOP, while the server that served it is still up (the operator stops the client only, never the server). It reads
  # the call's own start record (CALL_REC: the run's manifest.json or the judge's <out>.start.json; '-' = none), when it
  # was written at or after the client's start (CALL_SINCE) and has no end yet, and writes the light end check into it
  # (server_identity_end with "at_stop": true; scripts/end_check_at_stop.py). "identity_changed" is an integrity stop for
  # that run (9.1): its own STOP line, exit 2. Any other record is left for the deferred check at the run's next start,
  # which comes after the full start check (e4_run.start_checks).
  local run="$1" out="" rc=0 kind rest
  { [ -n "${CALL_REC:-}" ] && [ "$CALL_REC" != "-" ]; } || return 0
  out="$(nice -n 19 "$PY" -I -B "$E4/scripts/end_check_at_stop.py" --record "$CALL_REC" --since "${CALL_SINCE:-}" \
    --base "$BASE" --expect-server "$PINS" 2>> "$RUNS/end_check_at_stop.stderr")" || rc=$?
  kind="${out%%$'\n'*}" rest="${out#*$'\n'}"
  case "$rc:$kind" in
    0:same|0:restart_same_identity)
      say "end check at the stop, run=$run: $kind (written into $CALL_REC as server_identity_end, at_stop; section 4.1)" ;;
    0:none)
      say "end check at the stop, run=$run: none ($CALL_REC is not this call's or already has its end; a record without its end gets the deferred check at the run's next start, after the full start check; section 4.1)" ;;
    0:identity_changed)
      say "STOP [identity_changed] run=$run: the end check at the stop found the server changed during the call: ${rest:0:300} (an integrity stop, section 9.1; see $CALL_REC)"
      exit 2 ;;
    *)
      say "end check at the stop, run=$run: could not run (exit $rc; see $RUNS/end_check_at_stop.stderr); the record keeps no end and gets the deferred check at the run's next start (section 4.1)" ;;
  esac
}

on_exit() {  # revision 5: every end is tagged: exit 0 'DONE'; a signal the operator's stop; any other untagged end [client]
  local rc=$?
  trap - EXIT
  if [ "$rc" = 0 ]; then say "DONE: stage $STAGE ended with exit 0"; exit 0; fi
  if [ "$STOPPED" = 0 ]; then
    case "$rc" in
      130|143) say "STOP [slot_end] run=${CUR_RUN:-none}: a signal ended stage $STAGE (exit $rc): the operator's stop, not a failure and never counted (section 11.6)"; exit 8 ;;
      *) say "STOP [client]: a step of stage $STAGE failed with exit $rc before any tagged stop$(failnote)"; exit 3 ;;
    esac
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'slot_end TERM' TERM
trap 'slot_end INT' INT

client() {  # run_name record command...: the client runs in the background, so that slot_end can stop it at once
  # (revision 5); record = the call's start record that the end check at the stop reads (revision 6: the run's
  # manifest.json or the judge's <out>.start.json; '-' when the client writes none); CALL_SINCE = the client's start
  local run="$1" rc=0
  CALL_REC="$2" CALL_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  shift 2
  "$@" &
  CUR_PID=$! CUR_RUN="$run"
  wait "$CUR_PID" || rc=$?
  CUR_PID="" CUR_RUN=""
  return "$rc"
}

cpu_ok() {   # every stage waits (max 20 min) while the cgroup uses more than CPU_GATE cores (default 3.5)
  local i L
  for i in $(seq 1 20); do
    L="$("$DLPY" -I -B ${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py 10)"; say "cpu: $L (gate $CPU_GATE)"
    awk -v c="$(echo "$L" | awk '{print $3}')" -v g="$CPU_GATE" 'BEGIN{exit !(c <= g)}' && return 0
    "$DLPY" -I -B -c "import time; time.sleep(50)"
  done
  say "STOP [cpu_gate]: cgroup still above $CPU_GATE cores"; exit 4
}

pins_ok() {  # the registration's literal hashes vs the disk (and, from a0 on, vs the freeze record), every stage
  local fz=() tag
  [ "$DRY" = 1 ] && fz=(--frozen "$FROZEN")
  if ! nice -n 19 "$PY" -I -B "$E4/scripts/check_pins.py" --registration "$REG" --stage "$STAGE" --campaign "$CAMPAIGN" \
    "${fz[@]}" --out "$RUNS/pins_check_${STAGE}.json" > "$RUNS/pins_check_${STAGE}.stdout" 2>&1; then
    # [pins_changed]: a pinned file or the registration differs from its literal or record (an integrity stop, 9.1)
    tag="$("$PY" -I -B -c "import json, sys; print('pins_changed' if json.load(open(sys.argv[1])).get('pins_changed') else 'refused')" "$RUNS/pins_check_${STAGE}.json" 2>/dev/null || echo refused)"
    say "STOP [$tag]: check_pins.py found a difference from the registration or its freeze record (see $RUNS/pins_check_${STAGE}.json)"; exit 7
  fi
  say "pins ok ($(tail -1 "$RUNS/pins_check_${STAGE}.stdout" | head -c 300))"
}

integrity_guard() {  # revision 4 (section 9.1): after an integrity stop that voids a bar, only judge and summary run
  local v
  case "$STAGE" in smoke|judge|summary) return 0 ;; esac
  v="$("$PY" -I -B "$E4/scripts/check_pins.py" --stop-effects "$RUNS/run_e4_gpu.log")" \
    || { say "STOP [refused]: the integrity guard could not read $RUNS/run_e4_gpu.log: nothing of stage $STAGE ran; not a failed attempt, never counted (revision 6; section 11.6)"; exit 7; }
  if [ "$v" != "[]" ]; then
    say "REFUSING: an integrity stop of $CAMPAIGN voids $v: only judge and summary run in this campaign; then the record (scripts/integrity_failure.py) and, when it requires it, the integrity re-run (section 9.1)"
    exit 7
  fi
}

attempts_guard() {  # revision 5 (section 11.6; round-4 review M1 (b)): at most 3 attempts (exit 3 or 6) per stage
  local n
  case "$STAGE" in smoke) return 0 ;; esac
  n="$("$PY" -I -B "$E4/scripts/check_pins.py" --attempts "$RUNS/run_e4_gpu.log" --stage "$STAGE")" \
    || { say "STOP [refused]: the failed attempts could not be read from $RUNS/run_e4_gpu.log: nothing of stage $STAGE ran; not a failed attempt, never counted (revision 6; section 11.6)"; exit 7; }
  case "$n" in ''|*[!0-9]*) say "STOP [refused]: check_pins.py --attempts gave '$n', not a count: nothing of stage $STAGE ran; never counted (revision 6; section 11.6)"; exit 7 ;; esac
  if [ "$n" -ge "$ATTEMPTS_MAX" ]; then
    say "SKIP [stage_failed]: stage $STAGE failed $n of $ATTEMPTS_MAX attempts (exit 3 or 6): its outcome is final and it does not run again; the loop goes on; every bar with an arm it left incomplete reads 'not shown (stage failed)' (section 11.6)"
    exit 0
  fi
  ATTEMPT=$((n + 1))
  say "attempt $ATTEMPT of $ATTEMPTS_MAX (failed attempts so far: $n; section 11.6)"
}

start_stage() { integrity_guard; attempts_guard; cpu_ok; pins_ok; }

healthy() {
  local h m
  h="$(curl -sf -m 5 --noproxy '*' "http://127.0.0.1:${PORT}/health" || true)"
  m="$(curl -sf -m 5 --noproxy '*' "$BASE/models" || true)"
  if ! printf '%s' "$h" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    say "REFUSING: server not healthy at http://127.0.0.1:${PORT}/health (answer: '${h:-none}'). llama-server must be started first; this script never starts it."
    exit 2
  fi
  if ! printf '%s' "$m" | grep -q '"id"'; then
    say "REFUSING: $BASE/models lists no model (answer: '${m:-none}')"; exit 2
  fi
  say "server healthy: $(printf '%s' "$m" | head -c 200)"
}

client_stop() {  # a client's exit code -> the stage's: 2 (the server) stays 2; 130/143 (a client ended by INT or TERM:
  # the operator's stop, revision 5) is 8; anything else is 3
  # the tag (section 11.6): exit 2 with "(identity_changed)" in the client's last stderr line = [identity_changed]
  # (an integrity stop, 9.1; also from e4_run.py's deferred end check at a run's next start), "(restart_same_identity)"
  # = [restart_same_identity], any other exit 2 = [server] (not healthy or not the pinned one when the call starts);
  # 130 or 143 = [slot_end] (never counted); exit 3 or other = [client] (a failed attempt)
  # revision 6 (section 4.1; round-5 review Major 1 (a)): after the operator's stop (130/143) and after a failed client
  # the stopped call gets its end check at the stop (stop_end_check); "identity_changed" there adds an [identity_changed]
  # STOP line and the stage exits 2 (an integrity stop, 9.1; the failed attempt still counts). Never after exit 2: a
  # server that is not the pinned one leaves the record untouched (the deferred check waits for the pinned server).
  local rc="$1" what="$2" run="$3" errf="${4:-}" last="" tag=client
  if [ -n "$errf" ] && [ -f "$errf" ]; then last="$(grep -v '^[[:space:]]*$' "$errf" | tail -n 1 | head -c 300 || true)"; fi
  if [ "$rc" = 2 ]; then
    case "$last" in *"(identity_changed)"*) tag=identity_changed ;; *"(restart_same_identity)"*) tag=restart_same_identity ;; *) tag=server ;; esac
  elif [ "$rc" = 130 ] || [ "$rc" = 143 ]; then
    say "STOP [slot_end] run=$run: $what was ended by a signal (exit $rc): the operator's stop, not a failure and never counted; the stage resumes (section 11.6)"
    stop_end_check "$run"
    exit 8
  fi
  say "STOP [$tag] run=$run: $what failed (exit $rc): $last$( [ "$rc" = 2 ] || failnote )"
  [ "$rc" = 2 ] && exit 2
  stop_end_check "$run"
  exit 3
}

run_arm() {  # campaign arm split run_name [extra args...]
  local c="$1" a="$2" s="$3" n="$4"; shift 4
  local d="$E4/runs/$c" rc=0
  mkdir -p "$d"
  healthy
  say "start arm $a split $s as $c/$n $*"
  client "$n" "$d/$n/manifest.json" nice -n 19 "$PY" -I -B "$E4/scripts/e4_run.py" --arm "$a" --split "$s" --base "$BASE" --campaign "$c" \
    --run-name "$n" --expect-server "$PINS" "$@" >> "$d/${n}.stdout" 2>> "$d/${n}.stderr" || rc=$?
  [ "$rc" = 0 ] || client_stop "$rc" "e4_run.py for $c/$n (see $d/${n}.stderr)" "$n" "$d/${n}.stderr"
  say "end arm $c/$n: $(tail -c 300 "$d/${n}.stdout" | tr '\n' ' ')"
}

rerun_arm() {  # rule 9 for one arm folder: more than 3 infra/timeout ends -> those tasks once more, from the store
  local dir="$1" a="$2" s="$3"; shift 3
  [ -f "$RUNS/$dir/results.json" ] || return 0
  local rr="$RUNS/${dir}_rerun" ids
  if [ -f "$rr/rerun_tasks.txt" ]; then   # revision 5: a re-run stopped before its end resumes on ITS task list (one re-run)
    ids="$(cat "$rr/rerun_tasks.txt")"
    # shellcheck disable=SC2086
    if "$PY" -I -B -c "import json, sys; d = json.load(open(sys.argv[1])); done = {str(x['task_id']) for x in d['simulations']}; sys.exit(0 if set(sys.argv[2:]) <= done else 1)" "$rr/results.json" $ids 2>/dev/null; then
      say "rerun: $dir already re-run once (never twice)"; return 0
    fi
    say "rerun: the one re-run of $dir (tasks $ids) ended before every task finished; resuming it on the same tasks"
    # shellcheck disable=SC2086
    run_arm "$CAMPAIGN" "$a" "$s" "${dir}_rerun" --task-ids $ids "$@"
    return 0
  fi
  [ -d "$rr" ] && { say "rerun: $dir already re-run once (never twice)"; return 0; }
  ids="$(nice -n 19 "$PY" -I -B -c "
import sys; sys.path.insert(0, '$E4')
from pathlib import Path
from e4guard.runs import load_arm, rerun_needed
print(' '.join(rerun_needed(load_arm(Path('$RUNS/$dir')))))")"
  if [ -z "$ids" ]; then say "rerun: $dir has at most 3 infra/timeout ends; no re-run"; return 0; fi
  say "rerun: $dir has more than 3 infra/timeout ends; re-running tasks $ids once from the store"
  mkdir -p "$rr" && printf '%s\n' "$ids" > "$rr/rerun_tasks.txt"
  # shellcheck disable=SC2086
  run_arm "$CAMPAIGN" "$a" "$s" "${dir}_rerun" --task-ids $ids "$@"
}

pick_tasks() {  # dry-run task overrides (empty for registered runs)
  local v="$1"; [ -n "${!v:-}" ] && echo "--task-ids ${!v}" || true
}

gobar_write() {  # STAGE=train_guard's outcomes that stop before gobar.py: gobar.json says so (summarize_e4.py reads it)
  "$PY" -I -B -c "import json, sys; json.dump(json.loads(sys.argv[2]), open(sys.argv[1], 'w'), indent=1)" \
    "$RUNS/gobar.json" "$1"
}

case "$STAGE" in
  smoke)
    start_stage
    mkdir -p "$SRUNS"
    ST="$SMOKE_TASKS"
    # shellcheck disable=SC2086
    run_arm "$SCAMP" A0 train A0_smoke --task-ids $ST --blind
    # the extension rule (revision 4; section 11.2, S0; round-3 review m6): while A0's label candidates (S8) are fewer
    # than SMOKE_MIN_CANDIDATES and the smoke holds fewer than SMOKE_MAX_TASKS tasks, the next train task in split
    # order is added and A0 runs on it (tau2 resumes the results file: only the new task runs). Counts only, no reward.
    while :; do
      nc="$(nice -n 19 "$PY" -I -B "$E4/scripts/smoke_report.py" --campaign "$SCAMP" --count-candidates)" \
        || { say "STOP [client]: smoke_report.py --count-candidates failed"; exit 3; }
      case "$nc" in ''|*[!0-9]*) say "STOP [client]: smoke_report.py --count-candidates gave '$nc', not a count"; exit 3 ;; esac
      nt="$(echo "$ST" | wc -w)"
      if [ "$nc" -ge "$SMOKE_MIN_CANDIDATES" ] || [ "$nt" -ge "$SMOKE_MAX_TASKS" ]; then
        say "smoke tasks: $nt ($ST); label candidates: $nc (S8 needs $SMOKE_MIN_CANDIDATES; at most $SMOKE_MAX_TASKS tasks)"
        break
      fi
      nx="$("$PY" -I -B "$E4/scripts/smoke_report.py" --campaign "$SCAMP" --next-task $ST)"
      [ -n "$nx" ] || { say "smoke tasks: no further train task in split order; $nt tasks, $nc candidates"; break; }
      say "smoke extension: $nc label candidates on $nt tasks (S8 needs $SMOKE_MIN_CANDIDATES); adding train task $nx"
      ST="$ST $nx"
      # shellcheck disable=SC2086
      run_arm "$SCAMP" A0 train A0_smoke --task-ids $ST --blind
    done
    # shellcheck disable=SC2086
    run_arm "$SCAMP" A1 train A1_smoke --task-ids $ST --blind
    # shellcheck disable=SC2086
    run_arm "$SCAMP" STOCK train STOCK_smoke --task-ids $ST --blind --store-mode replay_only
    # shellcheck disable=SC2086
    run_arm "$SCAMP" A0 train A0_smoke_replay --task-ids $ST --blind --store-mode replay_only
    # shellcheck disable=SC2086
    run_arm "$SCAMP" A1 train A1_smoke_replay --task-ids $ST --blind --store-mode replay_only
    healthy
    rc=0
    client repeat_check - nice -n 19 "$PY" -I -B "$E4/scripts/repeat_check.py" --campaign "$SCAMP" --base "$BASE" --expect-server "$PINS" \
      --n-agent 50 --n-user 50 --out "$SRUNS/repeat_check.json" >> "$SRUNS/smoke.stdout" 2>> "$SRUNS/smoke.stderr" || rc=$?
    [ "$rc" = 0 ] || client_stop "$rc" "repeat_check.py" repeat_check "$SRUNS/smoke.stderr"
    for a in A0 A1; do
      rc=0
      client "judge_health:${a}_smoke" "$SRUNS/judge_${a}_smoke.start.json" nice -n 19 "$PY" -I -B "$E4/scripts/judge_posthoc.py" --results "$SRUNS/${a}_smoke/results.json" --base "$BASE" \
        --campaign "$SCAMP" --expect-server "$PINS" --health-only --out "$SRUNS/judge_${a}_smoke.json" \
        >> "$SRUNS/smoke.stdout" 2>> "$SRUNS/smoke.stderr" || rc=$?
      [ "$rc" = 0 ] || client_stop "$rc" "judge health for $a" "judge_health:${a}_smoke" "$SRUNS/smoke.stderr"
    done
    SR_EXTRA=()
    [ "$DRY" = 1 ] && SR_EXTRA=(--min-tasks "$SMOKE_MIN_TASKS" --min-candidates "$SMOKE_MIN_CANDIDATES" --max-tasks "$SMOKE_MAX_TASKS")
    nice -n 19 "$PY" -I -B "$E4/scripts/smoke_report.py" --campaign "$SCAMP" "${SR_EXTRA[@]}" \
      --out "$SRUNS/smoke_report.json" > "$SRUNS/smoke_report.stdout" 2>> "$SRUNS/smoke.stderr" \
      || { say "STOP [client]: smoke_report.py failed"; exit 3; }
    say "smoke report (blind; no slot yet, so PENDING at best): $(head -c 600 "$SRUNS/smoke_report.stdout" | tr '\n' ' ')"
    ;;
  a0)
    start_stage
    [ -f "$RUNS/.a0_started" ] || touch "$RUNS/.a0_started"
    # shellcheck disable=SC2046
    run_arm "$CAMPAIGN" A0 base A0_base $(pick_tasks BASE_TASKS)
    ;;
  a1)
    start_stage
    # shellcheck disable=SC2046
    run_arm "$CAMPAIGN" A1 base A1_base $(pick_tasks BASE_TASKS)
    ;;
  rerun)
    start_stage
    rerun_arm A0_base A0 base
    rerun_arm A1_base A1 base
    for sd in 1 2 3; do
      [ -f "$RUNS/models/tm_seed${sd}.json" ] && rerun_arm "A2s${sd}_test" A2 test --learned "$RUNS/models/tm_seed${sd}.json"
    done
    [ -f "$RUNS/models/lr.json" ] && rerun_arm A3_test A3 test --learned "$RUNS/models/lr.json"
    [ -f "$RUNS/models/tree.json" ] && rerun_arm A3t_test A3t test --learned "$RUNS/models/tree.json"
    true
    ;;
  train_guard)
    start_stage
    say "rows: veto-clean executed writes of A0 + A1 on the 74 train tasks, deduplicated (A0 first)"
    nice -n 19 "$PY" -I -B "$E4/scripts/build_rows_from_results.py" --results "$RUNS/A0_base" "$RUNS/A1_base" \
      --sources qwen_A0 qwen_A1 --split train --split-file "$SPLIT_FILE" --out "$RUNS/rows_train.json" \
      --all-out "$RUNS/rows_train_all.json" > "$RUNS/rows_train.stdout"
    say "rows: $(head -c 400 "$RUNS/rows_train.stdout" | tr '\n' ' ')"
    nice -n 19 "$PY" -I -B "$E4/scripts/dump_call_bits.py" --results "$RUNS/A0_base" "$RUNS/A1_base" \
      --out "$RUNS/bits_allcalls.json" > "$RUNS/bits_allcalls.stdout"
    rc=0
    nice -n 19 "$DLPY" -I -B "$E4/scripts/train_guard.py" --rows "$RUNS/rows_train.json" --out-dir "$RUNS/models" \
      --tag "$CAMPAIGN" --seeds 1 2 3 --min-class "$MIN_CLASS" --oof-rows "$RUNS/rows_train_all.json" \
      --oof-sources qwen_A1 --oof-out "$RUNS/models/oof_rows.json" > "$RUNS/train_guard.stdout" \
      2> "$RUNS/train_guard.stderr" || rc=$?
    if [ "$rc" = 3 ]; then     # section 8: a registered outcome, not a failure; the loop goes on (E4.2 needs judge, summary)
      gobar_write "{\"GO\": false, \"outcome\": \"not testable\", \"train_guard_exit\": 3, \"power\": null, \"decision\": \"NOT TESTABLE: fewer than $MIN_CLASS veto-clean, deduplicated train rows of a class; the learned part is not trained; A2, A3, A3t are not run (section 8)\"}"
      say "E4.3 and E4.5 NOT TESTABLE (train_guard.py exit 3; see train_guard.stderr); A2, A3, A3t are not run; gobar.json written"
      exit 0
    elif [ "$rc" != 0 ]; then
      say "STOP [train_guard]: train_guard.py failed (exit $rc; see train_guard.stderr)$(failnote)"; exit 6
    fi
    for sd in 1 2 3; do
      rc=0
      nice -n 19 "$DLPY" -I -B "$E4/scripts/milp_proof.py" --model "$RUNS/models/tm_seed${sd}.json" \
        --rows "$RUNS/rows_train.json" "$RUNS/bits_allcalls.json" --out "$RUNS/milp_tm_seed${sd}.json" \
        > "$RUNS/milp_seed${sd}.stdout" 2>&1 || rc=$?
      if [ "$rc" != 0 ]; then
        gobar_write "{\"GO\": false, \"outcome\": \"void\", \"failed_check\": \"IC2b\", \"seed\": $sd, \"power\": null, \"decision\": \"STOP: IC2(b) failed (the MILP check, TM seed $sd: a guarded proof failed or the domain does not hold); E4.3 and E4.5 are void (section 9.1); E4.2 stands\"}"
        say "STOP [milp]: the MILP check failed for seed $sd (IC2(b)): E4.3 and E4.5 are void (section 9.1); gobar.json written"; exit 5
      fi
    done
    rc=0
    GB_EXTRA=()
    if [ "$DRY" = 1 ] && [ -n "${BASE_TASKS:-}" ]; then read -r -a bt <<< "$BASE_TASKS"; GB_EXTRA+=(--train-tasks "${bt[@]}"); fi
    nice -n 19 "$PY" -I -B "$E4/scripts/gobar.py" --oof "$RUNS/models/oof_rows.json" --a1 "$RUNS/A1_base" \
      --split-file "$SPLIT_FILE" --out "$RUNS/gobar.json" "${GB_EXTRA[@]}" > "$RUNS/gobar.stdout" 2>&1 || rc=$?
    if [ "$DRY" = 1 ] && [ "${DRY_FORCE_GO:-0}" = 1 ] && [ "$rc" = 10 ]; then   # test hook: exercise a2a3 and the bars
      "$PY" -I -B -c "import json; p='$RUNS/gobar.json'; d=json.load(open(p)); d.update(GO=True, outcome='go', forced='DRY RUN test hook (DRY_FORCE_GO)'); json.dump(d, open(p, 'w'), indent=1)"
      say "DRY RUN: the go-bar said NO-GO; forced to GO by the test hook (gobar.json marks it)"; rc=0
    fi
    case "$rc" in
      0) say "go-bar: GO $(head -c 300 "$RUNS/gobar.stdout" | tr '\n' ' ')" ;;
      10) say "go-bar: NO-GO: E4.3 and E4.5 are a dev-only line; A2, A3, A3t are not run on the test tasks" ;;
      *) say "STOP [train_guard]: gobar.py could not decide (exit $rc)$(failnote)"; exit 6 ;;
    esac
    say "train_guard done: $(tail -c 300 "$RUNS/train_guard.stdout" | tr '\n' ' ')"
    ;;
  a2a3)
    start_stage
    GO="$("$PY" -I -B -c "import json; d=json.load(open('$RUNS/gobar.json')); print(int(bool(d.get('GO')) and d.get('outcome', 'go') == 'go'))" 2>/dev/null || echo 0)"
    if [ "$GO" != 1 ]; then
      say "REFUSING: the go-bar did not say GO (runs/$CAMPAIGN/gobar.json); A2, A3, A3t are not run (a registered outcome)"; exit 0
    fi
    CUT="$("$PY" -I -B -c "import json, sys; print(' '.join(json.load(open(sys.argv[1])).get('cut') or []))" "$FROZEN")"
    say "the frozen cut: '${CUT:-none}' (runs/E4_frozen.json)"
    for f in tm_seed1.json tm_seed2.json tm_seed3.json lr.json tree.json; do
      [ -f "$RUNS/models/$f" ] || { say "STOP [client]: $RUNS/models/$f missing (run STAGE=train_guard)$(failnote)"; exit 3; }
    done
    # protected arms first (never cut): A2 seed 1, A3; then A2 seeds 2-3 and A3t unless the frozen cut names them
    # shellcheck disable=SC2046
    run_arm "$CAMPAIGN" A2 test A2s1_test --learned "$RUNS/models/tm_seed1.json" $(pick_tasks TEST_TASKS)
    # shellcheck disable=SC2046
    run_arm "$CAMPAIGN" A3 test A3_test --learned "$RUNS/models/lr.json" $(pick_tasks TEST_TASKS)
    for sd in 2 3; do
      case " $CUT " in *" A2s$sd "*) say "budget cut: A2 seed $sd not run" ;;
        *) run_arm "$CAMPAIGN" A2 test "A2s${sd}_test" --learned "$RUNS/models/tm_seed${sd}.json" $(pick_tasks TEST_TASKS) ;;
      esac
    done
    case " $CUT " in *" A3t "*) say "budget cut: A3t not run" ;;
      *) run_arm "$CAMPAIGN" A3t test A3t_test --learned "$RUNS/models/tree.json" $(pick_tasks TEST_TASKS) ;;
    esac
    ;;
  judge)
    start_stage
    for d in A0_base A1_base A2s1_test A2s2_test A2s3_test A3_test A3t_test A0_base_rerun A1_base_rerun \
             A2s1_test_rerun A2s2_test_rerun A2s3_test_rerun A3_test_rerun A3t_test_rerun; do
      [ -f "$RUNS/$d/results.json" ] || continue
      healthy
      rc=0
      client "judge:$d" "$RUNS/$d/judged.start.json" nice -n 19 "$PY" -I -B "$E4/scripts/judge_posthoc.py" --results "$RUNS/$d/results.json" --base "$BASE" \
        --campaign "$CAMPAIGN" --expect-server "$PINS" --out "$RUNS/$d/judged.json" >> "$RUNS/judge.stdout" \
        2>> "$RUNS/judge.stderr" || rc=$?
      [ "$rc" = 0 ] || client_stop "$rc" "the judge for $d" "judge:$d" "$RUNS/judge.stderr"
      say "judged $d (self-judged, unvalidated): $("$PY" -I -B -c "import json;d=json.load(open('$RUNS/$d/judged.json'));print({k:d[k] for k in ('n','judge_errors')})" 2>/dev/null || echo '?')"
    done
    ;;
  summary)
    start_stage
    EXTRA=(--registration "$REG")
    if [ "$DRY" = 1 ]; then
      EXTRA+=(--frozen "$FROZEN")
      [ -n "${BASE_TASKS:-}" ] && read -r -a bt <<< "$BASE_TASKS" && EXTRA+=(--base-tasks "${bt[@]}")
      [ -n "${TEST_TASKS:-}" ] && read -r -a tt <<< "$TEST_TASKS" && EXTRA+=(--test-tasks "${tt[@]}")
    fi                                     # the smoke tasks come from the freeze record (revision 4: 10 to 20 tasks)
    rc=0
    nice -n 19 "$PY" -I -B "$E4/scripts/summarize_e4.py" --campaign "$CAMPAIGN" --out "$RUNS/summary.json" \
      "${EXTRA[@]}" > "$RUNS/summary.stdout" 2> "$RUNS/summary.stderr" || rc=$?
    [ "$rc" = 7 ] && { say "STOP [pins_changed]: summarize_e4.py refused: the registration differs from its freeze record"; exit 7; }
    [ "$rc" = 0 ] || { say "STOP [client]: summarize_e4.py failed (exit $rc)$(failnote)"; exit 3; }
    say "summary written: $RUNS/summary.json"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

# THE REPORT STEP (registration revision 6, section 11.5: the same text, line for line; tests/test_report_step.py)
# cd studies/agent_guards/e4
# C=e4reg_<yyyymmdd>    # the one campaign name: e4reg_ and the UTC date of the smoke; its smoke is SMOKE_$C
# # 1. the smoke (E4-0), before the freeze and with no slot: its decision reads PENDING until step 2
# CAMPAIGN=$C STAGE=smoke bash scripts/run_e4_gpu.sh
# # 2. the freeze (section 11.4): the labels into runs/SMOKE_$C/confirmations_labels.json, the slot, then
# scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/smoke_report.py --campaign SMOKE_$C \
#     --slot-minutes <slot> --out runs/SMOKE_$C/smoke_report.json
# #    on GO: REGISTRATION_E4.md = this file with H6, the F table and the status line filled (section 11.4), then
# scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/freeze_e4.py --registration REGISTRATION_E4.md
# # 3. the registered stages, in order; the loop stops after a failed stage (a non-zero exit)
# for s in a0 a1 rerun train_guard a2a3 rerun judge summary; do
#   CAMPAIGN=$C STAGE=$s bash scripts/run_e4_gpu.sh || break
# done
# # 4. after a stop: section 11.6, by the exit code. After exit 2, 3, 4, 6 or 8 the cause is logged, then the loop of
# #    step 3 runs again with its list starting at the stopped stage (after exit 8: in the next GPU slot); a stage
# #    with 3 failed attempts (exit 3 or 6) logs SKIP [stage_failed] and exits 0, so the loop goes on. The operator
# #    stops a stage with kill -TERM <pid of run_e4_gpu.sh> or Ctrl-C, never the server (exit 8, never counted).
# #    After an integrity stop (section 9.1) only judge and summary run:
# for s in judge summary; do
#   CAMPAIGN=$C STAGE=$s bash scripts/run_e4_gpu.sh || break
# done
# #    a void bar in runs/$C/summary.json: the cause found and logged, then the record and its mechanical decision
# scripts/offline_env.sh ../taubench/.venv/bin/python -I -B scripts/integrity_failure.py --campaign $C \
#     --cause '<the logged cause>' --fix-changes <every file the logged fix changes; none is a valid answer>
# #    the summary of $C again (it shows the decision); when the record requires it, every stage of ${C}_icr
# CAMPAIGN=$C STAGE=summary bash scripts/run_e4_gpu.sh
# for s in a0 a1 rerun train_guard a2a3 rerun judge summary; do
#   CAMPAIGN=${C}_icr STAGE=$s bash scripts/run_e4_gpu.sh || break
# done
# # 5. the outputs: runs/$C/summary.json ("table": E4.2, E4.3 with its power, E4.5; "readings"; "integrity" with
# #    "checks", "stop_events" and "by_bar"; "gobar"; "arms"; "anomalies"); runs/$C/gobar.json;
# #    runs/$C/models/train_report.json (meta.cv.picks, meta.oof_at_picks); runs/SMOKE_$C/smoke_report.json;
# #    runs/E4_frozen.json; runs/$C/integrity_failure.json (only after an integrity failure); the pins checks
# #    runs/$C/pins_check_<stage>.json; the arm folders runs/$C/{A0_base,A1_base,A2s1_test,A2s2_test,A2s3_test,
# #    A3_test,A3t_test}[_rerun]/ with results.json, guard_log.jsonl, manifest.json, run_summary.json, judged.json
