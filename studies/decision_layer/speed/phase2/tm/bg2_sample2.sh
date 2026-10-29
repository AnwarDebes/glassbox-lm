#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# bg2_sample2.sh ROOT RUNNER_PID OUT_PREFIX LOG_GLOB LINE_REGEX
#   Session-2 successor of bg2_sample.sh (that file is kept unchanged as reviewed). BG2 (PROTOCOL 4.3 / A3): CPU use of
#   one GPU TM job's own python WHILE IT TRAINS. Reads /proc and the job log only; starts nothing on the GPU and signals
#   no process. Start it right after the job: logs not newer than the sampler's start are stale and ignored.
#   1. waits (every POLL s, default 2; at most WAIT_MAX s, default 900) until a log ROOT/logs/LOG_GLOB newer than the
#      sampler's start holds a line matching LINE_REGEX (grep -E). Give a glob that only this job's log can match
#      (README: '<JOB>_<IMPL>_2*.log'); check_job_log.py --sample-log confirms afterwards that the log was the job's own;
#   2. waits SETTLE s (default 20: CUDA start, upload, first steps);
#   3. takes N samples (default 3) of WINDOW s (default 10), GAP s apart (default 20). Each sample runs two tools side
#      by side over the same window, both under nice -n 19:
#      - PROTOCOL 4.3's own tool, tmcode/spin_check.py PID WINDOW, with PID = the runner's only child from
#        `pgrep -P RUNNER_PID` (no name pattern; exactly one child, whose comm starts with "python")
#        -> OUT_PREFIX.proto_K.txt;
#      - this folder's per-thread tool, spin_check.py --descendant-of RUNNER_PID (the only python in the runner's
#        subtree) -> OUT_PREFIX.spin_K.json; its pid must equal the pgrep child;
#   4. writes OUT_PREFIX.sample.json: per sample both tools' main-thread cores (plus the per-thread tool's all-thread and
#      process cores), the max of each over the samples, and the A3 checks: a3_protocol_tool_le_0.2,
#      a3_per_thread_tool_le_0.2 and a3_main_thread_le_0.2 (true only if both are true).
#   Both tools are pinned by SHA-256 (PROTO_TOOL / MINE_TOOL override the paths for the self-test; a file that does not
#   hash to its pin is refused).
#   Exit 0: all N samples taken by both tools on one pid (whatever the A3 value). 1: the runner ended first, the line
#   never came, pgrep -P did not give exactly one python child, a tool failed, or the pids differ. 2: usage / refused
#   (OUT_PREFIX files exist, a tool is not the pinned file). Every step is logged in OUT_PREFIX.sample.log (UTC).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
proto="${PROTO_TOOL:-${REPO_ROOT}/studies/decision_layer/speed/tmcode/spin_check.py}"
mine="${MINE_TOOL:-$here/spin_check.py}"
PROTO_SHA=74b42cbd05ed326bbd773e3205b794ba3505f03b767b1717aa4b5709444dd867
MINE_SHA=ff281a079d5fc30f35b46ec6dc9a589a775709729103783f91419e5ee47a4e36
usage="usage: bg2_sample2.sh ROOT RUNNER_PID OUT_PREFIX LOG_GLOB LINE_REGEX"
root="${1:?$usage}"; runner="${2:?$usage}"; pre="${3:?$usage}"; glob="${4:?$usage}"; rx="${5:?$usage}"
POLL="${POLL:-2}"; WAIT_MAX="${WAIT_MAX:-900}"; SETTLE="${SETTLE:-20}"; N="${N:-3}"; WINDOW="${WINDOW:-10}"; GAP="${GAP:-20}"
log="$pre.sample.log"; mark="$pre.sample.start"
for f in "$log" "$mark" "$pre.sample.json"; do
  [ -e "$f" ] && { echo "refused: $f exists" >&2; exit 2; }
done
[ "$(sha256sum "$proto" 2> /dev/null | cut -c1-64)" = "$PROTO_SHA" ] || { echo "refused: $proto is not the pinned protocol tool" >&2; exit 2; }
[ "$(sha256sum "$mine" 2> /dev/null | cut -c1-64)" = "$MINE_SHA" ] || { echo "refused: $mine is not the pinned per-thread tool" >&2; exit 2; }
: > "$mark"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
alive() { [ -d "/proc/$runner" ]; }
say "start: root $root runner $runner glob $glob line /$rx/ poll $POLL wait_max $WAIT_MAX settle $SETTLE n $N window $WINDOW gap $GAP"
say "tools: protocol $proto sha256 $PROTO_SHA; per-thread $mine sha256 $MINE_SHA; started with nice -n 19 (niceness of such a child: $(nice -n 19 nice))"
found=""; waited=0
while :; do
  while IFS= read -r f; do
    if grep -qE -- "$rx" "$f"; then found="$f"; break; fi
  done < <(find "$root/logs" -maxdepth 1 -name "$glob" -newer "$mark" 2> /dev/null)
  [ -n "$found" ] && break
  alive || { say "runner $runner ended before the line came"; exit 1; }
  [ "$waited" -ge "$WAIT_MAX" ] && { say "the line never came within $WAIT_MAX s"; exit 1; }
  sleep "$POLL"; waited=$((waited + POLL))
done
say "line found in $found after about $waited s; settling $SETTLE s"
sleep "$SETTLE"
for k in $(seq 1 "$N"); do
  alive || { say "runner $runner ended before sample $k"; exit 1; }
  kids=$(pgrep -P "$runner" | tr '\n' ' ')
  read -r -a kid_list <<< "$kids"
  if [ "${#kid_list[@]}" -ne 1 ]; then
    say "sample $k: pgrep -P $runner gave ${#kid_list[@]} children ($kids): not exactly one"; exit 1
  fi
  child="${kid_list[0]}"
  comm=$(cat "/proc/$child/comm" 2> /dev/null)
  case "$comm" in
    python*) ;;
    *) say "sample $k: the runner's only child $child is '$comm', not a python"; exit 1 ;;
  esac
  nice -n 19 "$py" -I -B "$proto" "$child" "$WINDOW" > "$pre.proto_$k.txt" 2>> "$log" &
  proto_job=$!
  nice -n 19 "$py" -I -B "$mine" --descendant-of "$runner" --seconds "$WINDOW" --out "$pre.spin_$k.json" > /dev/null 2>> "$log"
  rc=$?
  wait "$proto_job"
  prc=$?
  say "sample $k: pgrep -P child $child ($comm); protocol tool exit $prc; per-thread tool exit $rc"
  if [ "$rc" -ne 0 ] || [ "$prc" -ne 0 ]; then exit 1; fi
  [ "$k" -lt "$N" ] && sleep "$GAP"
done
"$py" -I -B - "$pre" "$N" "$proto" "$mine" >> "$log" 2>&1 <<'EOF'
import json, re, sys
pre, n, proto, mine = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
rx = re.compile(r"^(\S+) pid (\d+) main-thread CPU ([0-9.]+) cores over ([0-9.]+) s; voluntary ctx switches \+(\d+); state (\S+);")
rows, parsed = [], True
for k in range(1, n + 1):
    s = json.load(open(f"{pre}.spin_{k}.json"))
    t = open(f"{pre}.proto_{k}.txt").read().strip()
    m = rx.match(t)
    parsed = parsed and m is not None
    pc = float(m[3]) if m else None
    rows.append({"sample": k, "protocol_tool": {"utc_end": m[1] if m else None, "pid": int(m[2]) if m else None,
                                                "main_thread_cores": pc, "seconds": float(m[4]) if m else None,
                                                "vol_switches": int(m[5]) if m else None, "state": m[6] if m else None,
                                                "line": t[:300]},
                 "per_thread_tool": {"utc_start": s["utc_start"], "pid": s["pid"], "main_thread_cores": s["main_thread_cores"],
                                     "all_threads_cores": s["all_threads_cores"], "process_cores": s["process_cores"],
                                     "n_threads": s["n_threads_end"]},
                 "abs_difference": None if pc is None or s["main_thread_cores"] is None else round(abs(pc - s["main_thread_cores"]), 4)})
pids = {r["per_thread_tool"]["pid"] for r in rows}
agree = all(r["protocol_tool"]["pid"] == r["per_thread_tool"]["pid"] for r in rows)
mp = max((r["protocol_tool"]["main_thread_cores"] for r in rows), default=None) if parsed else None
mt = max(r["per_thread_tool"]["main_thread_cores"] for r in rows)
out = {"samples": rows, "protocol_tool": proto, "per_thread_tool": mine,
       "max_main_thread_cores_protocol_tool": mp, "max_main_thread_cores_per_thread_tool": mt,
       "a3_protocol_tool_le_0.2": mp is not None and mp <= 0.2, "a3_per_thread_tool_le_0.2": mt <= 0.2,
       "a3_main_thread_le_0.2": mp is not None and mp <= 0.2 and mt <= 0.2,
       "pids_agree": agree, "same_pid_in_all_samples": len(pids) == 1, "protocol_lines_parsed": parsed,
       "max_abs_difference": max((r["abs_difference"] for r in rows if r["abs_difference"] is not None), default=None)}
open(f"{pre}.sample.json", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
sys.exit(0 if parsed and agree and len(pids) == 1 else 1)
EOF
src=$?
[ "$src" -eq 0 ] || { say "summary: the samples are not consistent (exit $src): see $pre.sample.json"; exit 1; }
say "done: $pre.sample.json"
exit 0
