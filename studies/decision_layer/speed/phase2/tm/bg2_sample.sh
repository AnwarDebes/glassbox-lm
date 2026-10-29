#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# bg2_sample.sh ROOT RUNNER_PID OUT_PREFIX LOG_GLOB LINE_REGEX
#   BG2 (PROTOCOL 4.3 / A3): CPU use of one GPU TM job's own python WHILE IT TRAINS. Reads /proc and the job log only;
#   starts nothing on the GPU and signals no process. Start it right after the job (the job's log must be written
#   after this sampler starts: logs not newer than the sampler's start are stale, from an earlier run, and ignored).
#   1. waits (every POLL s, default 2; at most WAIT_MAX s, default 900) until a log ROOT/logs/LOG_GLOB newer than the
#      sampler's start holds a line matching LINE_REGEX (grep -E). Patch jobs print "speed_patch <CONFIG>: make_tm"
#      just before the TM is built; an orig job prints "^view [0-9]+ bits" just before make_tm;
#   2. waits SETTLE s (default 20: CUDA start, upload, first steps);
#   3. takes N samples (default 3) of WINDOW s (default 10), GAP s apart (default 20), of the only python in
#      RUNNER_PID's subtree (spin_check.py --descendant-of RUNNER_PID) -> OUT_PREFIX.spin_K.json;
#   4. writes OUT_PREFIX.sample.json: every sample's main-thread, all-thread and process cores, the max main-thread
#      value and the A3 check (max main-thread cores <= 0.2).
#   Exit 0 all N samples taken (whatever the A3 value); 1 the runner ended first, the line never came, a sample
#   failed; 2 usage / refused (OUT_PREFIX files exist). Every step is logged in OUT_PREFIX.sample.log (UTC).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
usage="usage: bg2_sample.sh ROOT RUNNER_PID OUT_PREFIX LOG_GLOB LINE_REGEX"
root="${1:?$usage}"; runner="${2:?$usage}"; pre="${3:?$usage}"; glob="${4:?$usage}"; rx="${5:?$usage}"
POLL="${POLL:-2}"; WAIT_MAX="${WAIT_MAX:-900}"; SETTLE="${SETTLE:-20}"; N="${N:-3}"; WINDOW="${WINDOW:-10}"; GAP="${GAP:-20}"
log="$pre.sample.log"; mark="$pre.sample.start"
for f in "$log" "$mark" "$pre.sample.json"; do
  [ -e "$f" ] && { echo "refused: $f exists" >&2; exit 2; }
done
: > "$mark"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
alive() { [ -d "/proc/$runner" ]; }
say "start: root $root runner $runner glob $glob line /$rx/ poll $POLL wait_max $WAIT_MAX settle $SETTLE n $N window $WINDOW gap $GAP"
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
  "$py" -I -B "$here/spin_check.py" --descendant-of "$runner" --seconds "$WINDOW" --out "$pre.spin_$k.json" > /dev/null 2>> "$log"
  rc=$?
  say "sample $k: spin_check exit $rc"
  [ "$rc" -eq 0 ] || exit 1
  [ "$k" -lt "$N" ] && sleep "$GAP"
done
"$py" -I -B - "$pre" "$N" >> "$log" 2>&1 <<'EOF'
import json, sys
pre, n = sys.argv[1], int(sys.argv[2])
s = [json.load(open(f"{pre}.spin_{k}.json")) for k in range(1, n + 1)]
rows = [{"sample": k + 1, "utc_start": x["utc_start"], "pid": x["pid"], "main_thread_cores": x["main_thread_cores"],
         "all_threads_cores": x["all_threads_cores"], "process_cores": x["process_cores"], "n_threads": x["n_threads_end"]}
        for k, x in enumerate(s)]
mx = max(r["main_thread_cores"] for r in rows)
out = {"samples": rows, "max_main_thread_cores": mx, "a3_main_thread_le_0.2": mx <= 0.2,
       "same_pid_in_all_samples": len({r["pid"] for r in rows}) == 1}
open(f"{pre}.sample.json", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
EOF
say "done: $pre.sample.json"
exit 0
