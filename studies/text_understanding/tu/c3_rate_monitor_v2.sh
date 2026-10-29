#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# 3c throughput monitor v2, for the restart after the 04:32Z order (an ops script, read-only: it signals
# nothing and stops nothing). v1 (c3_rate_monitor.sh, kept unchanged) measures its projection from the FIRST server
# start (01:21:41Z), as the 01:28Z sharing flags asked; after a restart that anchor would flag at once.
# v2 differs from v1 only in:
# - the anchor: the restart's server start, i.e. the first 'server health' line of runs_c3.log written at or after
#   10 min before this monitor started; or the ANCHOR argument (an ISO time), for a monitor restarted later;
# - any second GPU compute process is a FLAG at once (Qwen and TM GPU sharing is OFF since 01:45Z);
# - an unparsable (half-written) line of gen.jsonl is skipped;
# - its own files: rate_monitor_v2.log and rate_baseline_v2.txt.
# Every WINDOW s it logs the rows, requests per minute, new tokens per second (the sets' answers differ in length, so
# tokens/s is the speed measure), the GPU compute processes, cpu_load.py over 10 s, and the projected end: the
# remaining rows' tokens (each set's mean answer length so far, or the pilot's mean x 1.04 below 200 rows) divided by
# the window's tokens/s. It EXITS 3 with a FLAG line on another GPU process or on a projected end later than the
# anchor + LIMIT_H - 0.25 h, exits 4 on any throttled period (with C3MON_THROTTLE=log a throttled period is only
# logged: the projection flag still catches a slowdown that threatens the limit), and exits 0 when gen.jsonl has all rows.
# Usage: c3_rate_monitor_v2.sh LIMIT_H [ANCHOR]
# Test hooks: C3MON_DIR (gen.jsonl, runs_c3.log, pilot/gen_pilot.jsonl), C3MON_OUT (its log and baseline),
# C3MON_WINDOW (default 600 s), C3MON_SETS (set:rows,...), C3MON_NVSMI, C3MON_LOAD (a command printing a cpu_load.py line);
# and the option C3MON_THROTTLE (exit, the default, or log).
T=${REPO_ROOT}/studies/text_understanding/tu
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
DIR=${C3MON_DIR:-$T/c3}
OUT=${C3MON_OUT:-$T/c3}
WINDOW=${C3MON_WINDOW:-600}
SETS=${C3MON_SETS:-clinc150:15000,banking77:9000,hwu64:8954}
NVSMI=${C3MON_NVSMI:-nvidia-smi}
read -r -a LOADCMD <<< "${C3MON_LOAD:-$py -I ${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py 10}"
LIMIT_H=${1:?limit in hours}
ANCHOR=${2:-}
THROTTLE=${C3MON_THROTTLE:-exit}
case $THROTTLE in exit|log) ;; *) echo "C3MON_THROTTLE must be exit or log" >&2; exit 2 ;; esac
log=$OUT/rate_monitor_v2.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
MON_START=$(ts)
TOTAL=$(echo "$SETS" | tr ',' '\n' | awk -F: '{ s += $2 } END { print s }')
echo "$MON_START monitor v2 start (limit ${LIMIT_H} h, anchor ${ANCHOR:-auto}, window ${WINDOW} s, ${TOTAL} rows, throttling: $THROTTLE)" >> "$log"
prev_n=$(grep -c "" "$DIR/gen.jsonl"); prev_t=$SECONDS
while true; do
  sleep "$WINDOW"
  apps=$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1)
  out=$("$py" -I - "$DIR/gen.jsonl" "$DIR/pilot/gen_pilot.jsonl" "$DIR/runs_c3.log" "$prev_n" "$((SECONDS - prev_t))" "$LIMIT_H" \
        "$OUT/rate_baseline_v2.txt" "$SETS" "$ANCHOR" "$MON_START" "$apps" <<'EOF'
import datetime as dt
import json
import math
import statistics as st
import sys

gen, pilot, runlog, prev_n, win, limit, basef, sets, anchor, mon_start, apps = sys.argv[1:12]
prev_n, win, limit = int(prev_n), max(float(win), 1.0), float(limit)
N = {s.split(":")[0]: int(s.split(":")[1]) for s in sets.split(",")}


def when(t):
    try:
        return dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def parsed(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


rows = parsed(gen)
new = rows[prev_n:]
tok = sum(r["usage"]["completion_tokens"] for r in new) / win
pil, by = {}, {}
for r in parsed(pilot):
    pil.setdefault(r["set"], []).append(r["usage"]["completion_tokens"])
for r in rows:
    by.setdefault(r["set"], []).append(r["usage"]["completion_tokens"])
rem = sum((N[s] - len(by.get(s, []))) * (st.mean(by[s]) if len(by.get(s, [])) >= 200 else st.mean(pil[s]) * 1.04) for s in N)
start = when(anchor) if anchor else None
if start is None:
    floor = when(mon_start) - dt.timedelta(minutes=10)
    with open(runlog) as f:
        for line in f:
            t = when(line[:20])
            if "server health" in line and t is not None and t >= floor:
                start = t
                break
now = dt.datetime.now(dt.timezone.utc)
end_h = ((now - start).total_seconds() + rem / max(tok, 1e-9)) / 3600 if start else float("nan")
apps = [a for a in apps.strip().splitlines() if a.strip()]
try:
    with open(basef) as f:
        base = [float(x) for x in f.read().split()]
except FileNotFoundError:
    base = []
if len(apps) <= 1:
    with open(basef, "a") as f:
        f.write(f"{tok:.2f}\n")
    base.append(tok)
b = st.median(base) if base else float("nan")
flags = []
if len(apps) > 1:
    flags.append(f"FLAG: {len(apps)} GPU compute processes (sharing is OFF): {apps}")
if not math.isnan(end_h) and end_h > limit - 0.25:
    flags.append(f"FLAG: projected end {end_h:.2f} h after the server start > {limit - 0.25:.2f} h")
print(f"rows {len(rows)} of {sum(N.values())} (+{len(new)} = {len(new) / win * 60:.1f} req/min), tokens/s {tok:.1f}, median {b:.1f}, "
      f"gpu apps {len(apps)} {apps}, anchor {start.strftime('%Y-%m-%dT%H:%M:%SZ') if start else 'none yet'}, "
      f"projected end {end_h:.2f} h after it" + (" | " + "; ".join(flags) if flags else ""))
EOF
)
  line=$("${LOADCMD[@]}" 2>&1); thr=$(echo "$line" | awk '{print $8}')
  echo "$(ts) ${out:-(no output from the check)} | cpu: $line" >> "$log"
  prev_n=$(grep -c "" "$DIR/gen.jsonl"); prev_t=$SECONDS
  if echo "$out" | grep -q "FLAG"; then echo "$(ts) EXIT on flag" >> "$log"; echo "$out"; exit 3; fi
  if [ "$THROTTLE" = exit ] && [ -n "$thr" ] && [ "$thr" -gt 0 ]; then echo "$(ts) EXIT on throttling: $line" >> "$log"; echo "THROTTLED: $line"; exit 4; fi
  if [ "$prev_n" -ge "$TOTAL" ]; then echo "$(ts) generation complete; monitor end" >> "$log"; echo "complete: $prev_n rows"; exit 0; fi
done
