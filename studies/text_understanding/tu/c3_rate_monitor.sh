#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# 3c throughput monitor (decided 01:28Z; an ops script, read-only: it signals nothing and stops nothing).
# Every 600 s it logs to tu/c3/rate_monitor.log:
# - rows done and requests per minute in the window;
# - new tokens per second in the window (the sets have different answer lengths, so tokens/s is the speed measure);
# - the GPU compute processes (nvidia-smi --query-compute-apps; more than one = e82 is sharing);
# - cpu_load.py over 10 s (throttled periods);
# - a projected end: the remaining tokens (per-set mean answer length so far; pilot means for a set not yet started)
#   divided by the window's tokens/s.
# It EXITS with a FLAG line, so that the run is flagged and logged, when:
# - a window has any throttled period;
# - another GPU process shares the GPU and the window's tokens/s is below 80% of the baseline (the median of the
#   windows before any sharing);
# - the projected end is later than generation start + LIMIT_H - 0.25 h.
# It exits quietly when gen.jsonl has all 32,954 rows.
# Usage: c3_rate_monitor.sh LIMIT_H
T=${REPO_ROOT}/studies/text_understanding/tu
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
LIMIT_H=${1:?limit in hours}
log=$T/c3/rate_monitor.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(ts) monitor start (limit ${LIMIT_H} h)" >> "$log"
prev_n=$(grep -c "" $T/c3/gen.jsonl); prev_t=$SECONDS
while true; do
  sleep 600
  out=$("$py" -I - "$T/c3/gen.jsonl" "$T/c3/pilot/gen_pilot.jsonl" "$T/c3/runs_c3.log" "$prev_n" "$((SECONDS - prev_t))" "$LIMIT_H" "$T/c3/rate_baseline.txt" <<'EOF'
import json, sys, statistics as st, subprocess, datetime as dt
gen, pilot, runlog, prev_n, win, limit, basef = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
rows = [json.loads(l) for l in open(gen)]
N = {"clinc150": 15000, "banking77": 9000, "hwu64": 8954}
new = rows[prev_n:]
tok = sum(r["usage"]["completion_tokens"] for r in new) / max(win, 1)
pil = {}
for l in open(pilot):
    r = json.loads(l); pil.setdefault(r["set"], []).append(r["usage"]["completion_tokens"])
by = {}
for r in rows:
    by.setdefault(r["set"], []).append(r["usage"]["completion_tokens"])
rem = sum((N[s] - len(by.get(s, []))) * (st.mean(by[s]) if len(by.get(s, [])) >= 200 else st.mean(pil[s]) * 1.04) for s in N)
start = None
for l in open(runlog):
    if "server health" in l and start is None:  # the FIRST server start (01:21:41Z), not the restart: conservative
        start = dt.datetime.strptime(l[:20], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
now = dt.datetime.now(dt.timezone.utc)
end_h = ((now - start).total_seconds() + rem / max(tok, 1e-9)) / 3600 if start else float("nan")
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
try:
    base = [float(x) for x in open(basef).read().split()]
except FileNotFoundError:
    base = []
sharing = len(apps) > 1
if not sharing:
    with open(basef, "a") as f:
        f.write(f"{tok:.2f}\n")
    base.append(tok)
b = st.median(base) if base else float("nan")
flag = ""
if sharing and base and tok < 0.8 * b:
    flag = f"FLAG: tokens/s {tok:.1f} is below 80% of the baseline {b:.1f} while sharing"
if end_h == end_h and end_h > limit - 0.25:
    flag = (flag + "; " if flag else "") + f"FLAG: projected end {end_h:.2f} h after the server start > {limit - 0.25:.2f} h"
print(f"rows {len(rows)} (+{len(new)} = {len(new) / win * 60:.1f} req/min), tokens/s {tok:.1f}, baseline {b:.1f}, gpu apps {len(apps)} {apps}, projected end {end_h:.2f} h after the server start {('| ' + flag) if flag else ''}")
EOF
)
  line=$("$py" -I "$load" 10); thr=$(echo "$line" | awk '{print $8}')
  echo "$(ts) $out | cpu: $line" >> "$log"
  prev_n=$(grep -c "" $T/c3/gen.jsonl); prev_t=$SECONDS
  if echo "$out" | grep -q "FLAG"; then echo "$(ts) EXIT on flag" >> "$log"; echo "$out"; exit 3; fi
  if [ -n "$thr" ] && [ "$thr" -gt 0 ]; then echo "$(ts) EXIT on throttling: $line" >> "$log"; echo "THROTTLED: $line"; exit 4; fi
  [ "$prev_n" -ge 32954 ] && { echo "$(ts) generation complete; monitor end" >> "$log"; exit 0; }
done
