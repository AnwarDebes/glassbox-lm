#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# W2 H-B step 0 part 1 (decided 07:36Z): starts after 2026-10-19T10:15:00Z, and only when this cgroup's CPU load
# (the decision-layer study's scripts/cpu_load.py over 10 s) is at most 3.0 cores without this job; re-checked every
# 5 minutes. Then pairs_scan.py, hash-checked (CPU only, one tokenizer thread). Log w2/runs_w2.log.
here=${REPO_ROOT}/studies/text_understanding/w2
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$here/runs_w2.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
target=$(date -u -d '2026-10-19T10:15:00Z' +%s)
until [ "$(date -u +%s)" -ge "$target" ]; do sleep 60; done
while true; do
  line=$("$py" -I "$load" 10)
  c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 3.0)}' && break
  sleep 300
done
[ "$(sha256sum $here/pairs_scan.py | cut -c1-16)" = b865208abb302aa0 ] || { echo "$(ts) STOP: pairs_scan.py changed" >> "$log"; exit 1; }
echo "$(ts) start pairs_scan.py" >> "$log"
"$py" -I -B $here/pairs_scan.py < /dev/null >> "$log" 2>&1
e=$?
echo "$(ts) end pairs_scan.py exit $e" >> "$log"
exit $e
