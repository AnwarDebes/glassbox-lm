#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# Throttle guard for the WebGuard CPU run (an ops script, not science). Rule set at 23:29Z: "If the cgroup shows
# throttling while 3a runs, pause it". Changed on the queue's go at 00:42Z: RESUME after 10 minutes with no
# throttled period, PAUSE again on any new throttling; the rest is kept.
# Every 120 s while 3a's chain (or R3's) is running on the GPU, it runs cpu_load.py over 10 s.
# - Any throttled period: SIGSTOP the WebGuard python process (wg_run.py or wg_eval.py), and restart the quiet clock.
# - Paused, and 10 minutes without a throttled period: SIGCONT.
# - 3a / R3 not running: SIGCONT if paused.
# At its start it reads the process state, so a process already stopped (state T) counts as paused. It ends when the
# WebGuard chain has ended (only log lines written after its start count). It signals only this job's own
# WebGuard process. Log webguard/throttle_guard.log.
W=${REPO_ROOT}/studies/agent_guards/webguard
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
a3log=${REPO_ROOT}/studies/text_understanding/tu/a3/runs_a3.log
r3log=${REPO_ROOT}/studies/text_understanding/tu/r3/runs_r3.log
log=$W/throttle_guard.log
QUIET=600
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
wgpids() { ps -eo pid,args | awk '$2 ~ /python/ && ($0 ~ /webguard\/wg_run\.py/ || $0 ~ /webguard\/wg_eval\.py/) {print $1}'; }
gpu_job() {  # 0 when 3a or R3 is running now
  if [ -e $a3log ] && grep -q "GPU go" $a3log && ! grep -q "3a done\|STOP" $a3log; then return 0; fi
  if [ -e $r3log ] && grep -q "3a verdict: PASS" $r3log && ! grep -q "R3 done\|STOP" $r3log; then return 0; fi
  return 1; }
resume() { local p; p=$(wgpids); [ -n "$p" ] && kill -CONT $p; paused=0; echo "$(ts) RESUMED pid $p ($1)" >> "$log"; }
paused=0
for p in $(wgpids); do [ "$(ps -o stat= -p $p | cut -c1)" = T ] && paused=1; done
quiet_since=$SECONDS
n0=$(grep -c "" $W/runs_wg.log 2>/dev/null)
echo "$(ts) guard start v2 (runs_wg.log had $n0 lines; paused at start: $paused; resume after ${QUIET}s quiet)" >> "$log"
while ! tail -n +$((n0 + 1)) $W/runs_wg.log 2>/dev/null | grep -q "WG done\|STOP"; do
  if gpu_job; then
    line=$("$py" -I "$load" 10); thr=$(echo "$line" | awk '{print $8}')
    if [ -n "$thr" ] && [ "$thr" -gt 0 ]; then
      quiet_since=$SECONDS
      if [ "$paused" = 0 ]; then p=$(wgpids); [ -n "$p" ] && kill -STOP $p && paused=1 && echo "$(ts) PAUSED pid $p: $line" >> "$log"; fi
    elif [ "$paused" = 1 ] && (( SECONDS - quiet_since >= QUIET )); then
      resume "no throttled period for $((SECONDS - quiet_since)) s; last check: $line"
    fi
  elif [ "$paused" = 1 ]; then
    resume "3a / R3 not running"
  fi
  sleep 120
done
[ "$paused" = 1 ] && resume "the WebGuard chain ended"
echo "$(ts) guard end" >> "$log"
