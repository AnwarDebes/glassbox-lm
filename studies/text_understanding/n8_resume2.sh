#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# N8 resume, second version (decided 07:37Z; n8_resume.sh fired at once on the dry-run queue's 'queue done' line of
# 05:47:57Z). Waits until the decision-layer study's logs/R21_queue.log shows 'queue done' AFTER the line
# 'queue start: 22 jobs from logs/jobs_R21.txt' (the real R21 queue), and until lane A of n8_chain_b.sh has exited;
# removes every claim that has no 'end <claim> exit' line in runs_n8.log, runs_n8b.log or runs_n8c.log (the claim taken
# at a pause: no run, no row), logged; writes the resume entry to LOG.md; then runs lane A from n8_chain_c.sh
# (= n8_chain.sh with its own log, runs_n8c.log).
emb=${REPO_ROOT}/studies/text_understanding
q=${REPO_ROOT}/studies/decision_layer/logs/R21_queue.log
real_done() { awk '/queue start: 22 jobs from logs\/jobs_R21.txt/{f=1} f && /queue done/{ok=1} END{exit !ok}' "$q" 2>/dev/null; }
[ "$1" = --dry ] && { real_done && echo "dry: WOULD FIRE" || echo "dry: would wait"; exit 0; }
until real_done; do sleep 60; done
until ! pgrep -f "glassbox-lm/studies/text_understanding/n8_chain_b.sh A" > /dev/null; do sleep 30; done
removed=""
for d in $emb/.n8/claim_*; do
  c=$(basename "$d")
  grep -q "end $c exit" $emb/runs_n8.log $emb/runs_n8b.log $emb/runs_n8c.log 2>/dev/null && continue
  rmdir "$d" && removed="$removed $c" && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) removed $c (claimed at a pause: no run, no row)" >> $emb/runs_n8c.log
done
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "- ${TS}. N8 RESUMED (second waiter, n8_resume2.sh): the decision-layer study's logs/R21_queue.log shows 'queue done' after 'queue start: 22 jobs from logs/jobs_R21.txt'. Claims without a row removed:${removed:- none}. Lane A restarts from n8_chain_c.sh ($(sha256sum $emb/n8_chain_c.sh | cut -c1-16); n8_chain.sh with its own log, runs_n8c.log), one lane." >> $emb/LOG.md
exec bash $emb/n8_chain_c.sh A
