#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# N8 resume (decided 07:34Z: N8 waits for the decision_layer registered R21 queue, GPU sharing). Waits until the decision-layer
# study's logs/R21_queue.log shows 'queue done' and lane A of n8_chain.sh has exited; removes every claim that has no
# 'end <claim> exit' line (the claim taken at the pause: no run, no row), logged; writes the resume entry to LOG.md;
# then runs lane A from n8_chain_b.sh (= n8_chain.sh with its own log, runs_n8b.log). Lane B is not started here.
emb=${REPO_ROOT}/studies/text_understanding
q=${REPO_ROOT}/studies/decision_layer/logs/R21_queue.log
until grep -q 'queue done' "$q" 2>/dev/null; do sleep 60; done
until grep -q 'STOP lane A: PAUSE' $emb/runs_n8.log && ! pgrep -f "glassbox-lm/studies/text_understanding/n8_chain.sh A" > /dev/null; do sleep 30; done
removed=""
for d in $emb/.n8/claim_*; do
  c=$(basename "$d")
  grep -q "end $c exit" $emb/runs_n8.log $emb/runs_n8b.log 2>/dev/null && continue
  rmdir "$d" && removed="$removed $c" && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) removed $c (claimed at the pause: no run, no row)" >> $emb/runs_n8b.log
done
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "- ${TS}. N8 RESUMED: the decision-layer study's logs/R21_queue.log shows 'queue done'. Claims without a row removed:${removed:- none}. Lane A restarts from n8_chain_b.sh ($(sha256sum $emb/n8_chain_b.sh | cut -c1-16); n8_chain.sh with its own log, runs_n8b.log), one lane." >> $emb/LOG.md
exec bash $emb/n8_chain_b.sh A
