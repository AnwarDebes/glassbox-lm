#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# rev2 checks, run one after the other (never two heavy jobs at once): (1) the stand-in traces relabelled with the rev2
# label rule (into a scratch folder; data/ is not touched), (2) the end-to-end smoke test on tiny data.
# Each step waits for the cgroup to use less than 3.5 cores; nice 19; 2 threads; no GPU; no LLM; no server.
set -u
E2=${REPO_ROOT}/studies/agent_guards/e2
S=${1:?usage: chain_relabel_smoke.sh <scratch dir>}
cd "$E2" || exit 2
echo "chain start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$E2/scripts/wait_cpu.sh" 3.5 360 || { echo "STOP: cgroup never quiet"; exit 1; }
rm -f "$S/rows_standin_rev2.jsonl" "$S/rows_standin_rev2.meta.json"
echo "relabel start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
env -i PATH=/usr/bin:/bin HOME=${HOME} LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
  ../agentdojo/.venv/bin/python -I -B scripts/traces_to_rows.py --source standin_trace --out "$S/rows_standin_rev2.jsonl" data/standin_traces > "$S/rows_standin_rev2.log" 2>&1
echo "relabel exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$E2/scripts/wait_cpu.sh" 3.5 360 || { echo "STOP: cgroup never quiet"; exit 1; }
echo "smoke start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash tests/smoke_e2e.sh "$E2/logs/rev2/smoke_run1" > "$E2/logs/rev2/smoke_run1.log" 2>&1
echo "smoke exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
