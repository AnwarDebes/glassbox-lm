#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Step 3c CPU pilot (tu/PLAN_3c.md "Tests before the GPU job"; training texts only, no probe, no GPU): a CPU
# llama-server (Qwen3.5-9B-Q8_0, -ngl 0, 2 threads, nice 19, 127.0.0.1:18082, thinking off), c3_generate.py on 20
# training texts per set (60 requests), the server stopped, then c3_filter.py --pilot on the CPU (bge fp32). Waits for
# the CPU gate (cpu_load.py over 10 s at most 4.0 cores without it: 2 threads). Log tu/c3/pilot/pilot.log.
tu=${REPO_ROOT}/studies/text_understanding/tu
P=$tu/c3/pilot
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
M=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$P/pilot.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p $P
[ -e "$log" ] && { echo "$log exists"; exit 1; }
for f in c3_common.py c3_generate.py c3_filter.py; do echo "$(ts) $f $(sha256sum $tu/$f | cut -c1-16)" >> "$log"; done
while true; do
  line=$("$PY" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.0)}' && break
  sleep 120
done
cd /tmp
CUDA_VISIBLE_DEVICES= nice -n 19 $B/llama-server -m $M --host 127.0.0.1 --port 18082 --no-webui -np 1 -c 2048 -ngl 0 -t 2 \
  --jinja --reasoning-budget 0 > $P/server_cpu.txt 2>&1 &
spid=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18082/health 2>/dev/null | grep -q '"ok"' && break; sleep 5; done
echo "$(ts) server health: $(curl -s -m 2 http://127.0.0.1:18082/health)" >> "$log"
CUDA_VISIBLE_DEVICES= $PY -I $tu/c3_generate.py --port 18082 --out $P/gen_pilot.jsonl --pilot 20 --workers 1 >> "$log" 2>&1
echo "$(ts) generate exit $?" >> "$log"
kill $spid; wait $spid 2>/dev/null
echo "$(ts) server stopped" >> "$log"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 $PY -I $tu/c3_filter.py --gen $P/gen_pilot.jsonl \
  --out-dir $P --device cpu --pilot >> "$log" 2>&1
echo "$(ts) filter exit $?" >> "$log"
echo "$(ts) pilot done" >> "$log"
