#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# CPU reference perplexity for the CUDA-vs-CPU smoke check: Qwen3.5-9B-Q8_0 (ff3d8ff1...) on llm/ppl_paragraphs.txt
# (72a8d08a1fab8ac7), context 512, all chunks, 2 threads, no GPU (CUDA_VISIBLE_DEVICES empty, -ngl 0), nice 19.
# Waits for the CPU gate (cpu_load.py over 10 s at most 4.0 cores without it: 2 threads). Log ppl_cpu.log.
H=${REPO_ROOT}/studies/agent_guards/llm
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
M=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf
log=$H/ppl_cpu.log
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
lpy=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -e "$log" ] && { echo "$log exists"; exit 1; }
[ "$(sha256sum $H/ppl_paragraphs.txt | cut -c1-16)" = 72a8d08a1fab8ac7 ] || { echo "paragraphs changed" >> "$log"; exit 1; }
while true; do
  line=$("$lpy" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.0)}' && break
  sleep 120
done
echo "$(ts) start" >> "$log"
cd /tmp && CUDA_VISIBLE_DEVICES= nice -n 19 $B/llama-perplexity -m $M -f $H/ppl_paragraphs.txt -c 512 -t 2 -ngl 0 >> "$log" 2>&1
echo "$(ts) end exit $?" >> "$log"
