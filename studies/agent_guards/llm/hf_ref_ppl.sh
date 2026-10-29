#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Conversion check (CPU only, nice 19, 2 threads, no GPU, offline): llama-tokenize's ids of llm/ppl_paragraphs.txt
# (72a8d08a1fab8ac7), then hf_ref_ppl.py (the original weights in transformers, fp32) on the first 8 chunks against
# llama.cpp's Q8_0 running perplexity in ppl_cpu.log. Waits for the CPU gate (cpu_load.py over 10 s at most 4.0 cores
# without it: 2 threads). Log hf_ref_ppl.log; result hf_ref_ppl.json.
H=${REPO_ROOT}/studies/agent_guards/llm
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
M=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
log=$H/hf_ref_ppl.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -e "$log" ] && { echo "$log exists"; exit 1; }
[ -e $H/hf_ref_ppl.json ] && { echo "hf_ref_ppl.json exists"; exit 1; }
[ "$(sha256sum $H/ppl_paragraphs.txt | cut -c1-16)" = 72a8d08a1fab8ac7 ] || { echo "$(ts) STOP: paragraphs changed" >> "$log"; exit 1; }
[ "$(sha256sum $H/hf_ref_ppl.py | cut -c1-16)" = "$1" ] || { echo "$(ts) STOP: hf_ref_ppl.py is not $1" >> "$log"; exit 1; }
while true; do
  line=$("$PY" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.0)}' && break
  sleep 120
done
echo "$(ts) start: llama-tokenize" >> "$log"
cd /tmp
CUDA_VISIBLE_DEVICES= nice -n 19 $B/llama-tokenize -m $M -f $H/ppl_paragraphs.txt --ids --no-parse-special --log-disable > $H/ppl_paragraphs.llama_ids.txt 2>> "$log"
echo "$(ts) llama-tokenize exit $?; ids file $(sha256sum $H/ppl_paragraphs.llama_ids.txt | cut -c1-16)" >> "$log"
echo "$(ts) start: hf_ref_ppl.py" >> "$log"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 19 \
  $PY -I $H/hf_ref_ppl.py $H/ppl_paragraphs.llama_ids.txt $H/hf_ref_ppl.json >> "$log" 2>&1
echo "$(ts) end exit $?" >> "$log"
