#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Qwen3.5-9B GPU smoke test: llama.cpp b11742 (commit 4c3c5810), CUDA sm_70, toolkit 12.9 with the system's
# forward-compatibility libcuda 12.9 (/usr/local/cuda-12.9/compat, listed first by ldconfig) on kernel driver 570.
# Runs only on the queue's go: it waits for llm/.smoke/gpu_go (one big GPU job at a time). Model
# Qwen3.5-9B-Q8_0.gguf (ff3d8ff1...) and mmproj Qwen3.5-9B-f16.gguf (d5b6bbd0...), both checked first.
# Steps (each logged; a failing step is marked FAIL and the next still runs where it can):
#   1. test-backend-ops -b CUDA0 (correctness against the CPU backend): the summary and every failed case;
#   2. llama-bench, -ngl 99, p 512 / n 128, 3 repetitions, with -fa on and with -fa off;
#   3. llama-perplexity on llm/ppl_paragraphs.txt (-c 512, -ngl 99, -fa auto); compared with the CPU reference in
#      ppl_cpu.log (the same file, model and context);
#   4. llama-server on 127.0.0.1:18080 only (--no-webui, -np 1, -c 8192, -fa auto, thinking off with
#      --reasoning-budget 0), then tool_smoke.py (50 prompts, temperature 0), then the server is stopped;
#   5. vision: llama-mtmd-cli with the mmproj on llm/test_shapes.png (a red circle and a blue square), temperature 0.
# Every log is grepped for "CUDA error". Log: smoke_gpu.log; tool rows: tool_smoke.jsonl.
H=${REPO_ROOT}/studies/agent_guards/llm
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
G=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742
M=$G/Qwen3.5-9B-Q8_0.gguf
MM=$G/Qwen3.5-9B-f16.gguf
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$H/smoke_gpu.log
go=$H/.smoke/gpu_go
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(ts) $*" >> "$log"; }
[ -e "$log" ] && { echo "$log exists"; exit 1; }
[ -e $H/tool_smoke.jsonl ] && { echo "tool_smoke.jsonl exists"; exit 1; }
until [ -e $go ]; do sleep 60; done
say "GPU go: $(head -c 300 $go)"
say "GPU at the start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
(cd $G && sha256sum -c --quiet FILES.sha256) >> "$log" 2>&1 || { say "STOP: GGUF files do not match"; exit 1; }
[ "$(sha256sum $H/tool_prompts.json | cut -c1-16)" = 4e54025403af3e03 ] || { say "STOP: tool_prompts.json changed"; exit 1; }
[ "$(sha256sum $H/tool_smoke.py | cut -c1-16)" = 1e47125c3f626a39 ] || { say "STOP: tool_smoke.py changed"; exit 1; }
cd /tmp
export CUDA_VISIBLE_DEVICES=0
say "step 1: test-backend-ops -b CUDA0"
timeout 3600 $B/test-backend-ops test -b CUDA0 > $H/smoke_backend_ops.txt 2>&1; e=$?
say "step 1 exit $e; $(grep -c 'OK$' $H/smoke_backend_ops.txt) OK lines, $(grep -c 'FAIL' $H/smoke_backend_ops.txt) FAIL lines; tail: $(tail -n 3 $H/smoke_backend_ops.txt | tr '\n' ' ')"
grep 'FAIL' $H/smoke_backend_ops.txt | head -n 40 >> "$log"
fa_ok=auto
for fa in on off; do
  say "step 2: llama-bench -fa $fa"
  timeout 1800 $B/llama-bench -m $M -ngl 99 -fa $fa -p 512 -n 128 -r 3 >> "$log" 2>&1; e=$?
  say "step 2 -fa $fa exit $e"
  [ $e -ne 0 ] && [ $fa = on ] && fa_ok=off
done
say "step 3: llama-perplexity on the GPU (-fa auto)"
timeout 1800 $B/llama-perplexity -m $M -f $H/ppl_paragraphs.txt -c 512 -ngl 99 -fa auto > $H/smoke_ppl_gpu.txt 2>&1; e=$?
say "step 3 exit $e; GPU: $(grep -i 'Final estimate' $H/smoke_ppl_gpu.txt); CPU reference: $(grep -i 'Final estimate' $H/ppl_cpu.log)"
say "step 4: llama-server on 127.0.0.1:18080 (-fa $fa_ok)"
$B/llama-server -m $M --host 127.0.0.1 --port 18080 --no-webui -np 1 -c 8192 -ngl 99 -fa $fa_ok --jinja --reasoning-budget 0 > $H/smoke_server.txt 2>&1 &
spid=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18080/health 2>/dev/null | grep -q '"ok"' && break; sleep 5; done
say "server health: $(curl -s -m 2 http://127.0.0.1:18080/health)"
say "libcuda loaded by the server: $(grep -o '/[^ ]*libcuda[^ ]*' /proc/$spid/maps | sort -u | tr '\n' ' ')"
timeout 3600 $PY -I $H/tool_smoke.py 18080 $H/tool_smoke.jsonl >> "$log" 2>&1; e=$?
say "step 4 tool smoke exit $e"
kill $spid; wait $spid 2>/dev/null
say "server stopped"
say "step 5: vision on test_shapes.png"
timeout 900 $B/llama-mtmd-cli -m $M --mmproj $MM --image $H/test_shapes.png -ngl 99 --temp 0 -n 96 \
  -p "Describe the shapes and their colours in this image in one sentence." > $H/smoke_vision.txt 2>&1; e=$?
say "step 5 exit $e; answer: $(grep -v '^\(llama\|load\|print_info\|clip\|main\|ggml\|mtmd\|encod\|decod\|image\|alloc\|srv\|common\|build\|system\|sampler\|generate\|llm_\|.*: \)' $H/smoke_vision.txt | tail -n 3 | tr '\n' ' ')"
say "CUDA errors in all logs: $(cat $log $H/smoke_*.txt | grep -c 'CUDA error')"
say "SMOKE DONE"
