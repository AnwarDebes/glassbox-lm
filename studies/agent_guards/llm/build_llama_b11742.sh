#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Build llama.cpp release b11742 (tag commit 4c3c5810f87efe177949f1b85ae509d38a2a311f) with CUDA for the V100 (sm_70),
# userspace only, two compile jobs at nice 19 / idle I/O. Decided 2026-10-19/20: system nvcc 12.9 first
# (driver 570 = CUDA 12.8; SASS for sm_70 is embedded, so no PTX JIT is needed; whether it runs is tested on the GPU only
# after 0b, on the queue's go); never CUDA 13; GGML_CUDA_FORCE_CUBLAS not set. No network during the build:
# LLAMA_USE_PREBUILT_UI=OFF (no web-UI download), LLAMA_BUILD_UI=OFF (no npm), LLAMA_OPENSSL=OFF (no HTTPS model
# downloads in the binaries), LLAMA_SUBPROCESS=OFF (no subprocess tools or router mode in the server), GGML_CUDA_NCCL=OFF.
# Waits until memory (this cgroup's anon + 8 GiB under 0.8 of memory.max) and CPU (cpu_load.py over 10 s at most 4.0
# cores without this job, the 4.5 gate minus the second job; re-checked every 2 min) allow it. Log: build_b11742.log.
# cmake: 4.4.4 from tools/kenlm-deps (conda-forge via micromamba, installed 2026-10-14 for KenLM; there is no system cmake),
# called by absolute path only (kenlm-deps/bin is not put on PATH); C, C++ and CUDA host compiler: system gcc/g++ 13.3.
D=${WS_ROOT}/tools/llama.cpp/b11742
S=$D/src
B=$D/build-cuda129
H=${REPO_ROOT}/studies/agent_guards/llm
CMAKE=${WS_ROOT}/tools/kenlm-deps/bin/cmake
log=$H/build_b11742.log
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
lpy=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
want=4c3c5810f87efe177949f1b85ae509d38a2a311f
targets="llama-cli llama-server llama-bench llama-perplexity llama-mtmd-cli llama-quantize llama-tokenize test-backend-ops"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
anon() { awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat; }
mem_ok() { local cur max; cur=$(anon); max=$(cat /sys/fs/cgroup/memory.max)
  awk -v c="$cur" -v m="$max" 'BEGIN{exit !(c + 8 * 2^30 < 0.8 * m)}'; }
[ -e "$B" ] && stop "$B exists"
[ "$(git -C $S rev-parse HEAD)" = "$want" ] || stop "HEAD is not $want"
[ "$(git -C $S ls-files -s | sha256sum | cut -c1-16)" = ddca1859f9f6a06c ] || stop "index listing changed"
[ -z "$(git -C $S status --porcelain)" ] || stop "work tree not clean"
echo "$(ts) waiting for the gates (memory, CPU at most 4.0 cores)" >> "$log"
while true; do
  if mem_ok; then
    line=$("$lpy" -I "$load" 10)
    c=$(echo "$line" | awk '{print $3}')
    echo "$(ts) load check: $line" >> "$log"
    awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.0)}' && break
  else
    echo "$(ts) memory check: anon $(anon) of $(cat /sys/fs/cgroup/memory.max)" >> "$log"
  fi
  sleep 120
done
export PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=
echo "$(ts) configure (nvcc: $(nvcc --version | tail -n 2 | tr '\n' ' '); gcc $(gcc -dumpfullversion); cmake $("$CMAKE" --version | head -n 1))" >> "$log"
cd /tmp && nice -n 19 "$CMAKE" -S $S -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_SUBPROCESS=OFF \
  -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=ON >> "$log" 2>&1 || stop "configure failed"
"$CMAKE" --build $B --target help > $B/targets.txt 2>&1
for t in $targets; do grep -q "\.\.\. $t\$\|: $t\$\|^$t:" $B/targets.txt || grep -qw "$t" $B/targets.txt || stop "no target $t"; done
grep -i "CUDA_ARCHITECTURES\|GGML_CUDA_FORCE_CUBLAS\|GGML_CUDA_FA\b\|GGML_NATIVE" $B/CMakeCache.txt >> "$log"
echo "$(ts) build start: $targets" >> "$log"
nice -n 19 ionice -c3 "$CMAKE" --build $B -j2 --target $targets >> "$log" 2>&1 || stop "build failed"
echo "$(ts) build done" >> "$log"
(cd $B/bin && sha256sum * 2>/dev/null) > $H/build_b11742_bin.sha256
ldd $B/bin/llama-server | grep -i "cuda\|cublas" >> "$log"
$B/bin/llama-cli --version >> "$log" 2>&1
echo "$(ts) BUILD OK" >> "$log"
