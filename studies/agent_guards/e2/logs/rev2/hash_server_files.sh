#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# rev2: full sha256 of the GGUF and the llama-server build files (READ-ONLY; no server started or contacted).
# Waits for the cgroup to be below 3.5 cores; nice 19, idle I/O class.
set -u
E2=${REPO_ROOT}/studies/agent_guards/e2
G=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$E2/scripts/wait_cpu.sh" 3.5 360 || { echo "STOP: cgroup never quiet"; exit 1; }
cd "$B" && nice -n 19 ionice -c3 sha256sum llama-server libllama-server-impl.so libllama.so.0.7.0 libllama-common.so.0.7.0 \
  libggml.so.0.27.0 libggml-base.so.0.27.0 libggml-cpu.so.0.27.0 libggml-cuda.so.0.27.0 libmtmd.so.0.7.0
echo "build-info: $(grep -E 'LLAMA_BUILD_NUMBER =|LLAMA_COMMIT =' ${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/common/build-info.cpp | tr '\n' ' ')"
echo "source HEAD: $(cat ${WS_ROOT}/tools/llama.cpp/b11742/src/.git/HEAD)"
"$E2/scripts/wait_cpu.sh" 3.5 360 || { echo "STOP: cgroup never quiet"; exit 1; }
cd "$G" && nice -n 19 ionice -c3 sha256sum Qwen3.5-9B-Q8_0.gguf
echo "FILES.sha256 says: $(grep Q8_0 "$G/FILES.sha256")"
stat -c '%n %s bytes, modified %y' "$G/Qwen3.5-9B-Q8_0.gguf"
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
