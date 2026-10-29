#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Convert Qwen3.5-9B (Qwen/Qwen3.5-9B @ c2022362, apache-2.0; every file verified, FILES.sha256) to GGUF with
# llama.cpp b11742's converter (tag commit 4c3c5810), CPU only, at nice 19, in its own uv venv
# (tools/llama.cpp/venv-convert-b11742: torch 2.11.0+cpu, transformers 4.57.6, numpy 2.2.6, from the converter's pinned
# requirements; never .venv-dl). Offline (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE). The converter imports its own
# conversion/ and gguf-py/ packages from the llama.cpp source folder, so it runs as a script there (no -I); cwd /tmp.
# Outputs (new folder): Qwen3.5-9B-Q8_0.gguf (text model with its MTP layer) and the vision projector in f16
# (mmproj-...). Waits for the CPU gate (cpu_load.py over 10 s at most 4.5 cores without it, re-checked every 2 min).
# Log: convert_qwen35_9b.log.
S=${WS_ROOT}/tools/llama.cpp/b11742/src
V=${WS_ROOT}/tools/llama.cpp/venv-convert-b11742
M=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/c2022362
O=${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742
H=${REPO_ROOT}/studies/agent_guards/llm
log=$H/convert_qwen35_9b.log
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
lpy=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
[ -e "$O" ] && stop "$O exists"
[ "$(git -C $S rev-parse HEAD)" = 4c3c5810f87efe177949f1b85ae509d38a2a311f ] || stop "llama.cpp HEAD changed"
[ -z "$(git -C $S status --porcelain)" ] || stop "llama.cpp work tree not clean"
(cd $M && sha256sum -c --quiet FILES.sha256) >> "$log" 2>&1 || stop "model files do not match FILES.sha256"
echo "$(ts) model files verified; waiting for the CPU gate" >> "$log"
while true; do
  line=$("$lpy" -I "$load" 10); c=$(echo "$line" | awk '{print $3}')
  echo "$(ts) load check: $line" >> "$log"
  awk -v c="$c" 'BEGIN{exit !(c != "" && c <= 4.5)}' && break
  sleep 120
done
mkdir -p $O
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd /tmp
echo "$(ts) convert: text model to Q8_0" >> "$log"
nice -n 19 ionice -c3 $V/bin/python $S/convert_hf_to_gguf.py $M --outtype q8_0 --outfile $O/Qwen3.5-9B-Q8_0.gguf >> "$log" 2>&1 || stop "text conversion failed"
echo "$(ts) convert: vision projector to f16" >> "$log"
nice -n 19 ionice -c3 $V/bin/python $S/convert_hf_to_gguf.py $M --mmproj --outtype f16 --outfile $O/Qwen3.5-9B-f16.gguf >> "$log" 2>&1 || stop "mmproj conversion failed"
(cd $O && sha256sum *.gguf) > $O/FILES.sha256
ls -la $O >> "$log"; cat $O/FILES.sha256 >> "$log"
echo "$(ts) CONVERT OK" >> "$log"
