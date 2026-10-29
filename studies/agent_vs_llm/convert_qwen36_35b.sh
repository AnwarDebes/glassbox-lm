#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# Convert Qwen3.6-35B-A3B (Qwen/Qwen3.6-35B-A3B @ 995ad96e, apache-2.0) to GGUF for the headline line's LLM-agent
# baseline, with llama.cpp b11742 (tag commit 4c3c5810), CPU only, nice 19, in the converter's own venv
# (tools/llama.cpp/venv-convert-b11742; never a study venv). Steps: verify every downloaded file against the Hub's
# LFS sha256 (FILES.sha256 written from the Hub metadata before this script), convert the text model to Q8_0, delete
# the safetensors (disk), requantize Q8_0 -> Q4_K_M (--allow-requantize) so that the model fits the 32 GB V100.
# Waits for the CPU gate. Log: convert_qwen36_35b.log next to this script.
set -uo pipefail
S=${WS_ROOT}/tools/llama.cpp/b11742/src
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
V=${WS_ROOT}/tools/llama.cpp/venv-convert-b11742
M=${WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/hf_995ad96e
O=${WS_ROOT}/tools/models_untrusted/qwen3.6-35b-a3b/gguf-b11742
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log="${here}/convert_qwen36_35b.log"
load=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
lpy=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) STOP: $1" >> "$log"; exit 1; }
[ -e "$O" ] && stop "$O exists"
[ "$(git -C $S rev-parse HEAD)" = 4c3c5810f87efe177949f1b85ae509d38a2a311f ] || stop "llama.cpp HEAD changed"
[ -z "$(git -C $S status --porcelain)" ] || stop "llama.cpp work tree not clean"
(cd $M && sha256sum -c --quiet FILES.sha256) >> "$log" 2>&1 || stop "model files do not match FILES.sha256"
echo "$(ts) model files verified ($(wc -l < $M/FILES.sha256) files); waiting for the CPU gate" >> "$log"
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
nice -n 19 ionice -c3 $V/bin/python $S/convert_hf_to_gguf.py $M --outtype q8_0 --outfile $O/Qwen3.6-35B-A3B-Q8_0.gguf >> "$log" 2>&1 || stop "conversion failed"
echo "$(ts) delete the safetensors (kept: config, licence, readme, FILES.sha256)" >> "$log"
rm -f $M/model-*.safetensors
echo "$(ts) requantize Q8_0 -> Q4_K_M" >> "$log"
nice -n 19 ionice -c3 $B/llama-quantize --allow-requantize $O/Qwen3.6-35B-A3B-Q8_0.gguf $O/Qwen3.6-35B-A3B-Q4_K_M.gguf Q4_K_M 2 >> "$log" 2>&1 || stop "quantize failed"
(cd $O && sha256sum *.gguf) > $O/FILES.sha256
ls -la $O >> "$log"; cat $O/FILES.sha256 >> "$log"
echo "$(ts) CONVERT OK" >> "$log"
