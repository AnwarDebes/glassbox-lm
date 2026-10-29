#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# llama-server for the headline line's LLM runs (dev only): the pinned build b11742, one model, port 18083.
#   serve_headline.sh start MODEL.gguf SHA256 [CTX] [NP] [REASONING_BUDGET]   (budget -1: unlimited; 0: thinking off)
#   serve_headline.sh stop
# Checks before a start: no llama-server is running, no GPU compute process, the CPU gate (at most 4.5 cores), the
# model file's sha256. Runs at nice 19 with 2 threads, -fa off (sm_70), -ngl 99, 127.0.0.1 only. The pinned E2/E4
# script (glassbox-lm/studies/agent_guards/llm/serve_qwen_agent.sh) is not used or changed.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin/llama-server
GATE=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
GPY=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
PORT=18083
pidf="${here}/serve_headline.pid"
log="${here}/serve_headline.log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
case "${1:-}" in
start)
    model="$2"; want="$3"; ctx="${4:-65536}"; np="${5:-1}"; budget="${6:-0}"
    if pgrep -x llama-server > /dev/null; then echo "$(stamp) refused: a llama-server runs" | tee -a "${log}"; exit 2; fi
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        echo "$(stamp) refused: a GPU compute process runs" | tee -a "${log}"; exit 2; fi
    g="$(OMP_NUM_THREADS=1 nice -n 19 "${GPY}" -I "${GATE}" 10)"
    c="$(sed -n 's/.* cores \([0-9.]*\) .*/\1/p' <<< "${g}")"
    awk -v c="${c}" 'BEGIN { exit !(c <= 4.5) }' || { echo "$(stamp) refused: CPU gate ${g}" | tee -a "${log}"; exit 2; }
    have="$(sha256sum "${model}" | cut -d' ' -f1)"
    [ "${have}" = "${want}" ] || { echo "$(stamp) refused: sha256 ${have} != ${want}" | tee -a "${log}"; exit 2; }
    rea=(-rea off --reasoning-budget 0)
    [ "${budget}" != "0" ] && rea=(-rea on --reasoning-budget "${budget}")
    echo "$(stamp) start ${model} ctx ${ctx} np ${np} reasoning ${budget} (gate: ${g})" >> "${log}"
    setsid nohup nice -n 19 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${BIN}" -m "${model}" --host 127.0.0.1 \
        --port "${PORT}" -a local --jinja "${rea[@]}" -fa off -np "${np}" -c "$((ctx * np))" -ngl 99 -t 2 \
        --no-webui --metrics >> "${log}" 2>&1 < /dev/null &
    echo $! > "${pidf}"
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "$(stamp) READY pid $(cat "${pidf}")" | tee -a "${log}"; exit 0; fi
        sleep 2
    done
    echo "$(stamp) not ready after 360 s" | tee -a "${log}"; kill "$(cat "${pidf}")" 2> /dev/null; exit 3 ;;
stop)
    if [ -f "${pidf}" ]; then
        p="$(cat "${pidf}")"
        if ps -p "${p}" -o args= | grep -q llama-server; then kill "${p}"; sleep 3; fi
        rm -f "${pidf}"
    fi
    echo "$(stamp) stopped" | tee -a "${log}" ;;
*) echo "usage: serve_headline.sh start MODEL SHA256 [CTX] [NP] [BUDGET] | stop"; exit 2 ;;
esac
