#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# Serial GPU queue for the headline line: one job at a time, each under gpu_run.sh exclusive, after the CPU gate.
#   run_queue.sh JOBS_FILE LOG
# JOBS_FILE: one job per line, "<name> <args for ft_baseline.py or another command after 'cmd:'>"; the file is
# re-read before each job, so lines can be appended while it runs; a line "END" stops the queue.
set -uo pipefail
jobs_file="$1"
log="$2"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR=${REPO_ROOT}/studies/tm_language_model/scripts/gpu_run.sh
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
GATE=${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py
GPY=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(stamp) queue start: ${jobs_file}" >> "${log}"
i=0
while :; do
    i=$((i + 1))
    n="$(wc -l < "${jobs_file}")"
    if [ "${i}" -gt "${n}" ]; then sleep 30; i=$((i - 1)); continue; fi
    line="$(sed -n "${i}p" "${jobs_file}")"
    [ -z "${line// /}" ] && continue
    [ "${line}" = "END" ] && { echo "$(stamp) END line read; queue end" >> "${log}"; break; }
    name="${line%% *}"
    args="${line#* }"
    while :; do
        g="$(OMP_NUM_THREADS=1 nice -n 19 "${GPY}" -I "${GATE}" 10 2>&1)"
        c="$(sed -n 's/.* cores \([0-9.]*\) .*/\1/p' <<< "${g}")"
        if [ -n "${c}" ] && awk -v c="${c}" 'BEGIN { exit !(c <= 4.5) }'; then break; fi
        echo "$(stamp) gate closed: ${g}" >> "${log}"
        sleep 60
    done
    echo "$(stamp) ${name} start: ${args} (gate: ${g})" >> "${log}"
    mkdir -p "${here}/logs"
    if [[ "${args}" == cmd:* ]]; then
        nice -n 19 "${GR}" exclusive -- bash -c "${args#cmd:}" > "${here}/logs/${name}.txt" 2>&1 < /dev/null
    else
        # shellcheck disable=SC2086
        nice -n 19 "${GR}" exclusive -- "${PY}" -I "${here}/ft_baseline.py" ${args} --out "${here}/runs/${name}" \
            > "${here}/logs/${name}.txt" 2>&1 < /dev/null
    fi
    rc=$?
    echo "$(stamp) ${name} exit ${rc}" >> "${log}"
done
