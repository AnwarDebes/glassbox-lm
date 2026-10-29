#!/usr/bin/env bash
# B1 full grid: 10 seeds x 7 team sizes x all arms, one process per run, each in one CPU slot.
#
#   scripts/b1_launch.sh registrations/B1_<date>.md [JOBS]
#   B1_DRY_RUN=1 scripts/b1_launch.sh registrations/B1_<date>.md [JOBS]   # print the jobs, run nothing
#
# Needs the frozen registration (registrations/<name>.frozen.json next to it). At most JOBS runs
# at a time (default 5), each through cpu_slot.sh 1 under nice 19, with every thread pool capped
# at 1 (NUMBA, OMP, MKL, OPENBLAS, RAYON). Machine rule: before every start (self-checks, each
# run, report) the launcher waits until the CPU gate, scripts/cpu_load.py of the decision_layer
# study over 10 s, reports at most 4.5 cores in use for the whole cgroup, checking again every
# 60 s while it does not; the readings go to gate.log. Jobs are started one after the other by
# this script, so two jobs never start on the same reading. A reading that fails (the gate
# command exits non-zero) counts as closed and is logged; the launcher reads again after the pause.
# Phase 1: self-checks, flat TM tuning (one per size), oracle, shared, selfish, marginal, dual,
#          greedy, random. Phase 2 (needs phase 1): flat (reads its tuning run) and ab (reads the
#          dual run of the same size and seed). Largest sizes first. Job logs: results/b1_launch_<UTC>/.
# Changed 2026-10-25: nice 19 instead of 10, thread caps, the CPU gate, the dry run, and a start
# loop in place of xargs. For tests the environment can replace the interpreter (B1_PY), the slot
# script (B1_SLOT), the gate command (CPU_GATE_CMD), the gate limit and pause (CPU_GATE_MAX,
# CPU_GATE_RETRY), the pause after a start (B1_SETTLE), the grid (B1_SIZES, B1_SEEDS) and the
# log folder (B1_LOGS).
# Changed 2026-10-26: a failed gate reading counts as closed. Before, `set -e` stopped the whole
# launcher at the first failed reading while its started runs went on.
set -euo pipefail

reg="${1:?usage: b1_launch.sh registrations/<file>.md [JOBS]}"
njobs="${2:-5}"
[[ "${njobs}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer, got ${njobs}" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study="$(cd "${here}/.." && pwd)"
slot="${B1_SLOT:-${study}/../tm_language_model/scripts/cpu_slot.sh}"
py="${B1_PY:-${study}/../tm_language_model/.venv-tm/bin/python}"
gate_script="${study}/../decision_layer/scripts/cpu_load.py"
gate_cmd="${CPU_GATE_CMD:-}"
gate_max="${CPU_GATE_MAX:-4.5}"
gate_retry="${CPU_GATE_RETRY:-60}"
settle="${B1_SETTLE:-2}"
sizes="${B1_SIZES:-512 256 128 64 32 16 8}"
seeds="${B1_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
reg="$(cd "$(dirname "${reg}")" && pwd)/$(basename "${reg}")"
frozen="${reg%.md}.frozen.json"

export PYTHONPATH="${study}/src"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1

job_list() {   # phase 1 and phase 2 lines: arm M seed
    local M arm s
    for M in ${sizes}; do
        echo "flat_tune ${M} 10000" >> "$1"
    done
    for M in ${sizes}; do
        for arm in dual marginal shared selfish greedy random oracle; do
            for s in ${seeds}; do
                echo "${arm} ${M} ${s}" >> "$1"
            done
        done
        for arm in flat ab; do
            for s in ${seeds}; do
                echo "${arm} ${M} ${s}" >> "$2"
            done
        done
    done
}

job_cmd() {    # the command of one run, as an array in JOB
    JOB=(nice -n 19 "${slot}" 1 -- "${py}" -m tally.b1.run --arm "$1" --M "$2" --seed "$3"
         --label reg --registration "${reg}")
}

if [ "${B1_DRY_RUN:-0}" = "1" ]; then
    tmpd="$(mktemp -d)"
    job_list "${tmpd}/phase1.txt" "${tmpd}/phase2.txt"
    [ -f "${frozen}" ] || echo "note: not frozen yet (${frozen} missing); a real start stops here"
    echo "env: NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}"
    echo "gate before every start: ${gate_cmd:-${py} -I ${gate_script} 10} at most ${gate_max} cores"
    for ph in phase1 phase2; do
        while read -r arm M s; do
            job_cmd "${arm}" "${M}" "${s}"
            echo "${ph}: ${JOB[*]}"
        done < "${tmpd}/${ph}.txt"
    done
    echo "$(wc -l < "${tmpd}/phase1.txt") phase 1 runs, $(wc -l < "${tmpd}/phase2.txt") phase 2 runs, ${njobs} at a time"
    rm -r "${tmpd:?}"
    exit 0
fi

[ -f "${frozen}" ] || { echo "not frozen: ${frozen} missing" >&2; exit 2; }
if [ -z "${gate_cmd}" ]; then
    [ -f "${gate_script}" ] || { echo "CPU gate not found: ${gate_script}" >&2; exit 2; }
    gate_cmd="$(printf '%q ' "${py}" -I "${gate_script}" 10)"
fi
logs="${B1_LOGS:-${study}/results/b1_launch_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${logs}"

wait_gate() {   # returns when the gate reads at most gate_max cores; a failed reading is closed
    local line cores rc
    while :; do
        rc=0
        line="$(bash -c "${gate_cmd}" < /dev/null)" || rc=$?
        cores=""
        if [ "${rc}" -eq 0 ]; then
            cores="$(sed -n 's/.* cores \([0-9.][0-9.]*\) .*/\1/p' <<< "${line}")"
        else
            line="gate command failed (exit ${rc}) ${line}"
        fi
        if [ -n "${cores}" ] && awk -v c="${cores}" -v m="${gate_max}" 'BEGIN { exit !(c <= m) }'; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) open: ${line}" >> "${logs}/gate.log"
            return 0
        fi
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) closed (limit ${gate_max}): ${line}" >> "${logs}/gate.log"
        sleep "${gate_retry}"
    done
}

run_one() {
    local arm="$1" M="$2" s="$3"
    job_cmd "${arm}" "${M}" "${s}"
    "${JOB[@]}" < /dev/null > "${logs}/${arm}_M${M}_s${s}.txt" 2>&1 \
        || echo "FAILED ${arm} ${M} ${s}" >> "${logs}/failed.txt"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end ${arm} ${M} ${s}" >> "${logs}/starts.log"
}

run_phase() {   # start the runs of one list, at most njobs at a time, each after the gate
    local arm M s
    while read -r arm M s; do
        while [ "$(jobs -rp | wc -l)" -ge "${njobs}" ]; do
            wait -n || true
        done
        wait_gate
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start ${arm} ${M} ${s}" >> "${logs}/starts.log"
        run_one "${arm}" "${M}" "${s}" &
        sleep "${settle}"
    done < "$1"
    wait
}

phase1="${logs}/phase1.txt"
phase2="${logs}/phase2.txt"
: > "${phase1}"
: > "${phase2}"
job_list "${phase1}" "${phase2}"

wait_gate
nice -n 19 "${py}" -m tally.b1.selfcheck < /dev/null > "${logs}/selfcheck.txt" \
    || { echo "self-checks failed" >&2; exit 1; }

echo "phase 1: $(wc -l < "${phase1}") runs, ${njobs} at a time, each after the CPU gate; logs in ${logs}"
run_phase "${phase1}"
echo "phase 2: $(wc -l < "${phase2}") runs"
run_phase "${phase2}"
wait_gate
nice -n 19 "${py}" -m tally.b1.report --label reg --out "${logs}/report.json" < /dev/null \
    > "${logs}/report.txt" 2>&1 || true
echo "done; failed runs: $(cat "${logs}/failed.txt" 2>/dev/null | wc -l)"
