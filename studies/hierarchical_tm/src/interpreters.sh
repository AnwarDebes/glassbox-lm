# shellcheck shell=bash
# Interpreters of the launchers of this study. Source it from a launcher; do not execute it.
#
# Added on 2026-10-25 (bug HTM-B1 of the finish pass). The recorded runs started the GPU jobs with
# ../tm_language_model/.venv-htm/bin/python, and they started scripts/mem_guard.py and scripts/run_logged.py of
# ../tm_language_model by name, through the line '#!/usr/bin/env python3' of both scripts. On the workstation
# of 2026-10-25 that virtualenv does not exist and no python3 is on PATH, so every launcher stopped before its
# job started. The launchers now take both interpreters from this file:
#
#   HTM_PY       interpreter of the job itself (the drivers import PyHierarchicalTsetlinMachineCUDA).
#                Default: ../tm_language_model/.venv-htm/bin/python when it is an executable file (the
#                interpreter of the recorded runs), else ../decision_layer/.venv-htm/bin/python, which pins the
#                same package commit 5b4a223. That virtualenv belongs to the decision_layer study and is only
#                used through its path: nothing is installed into it.
#   HTM_TOOL_PY  interpreter that starts scripts/mem_guard.py and scripts/run_logged.py of ../tm_language_model.
#                Default: ../tm_language_model/.venv-tm/bin/python, the interpreter of the run logger in the
#                recorded runs.
#
# A value that the caller sets wins; a relative path is read from the caller's working directory. Both values
# are exported. A launcher that sources this file stops with exit status 127 when either interpreter is not an
# executable file. When HTM_PY lies in ../decision_layer/, PYTHONDONTWRITEBYTECODE=1 is exported as well. The launchers write HTM_PY into the run config as "job_python", because the environment
# record of the run logger (env.json) describes HTM_TOOL_PY and not the interpreter of the job.

_htm_studies="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${HTM_PY:-}" ]; then
    if [ -f "${_htm_studies}/tm_language_model/.venv-htm/bin/python" ] &&
        [ -x "${_htm_studies}/tm_language_model/.venv-htm/bin/python" ]; then
        HTM_PY="${_htm_studies}/tm_language_model/.venv-htm/bin/python"
    else
        HTM_PY="${_htm_studies}/decision_layer/.venv-htm/bin/python"
    fi
fi
HTM_TOOL_PY="${HTM_TOOL_PY:-${_htm_studies}/tm_language_model/.venv-tm/bin/python}"
case "${HTM_PY}" in /*) ;; *) HTM_PY="${PWD}/${HTM_PY}" ;; esac
case "${HTM_TOOL_PY}" in /*) ;; *) HTM_TOOL_PY="${PWD}/${HTM_TOOL_PY}" ;; esac
for _htm_var in HTM_PY HTM_TOOL_PY; do
    if [ ! -f "${!_htm_var}" ] || [ ! -x "${!_htm_var}" ]; then
        echo "$(basename "$0"): ${_htm_var}=${!_htm_var} is not an executable file; set ${_htm_var} to a Python interpreter" >&2
        exit 127
    fi
done
export HTM_PY HTM_TOOL_PY
# The virtualenv of the decision_layer study is only borrowed: no bytecode may be written into it (added 2026-10-25,
# second round of the finish pass; without it the first job would write __pycache__ folders into its site-packages).
case "${HTM_PY}" in
    "${_htm_studies}/decision_layer/"*) export PYTHONDONTWRITEBYTECODE=1 ;;
esac
unset _htm_studies _htm_var
