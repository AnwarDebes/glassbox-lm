# shellcheck shell=bash
# Helpers shared by scripts/gpu_run.sh and scripts/cpu_slot.sh. Source it, do not run it.
# Linux only (/proc). Safe under `set -euo pipefail`. Defines functions, nothing else.

# tmlm_pid_alive <pid>: 0 when /proc/<pid> is a process that still runs. A zombie
# (killed, not yet reaped by its parent) counts as dead: it will never take a lock.
# Same rule as tmlm.common.locks.pid_alive.
tmlm_pid_alive() {
    local stat="" state
    if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        return 1
    fi
    if ! { IFS= read -r stat <"/proc/$1/stat"; } 2>/dev/null && [ -z "${stat}" ]; then
        return 1
    fi
    # "<pid> (<comm>) <state> ...": comm may contain spaces and parentheses.
    state="${stat##*) }"
    state="${state%% *}"
    case "${state}" in
        "" | Z | X | x) return 1 ;;
    esac
    return 0
}

# tmlm_ancestor_holds exact <path>
# tmlm_ancestor_holds glob <prefix> <suffix>
# 0 when this shell or one of its ancestors has a descriptor open on <path>, or on a
# file named <prefix>*<suffix>. Used for nested calls whose own lock descriptor was
# closed by a wrapper in between (Python's subprocess closes descriptors by default):
# the lock such a call would wait for is held by its own ancestors, so waiting would
# deadlock. Ancestors whose descriptors cannot be read are skipped.
tmlm_ancestor_holds() {
    local kind="$1" first="$2" second="${3:-}" pid="$$" target line ppid hops=0
    while [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && [ "${hops}" -lt 64 ]; do
        hops=$((hops + 1))
        while IFS= read -r target; do
            if [ "${kind}" = "exact" ]; then
                if [ "${target}" = "${first}" ]; then
                    return 0
                fi
            elif [[ "${target}" == "${first}"*"${second}" ]]; then
                return 0
            fi
        done < <(readlink -- "/proc/${pid}/fd/"* 2>/dev/null || true)
        ppid=""
        while IFS= read -r line; do
            if [[ "${line}" == PPid:* ]]; then
                ppid="${line##*[[:space:]]}"
                break
            fi
        done < <(cat "/proc/${pid}/status" 2>/dev/null || true)
        pid="${ppid}"
    done
    return 1
}
