#!/usr/bin/env bash
# CPU lane: a counting semaphore over the 6 CPUs of the cgroup quota (design decision).
#
#   cpu_slot.sh N -- <cmd...>      N in 1..6
#
# Takes N of the 6 slot locks, caps every thread pool at N (by exporting
# TMLM_CPU_SLOTS=N and sourcing env/env.sh), then runs the command.
#
# Mechanism: 6 lock files, one flock(2) each, held through file descriptors that the
# wrapped command INHERITS (this script ends with `exec`). The kernel releases the
# slots when the command is gone, however it dies, and the command's exit code is the
# exit code.
#
# No deadlock, no starvation: a single gate lock admits one waiter at a time. The
# waiter at the gate takes slots only when N are free at once (all or nothing, it never
# sleeps while holding a partial set), and nobody else can take slots in the meantime,
# so free slots only accumulate until it is served.
#
# Nesting: inside a command that already holds M slots, `cpu_slot.sh N` with N <= M
# runs the command directly with the caps lowered to N (the outer job owns its M slots
# and is responsible for what it runs inside them). N > M fails with exit 2 instead of
# deadlocking. A nested call is recognized by its inherited descriptors
# TMLM_CPU_SLOT_FDS. When a wrapper in between closed them (Python's subprocess does,
# unless the wrapper passes pass_fds=tmlm.common.locks.inherited_lock_fds()),
# TMLM_CPU_SLOTS is still set: the call is then nested exactly when an ANCESTOR process
# has a slot lock open, and the same rules apply. With no such ancestor, TMLM_CPU_SLOTS
# is only a thread cap somebody exported and the call is an ordinary request.
#
# Thread caps follow the min rule of env/env.sh: a *_NUM_THREADS value that is already
# an integer in 1..N is kept, anything else becomes N.
#
# Exported to the command: TMLM_CPU_SLOTS, TMLM_CPU_SLOT_FDS, the *_NUM_THREADS caps.
# Environment: TMLM_LOCK_DIR overrides the lock directory (default results/.locks).
# Exit codes of this script itself: 2 usage, 1 lock failure, 129/130/143 on HUP/INT/TERM
# while still waiting.

set -euo pipefail

readonly TOTAL_SLOTS=6

usage() {
    echo "usage: cpu_slot.sh N -- <command...>   (N in 1..${TOTAL_SLOTS})" >&2
    exit 2
}

[ "$#" -ge 3 ] || usage
want="$1"
[ "$2" = "--" ] || usage
shift 2
[[ "${want}" =~ ^[1-9][0-9]*$ ]] || usage
[ "${want}" -le "${TOTAL_SLOTS}" ] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${TMLM_ROOT:-$(cd "${script_dir}/.." && pwd)}"
lock_dir="${TMLM_LOCK_DIR:-${root}/results/.locks}"
mkdir -p "${lock_dir}"
lock_dir="$(cd "${lock_dir}" && pwd -P)"
export TMLM_LOCK_DIR="${lock_dir}"

run_with_caps() {
    export TMLM_CPU_SLOTS="${want}"
    # shellcheck source=../env/env.sh
    source "${script_dir}/../env/env.sh"
    exec "$@"
}

# shellcheck source=lib_locks.sh
source "${script_dir}/lib_locks.sh"

# ---- nested call: our ancestors already hold slots.
outer_slots="${TMLM_CPU_SLOTS:-}"
outer_fds="${TMLM_CPU_SLOT_FDS:-}"
first_fd="${outer_fds%% *}"
nested=0
if [[ "${outer_slots}" =~ ^[1-6]$ ]]; then
    if [[ "${first_fd}" =~ ^[0-9]+$ ]] && [ -e "/proc/$$/fd/${first_fd}" ] &&
        [[ "$(readlink "/proc/$$/fd/${first_fd}")" == "${lock_dir}/cpu_slot_"*.lock ]]; then
        nested=1
    elif tmlm_ancestor_holds glob "${lock_dir}/cpu_slot_" ".lock"; then
        nested=1
        echo "cpu_slot.sh: note: the slot descriptors were not inherited (closed by a" \
            "wrapper), an ancestor still holds ${outer_slots} slot(s)" >&2
        # The numbers now mean nothing here, or name unrelated files.
        unset TMLM_CPU_SLOT_FDS
    fi
fi
if [ "${nested}" -eq 1 ]; then
    if [ "${want}" -le "${outer_slots}" ]; then
        run_with_caps "$@"
    fi
    echo "cpu_slot.sh: ${want} slots requested inside a job that holds ${outer_slots}" >&2
    exit 2
fi

trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# try_lock <fd> [flock options]: 0 acquired, 1 busy. Any other flock failure is fatal.
try_lock() {
    local fd="$1" rc=0
    shift
    flock -E 75 "$@" "${fd}" || rc=$?
    case "${rc}" in
        0) return 0 ;;
        75) return 1 ;;
        *)
            echo "cpu_slot.sh: flock failed with exit ${rc}" >&2
            exit 1
            ;;
    esac
}

# The gate, waited for in slices of 1 s so pending traps run promptly.
exec {gate_fd}>>"${lock_dir}/cpu_gate.lock"
until try_lock "${gate_fd}" -x -w 1; do :; done

held=()
while :; do
    held=()
    for ((slot = 1; slot <= TOTAL_SLOTS; slot++)); do
        exec {slot_fd}>>"${lock_dir}/cpu_slot_${slot}.lock"
        if try_lock "${slot_fd}" -x -n; then
            held+=("${slot_fd}")
            [ "${#held[@]}" -lt "${want}" ] || break
        else
            exec {slot_fd}>&-
        fi
    done
    [ "${#held[@]}" -lt "${want}" ] || break
    for slot_fd in "${held[@]}"; do
        exec {slot_fd}>&-
    done
    sleep 0.2
done

exec {gate_fd}>&-
trap - HUP INT TERM
export TMLM_CPU_SLOT_FDS="${held[*]}"
run_with_caps "$@"
