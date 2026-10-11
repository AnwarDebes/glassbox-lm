#!/usr/bin/env bash
# GPU lane lock (design decision). One GPU, two kinds of jobs:
#
#   gpu_run.sh shared    -- <cmd...>   training and correctness runs, may overlap
#   gpu_run.sh exclusive -- <cmd...>   timing, energy, profiling: alone on the GPU
#
# Mechanism: flock(2) on $LOCK_DIR/gpu.lock through a file descriptor that the wrapped
# command INHERITS (this script ends with `exec`). Consequences:
#   - The kernel drops the lock when the command and all children that inherited the
#     descriptor are gone: normal exit, crash, SIGKILL, all the same. No stale locks.
#   - The command's exit code is the exit code, and signals reach it directly, because
#     it replaces this shell.
#
# Exclusive requests: while an exclusive job waits, $LOCK_DIR/gpu_exclusive_request
# exists. New shared jobs queue behind it (flock alone gives writers no priority), and
# running shared jobs can yield at a checkpoint: see src/tmlm/common/locks.py. The file
# is removed once every waiting exclusive job has acquired the lock. Each waiter also
# has a marker $LOCK_DIR/gpu_exclusive_waiters/<pid>. Markers of dead waiters (killed
# with SIGKILL before cleanup, zombies included) are pruned by whoever looks next, from
# this script and from tmlm.common.locks alike.
#
# Nesting: a command that already runs under gpu_run.sh and calls it again would
# deadlock on its own lock. So a nested call runs the command directly when the lock
# already held is strong enough, and fails with exit 2 for exclusive inside shared.
# A nested call is recognized by its inherited descriptor TMLM_GPU_LOCK_FD. When a
# wrapper in between closed that descriptor (Python's subprocess does, unless the
# wrapper passes pass_fds=tmlm.common.locks.inherited_lock_fds()), TMLM_GPU_LOCK_MODE is
# still set: the call is then nested exactly when an ANCESTOR process has gpu.lock
# open, and the same rules apply. With no such ancestor the variables are stale (a job
# that outlived its holder) and the call is an ordinary top-level request.
#
# Exported to the command: TMLM_GPU_LOCK_MODE, TMLM_GPU_LOCK_FD, TMLM_LOCK_DIR.
# Environment: TMLM_LOCK_DIR overrides the lock directory (default results/.locks).
# Exit codes of this script itself: 2 usage, 1 lock failure, 129/130/143 on HUP/INT/TERM
# while still waiting.

set -euo pipefail

usage() {
    echo "usage: gpu_run.sh shared|exclusive -- <command...>" >&2
    exit 2
}

[ "$#" -ge 3 ] || usage
mode="$1"
[ "$2" = "--" ] || usage
shift 2
case "${mode}" in
    shared | exclusive) ;;
    *) usage ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${TMLM_ROOT:-$(cd "${script_dir}/.." && pwd)}"
lock_dir="${TMLM_LOCK_DIR:-${root}/results/.locks}"
mkdir -p "${lock_dir}/gpu_exclusive_waiters"
lock_dir="$(cd "${lock_dir}" && pwd -P)"
lock_file="${lock_dir}/gpu.lock"
mutex_file="${lock_dir}/gpu_request.mutex"
request_flag="${lock_dir}/gpu_exclusive_request"
waiters_dir="${lock_dir}/gpu_exclusive_waiters"
export TMLM_LOCK_DIR="${lock_dir}"

# shellcheck source=lib_locks.sh
source "${script_dir}/lib_locks.sh"

# ---- nested call: the lock we would wait for is held by our own ancestors.
outer_fd="${TMLM_GPU_LOCK_FD:-}"
outer_mode="${TMLM_GPU_LOCK_MODE:-}"
nested=0
if [[ "${outer_fd}" =~ ^[0-9]+$ ]] && [ -e "/proc/$$/fd/${outer_fd}" ] &&
    [ "$(readlink "/proc/$$/fd/${outer_fd}")" = "${lock_file}" ]; then
    nested=1
elif [ "${outer_mode}" = "shared" ] || [ "${outer_mode}" = "exclusive" ]; then
    if tmlm_ancestor_holds exact "${lock_file}"; then
        nested=1
        echo "gpu_run.sh: note: the ${outer_mode} GPU lock descriptor was not inherited" \
            "(closed by a wrapper), an ancestor still holds the lock" >&2
        # The number now means nothing here, or names an unrelated file.
        unset TMLM_GPU_LOCK_FD
    fi
fi
if [ "${nested}" -eq 1 ]; then
    if [ "${mode}" = "shared" ] || [ "${outer_mode}" = "exclusive" ]; then
        exec "$@"
    fi
    echo "gpu_run.sh: exclusive requested inside a shared GPU lock: this would deadlock" >&2
    exit 2
fi

# ---- exclusive request bookkeeping, serialized by a small mutex.
# sync_requests: prune markers of dead waiters, then make the flag match the markers.
sync_requests() {
    local marker pid alive=0
    for marker in "${waiters_dir}"/*; do
        [ -e "${marker}" ] || continue
        pid="${marker##*/}"
        if tmlm_pid_alive "${pid}"; then
            alive=1
        else
            rm -f -- "${marker}"
        fi
    done
    if [ "${alive}" -eq 1 ]; then
        : >"${request_flag}"
    else
        rm -f -- "${request_flag}"
    fi
}

# with_mutex <add|drop|sync>: update this process's marker and the flag atomically.
with_mutex() {
    (
        flock -x 200
        case "$1" in
            add) : >"${waiters_dir}/$$" ;;
            drop) rm -f -- "${waiters_dir}/$$" ;;
            sync) ;;
        esac
        sync_requests
    ) 200>>"${mutex_file}"
}

registered=0
cleanup() {
    if [ "${registered}" -eq 1 ]; then
        registered=0
        with_mutex drop || true
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# acquire <-s|-x>: blocking flock in slices of 1 s, so pending traps run promptly.
acquire() {
    local rc
    while :; do
        rc=0
        flock -E 75 -w 1 "$1" "${lock_fd}" || rc=$?
        case "${rc}" in
            0) return 0 ;;
            75) ;;
            *)
                echo "gpu_run.sh: flock failed with exit ${rc}" >&2
                exit 1
                ;;
        esac
    done
}

exec {lock_fd}>>"${lock_file}"

if [ "${mode}" = "exclusive" ]; then
    registered=1
    with_mutex add
    acquire -x
    cleanup
else
    # Writer priority: do not start a shared job while an exclusive job is waiting.
    while [ -e "${request_flag}" ]; do
        with_mutex sync
        [ -e "${request_flag}" ] || break
        sleep 0.5
    done
    acquire -s
fi

export TMLM_GPU_LOCK_MODE="${mode}"
export TMLM_GPU_LOCK_FD="${lock_fd}"
trap - EXIT HUP INT TERM
exec "$@"
