#!/bin/bash
# lock_held.sh <file>: exit 0 when another open file description holds an exclusive flock on <file>, else 1
# (also 1 when the file does not exist). The probe opens the file read-only, so it never creates it, and asks for a
# shared lock without waiting; a shared lock it gets is released when this script exits. Amendment 1: run_job.sh uses
# it to check that a real A job runs under the Part 3 chain lock (chain/part3_cpu_job.sh).
[ -f "${1:-}" ] || exit 1
exec 8<"$1" || exit 1
flock -n -s -E 75 8
rc=$?
[ "$rc" = "75" ] && exit 0
exit 1
