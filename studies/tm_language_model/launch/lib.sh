# Shared helpers for the A7 chains (sourced). Recovered 2026-09-28 from the saved record (the /tmp original was lost
# in the host reboot of 2026-09-27); text identical to the 2026-09-25 08:12 UTC version except for the study folder, which is now derived from this file's location. Each chain runs inside one CPU slot (cpu_slot.sh 1 -- bash chain.sh).
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -u
cd "$LAUNCH_DIR/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1
source env/env.sh >/dev/null 2>&1
PY=.venv-tm/bin/python
headroom() {  # wait until memory.current + need_gib < 0.80 * memory.max (at most 60 min)
  local need=$1 i=0 cur max
  while :; do
    # non-reclaimable memory only: anon + shmem + kernel + mapped file pages (clean page cache is reclaimable and was
    # stalling chains: 2026-09-25 59 GB of cache against 1.6 GB of anon)
    cur=$(awk '$1=="anon"||$1=="shmem"||$1=="kernel"||$1=="file_mapped"{s+=$2} END{printf "%d", s}' /sys/fs/cgroup/memory.stat); max=$(cat /sys/fs/cgroup/memory.max)
    if [ $(( cur + need * 1073741824 )) -lt $(( max * 8 / 10 )) ]; then return 0; fi
    i=$((i + 1)); [ $i -gt 120 ] && { echo "HEADROOM not reached in 60 min (need $need GiB)"; return 1; }
    sleep 30
  done
}
latest() { ls -td results/*_"$1" 2>/dev/null | head -1; }
step() {  # step NAME CFG LIMIT_GIB GPU(0|1) -- command...
  local name=$1 cfg=$2 lim=$3 gpu=$4; shift 5
  headroom $(( lim < 30 ? lim : 30 )) || return 1
  echo "STEP $name start $(date -u +%FT%TZ)"
  if [ "$gpu" = 1 ]; then
    scripts/gpu_run.sh shared -- nice -n 19 $PY scripts/mem_guard.py --scope group --limit-gib "$lim" -- scripts/run_logged.py --name "$name" --config "$cfg" -- "$@"
  else
    nice -n 19 $PY scripts/mem_guard.py --scope group --limit-gib "$lim" -- scripts/run_logged.py --no-gpu-sampler --name "$name" --config "$cfg" -- "$@"
  fi
  local rc=$?; echo "STEP $name exit $rc $(date -u +%FT%TZ) dir $(latest "$name")"; return $rc
}
