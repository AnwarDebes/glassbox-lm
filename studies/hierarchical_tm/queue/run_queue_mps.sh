#!/usr/bin/env bash
# MPS queue: at most 2 GPU jobs at once under NVIDIA MPS (study rule 4, changed on 2 Oct 2026 after the sharing test).
# jobs.txt lines: NAME|MINUTES|ARGS   ('#' lines are comments)
#   ARGS are htm_driver.py args, or '@DRIVER.py args' to run another driver from src/ (e.g. '@text_driver.py --data ...').
#   Every driver must accept --max-minutes.
# Whenever a slot is free, the FIRST not-yet-started line starts, so a line inserted above pending ones gets priority.
# Each job: MPS env -> timeout (3x cap + 10 min: last resort only) -> mem_guard 12 GiB -> run_logged.py ->
#           gpu_run.sh shared -> DRIVER ARGS --max-minutes CAP   (under MPS jobs stop only through --max-minutes)
# PAUSE: create queue/STOP. Running jobs finish; nothing new starts. The runner stops the MPS daemon when it exits.
# Start it only when no queue job is running (checked from queue.log) and no MPS daemon is up.
# Interpreters (changed on 2026-10-25, bug HTM-B1): jobs run under $HTM_PY, and mem_guard.py and run_logged.py are
# started through $HTM_TOOL_PY instead of by name; src/interpreters.sh sets both, before the MPS daemon starts.
# Path variables are quoted since the same date (HTM-B5); $args stays unquoted, because it is split into arguments.
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$STUDY/src/interpreters.sh"
Q="$STUDY/queue"; H="$STUDY"; SLOTS=2
MPS="$Q/mps"; mkdir -p "$MPS/pipe" "$MPS/log"
export CUDA_MPS_PIPE_DIRECTORY="$MPS/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS/log"
touch "$Q/done.txt"
[ -f "$Q/started.txt" ] || cp "$Q/done.txt" "$Q/started.txt"   # never re-run jobs the serial runner already did
log() { echo "$(date -u +%FT%TZ) $*" >> "$Q/queue.log"; }
open=$(awk '/ START /{s[$3]=1} / END /{delete s[$3]} END{for (k in s) print k}' "$Q/queue.log")
if [ -n "$open" ]; then log "runner not started: queue.log shows jobs still running: $(echo $open)"; exit 1; fi
if nvidia-cuda-mps-control -d; then log "MPS daemon started"; else log "MPS daemon could not start; runner exits"; exit 1; fi
run_job() {  # name cap args
  local name=$1 cap=$2 args=$3 drv="$H/src/htm_driver.py" first cfg out rc rd fin stop
  first=${args%% *}
  if [ "${first:0:1}" = "@" ]; then drv="$H/src/${first:1}"; args=${args#"$first"}; fi
  if [ ! -f "$drv" ]; then
    log "START $name (driver missing)"; log "END $name exit=127 driver not found: $drv"; echo "$name" >> "$Q/done.txt"; return
  fi
  cfg=$(mktemp "$Q/cfg_XXXX.json")
  echo "{\"name\": \"$name\", \"timeout_min\": $cap, \"driver\": \"$(basename "$drv")\", \"argv\": \"$args\", \"launcher\": \"queue/run_queue_mps.sh (MPS, 2 slots)\", \"job_python\": \"$HTM_PY\"}" > "$cfg"
  log "START $name (cap $cap min, MPS, $(basename "$drv"))"
  out=$(cd "$STUDY/../tm_language_model" && source env/env.sh >/dev/null 2>&1 && \
        timeout --kill-after=60 "$((3 * cap + 10))m" "$HTM_TOOL_PY" scripts/mem_guard.py --scope group --limit-gib 12 -- \
        "$HTM_TOOL_PY" scripts/run_logged.py --name "$name" --config "$cfg" --results-dir "$H/results" -- \
        scripts/gpu_run.sh shared -- "$HTM_PY" -u "$drv" $args --max-minutes "$cap" 2>&1); rc=$?
  rd=$(echo "$out" | grep -o 'run dir: .*' | head -1 | sed 's/run dir: //')
  fin=$(echo "$out" | grep '^FINAL' | grep -o '"dev_acc": [0-9.]*\|"dev_auc": [0-9.]*\|"dev_macro_f1": [0-9.]*\|"partner_shuffle_auc": [0-9.]*' | tr '\n' ' ')
  stop=$(echo "$out" | grep -o 'STOP: [^"]*' | head -1)
  log "END $name exit=$rc $fin${stop:+[$stop] }dir=$(basename "$rd")"
  echo "$name" >> "$Q/done.txt"
  rm -f "$cfg"
}
while true; do
  running=$(jobs -rp | wc -l)
  if [ ! -f "$Q/STOP" ] && [ "$running" -lt "$SLOTS" ]; then
    job=$(grep -v '^\s*#' "$Q/jobs.txt" | grep -v '^\s*$' | while IFS= read -r l; do n=${l%%|*}; grep -qxF "$n" "$Q/started.txt" || { echo "$l"; break; }; done)
    if [ -n "$job" ]; then
      name=${job%%|*}; rest=${job#*|}; cap=${rest%%|*}; args=${rest#*|}
      echo "$name" >> "$Q/started.txt"
      run_job "$name" "$cap" "$args" &
      sleep 20   # stagger starts (kernel compile, data loading)
      continue
    fi
    if [ "$running" -eq 0 ]; then log "QUEUE EMPTY"; break; fi
  fi
  sleep 15
done
wait
echo quit | nvidia-cuda-mps-control && log "MPS daemon stopped"
