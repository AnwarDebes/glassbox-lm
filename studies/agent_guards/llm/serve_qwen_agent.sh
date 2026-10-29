#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# The local Qwen for the agent runs (E2 AgentDojo, E4 tau2-bench): llama-server b11742 with Qwen3.5-9B Q8_0.
# It is the ONE shared Qwen instance. It is started only in a GPU
# slot of the queue: one big GPU job at a time, never beside a TM job or a second llama-server (sharing OFF since 01:45Z).
#
#   serve_qwen_agent.sh print  [options]   print the command; nothing is started or written
#   serve_qwen_agent.sh start  [options]   checks, start, wait for /health, check /v1/models and /props; the server
#                                          keeps running (in its own session) after the script returns
#   serve_qwen_agent.sh status
#   serve_qwen_agent.sh stop               stops only the server this script started (pid file + command line)
#
# Options (defaults = E2's settings, decided 03:39Z):
#   --np N          slots, 1-4 (default 1; E4 may take 2)
#   --ctx-slot T    context per slot, 32768-262144, a multiple of 1024 (default 65536). With -np set, kv_unified is
#                   off and each slot gets -c / N, so the script passes -c = N * T.
#   --port P        default 18082; never 18081 (3c's registered chain), so a server left running here can never
#                   answer 3c's health check
#   --alias A       default qwen3.5-9b-q8_0 (the id E2's slot script expects; E4 takes the first id listed)
#   --reasoning R   off (default since 04:33Z) or auto. off adds -rea off: every request renders with the
#                   template's thinking off, also one that sends no chat_template_kwargs.enable_thinking (E2's upstream
#                   D1 filter call). auto: such a request renders with thinking on ('<think>\n') and
#                   --reasoning-budget 0 ends the thinking at once (CPU smoke, 2026-10-21).
#   --note TEXT     who gave the go, for serve.log (free text)
# Fixed: --host 127.0.0.1, --jinja, --reasoning-budget 0, -fa off (sm_70; never -fa with a quantized KV cache, so the
# KV cache stays f16), -ngl 99, -t 2 (the 2-thread rule; every op of this model has a CUDA kernel), --no-webui,
# --metrics; the rest at b11742's defaults (32 context checkpoints per slot, 8 GiB host prompt cache, timeout 3600 s).
# The server runs at nice 19 with OMP_NUM_THREADS=2. Every start first checks: no live server of ours, no other
# llama-server, the port free, no GPU compute process, the CPU gate (cpu_load.py 10 at most 4.5 cores), the build's
# hashes (build_b11742_bin.sha256, itself pinned) and the Q8_0 file's hash (pinned; FILES.sha256 must list the same).
# After the start, /props must report the pinned build_info b1-4c3c581 (commit 4c3c5810 = b11742; a shallow clone
# has no build count). The pins are below (decided 04:33Z).
# State in llm/serve/: current.pid, current.env, serve.log (every start, check and stop, UTC), and
# runs/<stamp>_p<port>_<script pid>/ (cmd.txt, server.txt, models.json, props.json, meta.json).
# Test hooks (tests only): SERVE_DIR, SERVE_BIN, SERVE_NVSMI, SERVE_LOAD (a command printing a cpu_load.py line),
# SERVE_GATE_TRIES, SERVE_GATE_SLEEP, SERVE_HEALTH_SECS, SERVE_HASH_LIST, SERVE_LIST_SHA, SERVE_GGUF_DIR, SERVE_Q8_SHA,
# SERVE_SKIP_HASH=1, and
# SERVE_CPU_TEST=1 (CUDA_VISIBLE_DEVICES= and -ngl 0: the GPU is never touched, so its free check is skipped; the CPU
# gate is then 4.0 cores, the limit for a 2-thread CPU job).
set -u
L=${REPO_ROOT}/studies/agent_guards/llm
B=${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin
G=${SERVE_GGUF_DIR:-${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742}
M=$G/Qwen3.5-9B-Q8_0.gguf
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
D=${SERVE_DIR:-$L/serve}
BIN=${SERVE_BIN:-$B/llama-server}
NVSMI=${SERVE_NVSMI:-nvidia-smi}
HASH_LIST=${SERVE_HASH_LIST:-$L/build_b11742_bin.sha256}
LIST_SHA=${SERVE_LIST_SHA:-5d9d8e33161234731c72f0cce7e449939420073eadb691244ed8b70ef06d9e2d}  # pinned: build_b11742_bin.sha256
Q8_SHA=${SERVE_Q8_SHA:-ff3d8ff122c21c37f2ca52502db705e36903c581657f1e8d3e930a9cfb4d50aa}  # pinned: Qwen3.5-9B-Q8_0.gguf
BUILD_INFO=b1-4c3c581  # pinned: /props build_info of commit 4c3c5810f87efe177949f1b85ae509d38a2a311f (b11742)
read -r -a LOADCMD <<< "${SERVE_LOAD:-$PY -I ${REPO_ROOT}/studies/decision_layer/scripts/cpu_load.py 10}"
CPU_TEST=${SERVE_CPU_TEST:-0}
GATE_MAX=4.5; [ "$CPU_TEST" = 1 ] && GATE_MAX=4.0
GATE_TRIES=${SERVE_GATE_TRIES:-10}
GATE_SLEEP=${SERVE_GATE_SLEEP:-60}
HEALTH_SECS=${SERVE_HEALTH_SECS:-900}
PIDF=$D/current.pid
CURF=$D/current.env
LEDGER=$D/serve.log
ACT=${1:-}
[ $# -gt 0 ] && shift

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(ts) $*" | tee -a "$LEDGER"; }
die() { local code=$1; shift
  echo "STOP: $*" >&2
  if [ "$ACT" != print ] && [ -d "$D" ]; then echo "$(ts) STOP ($ACT, exit $code): $*" >> "$LEDGER"; fi
  exit "$code"; }
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
http() { curl -s --noproxy '*' -m "${2:-5}" "http://127.0.0.1:$PORT$1" 2>/dev/null; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
gpu_line() { "$NVSMI" --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -c 120; }
cur() { sed -n "s/^$1=//p" "$CURF" 2>/dev/null; }

NP=1; CTX=65536; PORT=18082; ALIAS=qwen3.5-9b-q8_0; REA=off; NOTE=""
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die 2 "missing value for $1"
  case $1 in
    --np) NP=$2 ;;
    --ctx-slot) CTX=$2 ;;
    --port) PORT=$2 ;;
    --alias) ALIAS=$2 ;;
    --reasoning) REA=$2 ;;
    --note) NOTE=$2 ;;
    *) die 2 "unknown option $1" ;;
  esac
  shift 2
done
case $ACT in
  print|start|status|stop) ;;
  *) echo "usage: $0 print|start|status|stop [--np N] [--ctx-slot T] [--port P] [--alias A] [--reasoning auto|off] [--note TEXT]" >&2; exit 2 ;;
esac
[[ $NP =~ ^[1-4]$ ]] || die 2 "--np must be 1-4 (got '$NP')"
if ! [[ $CTX =~ ^[0-9]+$ ]] || [ "$CTX" -lt 32768 ] || [ "$CTX" -gt 262144 ] || [ $((CTX % 1024)) -ne 0 ]; then
  die 2 "--ctx-slot must be 32768-262144 and a multiple of 1024 (got '$CTX')"
fi
if ! [[ $PORT =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then die 2 "--port must be 1024-65535 (got '$PORT')"; fi
[ "$PORT" != 18081 ] || die 2 "port 18081 belongs to 3c's registered chain; take another"
[[ $ALIAS =~ ^[A-Za-z0-9._:-]+$ ]] || die 2 "--alias must be one id of letters, digits and ._:- (got '$ALIAS')"
case $REA in auto|off) ;; *) die 2 "--reasoning must be auto or off (got '$REA')" ;; esac

NGL=99; [ "$CPU_TEST" = 1 ] && NGL=0
ARGS=(-m "$M" --host 127.0.0.1 --port "$PORT" -a "$ALIAS" --jinja --reasoning-budget 0 -fa off -np "$NP" -c "$((NP * CTX))"
      -ngl "$NGL" -t 2 --no-webui --metrics)
[ "$REA" = off ] && ARGS+=(-rea off)
ENVV=(OMP_NUM_THREADS=2 MKL_NUM_THREADS=2)
[ "$CPU_TEST" = 1 ] && ENVV+=(CUDA_VISIBLE_DEVICES=)
CMD=(nice -n 19 env "${ENVV[@]}" "$BIN" "${ARGS[@]}")

is_ours() { local pid=$1 cl
  alive "$pid" || return 1
  cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  [[ $cl == *"$BIN "* && $cl == *" --port $PORT "* ]]; }

stop_pid() { local pid=$1 i
  is_ours "$pid" || die 8 "pid $pid is not the llama-server this script started on port $PORT; nothing signalled"
  kill -TERM "$pid" 2>/dev/null
  for ((i = 0; i < 60; i++)); do alive "$pid" || return 0; sleep 1; done
  say "pid $pid still running 60 s after SIGTERM: SIGKILL"
  kill -KILL "$pid" 2>/dev/null
  for ((i = 0; i < 10; i++)); do alive "$pid" || return 0; sleep 1; done
  die 8 "pid $pid did not exit"; }

gate() { local i line c
  for ((i = 1; i <= GATE_TRIES; i++)); do
    line=$("${LOADCMD[@]}" 2>&1); c=$(echo "$line" | awk '{print $3}')
    if awk -v c="$c" -v m="$GATE_MAX" 'BEGIN { exit !(c ~ /^[0-9]+(\.[0-9]+)?$/ && c + 0 <= m + 0) }'; then
      say "CPU gate ok (at most $GATE_MAX cores): $line"; return 0
    fi
    say "CPU gate, try $i of $GATE_TRIES (at most $GATE_MAX cores): $line"
    [ "$i" -lt "$GATE_TRIES" ] && sleep "$GATE_SLEEP"
  done
  die 4 "the CPU gate did not pass in $GATE_TRIES tries"; }

hashes() { local out line
  if [ "${SERVE_SKIP_HASH:-0}" = 1 ]; then say "hash checks skipped (test hook)"; return 0; fi
  [ "$(sha256sum "$HASH_LIST" | cut -d' ' -f1)" = "$LIST_SHA" ] || die 5 "$HASH_LIST is not the pinned list (sha256 $LIST_SHA)"
  out=$(cd "$B" && sha256sum -c --quiet "$HASH_LIST" 2>&1) || die 5 "the b11742 build does not match $HASH_LIST: $out"
  line=$(grep ' Qwen3.5-9B-Q8_0.gguf$' "$G/FILES.sha256")
  [ "${line%% *}" = "$Q8_SHA" ] || die 5 "$G/FILES.sha256 does not list the pinned Q8_0 sha256 $Q8_SHA (it lists '${line%% *}')"
  out=$(cd "$G" && echo "$line" | sha256sum -c --quiet - 2>&1) || die 5 "the Q8_0 GGUF does not match $G/FILES.sha256: $out"
  say "hashes ok: the pinned list ${LIST_SHA:0:16}, $(grep -c '' "$HASH_LIST") build files, the pinned Q8_0 ${Q8_SHA:0:16}"; }

start() { local old others apps stamp run pid i h t0 t_start chk
  mkdir -p "$D/runs" || exit 1
  if [ -e "$PIDF" ]; then
    old=$(cat "$PIDF")
    alive "$old" && die 3 "the server this script started is still running (pid $old, port $(cur PORT)); stop it first"
    say "stale pid file removed (pid $old is not running)"; rm -f "$PIDF" "$CURF"
  fi
  others=$(pgrep -x llama-server | tr '\n' ' ')
  [ -z "$others" ] || die 3 "a llama-server is already running (pids $others): one shared Qwen instance only"
  port_open "$PORT" && die 3 "port $PORT is in use"
  if [ "$CPU_TEST" = 1 ]; then
    say "CPU test: CUDA_VISIBLE_DEVICES= and -ngl 0, so the GPU is not used and its free check is skipped"
  else
    apps=$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1) || die 3 "nvidia-smi failed: $apps"
    [ -z "$apps" ] || die 3 "the GPU has compute processes (host PIDs): $(echo "$apps" | tr '\n' ';') - no sharing with other GPU jobs (decided 01:45Z)"
    say "GPU free: no compute process; $(gpu_line)"
  fi
  gate
  hashes
  stamp=$(date -u +%Y%m%dT%H%M%SZ); run=$D/runs/${stamp}_p${PORT}_$$
  mkdir "$run" || die 6 "run folder $run exists (never overwritten)"
  printf '%q ' "${CMD[@]}" > "$run/cmd.txt"; echo >> "$run/cmd.txt"
  t_start=$(ts)
  say "start (note: ${NOTE:-none}): $(cat "$run/cmd.txt")"
  # shellcheck disable=SC2016  # $$ and $0 expand in the inner bash: it writes its own pid, then becomes the server
  ( cd /tmp && exec setsid bash -c 'echo $$ > "$0"; exec "$@"' "$run/pid" "${CMD[@]}" > "$run/server.txt" 2>&1 < /dev/null ) &
  for ((i = 0; i < 50; i++)); do [ -s "$run/pid" ] && break; sleep 0.2; done
  pid=$(cat "$run/pid" 2>/dev/null)
  alive "$pid" || die 6 "the server did not start (see $run/server.txt): $(tail -c 600 "$run/server.txt")"
  echo "$pid" > "$PIDF"
  printf 'PID=%s\nPORT=%s\nALIAS=%s\nNP=%s\nCTX_SLOT=%s\nREASONING=%s\nCPU_TEST=%s\nRUN=%s\n' \
    "$pid" "$PORT" "$ALIAS" "$NP" "$CTX" "$REA" "$CPU_TEST" "$run" > "$CURF"
  t0=$SECONDS
  while true; do
    alive "$pid" || { rm -f "$PIDF" "$CURF"; die 6 "the server exited while loading (see $run/server.txt): $(tail -c 800 "$run/server.txt")"; }
    h=$(http /health 2)
    echo "$h" | grep -Eq '"status" *: *"ok"' && break
    if (( SECONDS - t0 > HEALTH_SECS )); then
      stop_pid "$pid"; rm -f "$PIDF" "$CURF"
      die 6 "no healthy /health after $HEALTH_SECS s (last answer: ${h:-none}); the server was stopped"
    fi
    sleep 3
  done
  say "health ok after $((SECONDS - t0)) s: $h"
  http /v1/models > "$run/models.json"
  http /props 10 > "$run/props.json"
  chk=$("$PY" -I - "$run/models.json" "$run/props.json" "$ALIAS" "$NP" "$CTX" "$BUILD_INFO" <<'EOF'
import json, sys
models, props, alias, n_slots, ctx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
build = sys.argv[6]
try:
    m, p = json.load(open(models)), json.load(open(props))
except ValueError as e:
    sys.exit(print(f"BAD: /v1/models or /props is not JSON ({e})"))
ids = [d.get("id") for d in m.get("data", [])]
slots, n_ctx = p.get("total_slots"), p.get("default_generation_settings", {}).get("n_ctx")
bad = [f"/v1/models lists {ids}, not {alias}"] * (alias not in ids)
bad += [f"total_slots {slots} != {n_slots}"] * (slots != n_slots) + [f"n_ctx per slot {n_ctx} != {ctx}"] * (n_ctx != ctx)
bad += [f"build_info {p.get('build_info')} != the pinned {build}"] * (p.get("build_info") != build)
print("BAD: " + "; ".join(bad) if bad else f"ok: ids {ids}, total_slots {slots}, n_ctx per slot {n_ctx}, build {p.get('build_info')}")
EOF
)
  case $chk in
    ok:*) say "props $chk" ;;
    *) stop_pid "$pid"; rm -f "$PIDF" "$CURF"; die 7 "${chk:-no props check output}; the server was stopped" ;;
  esac
  "$PY" -I -c 'import json, sys; a = sys.argv[2:]; json.dump(dict(zip(a[::2], a[1::2])), open(sys.argv[1], "w"), indent=1)' "$run/meta.json" \
    started "$t_start" ready "$(ts)" pid "$pid" port "$PORT" alias "$ALIAS" np "$NP" ctx_slot "$CTX" n_ctx_total "$((NP * CTX))" \
    reasoning "$REA" cpu_test "$CPU_TEST" note "$NOTE" cmd "$(cat "$run/cmd.txt")" bin "$BIN" bin_sha256 "$(sha256sum "$BIN" | cut -d' ' -f1)" \
    model "$M" model_sha256 "$(grep ' Qwen3.5-9B-Q8_0.gguf$' "$G/FILES.sha256" | cut -d' ' -f1)" health_seconds "$((SECONDS - t0))" \
    props_check "$chk" gpu_after "$(gpu_line)" gpu_apps_after "$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')"
  say "READY pid $pid port $PORT alias $ALIAS np $NP ctx_slot $CTX reasoning $REA cpu_test $CPU_TEST run $run"; }

stop() { local pid run t0 others
  mkdir -p "$D" || exit 1
  if [ ! -e "$PIDF" ]; then
    echo "no server started by this script is running (no pid file)"
    others=$(pgrep -x llama-server | tr '\n' ' ')
    [ -z "$others" ] || echo "note: llama-server pids $others run without our pid file; not touched"
    return 0
  fi
  pid=$(cat "$PIDF"); PORT=$(cur PORT); run=$(cur RUN)
  if ! alive "$pid"; then say "pid $pid (port $PORT) was not running; stale pid file removed"; rm -f "$PIDF" "$CURF"; return 0; fi
  t0=$SECONDS
  stop_pid "$pid"
  rm -f "$PIDF" "$CURF"
  say "STOPPED pid $pid port $PORT after $((SECONDS - t0)) s; run $run; GPU after: $(gpu_line)"
  port_open "$PORT" && die 8 "port $PORT still answers after the stop"
  return 0; }

status() { local pid
  pid=$(cat "$PIDF" 2>/dev/null)
  if alive "$pid"; then
    PORT=$(cur PORT)
    echo "running: $(tr '\n' ' ' < "$CURF")"
    echo "health: $(http /health 2)"
    echo "props: $(http /props 10 | "$PY" -I -c 'import json, sys; p = json.load(sys.stdin); print("total_slots", p.get("total_slots"), "n_ctx per slot", p.get("default_generation_settings", {}).get("n_ctx"), "build", p.get("build_info"))' 2>&1)"
  else
    echo "not running (no live pid file)"
  fi
  echo "llama-server processes: $(pgrep -x llama-server | tr '\n' ' ')"
  echo "GPU: $(gpu_line); compute apps: $("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')"; }

case $ACT in
  print) printf '%q ' "${CMD[@]}"; echo ;;
  start) start ;;
  stop) stop ;;
  status) status ;;
esac
