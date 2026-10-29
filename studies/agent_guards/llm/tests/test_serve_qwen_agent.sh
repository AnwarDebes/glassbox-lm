#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
WS_ROOT="$(dirname "$REPO_ROOT")"  # the folder that holds the repository and tools/
# shellcheck disable=SC2015
# Unit tests for serve_qwen_agent.sh with stubs (no model, no GPU, no real server): print, the option checks,
# start/status/stop with a stub server, and every refusal. Ports 18096 and 18097 only; state in tests/runs/unit_<stamp>/.
# Run: bash tests/test_serve_qwen_agent.sh
set -u
L=${REPO_ROOT}/studies/agent_guards/llm
S=${SERVE_SCRIPT:-$L/serve_qwen_agent.sh}  # SERVE_SCRIPT: a mutant copy, for tests/mutation_check.sh only
T=$L/tests
PY=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
export PATH="$(dirname "$PY"):$PATH"  # the stub server runs as "env python3"
W=$T/runs/unit_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$W" || exit 1
P=18097
pass=0; fail=0
ok() { pass=$((pass + 1)); echo "PASS $1"; }
bad() { fail=$((fail + 1)); echo "FAIL $1"; }
expect() { if [ "$3" = "$2" ]; then ok "$1 (exit $3)"; else bad "$1: exit $3, wanted $2"; fi; }
has() { case "$2" in *"$3"*) ok "$1" ;; *) bad "$1: '$3' not in: $2" ;; esac; }
hasnt() { case "$2" in *"$3"*) bad "$1: '$3' found in: $2" ;; *) ok "$1" ;; esac; }
servers() { pgrep -x llama-server | wc -l; }
none_left() { if [ "$(servers)" = 0 ] && [ ! -e "$SERVE_DIR/current.pid" ]; then ok "$1: no server and no pid file left"; else bad "$1: servers $(servers), pid file $(ls "$SERVE_DIR"/current.pid 2>/dev/null)"; fi; }
stubenv() {
  export SERVE_DIR=$W/$1 SERVE_BIN=$T/stub/llama-server SERVE_NVSMI=$T/stub/nvidia-smi SERVE_LOAD="$T/stub/cpu_load 1.00" \
    SERVE_SKIP_HASH=1 SERVE_GATE_TRIES=2 SERVE_GATE_SLEEP=0 SERVE_HEALTH_SECS=20
  unset SERVE_CPU_TEST SERVE_HASH_LIST SERVE_GGUF_DIR STUB_APPS STUB_DIE STUB_DELAY STUB_NCTX STUB_IDS STUB_NVSMI_FAIL; }
trap 'pkill -f "$T/stub/llama-server" 2>/dev/null' EXIT
[ "$(servers)" = 0 ] || { echo "a llama-server is running; the tests need none"; exit 1; }

echo "== print and the option checks"
stubenv print
out=$(bash "$S" print); expect "print" 0 $?
for pat in "nice -n 19 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $T/stub/llama-server " \
    " -m ${WS_ROOT}/tools/models_untrusted/qwen3.5-9b/gguf-b11742/Qwen3.5-9B-Q8_0.gguf " " --host 127.0.0.1 " \
    " --port 18082 " " -a qwen3.5-9b-q8_0 " " --jinja " " --reasoning-budget 0 " " -fa off " " -np 1 " " -c 65536 " \
    " -ngl 99 " " -t 2 " " --no-webui " " --metrics " " -rea off "; do
  has "print has '$pat'" "$out " "$pat"
done
for pat in "CUDA_VISIBLE_DEVICES" " -ctk" " -ctv" " -kvu" " --kv-unified" " -ngl 0 "; do hasnt "print lacks '$pat'" "$out " "$pat"; done
[ ! -e "$SERVE_DIR" ] && ok "print writes nothing" || bad "print created $SERVE_DIR"
out=$(env -u SERVE_BIN bash "$S" print); has "the real binary by default" "$out" "${WS_ROOT}/tools/llama.cpp/b11742/build-cuda129/bin/llama-server "
out=$(bash "$S" print --np 2); has "np 2" "$out " " -np 2 "; has "np 2 doubles -c" "$out " " -c 131072 "
out=$(bash "$S" print --np 2 --ctx-slot 32768); has "np 2 x 32768" "$out " " -c 65536 "
out=$(bash "$S" print --reasoning auto); hasnt "reasoning auto has no -rea" "$out " " -rea "; has "auto keeps the budget" "$out " " --reasoning-budget 0 "
out=$(bash "$S" print --reasoning off); has "reasoning off: -rea off" "$out " " -rea off "
out=$(SERVE_CPU_TEST=1 bash "$S" print); has "CPU test: -ngl 0" "$out " " -ngl 0 "; has "CPU test: no GPU" "$out " " CUDA_VISIBLE_DEVICES= "
for a in "--np 0" "--np 5" "--np x" "--ctx-slot 16384" "--ctx-slot 32769" "--ctx-slot 300000" "--ctx-slot abc" "--port 18081" \
    "--port 80" "--port 70000" "--port abc" "--reasoning on" "--foo 1" "--np"; do
  # shellcheck disable=SC2086
  bash "$S" print $a > /dev/null 2>&1; expect "refused: print $a" 2 $?
done
bash "$S" print --alias "a b" > /dev/null 2>&1; expect "refused: an alias with a space" 2 $?
bash "$S" print --alias "a,b" > /dev/null 2>&1; expect "refused: two aliases" 2 $?
bash "$S" print --alias "" > /dev/null 2>&1; expect "refused: an empty alias" 2 $?
bash "$S" begin > /dev/null 2>&1; expect "refused: an unknown action" 2 $?
bash "$S" > /dev/null 2>&1; expect "refused: no action" 2 $?

echo "== start, status and stop with the stub"
stubenv flow
out=$(bash "$S" start --port $P --note "unit test" 2>&1); expect "start" 0 $?
has "start says READY" "$out" "READY pid"
pid=$(cat "$SERVE_DIR/current.pid" 2>/dev/null)
kill -0 "$pid" 2>/dev/null && ok "the server pid is alive" || bad "the server pid '$pid' is not alive"
has "its command line has the port" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" " --port $P "
[ "$(pgrep -x llama-server | tr '\n' ' ')" = "$pid " ] && ok "pgrep -x llama-server finds exactly it" || bad "pgrep -x llama-server: '$(pgrep -x llama-server | tr '\n' ' ')', pid $pid"
curl -s --noproxy '*' -m 2 "http://127.0.0.1:$P/health" | grep -q '"ok"' && ok "health ok" || bad "health"
run=$(sed -n 's/^RUN=//p' "$SERVE_DIR/current.env")
for f in pid cmd.txt server.txt models.json props.json meta.json; do [ -s "$run/$f" ] && ok "run file $f" || bad "run file $f missing or empty"; done
"$PY" -I -c '
import json, sys
m = json.load(open(sys.argv[1]))
need = {"started", "ready", "pid", "port", "alias", "np", "ctx_slot", "n_ctx_total", "reasoning", "cpu_test", "note", "cmd", "bin",
        "bin_sha256", "model", "model_sha256", "health_seconds", "props_check", "gpu_after", "gpu_apps_after"}
sys.exit(0 if not need - set(m) and m["port"] == sys.argv[2] and m["note"] == "unit test" and m["n_ctx_total"] == "65536" else 1)
' "$run/meta.json" "$P" && ok "meta.json keys and values" || bad "meta.json: $(head -c 300 "$run/meta.json")"
grep -q "READY pid $pid port $P alias qwen3.5-9b-q8_0 np 1 ctx_slot 65536" "$SERVE_DIR/serve.log" && ok "serve.log READY line" || bad "serve.log READY line"
grep -q "start (note: unit test)" "$SERVE_DIR/serve.log" && ok "serve.log start line with the note" || bad "serve.log start line"
out=$(bash "$S" start --port $P 2>&1); expect "a second start is refused" 3 $?
has "it names the running server" "$out" "still running (pid $pid"
[ "$(servers)" = 1 ] && ok "still exactly one server" || bad "servers: $(servers)"
out=$(bash "$S" status 2>&1); has "status: running" "$out" "running: PID=$pid"; has "status: props" "$out" "total_slots 1 n_ctx per slot 65536"
out=$(bash "$S" stop 2>&1); expect "stop" 0 $?
kill -0 "$pid" 2>/dev/null && bad "pid $pid alive after the stop" || ok "pid gone after the stop"
(exec 3<>"/dev/tcp/127.0.0.1/$P") 2>/dev/null && bad "port $P still open" || ok "port free after the stop"
none_left "after the stop"
grep -q "STOPPED pid $pid port $P" "$SERVE_DIR/serve.log" && ok "serve.log STOPPED line" || bad "serve.log STOPPED line"
out=$(bash "$S" stop 2>&1); expect "stop again" 0 $?; has "stop again: none running" "$out" "no server started by this script"
out=$(bash "$S" status 2>&1); has "status: not running" "$out" "not running"
out=$(SERVE_LOAD="$T/stub/cpu_load 4.50" bash "$S" start --port $P --np 2 --ctx-slot 32768 2>&1); expect "start at the gate limit 4.50, np 2" 0 $?
out=$(bash "$S" status 2>&1); has "np 2: two slots of 32768" "$out" "total_slots 2 n_ctx per slot 32768"
bash "$S" stop > /dev/null 2>&1; expect "stop (np 2)" 0 $?
none_left "after the np 2 run"

echo "== refusals before the launch"
stubenv refuse
"$T/stub/llama-server" --port 18096 -a other > "$W/other_stub.txt" 2>&1 &
opid=$!
for _ in $(seq 1 50); do curl -s --noproxy '*' -m 1 http://127.0.0.1:18096/health | grep -q ok && break; sleep 0.1; done
out=$(bash "$S" start --port $P 2>&1); expect "refused beside another llama-server" 3 $?
has "it says one instance only" "$out" "one shared Qwen instance only"
out=$(bash "$S" stop 2>&1); expect "stop without a pid file" 0 $?; has "stop leaves the other server alone" "$out" "not touched"
kill -0 $opid 2>/dev/null && ok "the other server was not signalled" || bad "the other server is gone"
kill $opid; wait $opid 2>/dev/null
mkdir -p "$W/www"
"$PY" -I -m http.server $P --bind 127.0.0.1 --directory "$W/www" > "$W/listener.txt" 2>&1 &
lpid=$!
for _ in $(seq 1 50); do (exec 3<>"/dev/tcp/127.0.0.1/$P") 2>/dev/null && break; sleep 0.1; done
out=$(bash "$S" start --port $P 2>&1); expect "refused on a busy port" 3 $?; has "it says the port is in use" "$out" "port $P is in use"
kill $lpid; wait $lpid 2>/dev/null
out=$(STUB_APPS="742362, 4184 MiB" bash "$S" start --port $P 2>&1); expect "refused with a GPU compute process" 3 $?
has "it says no sharing" "$out" "no sharing"
out=$(STUB_NVSMI_FAIL=1 bash "$S" start --port $P 2>&1); expect "refused when nvidia-smi fails" 3 $?
out=$(SERVE_LOAD="$T/stub/cpu_load 5.20" bash "$S" start --port $P 2>&1); expect "refused by the CPU gate (5.20 > 4.5)" 4 $?
[ "$(grep -c 'CPU gate, try' "$SERVE_DIR/serve.log")" = 2 ] && ok "the gate tried twice" || bad "gate tries: $(grep -c 'CPU gate, try' "$SERVE_DIR/serve.log")"
out=$(SERVE_LOAD="$T/stub/cpu_load nan" bash "$S" start --port $P 2>&1); expect "refused on a non-numeric load" 4 $?
out=$(SERVE_CPU_TEST=1 SERVE_LOAD="$T/stub/cpu_load 4.20" bash "$S" start --port $P 2>&1); expect "CPU test: refused at 4.20 > 4.0" 4 $?
has "CPU test skips the GPU check" "$out" "its free check is skipped"
none_left "after the refusals"

echo "== failures after the launch"
stubenv after
out=$(STUB_NCTX=1024 bash "$S" start --port $P 2>&1); expect "a wrong n_ctx in /props is refused" 7 $?
has "it names n_ctx" "$out" "n_ctx per slot 1024 != 65536"; none_left "after the n_ctx refusal"
out=$(STUB_BUILD=b1-0000000 bash "$S" start --port $P 2>&1); expect "a wrong build_info in /props is refused" 7 $?
has "it names the build" "$out" "build_info b1-0000000 != the pinned b1-4c3c581"; none_left "after the build refusal"
out=$(STUB_IDS=other-model bash "$S" start --port $P 2>&1); expect "an unlisted alias is refused" 7 $?
has "it names the ids" "$out" "lists ['other-model']"; none_left "after the alias refusal"
out=$(STUB_DIE=1 bash "$S" start --port $P 2>&1); expect "a server that dies while loading" 6 $?; none_left "after the dead server"
out=$(STUB_DELAY=1000 SERVE_HEALTH_SECS=4 bash "$S" start --port $P 2>&1); expect "no health in time" 6 $?
has "it says the server was stopped" "$out" "the server was stopped"; none_left "after the health timeout"

echo "== stale and foreign pid files"
stubenv pidfiles
mkdir -p "$SERVE_DIR"
true & dpid=$!; wait $dpid
echo $dpid > "$SERVE_DIR/current.pid"; printf 'PID=%s\nPORT=%s\nRUN=none\n' $dpid $P > "$SERVE_DIR/current.env"
out=$(bash "$S" stop 2>&1); expect "stop with a stale pid file" 0 $?; has "it removes the stale file" "$out" "stale pid file removed"
sleep 120 & spid=$!
echo $spid > "$SERVE_DIR/current.pid"; printf 'PID=%s\nPORT=%s\nRUN=none\n' $spid $P > "$SERVE_DIR/current.env"
out=$(bash "$S" stop 2>&1); expect "stop refuses a foreign pid" 8 $?; has "nothing signalled" "$out" "nothing signalled"
kill -0 $spid 2>/dev/null && ok "the foreign process was not signalled" || bad "the foreign process is gone"
kill $spid; wait $spid 2>/dev/null
rm -f "$SERVE_DIR/current.pid" "$SERVE_DIR/current.env"
true & dpid=$!; wait $dpid
echo $dpid > "$SERVE_DIR/current.pid"
out=$(bash "$S" start --port $P 2>&1); expect "start over a stale pid file" 0 $?; has "start removes the stale file" "$out" "stale pid file removed"
bash "$S" stop > /dev/null 2>&1; none_left "after the stale-file start"

echo "== hash checks"
stubenv hash; unset SERVE_SKIP_HASH
sed '1s/^[0-9a-f]\{64\}/0000000000000000000000000000000000000000000000000000000000000000/' "$L/build_b11742_bin.sha256" > "$W/bad_build.sha256"
out=$(SERVE_HASH_LIST=$W/bad_build.sha256 bash "$S" start --port $P 2>&1); expect "a build list other than the pin is refused" 5 $?
has "it names the pin" "$out" "is not the pinned list"
head -n -1 "$L/build_b11742_bin.sha256" > "$W/short_build.sha256"
out=$(SERVE_HASH_LIST=$W/short_build.sha256 bash "$S" start --port $P 2>&1); expect "a valid but different build list is refused by the pin" 5 $?
has "the pin refuses it" "$out" "is not the pinned list"; none_left "after the list refusals"
out=$(SERVE_HASH_LIST=$W/bad_build.sha256 SERVE_LIST_SHA=$(sha256sum "$W/bad_build.sha256" | cut -d" " -f1) bash "$S" start --port $P 2>&1)
expect "a build file that does not match its list is refused" 5 $?; has "it names the mismatch" "$out" "does not match"
mkdir -p "$W/gguf_bad" "$W/gguf_good"
echo "not a model" > "$W/gguf_bad/Qwen3.5-9B-Q8_0.gguf"
echo "0000000000000000000000000000000000000000000000000000000000000000  Qwen3.5-9B-Q8_0.gguf" > "$W/gguf_bad/FILES.sha256"
out=$(SERVE_GGUF_DIR=$W/gguf_bad bash "$S" start --port $P 2>&1); expect "a Q8_0 hash other than the pin is refused" 5 $?
has "it names the Q8_0 pin" "$out" "does not list the pinned Q8_0"
out=$(SERVE_GGUF_DIR=$W/gguf_bad SERVE_Q8_SHA=0000000000000000000000000000000000000000000000000000000000000000 bash "$S" start --port $P 2>&1)
expect "a Q8_0 file that does not match FILES.sha256 is refused" 5 $?; has "it names the file mismatch" "$out" "the Q8_0 GGUF does not match"
echo "not a model either" > "$W/gguf_good/Qwen3.5-9B-Q8_0.gguf"
(cd "$W/gguf_good" && sha256sum Qwen3.5-9B-Q8_0.gguf > FILES.sha256)
out=$(SERVE_GGUF_DIR=$W/gguf_good bash "$S" start --port $P 2>&1); expect "a matching file that is not the pinned Q8_0 is refused" 5 $?
out=$(SERVE_GGUF_DIR=$W/gguf_good SERVE_Q8_SHA=$(cut -d" " -f1 "$W/gguf_good/FILES.sha256") bash "$S" start --port $P 2>&1); expect "matching hashes: start" 0 $?
has "it logs the hashes" "$out" "$(grep -c "" "$L/build_b11742_bin.sha256") build files, the pinned Q8_0"
bash "$S" stop > /dev/null 2>&1; none_left "after the hash runs"

echo "unit tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
