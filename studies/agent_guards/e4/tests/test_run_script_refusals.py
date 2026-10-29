"""run_e4_gpu.sh's refusals that happen before anything is created or contacted (revision 3): a campaign name that is
not e4reg_<yyyymmdd>, a slot given to a stage (the slot enters only at the freeze), a dry-run hook in a registered
campaign, a smoke for an integrity re-run, a CPU gate above 4.5, and a registered stage before the freeze (exit 7, no
folder made). Revision 4: STAGE=smoke once the freeze record exists (exit 7, round-3 review m3), and the integrity
guard (after a logged integrity stop that voids a bar, every stage but judge and summary exits 7; section 9.1).
Revision 5 (round-4 review M1 (e)): the operator's stop (TERM or INT to the script stops its client, logs
'STOP [slot_end] run=<n>' and exits 8; never counted), the exit trap (DONE; an untagged failure is [client] exit 3; a
signal is [slot_end]), the attempt guard (a stage with 3 failed attempts logs SKIP [stage_failed] and exits 0), a
client ended by TERM or INT (exit 8), the note of every failed attempt, and a stopped rule-9 re-run that resumes on its
own task list. Revision 6 (round-5 review Major 1 (d), minor 2): against a tiny local HTTP server process (not
llama-server; it serves /health, /v1/models and /props with '--port <p>' in its argv), the end check AT THE STOP writes
the stopped call's own record (TERM to the script, or to the client; "identity_changed" there is [identity_changed],
exit 2); a stage's third failed attempt has its call's end record before its next start logs SKIP; a run's next start
that meets a server that is not the pinned one is [server] exit 2 with the record untouched, no integrity event and
nothing counted, and with the pinned server back (restarted: a new pid) the start checks pass and the deferred check
writes "restart_same_identity"; a run log the guards cannot read is [refused] (exit 7), never counted. No GPU.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_run_script_refusals.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
SH = E4 / "scripts/run_e4_gpu.sh"
PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"


def run(**env):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run(["bash", str(SH)], env={**base, **env}, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def test_refusals_before_anything_runs():
    made = [E4 / "runs" / c for c in ("e4_20261022", "e4reg_00000000", "e4reg_00000000_icr", "DRYRUN_refusaltest")]
    assert not any(m.exists() for m in made)
    try:
        rc, out = run(CAMPAIGN="e4_20261022", STAGE="a0")
        assert rc == 1 and "e4reg_<yyyymmdd>" in out
        rc, out = run(CAMPAIGN="e4reg_00000000", STAGE="smoke", SLOT_MINUTES="100")
        assert rc == 1 and "set at the freeze" in out
        rc, out = run(CAMPAIGN="e4reg_00000000", STAGE="a0", MIN_CLASS="1")
        assert rc == 1 and "dry-run hook" in out
        rc, out = run(CAMPAIGN="e4reg_00000000", STAGE="a0", FROZEN="/tmp/x.json")
        assert rc == 1 and "dry-run hook" in out
        rc, out = run(CAMPAIGN="e4reg_00000000_icr", STAGE="smoke")
        assert rc == 1 and "no smoke" in out
        rc, out = run(CAMPAIGN="DRYRUN_refusaltest", STAGE="a0", CPU_GATE="5")
        assert rc == 1 and "CPU_GATE" in out
        if not (E4 / "runs" / "E4_frozen.json").exists():
            rc, out = run(CAMPAIGN="e4reg_00000000", STAGE="a0")
            assert rc == 7 and "after the freeze" in out
        assert not any(m.exists() for m in made)
    finally:
        for m in made:
            if m.exists():
                shutil.rmtree(m)


@pytest.mark.parametrize("stage", ["nope", "freeze"])
def test_unknown_stage_is_refused(stage):
    rc, out = run(CAMPAIGN="DRYRUN_refusaltest", STAGE=stage)
    assert rc == 1 and "unknown STAGE" in out and not (E4 / "runs" / "DRYRUN_refusaltest").exists()


def test_smoke_after_the_freeze_is_refused(tmp_path):
    """Round-3 review m3: the smoke's files are H6 files; once the freeze record exists no smoke runs (exit 7). The
    registered path (runs/E4_frozen.json) cannot be created here, so the DRYRUN_ FROZEN hook points at a record."""
    rec = tmp_path / "E4_frozen_dry.json"
    camp = E4 / "runs" / "DRYRUN_refusaltest"
    assert not camp.exists()
    rec.write_text("{}")
    rc, out = run(CAMPAIGN="DRYRUN_refusaltest", STAGE="smoke", FROZEN=str(rec))
    assert rc == 7 and "only before the freeze" in out and not camp.exists(), out
    sh = SH.read_text()                                                # the registered path takes the same branch
    assert 'FROZEN="${FROZEN:-$E4/runs/E4_frozen.json}"' in sh
    i = sh.index('if [ "$STAGE" = smoke ]; then')
    assert sh.index('[ -f "$FROZEN" ] && {', i) < sh.index("elif", i)


@pytest.mark.parametrize("line,refused", [
    ("2026-10-22T10:00:00Z [a1] STOP [identity_changed] run=A1_base: e4_run.py for c/A1_base failed (exit 2): x", True),
    ("2026-10-22T10:00:00Z [a2a3] STOP [identity_changed] run=A2s1_test: e4_run.py failed (exit 2): x", True),
    ("2026-10-22T10:00:00Z [a1] STOP [restart_same_identity] run=A1_base: e4_run.py failed (exit 2): x", False),
    ("2026-10-22T10:00:00Z [smoke] STOP [identity_changed] run=A0_smoke: e4_run.py failed (exit 2): x", False)])
def test_integrity_guard(line, refused):
    """After an integrity stop that voids a bar, a0..a2a3 exit 7 before the CPU gate (judge and summary are exempt: the
    guard's own case line, checked below); a stop that is not an integrity stop, or a smoke stop, refuses nothing."""
    sh = SH.read_text()
    assert 'case "$STAGE" in smoke|judge|summary) return 0 ;; esac' in sh
    assert "start_stage() { integrity_guard; attempts_guard; cpu_ok; pins_ok; }" in sh
    camp = E4 / "runs" / "DRYRUN_guardtest"
    assert not camp.exists()
    camp.mkdir(parents=True)
    try:
        (camp / "run_e4_gpu.log").write_text(line + "\n")
        p = subprocess.run([PY, "-I", "-B", str(E4 / "scripts/check_pins.py"), "--stop-effects",
                            str(camp / "run_e4_gpu.log")], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0 and (p.stdout.strip() != "[]") == refused, p.stdout + p.stderr
        if refused:
            rc, out = run(CAMPAIGN="DRYRUN_guardtest", STAGE="a2a3")
            assert rc == 7 and "only judge and summary run" in out, out
            assert "cpu:" not in out                                       # refused before the CPU gate
    finally:
        shutil.rmtree(camp, ignore_errors=True)


def _fns(*names):
    """The named functions of run_e4_gpu.sh, as the script defines them."""
    sh, out = SH.read_text(), []
    for nm in names:
        i = sh.index(f"\n{nm}() {{") + 1
        out.append(sh[i:sh.index("\n}\n", i) + 3])
    return "\n".join(out)


def test_client_stop_tags_are_the_lines_stop_effects_reads(tmp_path):
    """Section 11.6's tags: run_e4_gpu.sh's own client_stop, run on a fake stderr file per case (only the last non-empty
    line counts); check_pins.stop_effects reads the identity_changed line back with its run and void bars."""
    import sys

    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    fn = _fns("failnote", "stop_end_check", "client_stop")
    cases = [(2, "e4_run.py: REFUSING: the server identity changed during the call (identity_changed): x", "identity_changed"),
             (2, "e4_run.py: REFUSING: the server identity changed during the call (restart_same_identity): x",
              "restart_same_identity"),
             (2, "e4_run.py: REFUSING: server not healthy: /health answered 503", "server"),
             (3, "e4_run.py: REFUSING: a STAND-IN model", "client"),
             (1, "Traceback (most recent call last): ...", "client"),
             (143, "", "slot_end"), (130, "", "slot_end")]                 # revision 5: a client ended by TERM / INT
    for rc, last, tag in cases:
        errf = tmp_path / "x.stderr"
        errf.write_text("an earlier line (identity_changed)\n" + last + "\n\n")
        script = 'say() { echo "2026-10-22T10:00:00Z [a2a3] $*"; }\nATTEMPTS_MAX=3 ATTEMPT=3\n' + fn + \
            f'\nclient_stop {rc} "e4_run.py for c/A2s1_test" A2s1_test "{errf}"\n'
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        assert p.returncode == {2: 2, 143: 8, 130: 8}.get(rc, 3), (tag, p.returncode, p.stderr)
        assert f"STOP [{tag}] run=A2s1_test: " in p.stdout and p.stderr == "", (tag, p.stdout, p.stderr)
        assert ("[failed attempt 3 of 3: final" in p.stdout) == (tag == "client"), p.stdout   # only a failure counts
        e = CP.stop_effects(p.stdout)
        if tag == "identity_changed":
            assert e["events"][0]["run"] == "A2s1_test" and e["void_bars"] == ["E4.3", "E4.5"]
        else:
            assert e["events"] == [] and e["void_bars"] == []
        assert CP.stage_attempts(p.stdout)["failed"] == ({"a2a3": 1} if tag == "client" else {})
    sh = SH.read_text()
    sh_count = [x for x in sh.splitlines() if "--count-candidates gave" in x]
    assert len(sh_count) == 1 and "*[!0-9]*" in sh_count[0] and "exit 3" in sh_count[0]   # a count or a STOP


# ------------------------------------------------------------------ revision 5 (round-4 review M1 (e))
def _harness(tmp_path, stage, attempt, body, runs=None, server=None, more=()):
    """run_e4_gpu.sh's own functions with its traps; server = a _tiny_server (BASE and PINS for the end check at the
    stop); more = further functions of the script (client_stop)."""
    srv = f"E4={E4}\nPY={PY}\nBASE={server.base}\nPINS={server.pins}\n" if server else ""
    return (f"set -euo pipefail\nSTAGE={stage} RUNS={runs or tmp_path}\n{srv}STOPPED=0 ATTEMPTS_MAX=3 ATTEMPT={attempt} "
            "CUR_PID='' CUR_RUN='' CALL_REC='' CALL_SINCE=''\n"
            + _fns("say", "failnote", "slot_end", "stop_end_check", "on_exit", "client", *more)
            + "\ntrap on_exit EXIT\ntrap 'slot_end TERM' TERM\ntrap 'slot_end INT' INT\n" + body)


@pytest.mark.parametrize("sig,name", [(signal.SIGTERM, "TERM"), (signal.SIGINT, "INT")])
def test_the_operators_stop_stops_the_client_logs_slot_end_and_is_never_counted(tmp_path, sig, name):
    """TERM or INT to run_e4_gpu.sh while a client runs: the trap stops the client (TERM), logs 'STOP [slot_end]
    run=<n>' and exits 8; the line is neither an integrity stop nor a failed attempt."""
    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    flag = tmp_path / "client_up"
    body = (f'client A0_base - {PY} -I -c "import os, sys, time, pathlib; pathlib.Path(sys.argv[1]).write_text('
            f'str(os.getpid())); time.sleep(120)" {flag}\nsay "unreachable"\n')
    p = subprocess.Popen(["bash", "-c", _harness(tmp_path, "a0", 1, body)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    for _ in range(300):
        if flag.exists() and flag.read_text():
            break
        time.sleep(0.1)
    child = int(flag.read_text())
    p.send_signal(sig)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 8, (p.returncode, out, err)
    text = (tmp_path / "run_e4_gpu.log").read_text()
    assert f"[a0] STOP [slot_end] run=A0_base: the operator stopped stage a0 (SIG{name})" in text, text
    assert "unreachable" not in text and "DONE" not in text and not Path(f"/proc/{child}/cmdline").exists()
    assert CP.stop_effects(text)["events"] == [] and CP.stage_attempts(text)["failed"] == {}


@pytest.mark.parametrize("attempt,body,rc,want", [
    (2, "false\n", 3, "STOP [client]: a step of stage a1 failed with exit 1 before any tagged stop [failed attempt 2 of 3: "
                     "resumed whatever the cause"),
    (3, "false\n", 3, "[failed attempt 3 of 3: final; this stage's next start logs SKIP [stage_failed]"),
    (1, "true\n", 0, "[a1] DONE: stage a1 ended with exit 0"),
    (1, 'say "STOP [refused]: x"; exit 7\n', 7, "[a1] STOP [refused]: x"),
    (1, f'{PY} -I -c "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"\n', 8,
     "STOP [slot_end] run=none: a signal ended stage a1 (exit 143)")])
def test_the_exit_trap_tags_every_end(tmp_path, attempt, body, rc, want):
    """Every end of a stage leaves a tagged line: exit 0 'DONE'; a step that fails under set -e is [client] (exit 3, a
    failed attempt with its note); a foreground step ended by a signal is the operator's stop (exit 8); a tagged stop
    keeps its own code."""
    p = subprocess.run(["bash", "-c", _harness(tmp_path, "a1", attempt, body)], capture_output=True, text=True,
                       timeout=60)
    text = (tmp_path / "run_e4_gpu.log").read_text()
    assert p.returncode == rc and want in text, (p.returncode, text, p.stderr)
    assert text.count("STOP [") == (0 if rc == 0 else 1), text


def test_the_attempt_guard_skips_a_stage_after_its_third_failed_attempt():
    """Three failed attempts (exit 3 or 6) of a stage in its log: its next start logs SKIP [stage_failed] and exits 0
    before the CPU gate, so the loop goes on; two leave the third attempt to run (the count the guard reads)."""
    camp = E4 / "runs" / "DRYRUN_attempttest"
    assert not camp.exists()
    camp.mkdir(parents=True)
    try:
        fails = "".join(f"2026-10-22T10:0{i}:00Z [a2a3] STOP [client] run=A3_test: e4_run.py for c/A3_test failed "
                        "(exit 1): x\n" for i in range(3))
        (camp / "run_e4_gpu.log").write_text(fails)
        rc, out = run(CAMPAIGN="DRYRUN_attempttest", STAGE="a2a3")
        assert rc == 0 and "SKIP [stage_failed]: stage a2a3 failed 3 of 3 attempts" in out and "cpu:" not in out, out
        assert "DONE: stage a2a3 ended with exit 0" in (camp / "run_e4_gpu.log").read_text()
        (camp / "run_e4_gpu.log").write_text(fails.split("\n", 1)[1])
        cli = [PY, "-I", "-B", str(E4 / "scripts/check_pins.py"), "--attempts", str(camp / "run_e4_gpu.log")]
        p = subprocess.run(cli + ["--stage", "a2a3"], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0 and p.stdout.strip() == "2", p.stdout + p.stderr
        p = subprocess.run(cli + ["--stage", "smoke"], capture_output=True, text=True, timeout=60)
        assert p.returncode == 2 and "registered --stage" in p.stdout
    finally:
        shutil.rmtree(camp, ignore_errors=True)


def test_a_stopped_rerun_resumes_on_its_own_task_list(tmp_path):
    """Rule 9's one re-run of an arm, stopped before its end (a slot end or a failure), resumes on the task list it
    started with; a finished one, or a folder from before revision 5 without the list, is never re-run."""
    (tmp_path / "A0_base").mkdir()
    (tmp_path / "A0_base" / "results.json").write_text(json.dumps({"simulations": [], "tasks": []}))
    script = (f"set -euo pipefail\nE4={E4}\nPY={PY}\nRUNS={tmp_path}\nCAMPAIGN=DRYRUN_x\n"
              'say() { echo "SAY $*"; }\nrun_arm() { echo "RUN_ARM $*"; }\n' + _fns("rerun_arm")
              + "\nrerun_arm A0_base A0 base\n")
    go = lambda: subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)  # noqa: E731
    p = go()                                                      # no re-run folder, no infra end: nothing to re-run
    assert p.returncode == 0 and "no re-run" in p.stdout and "RUN_ARM" not in p.stdout, p.stdout + p.stderr
    rr = tmp_path / "A0_base_rerun"
    rr.mkdir()
    p = go()                                                      # a folder without the list (before revision 5)
    assert "already re-run once (never twice)" in p.stdout and "RUN_ARM" not in p.stdout
    (rr / "rerun_tasks.txt").write_text("3 7\n")
    (rr / "results.json").write_text(json.dumps({"simulations": [{"task_id": "3"}]}))
    p = go()                                                      # stopped after task 3: resumed on 3 and 7
    assert "RUN_ARM DRYRUN_x A0 base A0_base_rerun --task-ids 3 7" in p.stdout, p.stdout + p.stderr
    (rr / "results.json").write_text(json.dumps({"simulations": [{"task_id": "3"}, {"task_id": 7}]}))
    p = go()                                                      # finished: never twice
    assert "already re-run once (never twice)" in p.stdout and "RUN_ARM" not in p.stdout
    sh = SH.read_text()
    assert 'mkdir -p "$rr" && printf \'%s\\n\' "$ids" > "$rr/rerun_tasks.txt"' in sh


# ------------------------------------------------------------------ revision 6 (round-5 review Major 1 (d), minor 2)
TINY = """import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, mp = int(sys.argv[sys.argv.index("--port") + 1]), sys.argv[sys.argv.index("--model-path") + 1]
STATE = {"props": {}}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        if self.path.rstrip("/") == "/v1/models":
            return self._send({"data": [{"id": "tiny"}]})
        if self.path == "/props":
            return self._send({"default_generation_settings": {"n_ctx": 65536}, "total_slots": 1, "model_alias": "tiny",
                               "build_info": "b1-4c3c581", "model_path": mp, "chat_template": "tiny", **STATE["props"]})
        self._send({}, 404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/stub/props":
            STATE["props"] = body
            return self._send({"ok": True})
        self._send({}, 404)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""

CLIENT = """import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[5])
import e4_run
rec, base, pins, then = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), sys.argv[4]
ident, diff = e4_run.server_identity(base, pins)
assert diff == [], diff
rec.parent.mkdir(parents=True, exist_ok=True)
rec.write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "server_identity": ident}))
Path(str(rec) + ".pid").write_text(str(os.getpid()))
if then == "sleep":
    time.sleep(120)
sys.exit(int(then))
"""


def _get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as r:
        return json.loads(r.read().decode())


def _post(url, obj):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=5) as r:
        return json.loads(r.read().decode())


class _Tiny:
    """A tiny local HTTP server PROCESS with '--port <p>' in its argv (what e4_run.processes_on_port looks for) and
    its pins file: the end check at the stop runs as its own process, so the server must be real."""

    def __init__(self, tmp_path):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        (tmp_path / "tiny.py").write_text(TINY)
        (tmp_path / "client.py").write_text(CLIENT)
        self.client = tmp_path / "client.py"
        gg = tmp_path / "tiny.gguf"
        gg.write_bytes(b"GGUF tiny")
        self.argv = [PY, "-I", "-B", str(tmp_path / "tiny.py"), "--port", str(self.port), "--model-path", str(gg)]
        self.root, self.base = f"http://127.0.0.1:{self.port}", f"http://127.0.0.1:{self.port}/v1"
        self.proc = None
        self.start()
        import hashlib

        exe = os.path.realpath(PY)
        p = _get(self.root + "/props")
        pins = {"what": "unit-test pins for a tiny local server (not the registered server)", "port": self.port,
                "props": {"total_slots": 1, "n_ctx_per_slot": 65536, "build_info": "b1-4c3c581", "model_alias": "tiny",
                          "model_path": str(gg),
                          "chat_template_sha256": hashlib.sha256(p["chat_template"].encode()).hexdigest()},
                "models_ids": ["tiny"], "gguf_sha256": hashlib.sha256(b"GGUF tiny").hexdigest(), "exe_path": exe,
                "exe_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(), "build_dir_files_sha256": {},
                "argv": self.argv}
        self.pins = tmp_path / "pins_tiny.json"
        self.pins.write_text(json.dumps(pins, indent=1))

    def start(self):
        self.proc = subprocess.Popen(self.argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                _get(self.root + "/health")
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        raise RuntimeError("the tiny server did not start")

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait(timeout=10)
            self.proc = None

    def restart(self):                    # the pinned server back after a restart: the same argv, a new pid
        self.stop()
        self.start()

    def set_props(self, d):
        _post(self.root + "/stub/props", d)


@contextlib.contextmanager
def _tiny_server(tmp_path):
    t = _Tiny(tmp_path)
    try:
        yield t
    finally:
        t.stop()


def _wait_for(path, key="server_identity", n=600):
    for _ in range(n):
        try:
            if key in json.loads(Path(path).read_text()):
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("who,changed", [("script", False), ("script", True), ("client", False)])
def test_the_end_check_at_the_stop_writes_the_stopped_calls_own_record(tmp_path, who, changed):
    """The operator's stop (TERM to the script; or TERM to the client itself, exit 143): the stopped call's own start
    record gets its end check AT THE STOP (server_identity_end, "at_stop": true), while the server is still up. "same"
    keeps the stop as it is (exit 8; nothing counted); a server changed during the call is [identity_changed] for that
    run (exit 2; section 9.1's bars), never a failed attempt."""
    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    with _tiny_server(tmp_path) as ts:
        rec = tmp_path / "A0_base" / "manifest.json"
        errf = tmp_path / "A0_base.stderr"
        body = (f"rc=0\nclient A0_base {rec} {PY} -I -B {ts.client} {rec} {ts.base} {ts.pins} sleep {E4 / 'scripts'} "
                f"2>> {errf} || rc=$?\n"
                f'client_stop "$rc" "e4_run.py for c/A0_base" A0_base {errf}\nsay "unreachable"\n')
        p = subprocess.Popen(["bash", "-c", _harness(tmp_path, "a0", 1, body, server=ts, more=("client_stop",))],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert _wait_for(rec), "the client wrote no start record"
        pidf = Path(str(rec) + ".pid")
        for _ in range(200):
            if pidf.exists() and pidf.read_text():
                break
            time.sleep(0.05)
        if changed:
            ts.set_props({"total_slots": 2})                  # the server changed during the call
        if who == "script":
            p.send_signal(signal.SIGTERM)
        else:
            os.kill(int(pidf.read_text()), signal.SIGTERM)
        out, err = p.communicate(timeout=120)
        text = (tmp_path / "run_e4_gpu.log").read_text()
        end = json.loads(rec.read_text())["server_identity_end"]
        assert end["at_stop"] is True and "deferred" not in end, end
        assert text.count("STOP [slot_end] run=A0_base: ") == 1 and "unreachable" not in text, text
        assert CP.stage_attempts(text)["failed"] == {}
        if changed:
            assert p.returncode == 2 and end["kind"] == "identity_changed", (p.returncode, text, err)
            assert "[a0] STOP [identity_changed] run=A0_base: the end check at the stop found the server changed" in text
            e = CP.stop_effects(text)
            assert e["void_bars"] == ["E4.2", "E4.3", "E4.5"] and [x["run"] for x in e["events"]] == ["A0_base"]
        else:
            assert p.returncode == 8 and end["kind"] == "same", (p.returncode, text, err)
            assert "[a0] end check at the stop, run=A0_base: same (" in text and CP.stop_effects(text)["events"] == []


def test_a_third_failed_attempts_call_has_its_end_record_before_the_skip(tmp_path):
    """Round-5 review Major 1, the related gap: a failed client (exit 3) gets its call's end check at the stop, so the
    call of a stage's third failed attempt has its end record before the stage's next start logs SKIP [stage_failed]."""
    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    camp = E4 / "runs" / "DRYRUN_atstoptest"
    assert not camp.exists()
    camp.mkdir(parents=True)
    try:
        with _tiny_server(tmp_path) as ts:
            (camp / "run_e4_gpu.log").write_text("".join(
                f"2026-10-22T10:0{i}:00Z [a2a3] STOP [client] run=A3_test: e4_run.py for c/A3_test failed (exit 1): x\n"
                for i in range(2)))
            rec = camp / "A3_test" / "manifest.json"
            errf = tmp_path / "A3_test.stderr"
            body = (f"rc=0\nclient A3_test {rec} {PY} -I -B {ts.client} {rec} {ts.base} {ts.pins} 1 {E4 / 'scripts'} "
                    f"2>> {errf} || rc=$?\n"
                    f'client_stop "$rc" "e4_run.py for c/A3_test" A3_test {errf}\n')
            p = subprocess.run(["bash", "-c", _harness(tmp_path, "a2a3", 3, body, runs=camp, server=ts,
                                                       more=("client_stop",))],
                               capture_output=True, text=True, timeout=120)
            text = (camp / "run_e4_gpu.log").read_text()
            assert p.returncode == 3, (p.returncode, text, p.stderr)
            assert "STOP [client] run=A3_test: e4_run.py for c/A3_test failed (exit 1)" in text
            assert "[failed attempt 3 of 3: final" in text and "[a2a3] end check at the stop, run=A3_test: same (" in text
            end = json.loads(rec.read_text())["server_identity_end"]
            assert end["kind"] == "same" and end["at_stop"] is True
            assert CP.stage_attempts(text)["failed"] == {"a2a3": 3}
        rc, out = run(CAMPAIGN="DRYRUN_atstoptest", STAGE="a2a3")
        assert rc == 0 and "SKIP [stage_failed]: stage a2a3 failed 3 of 3 attempts" in out, out
        lines = (camp / "run_e4_gpu.log").read_text().splitlines()
        i = next(k for k, x in enumerate(lines) if "end check at the stop, run=A3_test: same" in x)
        assert i < next(k for k, x in enumerate(lines) if " [a2a3] SKIP [stage_failed]: " in x)   # the end record first
        assert json.loads(rec.read_text())["server_identity_end"] == end                      # the skip leaves it
    finally:
        shutil.rmtree(camp, ignore_errors=True)


def test_a_next_start_that_meets_a_server_that_is_not_the_pinned_one_is_a_server_stop(tmp_path):
    """Round-5 review Major 1 (b), (d): a run's earlier call has no end record (a kill -9). Its next start meets a server
    that is not the pinned one: e4_run.py's full start check comes first, so the stage stops [server] (exit 2), the
    record is byte for byte unchanged, there is no integrity event and nothing is counted, and no end check runs at
    this stop. With the pinned server back after a restart (a new pid), e4_run.py's start checks pass and the deferred
    check writes "restart_same_identity" into the record; the run goes on (here it stops right after them, at the
    learned-model check of an A2 call without --learned: exit 3, before any task)."""
    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    camp = E4 / "runs" / "DRYRUN_notpinnedtest"
    assert not camp.exists()
    try:
        with _tiny_server(tmp_path) as ts:
            rec = camp / "A0_base" / "manifest.json"
            q = subprocess.run([PY, "-I", "-B", str(ts.client), str(rec), ts.base, str(ts.pins), "0", str(E4 / "scripts")],
                               capture_output=True, text=True, timeout=120)        # the killed call's record, no end
            assert q.returncode == 0 and "server_identity_end" not in json.loads(rec.read_text()), q.stderr
            before = rec.read_bytes()
            ts.set_props({"chat_template": "a chat template that is not the pinned one"})
            errf = tmp_path / "A0_base.stderr"
            cmd = (f"{PY} -I -B {E4 / 'scripts/e4_run.py'} --base {ts.base} --campaign DRYRUN_notpinnedtest "
                   f"--run-name A0_base --expect-server {ts.pins}")
            body = (f"rc=0\nclient A0_base {rec} {cmd} --arm A0 --split base 2>> {errf} || rc=$?\n"
                    f'client_stop "$rc" "e4_run.py for c/A0_base" A0_base {errf}\n')
            p = subprocess.run(["bash", "-c", _harness(tmp_path, "a0", 1, body, server=ts, more=("client_stop",))],
                               capture_output=True, text=True, timeout=180)
            text = (tmp_path / "run_e4_gpu.log").read_text()
            assert p.returncode == 2, (p.returncode, text, p.stderr)
            line = next(x for x in text.splitlines() if "STOP [" in x)
            assert line.split(" ", 2)[2].startswith("STOP [server] run=A0_base: ") and "not the pinned one" in line
            assert "(identity_changed)" not in text and "end check at the stop" not in text
            assert rec.read_bytes() == before                                       # untouched
            assert CP.stop_effects(text)["events"] == [] and CP.stage_attempts(text)["failed"] == {}
            ts.set_props({})
            ts.restart()                                                            # the pinned server back
            r = subprocess.run(cmd.split() + ["--arm", "A2", "--split", "test"], capture_output=True, text=True,
                               timeout=180, env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})
            assert r.returncode == 3 and "needs --learned" in r.stderr, (r.returncode, r.stderr[-600:])
            end = json.loads(rec.read_text())["server_identity_end"]
            assert end["kind"] == "restart_same_identity" and end["deferred"] is True and "at_stop" not in end
    finally:
        shutil.rmtree(camp, ignore_errors=True)


@pytest.mark.parametrize("guard,fake_py", [("attempts_guard", "/bin/false"), ("attempts_guard", "/bin/echo"),
                                           ("integrity_guard", "/bin/false")])
def test_a_run_log_the_guards_cannot_read_is_refused_and_never_counted(tmp_path, guard, fake_py):
    """Round-5 review minor 2: when check_pins.py cannot read run_e4_gpu.log for the integrity or the attempt guard (or
    gives no count), nothing of the stage ran: 'STOP [refused]' (exit 7), not a failed attempt."""
    sys.path.insert(0, str(E4 / "scripts"))
    import check_pins as CP

    script = (f"set -euo pipefail\nSTAGE=a1 RUNS={tmp_path} E4={E4} PY={fake_py}\nSTOPPED=0 ATTEMPTS_MAX=3 ATTEMPT='' "
              "CUR_PID='' CUR_RUN='' CALL_REC='' CALL_SINCE=''\n" + _fns("say", "failnote", "on_exit", guard)
              + f"\ntrap on_exit EXIT\n{guard}\nsay unreachable\n")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    text = (tmp_path / "run_e4_gpu.log").read_text()
    assert p.returncode == 7 and text.count("STOP [refused]: ") == 1 and "unreachable" not in text, (p.returncode, text)
    assert "never counted" in text and CP.stage_attempts(text)["failed"] == {} and CP.stop_effects(text)["events"] == []
    sh = SH.read_text()
    assert "STOP [client]: the integrity guard" not in sh and "STOP [client]: the failed attempts" not in sh
    assert "STOP [client]: check_pins.py --attempts gave" not in sh
