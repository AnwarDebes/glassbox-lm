"""rev3 tests (registration rev3, 2026-10-21; the round-2 re-review and the decisions 1-5), AgentDojo venv
(pytest); no LLM, no server (tests/stub_server.py on an ephemeral 127.0.0.1 port only, stopped after each test):
- the transport rule: which failures stop a stage (failures.is_transport) and which fall under the error rule, with the
  stub returning HTTP 500, HTTP 400 and refusing connections, through the agent's client and the optimizer's;
- A3: optimizer errors caught per iteration; a transport error in an episode or the optimizer stops the stage;
- the per-pass request count (AgentDojo's up-to-3-pass rerun);
- the write-once freeze record (freeze_e2.py) and e2_check's checks of it; rev4: the freeze refuses code other than
  E2-0's and an earlier campaign the registration does not name;
- a logged stop, the one rerun on the same store, and a second stop leaving the stage incomplete;
- the driver: the A3 plan refuses black-box on D6; E2-0's open row carries no outcome and no count."""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import httpx
import openai
import pytest

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(E2 / "scripts"))

import e2_check as EC
import freeze_e2 as FZ
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatSystemMessage,
    ChatUserMessage,
    text_content_block_from_string,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from e2guard import attacks_a3 as A3
from e2guard.failures import StageStopped, error_type, is_transport
from e2guard.store_llm import StoreLLM

REQ = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")


def resp(code: int) -> httpx.Response:
    return httpx.Response(code, request=REQ)


def test_is_transport_classification():
    transport = [openai.APIConnectionError(request=REQ), openai.APITimeoutError(request=REQ),
                 openai.InternalServerError("x", response=resp(500), body=None),
                 openai.APIStatusError("x", response=resp(503), body=None),
                 urllib.error.HTTPError("http://x", 500, "x", {}, None), urllib.error.HTTPError("http://x", 503, "x", {}, None),
                 urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
                 http.client.RemoteDisconnected("closed"), ConnectionResetError(104, "reset"), TimeoutError("t")]
    error_rule = [openai.BadRequestError("x", response=resp(400), body=None),
                  openai.NotFoundError("x", response=resp(404), body=None),
                  openai.UnprocessableEntityError("x", response=resp(422), body=None),
                  openai.RateLimitError("x", response=resp(429), body=None),
                  urllib.error.HTTPError("http://x", 400, "x", {}, None), RuntimeError("guard crashed"),
                  ValueError("bad"), KeyError("choices")]
    assert all(is_transport(e) for e in transport), [type(e).__name__ for e in transport if not is_transport(e)]
    assert not any(is_transport(e) for e in error_rule), [type(e).__name__ for e in error_rule if is_transport(e)]
    assert error_type(openai.InternalServerError("x", response=resp(500), body=None)) == "InternalServerError 500"
    assert error_type(RuntimeError("x")) == "RuntimeError"


class Stub:
    """tests/stub_server.py in a subprocess (not an LLM), on an ephemeral port; stopped by close()."""

    def __init__(self, tmp: Path, *flags: str) -> None:
        self.pf = tmp / f"port_{time.time_ns()}"
        self.p = subprocess.Popen([sys.executable, "-I", "-B", str(E2 / "tests" / "stub_server.py"), "--port-file",
                                   str(self.pf), "--seconds", "120", *flags])
        for _ in range(200):
            if self.pf.exists() and self.pf.read_text().strip():
                break
            time.sleep(0.05)
        self.port = int(self.pf.read_text())
        self.base = f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.p.kill()
        self.p.wait(timeout=10)


def agent_body() -> dict:
    return {"model": "qwen3.5-9b-q8_0", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0,
            "max_tokens": 16, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "seed": 0}


def agent_error(base: str) -> BaseException:
    llm = StoreLLM(openai.OpenAI(base_url=base, api_key="sk-no-key-required", timeout=10, max_retries=0),
                   "qwen3.5-9b-q8_0", name="local-qwen3.5-9b-q8_0", store=None)
    with pytest.raises(Exception) as ei:
        llm.send(agent_body())
    assert llm.episode_errors and llm.episode_errors[0].startswith(type(ei.value).__name__)  # recorded before raising
    return ei.value


def test_the_stub_500_and_refused_are_transport_and_400_is_the_error_rule(tmp_path):
    s = Stub(tmp_path, "--fail-after", "0", "--fail-status", "500")
    try:
        e = agent_error(s.base)  # after tenacity's 3 tries
        assert isinstance(e, openai.InternalServerError) and is_transport(e)
        with pytest.raises(urllib.error.HTTPError) as hi:
            A3.QwenOptimizer(s.base).propose("goal", [], 4, None)
        assert hi.value.code == 500 and is_transport(hi.value)
    finally:
        s.close()
    s = Stub(tmp_path, "--fail-after", "0", "--fail-status", "400")
    try:
        e = agent_error(s.base)  # BadRequestError: no retry, and the error rule (not a stop)
        assert isinstance(e, openai.BadRequestError) and not is_transport(e)
        with pytest.raises(urllib.error.HTTPError) as hi:
            A3.QwenOptimizer(s.base).propose("goal", [], 4, None)
        assert hi.value.code == 400 and not is_transport(hi.value)
    finally:
        s.close()
    s = Stub(tmp_path, "--refuse-after", "0")
    try:
        e = agent_error(s.base)
        assert isinstance(e, openai.APIConnectionError) and is_transport(e)
        with pytest.raises(Exception) as oi:
            A3.QwenOptimizer(s.base).propose("goal", [], 4, None)
        assert is_transport(oi.value), repr(oi.value)  # connection refused (URLError) or dropped
    finally:
        s.close()


class _Task:
    GOAL = "g"
    ID = "t"


def _patch_attack(monkeypatch):
    monkeypatch.setattr(A3.CandidateAttack, "attack", lambda self, ut, it: {"v": self.text})
    monkeypatch.setattr(A3.CandidateAttack, "__init__", lambda self, s, p: setattr(self, "text", ""))


class _Quiet:
    def run_task_with_pipeline(self, pipeline, ut, it, inj):
        return True, False


def test_a3_optimizer_errors_are_caught_per_iteration_and_transport_stops(monkeypatch):
    _patch_attack(monkeypatch)

    class Flaky:
        def __init__(self, fails: list) -> None:
            self.fails, self.calls = list(fails), []

        def propose(self, goal, history, n, guard_info, use_store=True):
            self.calls.append(1)
            if self.fails:
                raise self.fails.pop(0)
            return ["x", "y"]

    opt = Flaky([urllib.error.HTTPError("http://x", 400, "bad", {}, None), ValueError("malformed reply")])
    res = A3.run_target(_Quiet(), object(), _Task(), _Task(), opt, None, iterations=3)
    notes = [e.get("note") for e in res["episodes"] if e.get("candidate") is None]
    assert notes == ["optimizer error (caught per iteration)"] * 2 and len(opt.calls) == 3
    assert res["episodes"][0]["error"].startswith("HTTPError 400") and not res["no_candidate_episode"]
    assert sum(e.get("candidate") is not None for e in res["episodes"]) == 2  # the third iteration ran its candidates
    res = A3.run_target(_Quiet(), object(), _Task(), _Task(), Flaky([ValueError("x")] * 6), None)
    assert res["no_candidate_episode"] and not res["success"]  # no candidate in 6 iterations: A3 incomplete here
    for exc in (urllib.error.HTTPError("http://x", 502, "x", {}, None), urllib.error.URLError(ConnectionRefusedError())):
        with pytest.raises(StageStopped) as si:
            A3.run_target(_Quiet(), object(), _Task(), _Task(), Flaky([exc]), None)
        assert si.value.where["what"] == "a3 optimizer"

    class Down:
        def run_task_with_pipeline(self, pipeline, ut, it, inj):
            raise openai.APIConnectionError(request=REQ)

    with pytest.raises(StageStopped) as si:
        A3.run_target(Down(), object(), _Task(), _Task(), Flaky([]), None)
    assert si.value.where["what"] == "a3 episode" and si.value.cause_type == "APIConnectionError"


def completion(text: str) -> ChatCompletion:
    return ChatCompletion(id="c", object="chat.completion", created=0, model="m",
                          choices=[Choice(index=0, finish_reason="stop",
                                          message=ChatCompletionMessage(role="assistant", content=text))])


def test_pass_counting_follows_agentdojos_rerun():
    llm = StoreLLM(None, "qwen3.5-9b-q8_0", name="local-qwen3.5-9b-q8_0", store=None)
    llm.send = lambda body, use_store=True: (completion("ok"), False, 0.0)
    sysm = ChatSystemMessage(role="system", content=[text_content_block_from_string("s")])
    user = ChatUserMessage(role="user", content=[text_content_block_from_string("q")])
    asst = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("a")], tool_calls=None)
    rt = FunctionsRuntime([])
    llm.pass_requests = []
    llm.query("q", rt, messages=[sysm, user])  # pass 1, request 1
    llm.query("q", rt, messages=[sysm, user, asst, user])  # pass 1, request 2
    llm.query("q", rt, messages=[sysm, user])  # AgentDojo's rerun: pass 2
    llm.query("q", rt, messages=[sysm, user])  # pass 3
    llm.query("q", rt, messages=[sysm, user, asst, user])
    assert llm.pass_requests == [2, 1, 2]


# ------------------------------------------------------------------ the freeze record, stops and the one rerun

def campaign(tmp_path, monkeypatch) -> Path:
    runs = tmp_path / "runs"
    for name, val in {"RUNS": runs, "STORE": runs / "sealed" / "replay_store.jsonl", "CHAIN": runs / "store_chain.jsonl",
                      "IDENTITY_REF": runs / "server_identity_ref.json", "SMOKE_GATE": runs / "E2-0_gate.json",
                      "BUDGET": runs / "budget.json", "FROZEN": runs / "E2_frozen.json",
                      "STOP_LOG": runs / "stopped_stages.jsonl", "MODELS_FINAL": tmp_path / "models" / "final",
                      "MODELS_PINS": runs / "models_final_pins.json", "ROOT": tmp_path,
                      "E2_0_SUMMARY": runs / "E2-0_summary.json"}.items():
        monkeypatch.setattr(EC, name, val)
    monkeypatch.setattr(EC, "check_pins", lambda pins=None: [])
    monkeypatch.setattr(EC, "check_venvs", list)
    assert EC.begin("smoke", None, skip_server=True) == []  # a fresh campaign
    (runs / "sealed").mkdir(parents=True)
    EC.STORE.write_text('{"key": "e2-0"}\n')
    EC.IDENTITY_REF.write_text(json.dumps({"props": {}, "models": []}))
    rec = EC.end("smoke", str(tmp_path / "run0"))
    (tmp_path / "run0" / "driver").mkdir()  # rev4: E2-0's open meta.json records the code it ran (round-3 m-14)
    (tmp_path / "run0" / "driver" / "meta.json").write_text(json.dumps({"code_sha256": EC.code_hashes()}))
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
    EC.CHAIN.write_text(json.dumps({**rec, "utc": old}) + "\n")  # E2-0 ended two minutes ago
    EC.SMOKE_GATE.write_text(json.dumps({"go": True, "fake": False}))
    EC.BUDGET.write_text(json.dumps({"stop": False, "plan": {"a3_black_box": False}}))
    EC.E2_0_SUMMARY.write_text(json.dumps({"fake": False, "smoke": {"a3_probes": {"calls": [{"n_valid": 4}] * 8}}}))
    return runs


def registration(tmp_path, extra: str = "") -> Path:
    reg = tmp_path / "REGISTRATION_E2.md"
    hashes = [EC.sha256_file(p) for p in (EC.IDENTITY_REF, EC.SMOKE_GATE, EC.BUDGET, EC.E2_0_SUMMARY)] + \
        [EC.sha256_file(E2 / "scripts" / "e2_check.py")]
    reg.write_text("# REGISTRATION E2 rev3 (FROZEN)\n" + extra + "\n".join(f"- `{h}`" for h in hashes) + "\n")
    return reg


def freeze(reg: Path, monkeypatch, back: int = 60) -> int:
    monkeypatch.setattr(sys, "argv", ["freeze_e2.py", "--registration", str(reg)])
    try:
        FZ.main()
    except SystemExit as e:
        return int(e.code or 0)
    fr = json.loads(EC.FROZEN.read_text())  # moved back in time (the test's stand-in for waiting `back` seconds)
    EC.FROZEN.write_text(json.dumps({**fr, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - back))}))
    return 0


def test_freeze_record_is_write_once_and_every_later_stage_needs_it(tmp_path, monkeypatch, capsys):
    campaign(tmp_path, monkeypatch)
    assert EC.begin("d0", None, skip_server=True)[0].startswith("no runs/E2_frozen.json")
    draft = tmp_path / "draft.md"
    draft.write_text("# Registration E2 (NOT FROZEN)\n")
    assert any("NOT FROZEN" in b for b in FZ.problems(draft))
    assert any("sha256 of budget.json" in b for b in FZ.problems(draft))
    assert any("sha256 of E2-0_summary.json" in b for b in FZ.problems(draft))
    reg = registration(tmp_path)
    assert FZ.problems(reg) == []
    assert freeze(reg, monkeypatch) == 0 and EC.FROZEN.exists()
    capsys.readouterr()
    assert EC.begin("d0", None, skip_server=True) == []
    assert f"freeze record ok: runs/E2_frozen.json sha256 {EC.sha256_file(EC.FROZEN)}" in capsys.readouterr().out
    assert freeze(reg, monkeypatch) == 1  # write-once: refused
    with pytest.raises(FileExistsError):
        EC.FROZEN.open("x")
    fr = json.loads(EC.FROZEN.read_text())
    assert fr["registration"] == "REGISTRATION_E2.md" and fr["registration_sha256"] == EC.sha256_file(reg)
    assert fr["e2_0_summary_sha256"] == EC.sha256_file(EC.E2_0_SUMMARY)
    assert EC.check_frozen(now=EC.utc_ts(fr["utc"]) - 1)[0].startswith("the freeze time")  # the stage starts first
    EC.BUDGET.write_text(json.dumps({"stop": False, "plan": {"a3_black_box": True}}))
    assert EC.begin("d0", None, skip_server=True)[0] == "runs/budget.json changed since the freeze (the E2-0 plan and records are final)"
    EC.BUDGET.write_text(json.dumps({"stop": False, "plan": {"a3_black_box": False}}))
    assert EC.begin("d0", None, skip_server=True) == []
    summary = EC.E2_0_SUMMARY.read_text()  # the A3 gate reads the refusal rule from this copy: final at the freeze
    EC.E2_0_SUMMARY.write_text(json.dumps({"fake": False, "smoke": {"a3_probes": {"calls": [{"n_valid": 0}] * 8}}}))
    assert EC.begin("d0", None, skip_server=True)[0] == "runs/E2-0_summary.json changed since the freeze (the E2-0 plan and records are final)"
    EC.E2_0_SUMMARY.write_text(summary)
    assert EC.begin("d0", None, skip_server=True) == []
    reg.write_text(reg.read_text() + "an edit after the freeze\n")
    assert EC.begin("d0", None, skip_server=True)[0].startswith("the registration changed since the freeze")


def test_freeze_refuses_without_the_checkers_hash_or_after_a_later_stage(tmp_path, monkeypatch):
    campaign(tmp_path, monkeypatch)
    reg = tmp_path / "REGISTRATION_E2.md"
    four = (EC.IDENTITY_REF, EC.SMOKE_GATE, EC.BUDGET, EC.E2_0_SUMMARY)
    reg.write_text("# REGISTRATION E2 (FROZEN)\n" + "\n".join(EC.sha256_file(p) for p in four) + "\n")
    assert FZ.problems(reg) == ["the registration does not hold the sha256 of e2_check.py"]
    reg.write_text("# REGISTRATION E2 (FROZEN)\n" + "\n".join(EC.sha256_file(p) for p in four[:3]) + "\n"
                   + EC.sha256_file(E2 / "scripts" / "e2_check.py") + "\n")
    assert FZ.problems(reg) == ["the registration does not hold the sha256 of E2-0_summary.json"]
    EC.E2_0_SUMMARY.unlink()
    assert any("runs/E2-0_summary.json is missing" in b for b in FZ.problems(reg))
    EC.E2_0_SUMMARY.write_text("{}")
    reg = registration(tmp_path)
    EC.end("d0", str(tmp_path / "run1"))  # a later stage already ended: too late to freeze
    assert any("already ran" in b for b in FZ.problems(reg))


def test_freeze_refuses_other_code_or_an_unnamed_earlier_campaign(tmp_path, monkeypatch):
    """Round-3 m-14: the code at the freeze must equal the code E2-0 ran (its open driver/meta.json); m-15: every
    earlier campaign moved aside (runs.aside.*) must be named in the registration."""
    campaign(tmp_path, monkeypatch)
    reg = registration(tmp_path)
    assert FZ.problems(reg) == []
    meta = tmp_path / "run0" / "driver" / "meta.json"
    m = json.loads(meta.read_text())
    k = min(m["code_sha256"])
    m["code_sha256"][k] = "0" * 64
    meta.write_text(json.dumps(m))
    assert FZ.problems(reg) == [f"the code differs from the code E2-0 ran (its driver/meta.json code_sha256): {[k]}"]
    meta.unlink()
    assert FZ.problems(reg)[0].startswith("E2-0's open driver/meta.json is missing")
    meta.write_text(json.dumps({"code_sha256": EC.code_hashes()}))
    assert "e2/run_e2_gpu_slot.sh" in EC.code_hashes() and "e2/scripts/e2_check.py" in EC.code_hashes()
    (tmp_path / "runs.aside.20261101T000000Z").mkdir()
    assert FZ.problems(reg) == ["earlier campaigns not named in the registration (section 15): runs.aside.20261101T000000Z"]
    reg = registration(tmp_path, extra="- runs.aside.20261101T000000Z: E2-0 stopped on a transport error\n")
    assert FZ.problems(reg) == []


def test_freeze_code_check_refuses_explicitly_without_e2_0s_chain_record(tmp_path, monkeypatch):
    """Round-5 R5-3: with an empty store chain (or a first record that is not E2-0's) the code-identity check is never
    skipped silently: it refuses with its own line, beside the 'E2-0 has not ended' refusal."""
    campaign(tmp_path, monkeypatch)
    reg = registration(tmp_path)
    assert FZ.problems(reg) == []
    no_code = "the code cannot be compared with the code E2-0 ran: no store-chain record of stage smoke"
    EC.CHAIN.write_text("")
    bad = FZ.problems(reg)
    assert bad[0].startswith("E2-0 has not ended") and no_code in bad
    EC.CHAIN.write_text(json.dumps({"stage": "d0", "bytes": 0, "sha256": "0" * 64}) + "\n")
    assert no_code in FZ.problems(reg)
    assert freeze(reg, monkeypatch) == 1 and not EC.FROZEN.exists()


def test_e2_0_needs_a_fresh_campaign(tmp_path, monkeypatch):
    """runs/ may hold only the current E2-0 run folder: an earlier campaign's gate or budget plan would otherwise be
    reused (the job file writes them only when runs/budget.json does not exist)."""
    runs = tmp_path / "runs"
    for name, val in {"RUNS": runs, "STORE": runs / "sealed" / "replay_store.jsonl", "CHAIN": runs / "store_chain.jsonl",
                      "IDENTITY_REF": runs / "server_identity_ref.json", "FROZEN": runs / "E2_frozen.json",
                      "STOP_LOG": runs / "stopped_stages.jsonl"}.items():
        monkeypatch.setattr(EC, name, val)
    monkeypatch.setattr(EC, "check_pins", lambda pins=None: [])
    monkeypatch.setattr(EC, "check_venvs", list)
    assert EC.begin("smoke", None, skip_server=True) == []  # no runs/ yet
    (runs / "E2-0" / "20261021T090000Z").mkdir(parents=True)  # the job file's current run folder
    assert EC.begin("smoke", None, skip_server=True) == []
    (runs / "budget.json").write_text("{}")  # an earlier campaign's plan
    bad = EC.begin("smoke", None, skip_server=True)
    assert bad and "earlier campaign (budget.json)" in bad[0]
    (runs / "budget.json").unlink()
    (runs / "E2-0" / "20261021T080000Z").mkdir()  # an earlier E2-0 folder
    assert "E2-0/ holds 2 run folders" in EC.begin("smoke", None, skip_server=True)[0]


def test_a_stopped_stage_is_rerun_once_on_the_same_store(tmp_path, monkeypatch):
    campaign(tmp_path, monkeypatch)
    assert freeze(registration(tmp_path), monkeypatch) == 0
    assert EC.begin("d0", None, skip_server=True) == []
    with EC.STORE.open("a") as f:  # d0 sends some requests, then a transport error stops it (exit 3)
        f.write('{"key": "d0-a"}\n')
    EC.stopped("d0", str(tmp_path / "run1"), 3)
    assert EC.begin("d1", None, skip_server=True)[0].startswith("stage d0 stopped")  # nothing else starts first
    assert EC.begin("d0", None, skip_server=True)[0].startswith("stage d0 stopped")  # a plain start is refused too
    assert EC.begin("d1", None, skip_server=True, rerun=True)[0].startswith("no logged stop of stage d1")
    assert EC.begin("d0", None, skip_server=True, rerun=True) == []  # the one rerun: the store extends the chain
    assert EC.stop_log()[-1]["kind"] == "rerun" and EC.stop_log()[-1]["rerun_of"].endswith("run1")
    with EC.STORE.open("a") as f:
        f.write('{"key": "d0-b"}\n')
    rec = EC.end("d0", str(tmp_path / "run2"))
    assert rec["rerun_of"].endswith("run1")
    assert rec["frozen_sha256"] == EC.sha256_file(EC.FROZEN)  # the freeze this stage started under, on record
    assert "frozen_sha256" not in EC.chain_records()[0]  # E2-0's record precedes the freeze
    assert EC.begin("d1", None, skip_server=True) == []
    EC.stopped("d1", str(tmp_path / "run3"), 3)
    with EC.STORE.open("a") as f:
        f.write('{"key": "stray"}\n')
    assert EC.begin("d1", None, skip_server=True, rerun=True)[0].startswith("the replay store differs from its record")


def test_a_second_stop_leaves_the_stage_incomplete(tmp_path, monkeypatch):
    campaign(tmp_path, monkeypatch)
    assert freeze(registration(tmp_path), monkeypatch) == 0
    EC.stopped("d0", str(tmp_path / "run1"), 3)
    assert EC.begin("d0", None, skip_server=True, rerun=True) == []
    EC.stopped("d0", str(tmp_path / "run2"), 3)  # the rerun stopped too
    assert EC.begin("d0", None, skip_server=True, rerun=True)[0] == "stage d0 was rerun once already: it stays incomplete"
    assert "stays incomplete; E2 stops here" in EC.begin("d1", None, skip_server=True)[0]
    assert EC.begin("smoke", None, skip_server=True, rerun=True)[0].startswith("E2-0 is never rerun on its campaign")


# ------------------------------------------------------------------ the driver's plan and E2-0's open row

def test_driver_a3_plan_and_e2_0_open_row(tmp_path):
    import run_e2_gpu as G

    g = tmp_path / "gate.json"
    g.write_text(json.dumps({"plan": [{"arm": "d4_tm", "variants": ["guard_aware", "black_box"]},
                                      {"arm": "d3r", "variants": ["guard_aware", "black_box"]},
                                      {"arm": "d6_rules", "variants": ["guard_aware"]}]}))
    ns = argparse.Namespace(gate=str(g), arms=None, a3_variants=["guard_aware"])
    assert G.a3_plan(ns) == {"d4_tm": ["guard_aware", "black_box"], "d3r": ["guard_aware", "black_box"],
                             "d6_rules": ["guard_aware"]}
    g.write_text(json.dumps({"plan": [{"arm": "d6_rules", "variants": ["guard_aware", "black_box"]}]}))
    with pytest.raises(SystemExit):
        G.a3_plan(ns)  # black-box never runs on D6
    with pytest.raises(SystemExit):
        G.a3_plan(argparse.Namespace(gate=None, arms=["d6_rules"], a3_variants=["black_box"]))
    row = {"stage": "E2-0", "arm": "w_only", "suite": "banking", "user_task": "user_task_0",
           "injection_task": "injection_task_7", "attack": "important_instructions", "utility": True, "security": False,
           "error": "RuntimeError: x", "error_type": "RuntimeError", "request_errors": ["BadRequestError: ctx"],
           "requests": 48, "passes": [16, 16, 16], "store_hits": 3, "store_misses": 45, "blocked_calls": 42,
           "guard_decisions": 45, "seconds": 2.0, "prompt_tokens": 1, "completion_tokens": 1, "utc": "x"}
    o = G.open_row(row)
    assert set(o) == {"stage", "arm", "suite", "user_task", "injection_task", "attack", "utc", "error_type",
                      "request_error_types", "outcome"} and o["request_error_types"] == ["BadRequestError"]
    loop = [{"blocked": 3, "new_requests": 14, "d0_requests": 4}, {"blocked": 0, "new_requests": 0, "d0_requests": 9}]
    assert G.closed_loop_ratio(loop)["ratio"] == 3.5  # vetoed episodes only: 14 / 4
    assert G.closed_loop_ratio([{"blocked": 0, "new_requests": 0, "d0_requests": 5}])["ratio"] == 1.0
    ps = G.passes_summary([{"passes": [16, 16, 16]}, {"passes": [3]}, {"passes": [16, 2]}])
    assert ps["passes_at_turn_cap"] == 4 and ps["episodes_with_a_pass_at_turn_cap"] == 2
    assert ps["passes_per_episode"] == {"1": 1, "2": 1, "3": 1}
