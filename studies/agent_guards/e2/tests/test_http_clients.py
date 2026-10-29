"""The HTTP paths of the D6 rule writer, the A3 optimizer and the driver's server check, against an in-process stub
on an ephemeral 127.0.0.1 port (not an LLM; closed at the end of each test)."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(E2 / "scripts"))

from e2guard.attacks_a3 import QwenOptimizer, guard_description
from e2guard.d6_actgov import QwenProposer


def stub(models=("qwen3.5-9b-q8_0",), health="ok"):
    seen: list[dict] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": health})
            elif self.path == "/v1/models":
                self._send({"object": "list", "data": [{"id": m} for m in models]})
            elif self.path == "/props":
                self._send({"build_info": "stub", "total_slots": 1, "default_generation_settings": {"n_ctx": 65536}})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            name = body["response_format"]["json_schema"]["name"]
            content = json.dumps({"rules": [["kind=read"]]} if name == "rules" else {"candidates": ["t1", "t2", " "]})
            self._send({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", seen


def test_rule_writer_and_optimizer_requests():
    srv, base, seen = stub()
    try:
        p = QwenProposer(base, model="alias-x")
        assert json.loads(p.propose([{"role": "user", "content": "hi"}])) == {"rules": [["kind=read"]]}
        o = QwenOptimizer(base, model="alias-x")
        info = guard_description("d4_tm", "banking")
        assert o.propose("goal", [("old", False)], 4, info) == ["t1", "t2"]  # blank candidate dropped
        b1, b2 = seen
        assert b1["model"] == b2["model"] == "alias-x"
        assert b1["temperature"] == 0.0 and b1["seed"] == 0 and b1["chat_template_kwargs"] == {"enable_thinking": False}
        assert b1["response_format"]["json_schema"]["name"] == "rules"
        assert b2["temperature"] == 0.7 and b2["seed"] == 0 and "checks every tool call before it runs" in b2["messages"][1]["content"]
        assert p.calls[0]["usage"]["prompt_tokens"] == 7 and o.calls[0]["n_valid"] == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_driver_server_check_accepts_and_refuses():
    import run_e2_gpu as G

    srv, base, _ = stub()
    try:
        r = G.server_check(base, "qwen3.5-9b-q8_0")
        assert r["health"]["status"] == "ok" and r["props"]["n_ctx"] == 65536
        with pytest.raises(SystemExit):
            G.server_check(base, "some-other-alias")  # the model is not listed
    finally:
        srv.shutdown()
        srv.server_close()
    srv, base, _ = stub(health="loading model")
    try:
        with pytest.raises(SystemExit):
            G.server_check(base, "qwen3.5-9b-q8_0")  # not healthy
    finally:
        srv.shutdown()
        srv.server_close()
